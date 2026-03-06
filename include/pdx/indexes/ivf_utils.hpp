#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "pdx/clustering.hpp"
#include "pdx/common.hpp"
#include "pdx/indexes/ivf_core.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"

#include <omp.h>

namespace PDX {

struct PDXIndexConfig {
    uint32_t num_dimensions;
    DistanceMetric distance_metric = DistanceMetric::L2SQ;
    uint32_t seed = 42;
    uint32_t num_clusters = 0; // 0 = auto-compute from num_embeddings
    uint32_t num_meso_clusters = 0;
    bool normalize = false;
    float sampling_fraction = 0.0f; // 0 = auto (1.0 if small dataset, 0.3 otherwise)
    uint32_t kmeans_iters = 10;
    bool hierarchical_indexing = true;
    uint32_t n_threads = 0; // 0 = omp_get_max_threads()

    void Validate() const {
        if (num_dimensions == 0 || num_dimensions > PDX_MAX_DIMS) {
            throw std::invalid_argument(
                "num_dimensions must be between 1 and " + std::to_string(PDX_MAX_DIMS) + ", got " +
                std::to_string(num_dimensions)
            );
        }
        if (sampling_fraction < 0.0f || sampling_fraction > 1.0f) {
            throw std::invalid_argument(
                "sampling_fraction must be between 0.0 and 1.0, got " +
                std::to_string(sampling_fraction)
            );
        }
        if (num_meso_clusters > 0 && num_clusters > 0 && num_meso_clusters >= num_clusters) {
            throw std::invalid_argument(
                "num_meso_clusters (" + std::to_string(num_meso_clusters) +
                ") must be smaller than num_clusters (" + std::to_string(num_clusters) + ")"
            );
        }
        if (kmeans_iters == 0 || kmeans_iters >= 100) {
            throw std::invalid_argument(
                "kmeans_iters must be between 1 and 99, got " + std::to_string(kmeans_iters)
            );
        }
    }

    void ValidateNumEmbeddings(size_t num_embeddings) const {
        if (num_clusters > 0 && num_clusters > num_embeddings) {
            throw std::invalid_argument(
                "num_clusters (" + std::to_string(num_clusters) + ") exceeds num_embeddings (" +
                std::to_string(num_embeddings) + ")"
            );
        }
    }
};

inline std::unique_ptr<float[]> NormalizeAndRotate(
    const float* embeddings,
    size_t num_embeddings,
    uint32_t num_dimensions,
    bool normalize,
    const ADSamplingPruner& pruner
) {
    const size_t total_floats = num_embeddings * num_dimensions;
    std::unique_ptr<float[]> normalized;
    const float* rotation_input = embeddings;
    if (normalize) {
        normalized.reset(new float[total_floats]);
        Quantizer quantizer(num_dimensions);
#pragma omp parallel for if (num_embeddings > 1) num_threads(PDX::g_n_threads)
        for (size_t i = 0; i < num_embeddings; i++) {
            quantizer.NormalizeQuery(
                embeddings + i * num_dimensions, normalized.get() + i * num_dimensions
            );
        }
        rotation_input = normalized.get();
    }
    std::unique_ptr<float[]> preprocessed(new float[total_floats]);
    pruner.PreprocessEmbeddings(rotation_input, preprocessed.get(), num_embeddings);
    return preprocessed;
}

// Store the embeddings into this cluster's preallocated buffers in the transposed PDX layout.
//
// See the README of the following for a description of the PDX layout:
// https://github.com/cwida/pdx
template <PDX::Quantization q, typename T>
inline void StoreClusterEmbeddings(
    typename PDX::IVF<q>::cluster_t& cluster,
    const PDX::IVF<q>& index,
    const T* embeddings,
    const size_t num_embeddings
);

template <>
inline void StoreClusterEmbeddings<PDX::Quantization::F32, float>(
    PDX::IVF<PDX::Quantization::F32>::cluster_t& cluster,
    const PDX::IVF<PDX::Quantization::F32>& index,
    const float* const embeddings,
    const size_t num_embeddings
) {
    using matrix_t = PDX::eigen_matrix_t;
    using h_matrix_t = Eigen::Matrix<float, Eigen::Dynamic, PDX::H_DIM_SIZE, Eigen::RowMajor>;

    const auto vertical_d = index.num_vertical_dimensions;
    const auto horizontal_d = index.num_horizontal_dimensions;
    const auto stride = static_cast<Eigen::Index>(cluster.max_capacity);

    Eigen::Map<const matrix_t> in(embeddings, num_embeddings, index.num_dimensions);

    // Vertical block: (vertical_d x num_embeddings) with row stride = max_capacity
    Eigen::Map<matrix_t, 0, Eigen::OuterStride<Eigen::Dynamic>> out(
        cluster.data, vertical_d, num_embeddings, Eigen::OuterStride<Eigen::Dynamic>(stride)
    );
    out.noalias() = in.leftCols(vertical_d).transpose();

    float* horizontal_out = cluster.data + stride * vertical_d;
    for (size_t j = 0; j < horizontal_d; j += PDX::H_DIM_SIZE) {
        Eigen::Map<h_matrix_t> out_h(horizontal_out, num_embeddings, PDX::H_DIM_SIZE);
        out_h.noalias() = in.block(0, vertical_d + j, num_embeddings, PDX::H_DIM_SIZE);
        horizontal_out += stride * PDX::H_DIM_SIZE;
    }
}

template <>
inline void StoreClusterEmbeddings<PDX::Quantization::U8, uint8_t>(
    PDX::IVF<PDX::Quantization::U8>::cluster_t& cluster,
    const PDX::IVF<PDX::Quantization::U8>& index,
    const uint8_t* const embeddings,
    const size_t num_embeddings
) {
    using u8_matrix_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using u8_v_matrix_t =
        Eigen::Matrix<uint8_t, Eigen::Dynamic, PDX::U8_INTERLEAVE_SIZE, Eigen::RowMajor>;
    using u8_h_matrix_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, PDX::H_DIM_SIZE, Eigen::RowMajor>;

    const auto vertical_d = index.num_vertical_dimensions;
    const auto horizontal_d = index.num_horizontal_dimensions;
    const auto stride = static_cast<size_t>(cluster.max_capacity);

    Eigen::Map<const u8_matrix_t> in(embeddings, num_embeddings, index.num_dimensions);

    size_t dim = 0;
    for (; dim + PDX::U8_INTERLEAVE_SIZE <= vertical_d; dim += PDX::U8_INTERLEAVE_SIZE) {
        Eigen::Map<u8_v_matrix_t> out_v(
            cluster.data + dim * stride, num_embeddings, PDX::U8_INTERLEAVE_SIZE
        );
        out_v.noalias() = in.block(0, dim, num_embeddings, PDX::U8_INTERLEAVE_SIZE);
    }
    if (dim < vertical_d) {
        auto remaining = static_cast<Eigen::Index>(vertical_d - dim);
        Eigen::Map<u8_matrix_t> out_v(cluster.data + dim * stride, num_embeddings, remaining);
        out_v.noalias() = in.block(0, dim, num_embeddings, remaining);
    }

    uint8_t* horizontal_out = cluster.data + stride * vertical_d;
    for (size_t j = 0; j < horizontal_d; j += PDX::H_DIM_SIZE) {
        Eigen::Map<u8_h_matrix_t> out_h(horizontal_out, num_embeddings, PDX::H_DIM_SIZE);
        out_h.noalias() = in.block(0, vertical_d + j, num_embeddings, PDX::H_DIM_SIZE);
        horizontal_out += stride * PDX::H_DIM_SIZE;
    }
}

template <Quantization Q>
void PopulateIVFClusters(
    IVF<Q>& ivf,
    const KMeansResult& kmeans_result,
    const float* source_data,
    const size_t* row_ids,
    uint32_t num_dimensions,
    uint32_t num_clusters,
    float quantization_base,
    float quantization_scale
) {
    using storage_t = pdx_data_t<Q>;

    size_t max_cluster_size = 0;
    for (size_t i = 0; i < num_clusters; i++) {
        max_cluster_size = std::max(max_cluster_size, kmeans_result.assignments[i].size());
    }

    // Pre-allocate all clusters sequentially
    for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
        ivf.clusters.emplace_back(kmeans_result.assignments[cluster_idx].size(), num_dimensions);
        ivf.clusters[cluster_idx].id = cluster_idx;
    }

    // Per-thread tmp buffers for gather + quantize
    const uint32_t n_threads = PDX::g_n_threads;
    std::vector<std::unique_ptr<storage_t[]>> tmp_buffers(n_threads);
    for (uint32_t t = 0; t < n_threads; t++) {
        tmp_buffers[t].reset(new storage_t[static_cast<uint64_t>(max_cluster_size) * num_dimensions]
        );
    }

#pragma omp parallel for num_threads(n_threads)
    for (size_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
        const auto cluster_size = kmeans_result.assignments[cluster_idx].size();
        auto& cluster = ivf.clusters[cluster_idx];
        auto* tmp = tmp_buffers[omp_get_thread_num()].get();

        for (size_t pos = 0; pos < cluster_size; pos++) {
            const auto emb_idx = kmeans_result.assignments[cluster_idx][pos];
            cluster.indices[pos] = row_ids[emb_idx];

            if constexpr (Q == U8) {
                ScalarQuantizer<Q> quantizer(num_dimensions);
                quantizer.QuantizeEmbedding(
                    source_data + (emb_idx * num_dimensions),
                    quantization_base,
                    quantization_scale,
                    tmp + (pos * num_dimensions)
                );
            } else {
                std::memcpy(
                    tmp + (pos * num_dimensions),
                    source_data + (emb_idx * num_dimensions),
                    num_dimensions * sizeof(float)
                );
            }
        }
        StoreClusterEmbeddings<Q, storage_t>(cluster, ivf, tmp, cluster_size);
    }

    ivf.ComputeClusterOffsets();
}

} // namespace PDX
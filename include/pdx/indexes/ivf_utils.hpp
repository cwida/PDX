#pragma once

#include <algorithm>
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
#include <vector>

#include "pdx/clustering.hpp"
#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/indexes/ivf_core.hpp"
#include "pdx/profiler.hpp"
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
    PDX_PROFILE_SCOPE("Search/NormalizeAndRotate");
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

// ******************************************
// Maintenance helpers (SPFresh-like Append/Delete), shared by PDXIndex and PDXTreeIndex.
// Everything here works on leaf clusters and knows nothing about meso-clusters.
// ******************************************

// Dequantize raw (Q-type) embeddings to float. For F32 this is a memcpy.
template <Quantization Q>
inline std::unique_ptr<float[]> DequantizeClusterEmbeddings(
    const IVF<Q>& index,
    ScalarQuantizer<Q>& quantizer,
    const pdx_data_t<Q>* raw_embeddings,
    uint32_t n_emb
) {
    PDX_PROFILE_SCOPE("Dequantize");
    const size_t d = index.num_dimensions;
    std::unique_ptr<float[]> result(new float[static_cast<size_t>(n_emb) * d]);
    if constexpr (Q == U8) {
        for (size_t i = 0; i < n_emb; i++) {
            quantizer.DequantizeEmbedding(
                raw_embeddings + i * d,
                index.quantization_base,
                index.quantization_scale,
                result.get() + i * d
            );
        }
    } else {
        std::memcpy(result.get(), raw_embeddings, static_cast<size_t>(n_emb) * d * sizeof(float));
    }
    return result;
}

// Quantize (if U8) and append a float embedding to a cluster. Returns its index in the cluster.
template <Quantization Q>
inline uint32_t QuantizeAndAppend(
    const IVF<Q>& index,
    ScalarQuantizer<Q>& quantizer,
    Cluster<Q>& cluster,
    uint32_t row_id,
    const float* embedding
) {
    if constexpr (Q == U8) {
        std::unique_ptr<pdx_data_t<Q>[]> quantized(new pdx_data_t<Q>[index.num_dimensions]);
        quantizer.QuantizeEmbedding(
            embedding, index.quantization_base, index.quantization_scale, quantized.get()
        );
        return cluster.AppendEmbedding(row_id, quantized.get());
    } else {
        return cluster.AppendEmbedding(row_id, embedding);
    }
}

// Gather the raw embeddings and row ids at the given positions of a cluster, accumulating
// their (float) sum into centroid_sum.
template <Quantization Q>
inline void GatherGroupEmbeddings(
    const IVF<Q>& index,
    const Cluster<Q>& cluster,
    const std::vector<uint32_t>& group_idx,
    const pdx_data_t<Q>* raw_embeddings,
    const float* float_embeddings,
    std::vector<pdx_data_t<Q>>& embs_out,
    std::vector<uint32_t>& ids_out,
    float* centroid_sum
) {
    const size_t d = index.num_dimensions;
    for (uint32_t idx : group_idx) {
        embs_out.insert(
            embs_out.end(),
            raw_embeddings + static_cast<size_t>(idx) * d,
            raw_embeddings + (static_cast<size_t>(idx) + 1) * d
        );
        ids_out.push_back(cluster.indices[idx]);
        const float* emb_f = float_embeddings + static_cast<size_t>(idx) * d;
        for (size_t j = 0; j < d; j++) {
            centroid_sum[j] += emb_f[j];
        }
    }
}

// Mean centroid from an accumulated sum; falls back to `fallback` when count == 0.
// Re-normalized when the index stores normalized vectors.
inline void ComputeCentroidMean(
    uint32_t num_dimensions,
    bool normalize,
    const float* centroid_sum,
    size_t count,
    const float* fallback,
    float* output
) {
    if (count == 0) {
        std::memcpy(output, fallback, num_dimensions * sizeof(float));
    } else {
        float inv = 1.0f / static_cast<float>(count);
        PDX_VECTORIZE_LOOP
        for (size_t j = 0; j < num_dimensions; j++) {
            output[j] = centroid_sum[j] * inv;
        }
    }
    if (normalize) {
        Quantizer q(num_dimensions);
        q.NormalizeQuery(output, output);
    }
}

// Distances from a (quantized) embedding to every used slot of a cluster, computed straight on
// the PDX layout (vertical block, then 64-dim horizontal blocks). Tombstoned slots are garbage.
template <Quantization Q>
inline std::unique_ptr<pdx_distance_t<Q>[]> CalculateDistanceFromEmbeddingToCluster(
    const IVF<Q>& index,
    const pdx_quantized_embedding_t<Q>* embedding,
    const Cluster<Q>& cluster
) {
    PDX_PROFILE_SCOPE("Split/CalculatePDXDistance");
    using distance_computer_t = DistanceComputer<DistanceMetric::L2SQ, Q>;
    using distance_t = pdx_distance_t<Q>;

    const size_t n_vectors = cluster.used_capacity;
    const size_t buffer_stride = cluster.max_capacity;
    // Vertical() accumulates, so the distances must start zeroed
    std::unique_ptr<distance_t[]> pruning_distances = std::make_unique<distance_t[]>(n_vectors);
    std::unique_ptr<uint32_t[]> pruning_positions(new uint32_t[n_vectors]);
    distance_computer_t::Vertical(
        embedding,
        cluster.data,
        n_vectors,
        buffer_stride,
        0,
        index.num_vertical_dimensions,
        pruning_distances.get(),
        pruning_positions.get()
    );
    const size_t vertical_block_size =
        static_cast<size_t>(index.num_vertical_dimensions) * buffer_stride;
    for (size_t horizontal_dimension = 0; horizontal_dimension < index.num_horizontal_dimensions;
         horizontal_dimension += H_DIM_SIZE) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; vector_idx++) {
            size_t data_pos = vertical_block_size + (horizontal_dimension * buffer_stride) +
                              (vector_idx * H_DIM_SIZE);
            pruning_distances[vector_idx] += distance_computer_t::Horizontal(
                embedding + index.num_vertical_dimensions + horizontal_dimension,
                cluster.data + data_pos,
                H_DIM_SIZE
            );
        }
    }
    return pruning_distances;
}

struct SplitPartition {
    std::unique_ptr<float[]> centroid_a;
    std::unique_ptr<float[]> centroid_b;
    std::vector<uint32_t> group_a_idx;
    std::vector<uint32_t> group_b_idx;
    std::vector<uint32_t> group_rest_idx; // closer to a neighboring cluster than to A or B
};

// 2-means over the (float) embeddings of a cluster that is about to split. Every position lands
// in group A, group B, or "rest" when one of the neighboring clusters' centroids is closer than
// both A and B (the caller reassigns those).
template <Quantization Q>
inline SplitPartition PartitionClusterForSplit(
    const IVF<Q>& index,
    const float* cluster_embeddings,
    uint32_t num_embeddings,
    const float* centroid_to_split,
    const std::vector<uint32_t>& neighboring_clusters_ids,
    DistanceMetric distance_metric,
    uint32_t seed
) {
    using distance_computer_f32_t = DistanceComputer<DistanceMetric::L2SQ, F32>;
    const size_t d = index.num_dimensions;
    SplitPartition partition;
    partition.centroid_a.reset(new float[d]);
    partition.centroid_b.reset(new float[d]);
    {
        PDX_PROFILE_SCOPE("Split/KMeans");
        KMeansResult split_result = ComputeKMeans(
            cluster_embeddings,
            num_embeddings,
            d,
            2,
            distance_metric,
            seed,
            true,
            1.0f,
            SPLIT_KMEANS_ITERS,
            false,
            1
        );
        std::memcpy(partition.centroid_a.get(), split_result.centroids.data(), d * sizeof(float));
        std::memcpy(
            partition.centroid_b.get(), split_result.centroids.data() + d, d * sizeof(float)
        );
        partition.group_a_idx.reserve(split_result.assignments[0].size());
        partition.group_b_idx.reserve(split_result.assignments[1].size());
    }

    // Assign each embedding to A, B, or rest (closer elsewhere)
    {
        PDX_PROFILE_SCOPE("Split/Partition");
        const float* centroid_a = partition.centroid_a.get();
        const float* centroid_b = partition.centroid_b.get();
        for (size_t i = 0; i < num_embeddings; i++) {
            const float* emb = cluster_embeddings + i * d;
            float dist_old = distance_computer_f32_t::Horizontal(emb, centroid_to_split, d);
            // TODO(@lkuffo, med): We could avoid one of these
            // since we have the distance from k-means, we just need to bring it here
            float dist_a = distance_computer_f32_t::Horizontal(emb, centroid_a, d);
            float dist_b = distance_computer_f32_t::Horizontal(emb, centroid_b, d);
            float min_ab = std::min(dist_a, dist_b);
            auto& group_ab = dist_a <= dist_b ? partition.group_a_idx : partition.group_b_idx;
            if (min_ab <= dist_old) {
                group_ab.push_back(i);
                continue;
            }
            bool closer_elsewhere = false;
            for (uint32_t c : neighboring_clusters_ids) {
                float dist = distance_computer_f32_t::Horizontal(
                    emb, index.centroids.data() + static_cast<size_t>(c) * d, d
                );
                if (dist < min_ab) {
                    closer_elsewhere = true;
                    break;
                }
            }
            if (closer_elsewhere) {
                partition.group_rest_idx.push_back(i);
            } else {
                group_ab.push_back(i);
            }
        }
    }
    return partition;
}

// Steal from the neighboring clusters the embeddings that are closer to centroid A or B than to
// their own centroid: they are tombstoned in the neighbor and appended to the A/B groups (raw
// embeddings, row ids, centroid sums). The caller must refresh the row-id mapping of the moved
// ids once the new clusters exist.
template <Quantization Q>
inline void StealNeighborEmbeddings(
    IVF<Q>& index,
    ScalarQuantizer<Q>& quantizer,
    const std::vector<uint32_t>& neighboring_clusters_ids,
    const float* centroid_a,
    const float* centroid_b,
    std::vector<pdx_data_t<Q>>& embs_a,
    std::vector<uint32_t>& ids_a,
    float* centroid_sum_a,
    std::vector<pdx_data_t<Q>>& embs_b,
    std::vector<uint32_t>& ids_b,
    float* centroid_sum_b
) {
    PDX_PROFILE_SCOPE("Split/NeighborReassign");
    using query_t = pdx_quantized_embedding_t<Q>;
    using distance_t = pdx_distance_t<Q>;
    const size_t d = index.num_dimensions;
    for (uint32_t neighbor_id : neighboring_clusters_ids) {
        auto& neighbor = index.clusters[neighbor_id];
        const float* neighbor_centroid =
            index.centroids.data() + static_cast<size_t>(neighbor_id) * d;

        // Quantize centroids for U8, or use directly for F32
        std::unique_ptr<query_t[]> q_own, q_a, q_b;
        const query_t* query_own;
        const query_t* query_a;
        const query_t* query_b;
        if constexpr (Q == U8) {
            q_own.reset(new query_t[d]);
            q_a.reset(new query_t[d]);
            q_b.reset(new query_t[d]);
            quantizer.QuantizeEmbedding(
                neighbor_centroid, index.quantization_base, index.quantization_scale, q_own.get()
            );
            quantizer.QuantizeEmbedding(
                centroid_a, index.quantization_base, index.quantization_scale, q_a.get()
            );
            quantizer.QuantizeEmbedding(
                centroid_b, index.quantization_base, index.quantization_scale, q_b.get()
            );
            query_own = q_own.get();
            query_a = q_a.get();
            query_b = q_b.get();
        } else {
            query_own = neighbor_centroid;
            query_a = centroid_a;
            query_b = centroid_b;
        }

        auto distances_to_own =
            CalculateDistanceFromEmbeddingToCluster<Q>(index, query_own, neighbor);
        auto distances_to_a = CalculateDistanceFromEmbeddingToCluster<Q>(index, query_a, neighbor);
        auto distances_to_b = CalculateDistanceFromEmbeddingToCluster<Q>(index, query_b, neighbor);

        for (uint32_t p = 0; p < neighbor.used_capacity; p++) {
            if (neighbor.HasTombstone(p))
                continue;

            distance_t dist_a = distances_to_a[p];
            distance_t dist_b = distances_to_b[p];
            distance_t dist_to_own = distances_to_own[p];

            if (dist_to_own < dist_a && dist_to_own < dist_b) {
                continue;
            }

            // We need the horizontal embedding (this happens in less than 1% of points)
            auto raw_emb = neighbor.GetHorizontalEmbeddingFromPDXBuffer(p);
            const float* emb_ptr;
            std::unique_ptr<float[]> emb_f32;
            if constexpr (Q == U8) {
                emb_f32.reset(new float[d]);
                quantizer.DequantizeEmbedding(
                    raw_emb.get(), index.quantization_base, index.quantization_scale, emb_f32.get()
                );
                emb_ptr = emb_f32.get();
            } else {
                emb_ptr = raw_emb.get();
            }

            const uint32_t row_id = neighbor.indices[p];
            neighbor.DeleteEmbedding(p);
            const bool goes_to_a = dist_a <= dist_b;
            auto& embs = goes_to_a ? embs_a : embs_b;
            auto& ids = goes_to_a ? ids_a : ids_b;
            float* centroid_sum = goes_to_a ? centroid_sum_a : centroid_sum_b;
            embs.insert(embs.end(), raw_emb.get(), raw_emb.get() + d);
            ids.push_back(row_id);
            for (size_t j = 0; j < d; j++) {
                centroid_sum[j] += emb_ptr[j];
            }
        }
    }
}

} // namespace PDX
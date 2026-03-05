#pragma once

#include "pdx/cluster.hpp"
#include "pdx/common.hpp"
#include "pdx/utils.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <vector>

namespace PDX {

template <Quantization Q>
class IVF {
  public:
    using cluster_t = Cluster<Q>;
    using data_t = pdx_data_t<Q>;

    uint32_t num_dimensions{};
    uint64_t total_num_embeddings{};
    uint32_t num_clusters{};
    uint32_t num_vertical_dimensions{};
    uint32_t num_horizontal_dimensions{};
    std::vector<cluster_t> clusters;
    size_t max_cluster_capacity{0};
    size_t total_capacity{0};
    std::unique_ptr<size_t[]> cluster_offsets;
    bool is_normalized{};
    std::vector<float> centroids;

    // U8-specific quantization parameters
    float quantization_scale = 1.0f;
    float quantization_scale_squared = 1.0f;
    float inverse_quantization_scale_squared = 1.0f;
    float quantization_base = 0.0f;

    IVF() = default;
    ~IVF() = default;
    IVF(IVF&&) = default;
    IVF& operator=(IVF&&) = default;

    IVF(uint32_t num_dimensions,
        uint64_t total_num_embeddings,
        uint32_t num_clusters,
        bool is_normalized)
        : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings),
          num_clusters(num_clusters),
          num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
          num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions),
          is_normalized(is_normalized) {
        clusters.reserve(num_clusters);
    }

    IVF(uint32_t num_dimensions,
        uint64_t total_num_embeddings,
        uint32_t num_clusters,
        bool is_normalized,
        float quantization_scale,
        float quantization_base)
        : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings),
          num_clusters(num_clusters),
          num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
          num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions),
          is_normalized(is_normalized), quantization_scale(quantization_scale),
          quantization_scale_squared(quantization_scale * quantization_scale),
          inverse_quantization_scale_squared(1.0f / (quantization_scale * quantization_scale)),
          quantization_base(quantization_base) {
        clusters.reserve(num_clusters);
    }

    // Compute cluster_offsets, total_capacity, and max_cluster_capacity from current clusters.
    // Must be called after all clusters have been created or after structural changes
    // (split/merge).
    void ComputeClusterOffsets() {
        PDX_PROFILE_SCOPE("ComputeClusterOffsets");
        cluster_offsets.reset(new size_t[num_clusters]);
        total_capacity = 0;
        max_cluster_capacity = 0;
        for (size_t i = 0; i < num_clusters; ++i) {
            cluster_offsets[i] = total_capacity;
            total_capacity += clusters[i].max_capacity;
            max_cluster_capacity =
                std::max(max_cluster_capacity, static_cast<size_t>(clusters[i].max_capacity));
        }
    }

    void Load(char* input) {
        char* next_value = input;
        num_dimensions = ((uint32_t*) input)[0];
        num_vertical_dimensions = ((uint32_t*) input)[1];
        num_horizontal_dimensions = ((uint32_t*) input)[2];

        next_value += sizeof(uint32_t) * 3;
        num_clusters = ((uint32_t*) next_value)[0];
        next_value += sizeof(uint32_t);
        auto* cluster_headers = (uint32_t*) next_value;
        next_value += num_clusters * 2 * sizeof(uint32_t);
        clusters.reserve(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            uint32_t n_emb = cluster_headers[i * 2];
            uint32_t max_cap = cluster_headers[i * 2 + 1];
            clusters.emplace_back(n_emb, max_cap, num_dimensions);
            clusters[i].id = i;
            clusters[i].LoadPDXData(next_value);
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            memcpy(clusters[i].indices, next_value, sizeof(uint32_t) * clusters[i].num_embeddings);
            next_value += sizeof(uint32_t) * clusters[i].num_embeddings;
        }

        is_normalized = ((char*) next_value)[0];
        next_value += sizeof(char);

        centroids.resize(num_clusters * num_dimensions);
        memcpy(
            centroids.data(), (float*) next_value, sizeof(float) * num_clusters * num_dimensions
        );
        next_value += sizeof(float) * num_clusters * num_dimensions;

        if constexpr (Q == U8) {
            quantization_base = ((float*) next_value)[0];
            next_value += sizeof(float);
            quantization_scale = ((float*) next_value)[0];
            next_value += sizeof(float);
            quantization_scale_squared = quantization_scale * quantization_scale;
            inverse_quantization_scale_squared = 1.0f / quantization_scale_squared;
        }
        ComputeClusterOffsets();
    }

    void Save(std::ostream& out) const {
        out.write(reinterpret_cast<const char*>(&num_dimensions), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&num_vertical_dimensions), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&num_horizontal_dimensions), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&num_clusters), sizeof(uint32_t));

        for (size_t i = 0; i < num_clusters; ++i) {
            out.write(reinterpret_cast<const char*>(&clusters[i].num_embeddings), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&clusters[i].max_capacity), sizeof(uint32_t));
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            clusters[i].SavePDXData(out);
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            out.write(
                reinterpret_cast<const char*>(clusters[i].indices),
                sizeof(uint32_t) * clusters[i].num_embeddings
            );
        }

        char norm = is_normalized;
        out.write(&norm, sizeof(char));

        out.write(
            reinterpret_cast<const char*>(centroids.data()),
            sizeof(float) * num_clusters * num_dimensions
        );

        if constexpr (Q == U8) {
            out.write(reinterpret_cast<const char*>(&quantization_base), sizeof(float));
            out.write(reinterpret_cast<const char*>(&quantization_scale), sizeof(float));
        }
    }

    size_t GetInMemorySizeInBytes() const {
        size_t in_memory_size_in_bytes = 0;
        in_memory_size_in_bytes += sizeof(*this);
        for (const auto& cluster : clusters) {
            in_memory_size_in_bytes += cluster.GetInMemorySizeInBytes();
        }
        in_memory_size_in_bytes +=
            (clusters.capacity() - clusters.size()) * sizeof(*clusters.data());
        in_memory_size_in_bytes += centroids.capacity() * sizeof(*centroids.data());
        in_memory_size_in_bytes += num_clusters * sizeof(size_t); // cluster_offsets
        return in_memory_size_in_bytes;
    }
};

template <Quantization Q>
class IVFTree : public IVF<Q> {
  public:
    using data_t = pdx_data_t<Q>;

    IVF<F32> l0; // Meso clusters

    IVFTree() = default;
    ~IVFTree() = default;
    IVFTree(IVFTree&&) = default;
    IVFTree& operator=(IVFTree&&) = default;

    IVFTree(
        uint32_t num_dimensions,
        uint64_t total_num_embeddings,
        uint32_t num_clusters,
        bool is_normalized
    )
        : IVF<Q>(num_dimensions, total_num_embeddings, num_clusters, is_normalized) {}

    IVFTree(
        uint32_t num_dimensions,
        uint64_t total_num_embeddings,
        uint32_t num_clusters,
        bool is_normalized,
        float quantization_scale,
        float quantization_base
    )
        : IVF<Q>(
              num_dimensions,
              total_num_embeddings,
              num_clusters,
              is_normalized,
              quantization_scale,
              quantization_base
          ) {}

    void Load(char* input) {
        char* next_value = input;

        // Header
        uint32_t dims = ((uint32_t*) input)[0];
        uint32_t v_dims = ((uint32_t*) input)[1];
        uint32_t h_dims = ((uint32_t*) input)[2];
        next_value += sizeof(uint32_t) * 3;

        uint32_t n_clusters_l1 = ((uint32_t*) next_value)[0];
        next_value += sizeof(uint32_t);
        uint32_t n_clusters_l0 = ((uint32_t*) next_value)[0];
        next_value += sizeof(uint32_t);

        // === L0 (meso-clusters, always F32) ===
        l0.num_dimensions = dims;
        l0.num_vertical_dimensions = v_dims;
        l0.num_horizontal_dimensions = h_dims;
        l0.num_clusters = n_clusters_l0;

        auto* l0_headers = (uint32_t*) next_value;
        next_value += n_clusters_l0 * 2 * sizeof(uint32_t);

        l0.clusters.reserve(n_clusters_l0);
        for (size_t i = 0; i < n_clusters_l0; ++i) {
            uint32_t n_emb = l0_headers[i * 2];
            uint32_t max_cap = l0_headers[i * 2 + 1];
            l0.clusters.emplace_back(n_emb, max_cap, dims);
            l0.clusters[i].id = i;
            l0.clusters[i].LoadPDXData(next_value);
        }
        for (size_t i = 0; i < n_clusters_l0; ++i) {
            memcpy(
                l0.clusters[i].indices, next_value, sizeof(uint32_t) * l0.clusters[i].num_embeddings
            );
            next_value += sizeof(uint32_t) * l0.clusters[i].num_embeddings;
        }

        // === L1 (data clusters, inherited fields) ===
        this->num_dimensions = dims;
        this->num_vertical_dimensions = v_dims;
        this->num_horizontal_dimensions = h_dims;
        this->num_clusters = n_clusters_l1;

        auto* l1_headers = (uint32_t*) next_value;
        next_value += n_clusters_l1 * 2 * sizeof(uint32_t);

        this->clusters.reserve(n_clusters_l1);
        for (size_t i = 0; i < n_clusters_l1; ++i) {
            uint32_t n_emb = l1_headers[i * 2];
            uint32_t max_cap = l1_headers[i * 2 + 1];
            this->clusters.emplace_back(n_emb, max_cap, dims);
            this->clusters[i].id = i;
            this->clusters[i].LoadPDXData(next_value);
        }
        for (size_t i = 0; i < n_clusters_l1; ++i) {
            memcpy(
                this->clusters[i].indices,
                next_value,
                sizeof(uint32_t) * this->clusters[i].num_embeddings
            );
            next_value += sizeof(uint32_t) * this->clusters[i].num_embeddings;
        }

        // === Shared metadata ===
        bool normalized = ((char*) next_value)[0];
        this->is_normalized = normalized;
        l0.is_normalized = normalized;
        next_value += sizeof(char);

        // === L0 centroids (centroids_pdx from file) ===
        l0.centroids.resize(n_clusters_l0 * dims);
        memcpy(l0.centroids.data(), (float*) next_value, sizeof(float) * n_clusters_l0 * dims);
        next_value += sizeof(float) * n_clusters_l0 * dims;

        // === U8 quantization params ===
        if constexpr (Q == U8) {
            this->quantization_base = ((float*) next_value)[0];
            next_value += sizeof(float);
            this->quantization_scale = ((float*) next_value)[0];
            next_value += sizeof(float);
            this->quantization_scale_squared = this->quantization_scale * this->quantization_scale;
            this->inverse_quantization_scale_squared = 1.0f / this->quantization_scale_squared;
        }
        // Set mesocluster_id on L1 clusters by scanning L0
        for (uint32_t mc = 0; mc < n_clusters_l0; mc++) {
            auto& l0c = l0.clusters[mc];
            for (uint32_t p = 0; p < l0c.num_embeddings; p++) {
                this->clusters[l0c.indices[p]].mesocluster_id = mc;
            }
        }

        l0.ComputeClusterOffsets();
        this->ComputeClusterOffsets();
    }

    void Save(std::ostream& out) const {
        // Header: dimensions (shared between L0 and L1)
        out.write(reinterpret_cast<const char*>(&this->num_dimensions), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&this->num_vertical_dimensions), sizeof(uint32_t));
        out.write(
            reinterpret_cast<const char*>(&this->num_horizontal_dimensions), sizeof(uint32_t)
        );

        // Number of clusters: L1 then L0
        out.write(reinterpret_cast<const char*>(&this->num_clusters), sizeof(uint32_t));
        uint32_t n_clusters_l0 = l0.num_clusters;
        out.write(reinterpret_cast<const char*>(&n_clusters_l0), sizeof(uint32_t));

        // === L0 (meso-clusters, always F32) ===
        for (size_t i = 0; i < n_clusters_l0; ++i) {
            out.write(
                reinterpret_cast<const char*>(&l0.clusters[i].num_embeddings), sizeof(uint32_t)
            );
            out.write(
                reinterpret_cast<const char*>(&l0.clusters[i].max_capacity), sizeof(uint32_t)
            );
        }
        for (size_t i = 0; i < n_clusters_l0; ++i) {
            l0.clusters[i].SavePDXData(out);
        }
        for (size_t i = 0; i < n_clusters_l0; ++i) {
            out.write(
                reinterpret_cast<const char*>(l0.clusters[i].indices),
                sizeof(uint32_t) * l0.clusters[i].num_embeddings
            );
        }

        // === L1 (data clusters) ===
        for (size_t i = 0; i < this->num_clusters; ++i) {
            out.write(
                reinterpret_cast<const char*>(&this->clusters[i].num_embeddings), sizeof(uint32_t)
            );
            out.write(
                reinterpret_cast<const char*>(&this->clusters[i].max_capacity), sizeof(uint32_t)
            );
        }
        for (size_t i = 0; i < this->num_clusters; ++i) {
            this->clusters[i].SavePDXData(out);
        }
        for (size_t i = 0; i < this->num_clusters; ++i) {
            out.write(
                reinterpret_cast<const char*>(this->clusters[i].indices),
                sizeof(uint32_t) * this->clusters[i].num_embeddings
            );
        }

        // === Shared metadata ===
        char norm = this->is_normalized;
        out.write(&norm, sizeof(char));

        // L0 centroids
        out.write(
            reinterpret_cast<const char*>(l0.centroids.data()),
            sizeof(float) * n_clusters_l0 * this->num_dimensions
        );

        // === U8 quantization params ===
        if constexpr (Q == U8) {
            out.write(reinterpret_cast<const char*>(&this->quantization_base), sizeof(float));
            out.write(reinterpret_cast<const char*>(&this->quantization_scale), sizeof(float));
        }
    }

    size_t GetInMemorySizeInBytes() const {
        size_t size = sizeof(*this);

        // L1 clusters (inherited from base)
        for (const auto& cluster : this->clusters) {
            size += cluster.GetInMemorySizeInBytes();
        }
        size += (this->clusters.capacity() - this->clusters.size()) * sizeof(Cluster<Q>);
        size += this->centroids.capacity() * sizeof(float);

        // L0 meso-clusters
        for (const auto& cluster : l0.clusters) {
            size += cluster.GetInMemorySizeInBytes();
        }
        size += (l0.clusters.capacity() - l0.clusters.size()) * sizeof(Cluster<F32>);
        size += l0.centroids.capacity() * sizeof(float);

        return size;
    }
};

} // namespace PDX

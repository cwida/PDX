#pragma once

#include "pdx/common.hpp"
#include "pdx/indexes/cluster.hpp"
#include "pdx/utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
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
    std::vector<uint64_t> cluster_data_offsets;
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
        BufferReader reader{input};
        Load(reader);
    }

    // Reads the num_clusters clusters of num_dimensions as Save writes them: the (num_embeddings,
    // max_capacity) of every cluster, then each cluster's PDX data, then each cluster's row ids.
    template <class Reader>
    void LoadClusters(Reader& reader) {
        std::unique_ptr<uint32_t[]> cluster_headers(
            new uint32_t[static_cast<size_t>(num_clusters) * 2]
        );
        reader.Read(
            cluster_headers.get(), static_cast<size_t>(num_clusters) * 2 * sizeof(uint32_t)
        );
        clusters.reserve(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            clusters.emplace_back(
                cluster_headers[i * 2], cluster_headers[i * 2 + 1], num_dimensions
            );
            clusters[i].id = i;
            clusters[i].LoadPDXData(reader);
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            reader.Read(clusters[i].indices, sizeof(uint32_t) * clusters[i].num_embeddings);
        }
    }

    template <class Reader>
    void Load(Reader& reader) {
        num_dimensions = ReadValue<uint32_t>(reader);
        num_vertical_dimensions = ReadValue<uint32_t>(reader);
        num_horizontal_dimensions = ReadValue<uint32_t>(reader);
        num_clusters = ReadValue<uint32_t>(reader);
        LoadClusters(reader);

        is_normalized = ReadValue<char>(reader) != 0;

        centroids.resize(static_cast<size_t>(num_clusters) * num_dimensions);
        reader.Read(centroids.data(), sizeof(float) * num_clusters * num_dimensions);

        if constexpr (Q == U8) {
            quantization_base = ReadValue<float>(reader);
            quantization_scale = ReadValue<float>(reader);
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

    [[nodiscard]] uint64_t GetClusterDataSizeInBytes(const uint32_t num_embeddings) const {
        return static_cast<uint64_t>(num_embeddings) *
               (num_dimensions * sizeof(data_t) + sizeof(uint32_t));
    }

    // The first half of the stream format (SaveToStream): what a search keeps in memory, with the
    // offset of each cluster in the cluster data that follows (SaveClusterData).
    void SaveResidentData(std::ostream& out) const {
        WriteValue(out, num_dimensions);
        WriteValue(out, num_vertical_dimensions);
        WriteValue(out, num_horizontal_dimensions);
        WriteValue(out, num_clusters);
        uint64_t cluster_data_offset = 0;
        for (size_t i = 0; i < num_clusters; ++i) {
            WriteValue(out, clusters[i].num_embeddings);
            WriteValue(out, clusters[i].max_capacity);
            WriteValue(out, cluster_data_offset);
            cluster_data_offset += GetClusterDataSizeInBytes(clusters[i].num_embeddings);
        }
        WriteValue(out, static_cast<char>(is_normalized));
        out.write(
            reinterpret_cast<const char*>(centroids.data()),
            static_cast<std::streamsize>(sizeof(float) * num_clusters * num_dimensions)
        );
        if constexpr (Q == U8) {
            WriteValue(out, quantization_base);
            WriteValue(out, quantization_scale);
        }
    }

    // Each cluster's row ids, then its compact PDX data, in cluster order.
    void SaveClusterData(std::ostream& out) const {
        for (size_t i = 0; i < num_clusters; ++i) {
            out.write(
                reinterpret_cast<const char*>(clusters[i].indices),
                static_cast<std::streamsize>(sizeof(uint32_t) * clusters[i].num_embeddings)
            );
            clusters[i].SavePDXData(out);
        }
    }

    template <class Reader>
    void LoadResidentData(Reader& reader, const bool allocate_cluster_data = true) {
        num_dimensions = ReadValue<uint32_t>(reader);
        num_vertical_dimensions = ReadValue<uint32_t>(reader);
        num_horizontal_dimensions = ReadValue<uint32_t>(reader);
        num_clusters = ReadValue<uint32_t>(reader);
        clusters.reserve(num_clusters);
        cluster_data_offsets.resize(num_clusters);
        for (uint32_t i = 0; i < num_clusters; ++i) {
            const auto num_embeddings = ReadValue<uint32_t>(reader);
            const auto max_capacity = ReadValue<uint32_t>(reader);
            clusters.emplace_back(num_embeddings, max_capacity, num_dimensions, allocate_cluster_data);
            clusters[i].id = i;
            cluster_data_offsets[i] = ReadValue<uint64_t>(reader);
        }
        is_normalized = ReadValue<char>(reader) != 0;
        centroids.resize(static_cast<size_t>(num_clusters) * num_dimensions);
        reader.Read(centroids.data(), sizeof(float) * centroids.size());
        if constexpr (Q == U8) {
            quantization_base = ReadValue<float>(reader);
            quantization_scale = ReadValue<float>(reader);
            quantization_scale_squared = quantization_scale * quantization_scale;
            inverse_quantization_scale_squared = 1.0f / quantization_scale_squared;
        }
        ComputeClusterOffsets();
    }

    // Reads the clusters in the order of their offsets, skipping the bytes between them.
    template <class Reader>
    void LoadClusterData(Reader& reader) {
        std::vector<uint32_t> clusters_in_stream_order(num_clusters);
        std::iota(clusters_in_stream_order.begin(), clusters_in_stream_order.end(), 0);
        std::sort(
            clusters_in_stream_order.begin(),
            clusters_in_stream_order.end(),
            [&](uint32_t a, uint32_t b) {
                return cluster_data_offsets[a] < cluster_data_offsets[b];
            }
        );
        uint64_t position = 0;
        for (const uint32_t cluster_id : clusters_in_stream_order) {
            if (cluster_data_offsets[cluster_id] < position) {
                throw std::runtime_error("Overlapping clusters in a PDX index stream");
            }
            reader.Skip(cluster_data_offsets[cluster_id] - position);
            auto& cluster = clusters[cluster_id];
            reader.Read(cluster.indices, sizeof(uint32_t) * cluster.num_embeddings);
            cluster.LoadPDXData(reader);
            position =
                cluster_data_offsets[cluster_id] + GetClusterDataSizeInBytes(cluster.num_embeddings);
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
        in_memory_size_in_bytes += cluster_data_offsets.capacity() * sizeof(uint64_t);
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
        BufferReader reader{input};
        Load(reader);
    }

    template <class Reader>
    void Load(Reader& reader) {
        // Header
        const auto dims = ReadValue<uint32_t>(reader);
        const auto v_dims = ReadValue<uint32_t>(reader);
        const auto h_dims = ReadValue<uint32_t>(reader);
        const auto n_clusters_l1 = ReadValue<uint32_t>(reader);
        const auto n_clusters_l0 = ReadValue<uint32_t>(reader);

        // === L0 (meso-clusters, always F32) ===
        l0.num_dimensions = dims;
        l0.num_vertical_dimensions = v_dims;
        l0.num_horizontal_dimensions = h_dims;
        l0.num_clusters = n_clusters_l0;
        l0.LoadClusters(reader);

        // === L1 (data clusters, inherited fields) ===
        this->num_dimensions = dims;
        this->num_vertical_dimensions = v_dims;
        this->num_horizontal_dimensions = h_dims;
        this->num_clusters = n_clusters_l1;
        this->LoadClusters(reader);

        // === Shared metadata ===
        const bool normalized = ReadValue<char>(reader) != 0;
        this->is_normalized = normalized;
        l0.is_normalized = normalized;

        // === L0 centroids (centroids_pdx from file) ===
        l0.centroids.resize(static_cast<size_t>(n_clusters_l0) * dims);
        reader.Read(l0.centroids.data(), sizeof(float) * n_clusters_l0 * dims);

        // === L1 centroids ===
        this->centroids.resize(static_cast<size_t>(n_clusters_l1) * dims);
        reader.Read(this->centroids.data(), sizeof(float) * n_clusters_l1 * dims);

        // === U8 quantization params ===
        if constexpr (Q == U8) {
            this->quantization_base = ReadValue<float>(reader);
            this->quantization_scale = ReadValue<float>(reader);
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

        // L1 centroids
        out.write(
            reinterpret_cast<const char*>(this->centroids.data()),
            sizeof(float) * this->num_clusters * this->num_dimensions
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

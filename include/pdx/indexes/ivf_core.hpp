#pragma once

#include "pdx/common.hpp"
#include "pdx/indexes/centroids.hpp"
#include "pdx/indexes/cluster.hpp"
#include "pdx/quantizers/saq.hpp"
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

// Gives searches the bytes of the clusters an index holds no data for (LoadResidentDataFromStream):
// Acquire returns them as SaveClusterData wrote them, valid until the matching Release. Called
// concurrently by the searches. AcquireSketch returns a cluster's sketch batches as SaveToStream
// wrote them (nullptr: none, the search scans the cluster), and IsClusterCached whether Acquire
// would find the cluster's bytes already in memory (a hint).
class IClusterSource {
  public:
    virtual ~IClusterSource() = default;
    virtual const char* Acquire(uint32_t cluster_id) = 0;
    virtual void Release(uint32_t cluster_id) = 0;
    virtual bool IsClusterCached(uint32_t /*cluster_id*/) const { return false; }
    virtual const char* AcquireSketch(uint32_t /*cluster_id*/) { return nullptr; }
    virtual void ReleaseSketch(uint32_t /*cluster_id*/) {}
};

template <Quantization Q>
class IVF : public skmeans::AllocatorHolder {
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
    std::vector<uint64_t> cluster_sketch_offsets;
    bool is_normalized{};
    Centroids centroids;
    std::unique_ptr<float[]> sketch_mean;

    // U8-specific quantization parameters
    float quantization_scale = 1.0f;
    float quantization_scale_squared = 1.0f;
    float inverse_quantization_scale_squared = 1.0f;
    float quantization_base = 0.0f;

    // SAQ-specific parameters
    SAQModel saq_model;
    std::unique_ptr<float[]> saq_chunk_epsilons;
    uint64_t num_embeddings_at_calibration = 0;
    std::unique_ptr<ClusterSketchRotation> sketch_rotation;
    std::unique_ptr<float[]> rotated_sketch_mean;

    IVF() = default;
    ~IVF() = default;
    IVF(IVF&&) = default;
    IVF& operator=(IVF&&) = default;

    IVF(uint32_t num_dimensions,
        uint64_t total_num_embeddings,
        uint32_t num_clusters,
        bool is_normalized,
        CentroidQuantization centroid_quantization = CentroidQuantization::F32)
        : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings),
          num_clusters(num_clusters),
          num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
          num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions),
          is_normalized(is_normalized), centroids(centroid_quantization, num_dimensions) {
        clusters.reserve(num_clusters);
    }

    IVF(uint32_t num_dimensions,
        uint64_t total_num_embeddings,
        uint32_t num_clusters,
        bool is_normalized,
        float quantization_scale,
        float quantization_base,
        CentroidQuantization centroid_quantization = CentroidQuantization::F32)
        : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings),
          num_clusters(num_clusters),
          num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
          num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions),
          is_normalized(is_normalized), centroids(centroid_quantization, num_dimensions),
          quantization_scale(quantization_scale),
          quantization_scale_squared(quantization_scale * quantization_scale),
          inverse_quantization_scale_squared(1.0f / (quantization_scale * quantization_scale)),
          quantization_base(quantization_base) {
        clusters.reserve(num_clusters);
    }

    // In data_t values, as ClusterFormat::GetValuesPerEmbedding
    uint32_t GetValuesPerEmbedding() const {
        if constexpr (Q == Quantization::SAQ) {
            return static_cast<uint32_t>(saq_model.GetEncodedEmbeddingSize());
        } else {
            return num_dimensions;
        }
    }

    bool HasSketches() const { return sketch_mean != nullptr; }

    uint32_t GetSketchNumDimensions() const {
        if (!HasSketches()) {
            return 0;
        }
        if constexpr (Q == Quantization::SAQ) {
            return AlignValue<uint32_t, CLUSTER_SKETCH_DIMENSION_ALIGNMENT>(num_dimensions);
        } else {
            return num_dimensions;
        }
    }

    ClusterFormat<Q> GetClusterFormat() const {
        if constexpr (Q == Quantization::SAQ) {
            return {
                saq_model.chunk_record_offsets.get(),
                saq_model.num_chunks,
                saq_model.num_segments,
                saq_model.HasPDXU8Head(),
                GetSketchNumDimensions()
            };
        } else {
            return {num_dimensions, GetSketchNumDimensions()};
        }
    }

    const float* ToSketchDomain(const float* vector, float* scratch) const {
        if constexpr (Q == Quantization::SAQ) {
            sketch_rotation->Rotate(vector, scratch);
            return scratch;
        } else {
            return vector;
        }
    }

    const float* GetSketchDomainMean() const {
        if constexpr (Q == Quantization::SAQ) {
            return rotated_sketch_mean.get();
        } else {
            return sketch_mean.get();
        }
    }

    std::unique_ptr<ClusterSketchQueryTable> CreateClusterSketchQueryTable(
        const float* transformed_query
    ) const {
        std::unique_ptr<float[]> scratch(
            Q == Quantization::SAQ ? new float[GetSketchNumDimensions()] : nullptr
        );
        return std::make_unique<ClusterSketchQueryTable>(
            ToSketchDomain(transformed_query, scratch.get()), GetSketchNumDimensions()
        );
    }

    void SetSketchRotation(std::unique_ptr<ClusterSketchRotation> rotation) {
        sketch_rotation = std::move(rotation);
        rotated_sketch_mean.reset(new float[sketch_rotation->padded_num_dimensions]);
        sketch_rotation->Rotate(sketch_mean.get(), rotated_sketch_mean.get());
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

    void SetAllocator(MemoryAllocator* allocator) {
        skmeans::AllocatorHolder::SetAllocator(allocator);
        centroids.SetAllocator(allocator);
    }

    void Load(char* input, CentroidQuantization centroid_quantization) {
        BufferReader reader{input};
        Load(reader, centroid_quantization);
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
        const ClusterFormat<Q> cluster_format = GetClusterFormat();
        for (size_t i = 0; i < num_clusters; ++i) {
            clusters.emplace_back(
                cluster_headers[i * 2], cluster_headers[i * 2 + 1], cluster_format, &GetAllocator()
            );
            clusters[i].id = i;
            clusters[i].LoadPDXData(reader);
            total_num_embeddings += clusters[i].num_embeddings;
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            reader.Read(clusters[i].indices.get(), sizeof(uint32_t) * clusters[i].num_embeddings);
        }
    }

    template <class Reader>
    void Load(Reader& reader, CentroidQuantization centroid_quantization) {
        num_dimensions = ReadValue<uint32_t>(reader);
        num_vertical_dimensions = ReadValue<uint32_t>(reader);
        num_horizontal_dimensions = ReadValue<uint32_t>(reader);
        num_clusters = ReadValue<uint32_t>(reader);
        if constexpr (Q == Quantization::SAQ) {
            saq_model.Load(reader);
            saq_chunk_epsilons.reset(new float[saq_model.num_chunks]);
            reader.Read(saq_chunk_epsilons.get(), saq_model.num_chunks * sizeof(float));
            num_embeddings_at_calibration = ReadValue<uint64_t>(reader);
        }
        LoadClusters(reader);

        is_normalized = ReadValue<char>(reader) != 0;

        centroids = Centroids(centroid_quantization, num_dimensions, &GetAllocator());
        centroids.Load(reader, num_clusters);

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
        if constexpr (Q == Quantization::SAQ) {
            saq_model.Save(out);
            out.write(
                reinterpret_cast<const char*>(saq_chunk_epsilons.get()),
                static_cast<std::streamsize>(saq_model.num_chunks * sizeof(float))
            );
            WriteValue(out, num_embeddings_at_calibration);
        }

        for (size_t i = 0; i < num_clusters; ++i) {
            out.write(reinterpret_cast<const char*>(&clusters[i].num_embeddings), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&clusters[i].max_capacity), sizeof(uint32_t));
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            clusters[i].SavePDXData(out);
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            out.write(
                reinterpret_cast<const char*>(clusters[i].indices.get()),
                sizeof(uint32_t) * clusters[i].num_embeddings
            );
        }

        char norm = is_normalized;
        out.write(&norm, sizeof(char));

        centroids.Save(out);

        if constexpr (Q == U8) {
            out.write(reinterpret_cast<const char*>(&quantization_base), sizeof(float));
            out.write(reinterpret_cast<const char*>(&quantization_scale), sizeof(float));
        }
    }

    [[nodiscard]] uint64_t GetClusterDataSizeInBytes(const uint32_t num_embeddings) const {
        return static_cast<uint64_t>(num_embeddings) *
               (GetValuesPerEmbedding() * sizeof(data_t) + sizeof(uint32_t));
    }

    // The first half of the stream format (SaveToStream): what a search keeps in memory, with the
    // offset of each cluster in the cluster data that follows (SaveClusterData).
    void SaveResidentData(std::ostream& out, const uint32_t* storage_order) const {
        WriteValue(out, num_dimensions);
        WriteValue(out, num_vertical_dimensions);
        WriteValue(out, num_horizontal_dimensions);
        WriteValue(out, num_clusters);
        if constexpr (Q == Quantization::SAQ) {
            saq_model.Save(out);
            out.write(
                reinterpret_cast<const char*>(saq_chunk_epsilons.get()),
                static_cast<std::streamsize>(saq_model.num_chunks * sizeof(float))
            );
            WriteValue(out, num_embeddings_at_calibration);
        }
        WriteValue(out, static_cast<uint8_t>(HasSketches()));
        if (HasSketches()) {
            out.write(
                reinterpret_cast<const char*>(sketch_mean.get()),
                static_cast<std::streamsize>(num_dimensions * sizeof(float))
            );
            if constexpr (Q == Quantization::SAQ) {
                sketch_rotation->Save(out);
            }
        }
        std::unique_ptr<uint64_t[]> cluster_data_offsets_to_save(new uint64_t[num_clusters]);
        uint64_t cluster_data_offset = 0;
        for (size_t position = 0; position < num_clusters; ++position) {
            const uint32_t cluster_id = storage_order[position];
            cluster_data_offsets_to_save[cluster_id] = cluster_data_offset;
            cluster_data_offset += GetClusterDataSizeInBytes(clusters[cluster_id].num_embeddings);
        }
        // The sketch section follows the cluster data, in the same storage order
        std::unique_ptr<uint64_t[]> cluster_sketch_offsets_to_save;
        if (HasSketches()) {
            cluster_sketch_offsets_to_save.reset(new uint64_t[num_clusters]);
            uint64_t cluster_sketch_offset = cluster_data_offset;
            for (size_t position = 0; position < num_clusters; ++position) {
                const uint32_t cluster_id = storage_order[position];
                cluster_sketch_offsets_to_save[cluster_id] = cluster_sketch_offset;
                cluster_sketch_offset += clusters[cluster_id].GetSavedSketchesSizeInBytes();
            }
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            WriteValue(out, clusters[i].num_embeddings);
            WriteValue(out, clusters[i].max_capacity);
            WriteValue(out, cluster_data_offsets_to_save[i]);
            if (HasSketches()) {
                WriteValue(out, cluster_sketch_offsets_to_save[i]);
            }
        }
        WriteValue(out, static_cast<char>(is_normalized));
        centroids.Save(out);
        if constexpr (Q == U8) {
            WriteValue(out, quantization_base);
            WriteValue(out, quantization_scale);
        }
    }

    // Each cluster's row ids, then its compact PDX data, in storage order.
    void SaveClusterData(std::ostream& out, const uint32_t* storage_order) const {
        for (size_t position = 0; position < num_clusters; ++position) {
            const cluster_t& cluster = clusters[storage_order[position]];
            out.write(
                reinterpret_cast<const char*>(cluster.indices.get()),
                static_cast<std::streamsize>(sizeof(uint32_t) * cluster.num_embeddings)
            );
            cluster.SavePDXData(out);
        }
    }

    // After the cluster data: each cluster's sketch batches, in storage order.
    void SaveSketchData(std::ostream& out, const uint32_t* storage_order) const {
        for (size_t position = 0; position < num_clusters; ++position) {
            clusters[storage_order[position]].SaveSketches(out);
        }
    }

    template <class Reader>
    void LoadResidentData(
        Reader& reader,
        CentroidQuantization centroid_quantization,
        const bool allocate_cluster_data = true
    ) {
        num_dimensions = ReadValue<uint32_t>(reader);
        num_vertical_dimensions = ReadValue<uint32_t>(reader);
        num_horizontal_dimensions = ReadValue<uint32_t>(reader);
        num_clusters = ReadValue<uint32_t>(reader);
        if constexpr (Q == Quantization::SAQ) {
            saq_model.Load(reader);
            saq_chunk_epsilons.reset(new float[saq_model.num_chunks]);
            reader.Read(saq_chunk_epsilons.get(), saq_model.num_chunks * sizeof(float));
            num_embeddings_at_calibration = ReadValue<uint64_t>(reader);
        }
        const bool has_sketches = ReadValue<uint8_t>(reader) != 0;
        if (has_sketches) {
            sketch_mean.reset(new float[num_dimensions]);
            reader.Read(sketch_mean.get(), num_dimensions * sizeof(float));
            if constexpr (Q == Quantization::SAQ) {
                SetSketchRotation(ClusterSketchRotation::Load(reader, num_dimensions));
            }
            cluster_sketch_offsets.resize(num_clusters);
        }
        clusters.reserve(num_clusters);
        cluster_data_offsets.resize(num_clusters);
        const ClusterFormat<Q> cluster_format = GetClusterFormat();
        for (uint32_t i = 0; i < num_clusters; ++i) {
            const auto num_embeddings = ReadValue<uint32_t>(reader);
            const auto max_capacity = ReadValue<uint32_t>(reader);
            clusters.emplace_back(
                num_embeddings, max_capacity, cluster_format, &GetAllocator(), allocate_cluster_data
            );
            clusters[i].id = i;
            cluster_data_offsets[i] = ReadValue<uint64_t>(reader);
            if (has_sketches) {
                cluster_sketch_offsets[i] = ReadValue<uint64_t>(reader);
            }
            total_num_embeddings += num_embeddings;
        }
        is_normalized = ReadValue<char>(reader) != 0;
        centroids = Centroids(centroid_quantization, num_dimensions, &GetAllocator());
        centroids.Load(reader, num_clusters);
        if constexpr (Q == U8) {
            quantization_base = ReadValue<float>(reader);
            quantization_scale = ReadValue<float>(reader);
            quantization_scale_squared = quantization_scale * quantization_scale;
            inverse_quantization_scale_squared = 1.0f / quantization_scale_squared;
        }
        ComputeClusterOffsets();
    }

    // Reads the clusters in the order of their offsets, skipping the bytes between them. An empty
    // cluster holds no bytes (it shares its offset with the next cluster), so it is not read.
    // Returns where it stopped, from the start of the cluster data.
    template <class Reader>
    uint64_t LoadClusterData(Reader& reader) {
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
            auto& cluster = clusters[cluster_id];
            if (cluster.num_embeddings == 0) {
                continue;
            }
            if (cluster_data_offsets[cluster_id] < position) {
                throw std::runtime_error("Overlapping clusters in a PDX index stream");
            }
            reader.Skip(cluster_data_offsets[cluster_id] - position);
            reader.Read(cluster.indices.get(), sizeof(uint32_t) * cluster.num_embeddings);
            cluster.LoadPDXData(reader);
            position = cluster_data_offsets[cluster_id] +
                       GetClusterDataSizeInBytes(cluster.num_embeddings);
        }
        return position;
    }

    // The sketch section, as LoadClusterData reads the cluster data, from position (where
    // LoadClusterData stopped).
    template <class Reader>
    void LoadSketchData(Reader& reader, uint64_t position) {
        std::vector<uint32_t> clusters_in_stream_order(num_clusters);
        std::iota(clusters_in_stream_order.begin(), clusters_in_stream_order.end(), 0);
        std::sort(
            clusters_in_stream_order.begin(),
            clusters_in_stream_order.end(),
            [&](uint32_t a, uint32_t b) {
                return cluster_sketch_offsets[a] < cluster_sketch_offsets[b];
            }
        );
        for (const uint32_t cluster_id : clusters_in_stream_order) {
            auto& cluster = clusters[cluster_id];
            if (cluster.num_embeddings == 0) {
                continue;
            }
            if (cluster_sketch_offsets[cluster_id] < position) {
                throw std::runtime_error("Overlapping cluster sketches in a PDX index stream");
            }
            reader.Skip(cluster_sketch_offsets[cluster_id] - position);
            cluster.LoadSketches(reader);
            position = cluster_sketch_offsets[cluster_id] + cluster.GetSavedSketchesSizeInBytes();
        }
    }

    size_t GetAllocatedSizeInBytes() const {
        size_t size = centroids.GetAllocatedSizeInBytes();
        for (const auto& cluster : clusters) {
            size += cluster.GetAllocatedSizeInBytes();
        }
        return size;
    }

    size_t GetInMemorySizeInBytes() const {
        size_t in_memory_size_in_bytes = 0;
        in_memory_size_in_bytes += sizeof(*this);
        for (const auto& cluster : clusters) {
            in_memory_size_in_bytes += cluster.GetInMemorySizeInBytes();
        }
        in_memory_size_in_bytes +=
            (clusters.capacity() - clusters.size()) * sizeof(*clusters.data());
        in_memory_size_in_bytes += centroids.GetAllocatedSizeInBytes();
        in_memory_size_in_bytes += num_clusters * sizeof(size_t); // cluster_offsets
        in_memory_size_in_bytes += cluster_data_offsets.capacity() * sizeof(uint64_t);
        in_memory_size_in_bytes += cluster_sketch_offsets.capacity() * sizeof(uint64_t);
        if (HasSketches()) {
            in_memory_size_in_bytes += num_dimensions * sizeof(float);
        }
        if (sketch_rotation != nullptr) {
            in_memory_size_in_bytes += sizeof(*sketch_rotation) +
                                       sketch_rotation->GetInMemorySizeInBytes() +
                                       sketch_rotation->padded_num_dimensions * sizeof(float);
        }
        if constexpr (Q == Quantization::SAQ) {
            // The model's sizeof is already in the IVF's
            in_memory_size_in_bytes += saq_model.GetInMemorySizeInBytes() - sizeof(saq_model) +
                                       saq_model.num_chunks * sizeof(float);
        }
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
        bool is_normalized,
        CentroidQuantization centroid_quantization = CentroidQuantization::F32
    )
        : IVF<Q>(
              num_dimensions,
              total_num_embeddings,
              num_clusters,
              is_normalized,
              centroid_quantization
          ) {}

    IVFTree(
        uint32_t num_dimensions,
        uint64_t total_num_embeddings,
        uint32_t num_clusters,
        bool is_normalized,
        float quantization_scale,
        float quantization_base,
        CentroidQuantization centroid_quantization = CentroidQuantization::F32
    )
        : IVF<Q>(
              num_dimensions,
              total_num_embeddings,
              num_clusters,
              is_normalized,
              quantization_scale,
              quantization_base,
              centroid_quantization
          ) {}

    // Binds L0 too
    void SetAllocator(MemoryAllocator* allocator) {
        IVF<Q>::SetAllocator(allocator);
        l0.SetAllocator(allocator);
    }

    void Load(char* input, CentroidQuantization centroid_quantization) {
        BufferReader reader{input};
        Load(reader, centroid_quantization);
    }

    template <class Reader>
    void Load(Reader& reader, CentroidQuantization centroid_quantization) {
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
        l0.centroids = Centroids(CentroidQuantization::F32, dims, &this->GetAllocator());
        l0.centroids.Load(reader, n_clusters_l0);

        // === L1 centroids ===
        this->centroids = Centroids(centroid_quantization, dims, &this->GetAllocator());
        this->centroids.Load(reader, n_clusters_l1);

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
                reinterpret_cast<const char*>(l0.clusters[i].indices.get()),
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
                reinterpret_cast<const char*>(this->clusters[i].indices.get()),
                sizeof(uint32_t) * this->clusters[i].num_embeddings
            );
        }

        // === Shared metadata ===
        char norm = this->is_normalized;
        out.write(&norm, sizeof(char));

        // L0 centroids
        l0.centroids.Save(out);

        // L1 centroids
        this->centroids.Save(out);

        // === U8 quantization params ===
        if constexpr (Q == U8) {
            out.write(reinterpret_cast<const char*>(&this->quantization_base), sizeof(float));
            out.write(reinterpret_cast<const char*>(&this->quantization_scale), sizeof(float));
        }
    }

    size_t GetAllocatedSizeInBytes() const {
        return IVF<Q>::GetAllocatedSizeInBytes() + l0.GetAllocatedSizeInBytes();
    }

    size_t GetInMemorySizeInBytes() const {
        size_t size = sizeof(*this);

        // L1 clusters (inherited from base)
        for (const auto& cluster : this->clusters) {
            size += cluster.GetInMemorySizeInBytes();
        }
        size += (this->clusters.capacity() - this->clusters.size()) * sizeof(Cluster<Q>);
        size += this->centroids.GetAllocatedSizeInBytes();
        size += this->num_clusters * sizeof(size_t); // cluster_offsets
        size += this->cluster_data_offsets.capacity() * sizeof(uint64_t);

        // L0 meso-clusters
        for (const auto& cluster : l0.clusters) {
            size += cluster.GetInMemorySizeInBytes();
        }
        size += (l0.clusters.capacity() - l0.clusters.size()) * sizeof(Cluster<F32>);
        size += l0.centroids.GetAllocatedSizeInBytes();
        size += l0.num_clusters * sizeof(size_t); // cluster_offsets
        size += l0.cluster_data_offsets.capacity() * sizeof(uint64_t);

        return size;
    }
};

} // namespace PDX

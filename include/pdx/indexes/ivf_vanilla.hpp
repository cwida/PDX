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

#include "pdx/common.hpp"
#include "pdx/indexes/ivf_utils.hpp"
#include "pdx/ivf_searcher.hpp"
#include "pdx/profiler.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"

namespace PDX {

class IPDXIndex : public skmeans::ExecutorHolder, public skmeans::AllocatorHolder {
  public:
    virtual ~IPDXIndex() = default;
    virtual std::vector<KNNCandidate> Search(const float* query_embedding, size_t knn) const = 0;
    virtual std::vector<KNNCandidate> FilteredSearch(
        const float* query_embedding,
        size_t knn,
        const std::vector<size_t>& passing_row_ids
    ) const = 0;
    virtual std::vector<std::vector<KNNCandidate>> BulkSearch(
        const float* /*query_embeddings*/,
        size_t /*num_queries*/,
        size_t /*knn*/,
        bool /*is_query_transformed*/ = false,
        ParallelExecutor* /*executor*/ = nullptr
    ) const {
        throw std::logic_error("BulkSearch: not implemented for this index type");
    }
    virtual void BuildIndex(const float* embeddings, size_t num_embeddings) = 0;
    virtual void SetNProbe(uint32_t n_probe) = 0;
    virtual void Save(const std::string& path) = 0;
    virtual void Restore(const std::string& path) = 0;
    // The stream holds no rotation matrix: LoadPDXIndexFromStream loads it into an index on the
    // pruner it was saved with (indexes that share one rotation persist it once). LoadFromStream
    // reads what follows the header.
    virtual void SaveToStream(std::ostream& out) = 0;
    virtual void LoadFromStream(std::istream& in) = 0;
    // LoadFromStream without the clusters' data, which searches then get from cluster_source (it
    // must outlive the index). Indexes that cannot page their clusters (Flat, tree) load
    // everything.
    virtual void LoadResidentDataFromStream(std::istream& in, IClusterSource& /*cluster_source*/) {
        LoadFromStream(in);
    }
    // The offset and size in bytes of a cluster in the cluster data of the stream the index was
    // loaded from (LoadResidentDataFromStream).
    virtual std::pair<uint64_t, uint64_t> GetClusterDataRange(uint32_t /*cluster_id*/) const {
        throw std::logic_error("GetClusterDataRange: the index does not page its clusters");
    }
    virtual uint32_t GetNumDimensions() const = 0;
    virtual uint32_t GetNumClusters() const = 0;
    virtual uint32_t GetClusterSize(uint32_t cluster_id) const = 0;
    virtual std::vector<uint32_t> GetClusterRowIds(uint32_t cluster_id) const = 0;
    virtual size_t GetInMemorySizeInBytes() const = 0;
    // The bytes of the buffers the index holds from its allocator.
    virtual size_t GetAllocatedSizeInBytes() const = 0;
    // What GetInMemorySizeInBytes counts outside the allocator.
    size_t GetSelfManagedMemorySizeInBytes() const {
        return GetInMemorySizeInBytes() - GetAllocatedSizeInBytes();
    }
    // Maintenance (SPFresh-like). Concurrent writes must go through a single writer thread.
    virtual void Append(size_t row_id, const float* embedding) = 0;
    // Appends num_embeddings rows at once, as Append does one: same validation, same split rule.
    // Basic exception guarantee.
    virtual void BulkAppend(
        const size_t* row_ids,
        const float* embeddings,
        size_t num_embeddings
    ) = 0;
    // Returns whether the row was in the index: deleting an absent row is a no-op.
    virtual bool Delete(size_t row_id) = 0;
    // Leaves one free slot per cluster, for an index that stops receiving appends.
    virtual void ShrinkToFit() {}
    virtual std::pair<uint32_t, uint32_t> GetRowIdMapping(size_t row_id) const = 0;
    bool Contains(size_t row_id) const { return GetRowIdMapping(row_id).first != DELETED_MARKER; }
    // Resumable search into the caller's TopKHeap (construct it thread_safe when several cursors
    // share it); passing_row_ids to filter (nullptr: unfiltered). clusters_access_order: the
    // query's GetClustersAccessOrder, so that several searches of one query rank the clusters once
    // (nullptr: rank).
    virtual std::unique_ptr<IIterativeSearch> BeginIterativeSearch(
        const float* query_embedding,
        uint32_t knn,
        TopKHeap& top_k_heap,
        const std::vector<size_t>* passing_row_ids,
        bool is_query_transformed = false,
        const std::vector<uint32_t>* clusters_access_order = nullptr
    ) const = 0;
    // Builds the evaluator of passing_row_ids once, for the filtered searches of many queries.
    virtual std::unique_ptr<PredicateEvaluator> CreateSharedPredicateEvaluator(
        const std::vector<size_t>& passing_row_ids
    ) const = 0;
    // BeginIterativeSearch filtered by an evaluator from CreateSharedPredicateEvaluator, which must
    // outlive the cursor.
    virtual std::unique_ptr<IIterativeSearch> BeginIterativeSearchWithSharedEvaluator(
        const float* query_embedding,
        uint32_t knn,
        TopKHeap& top_k_heap,
        const PredicateEvaluator& evaluator,
        bool is_query_transformed = false,
        const std::vector<uint32_t>* clusters_access_order = nullptr
    ) const = 0;
    // Writes the embeddings of row_ids as the index stores them (normalized and rotated; u8
    // dequantized), row-major into out (row_ids.size() x num_dimensions), in the order of row_ids.
    // Every row id must be in the index.
    virtual void GetEmbeddingsFromIndexByRowIds(const std::vector<size_t>& row_ids, float* out)
        const = 0;
    // The query's distance to each centroid, unsorted (out holds GetNumClusters() floats), to rank
    // the clusters of several indexes together. Flat: its one centroid is the mean of its live
    // embeddings.
    virtual void GetDistancesToCentroids(
        const float* query_embedding,
        bool is_query_transformed,
        float* out
    ) const = 0;
    // All clusters, nearest centroid to the query first.
    virtual std::vector<uint32_t> GetClustersAccessOrder(
        const float* query_embedding,
        bool is_query_transformed = false
    ) const = 0;

  protected:
    void ValidateBulkAppendRowIds(
        const size_t* row_ids,
        const size_t num_embeddings,
        const size_t base_row_id
    ) const {
        auto sorted_row_ids = skmeans::AllocateBuffer<size_t>(GetAllocator(), num_embeddings);
        std::memcpy(sorted_row_ids.get(), row_ids, num_embeddings * sizeof(size_t));
        std::sort(sorted_row_ids.get(), sorted_row_ids.get() + num_embeddings);
        for (size_t i = 0; i < num_embeddings; i++) {
            const size_t row_id = sorted_row_ids[i];
            if (row_id < base_row_id || row_id > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument(
                    "BulkAppend: row_id " + std::to_string(row_id) +
                    " is below the index's base_row_id or above the uint32 range"
                );
            }
            if (i > 0 && row_id == sorted_row_ids[i - 1]) {
                throw std::invalid_argument(
                    "BulkAppend: row_id " + std::to_string(row_id) + " appears twice in the batch"
                );
            }
            if (Contains(row_id)) {
                throw std::invalid_argument(
                    "BulkAppend: row_id " + std::to_string(row_id) + " already exists in the index"
                );
            }
        }
    }

    void BindExecutor(const PDXIndexConfig& config) {
        if (config.executor != nullptr) {
            SetExecutor(config.executor);
        } else {
            owned_executor = skmeans::MakeDefaultExecutor(config.n_threads);
            SetExecutor(owned_executor.get());
        }
    }

  private:
    std::unique_ptr<ParallelExecutor> owned_executor;
};

template <PDX::Quantization Q>
class PDXIndex : public IPDXIndex {
  public:
    using embedding_storage_t = PDX::pdx_data_t<Q>;
    using cluster_t = PDX::Cluster<Q>;
    using batch_computer =
        skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;

  private:
    PDXIndexConfig config{};
    PDX::IVF<Q> index;
    std::unique_ptr<PDX::ADSamplingPruner> owned_pruner;
    PDX::ADSamplingPruner* pruner = nullptr;
    std::unique_ptr<PDX::PDXearch<Q>> searcher;
    RowIdClusterMapping row_id_cluster_mapping;
    uint32_t split_depth = 0;

  public:
    PDXIndex() = default;

    explicit PDXIndex(PDXIndexConfig config) : config(config) {
        config.Validate();
        BindExecutor(config);
        SetAllocator(config.allocator);
        index.SetAllocator(config.allocator);
        owned_pruner = std::make_unique<PDX::ADSamplingPruner>(
            config.num_dimensions, config.seed, config.allocator
        );
        pruner = owned_pruner.get();
        row_id_cluster_mapping.base_row_id = config.base_row_id;
    }

    PDXIndex(PDXIndexConfig config, PDX::ADSamplingPruner& external_pruner)
        : config(config), pruner(&external_pruner) {
        config.Validate();
        BindExecutor(config);
        SetAllocator(config.allocator);
        index.SetAllocator(config.allocator);
        row_id_cluster_mapping.base_row_id = config.base_row_id;
    }

    void Save(const std::string& path) override {
        CompactClusters();

        std::ofstream out(path, std::ios::binary);
        WriteValue(out, static_cast<uint8_t>(GetIndexType()));
        WriteRotationMatrix(out, *pruner);
        index.Save(out);
    }

    void Restore(const std::string& path) override {
        auto buffer = MmapFile(path);
        char* ptr = buffer.get();
        BufferReader reader{ptr};

        // Index type flag
        ReadValue<uint8_t>(reader);
        const auto matrix = ReadRotationMatrix(reader, GetAllocator());
        index.SetAllocator(&GetAllocator());
        index.Load(reader);
        config.num_dimensions = index.num_dimensions;
        config.normalize = index.is_normalized;

        // Create pruner and searcher
        owned_pruner = std::make_unique<PDX::ADSamplingPruner>(
            index.num_dimensions, matrix.get(), &GetAllocator()
        );
        pruner = owned_pruner.get();
        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
        BuildRowIdClusterMapping();
    }

    void SaveToStream(std::ostream& out) override {
        CompactClusters();
        WriteStreamHeader(out, GetIndexType(), config);
        index.SaveResidentData(out);
        row_id_cluster_mapping.Save(out);
        index.SaveClusterData(out);
    }

    void LoadFromStream(std::istream& in) override {
        StreamReader reader{in};
        index.LoadResidentData(reader);
        row_id_cluster_mapping.Load(reader);
        index.LoadClusterData(reader);
        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
    }

    void LoadResidentDataFromStream(std::istream& in, IClusterSource& cluster_source) override {
        StreamReader reader{in};
        index.LoadResidentData(reader, /*allocate_cluster_data=*/false);
        row_id_cluster_mapping.Load(reader);
        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
        searcher->cluster_source = &cluster_source;
    }

    std::pair<uint64_t, uint64_t> GetClusterDataRange(uint32_t cluster_id) const override {
        return {
            index.cluster_data_offsets.at(cluster_id),
            index.GetClusterDataSizeInBytes(index.clusters[cluster_id].used_capacity)
        };
    }

    std::vector<PDX::KNNCandidate> Search(const float* query_embedding, size_t knn) const override {
        return searcher->Search(query_embedding, knn);
    }

    std::vector<PDX::KNNCandidate> FilteredSearch(
        const float* query_embedding,
        size_t knn,
        const std::vector<size_t>& passing_row_ids
    ) const override {
        auto evaluator = CreatePredicateEvaluator(passing_row_ids);
        {
            PDX_PROFILE_SCOPE("Search");
            return searcher->FilteredSearch(query_embedding, knn, evaluator);
        }
    }

    std::unique_ptr<IIterativeSearch> BeginIterativeSearch(
        const float* query_embedding,
        uint32_t knn,
        TopKHeap& top_k_heap,
        const std::vector<size_t>* passing_row_ids,
        bool is_query_transformed = false,
        const std::vector<uint32_t>* clusters_access_order = nullptr
    ) const override {
        const uint32_t* preset_clusters_access_order =
            clusters_access_order ? clusters_access_order->data() : nullptr;
        if (!passing_row_ids) {
            return std::make_unique<typename PDXearch<Q>::template IterativeSearch<false>>(
                searcher->BeginIterativeSearch(
                    query_embedding,
                    knn,
                    top_k_heap,
                    is_query_transformed,
                    preset_clusters_access_order
                )
            );
        }
        auto evaluator =
            std::make_unique<PredicateEvaluator>(CreatePredicateEvaluator(*passing_row_ids));
        return std::make_unique<typename PDXearch<Q>::template IterativeSearch<true>>(
            searcher->BeginFilteredIterativeSearch(
                query_embedding,
                knn,
                std::move(evaluator),
                top_k_heap,
                is_query_transformed,
                preset_clusters_access_order
            )
        );
    }

    std::unique_ptr<PredicateEvaluator> CreateSharedPredicateEvaluator(
        const std::vector<size_t>& passing_row_ids
    ) const override {
        return std::make_unique<PredicateEvaluator>(CreatePredicateEvaluator(passing_row_ids));
    }

    std::unique_ptr<IIterativeSearch> BeginIterativeSearchWithSharedEvaluator(
        const float* query_embedding,
        uint32_t knn,
        TopKHeap& top_k_heap,
        const PredicateEvaluator& evaluator,
        bool is_query_transformed = false,
        const std::vector<uint32_t>* clusters_access_order = nullptr
    ) const override {
        return std::make_unique<typename PDXearch<Q>::template IterativeSearch<true>>(
            searcher->BeginFilteredIterativeSearch(
                query_embedding,
                knn,
                evaluator,
                top_k_heap,
                is_query_transformed,
                clusters_access_order ? clusters_access_order->data() : nullptr
            )
        );
    }

    void GetEmbeddingsFromIndexByRowIds(const std::vector<size_t>& row_ids, float* out)
        const override {
        GetEmbeddingsFromIndexByRowIdsImpl<Q>(
            index,
            searcher->quantizer,
            row_id_cluster_mapping,
            row_ids,
            out,
            searcher->cluster_source
        );
    }

    void GetDistancesToCentroids(
        const float* query_embedding,
        bool is_query_transformed,
        float* out
    ) const override {
        searcher->GetDistancesToCentroids(query_embedding, is_query_transformed, out);
    }

    std::vector<uint32_t> GetClustersAccessOrder(
        const float* query_embedding,
        bool is_query_transformed = false
    ) const override {
        return searcher->GetClustersAccessOrder(query_embedding, is_query_transformed);
    }

    void SetNProbe(uint32_t n_probe) override { searcher->SetNProbe(n_probe); }

    const PDX::PDXearch<Q>& GetSearcher() const { return *searcher; }

    std::pair<uint32_t, uint32_t> GetRowIdMapping(size_t row_id) const override {
        return row_id_cluster_mapping.Get(row_id);
    }

    uint32_t GetNumDimensions() const override { return index.num_dimensions; }

    uint32_t GetNumClusters() const override { return index.num_clusters; }

    size_t GetNumVectorsAccessed() const {
        size_t total = 0;
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            total += index.clusters[c].n_accessed * index.clusters[c].num_embeddings;
        }
        return total;
    }

    uint32_t GetClusterSize(uint32_t cluster_id) const override {
        return index.clusters[cluster_id].num_embeddings;
    }

    std::vector<uint32_t> GetClusterRowIds(uint32_t cluster_id) const override {
        const auto& cluster = index.clusters[cluster_id];
        std::vector<uint32_t> row_ids;
        row_ids.reserve(cluster.num_embeddings);
        for (uint32_t i = 0; i < cluster.used_capacity; i++) {
            if (!cluster.HasTombstone(i)) {
                row_ids.push_back(cluster.indices[i]);
            }
        }
        return row_ids;
    }

    size_t GetInMemorySizeInBytes() const override {
        size_t size = sizeof(*this);
        // IVF heap allocations (sizeof(IVF<Q>) is inline in sizeof(*this))
        size += index.GetInMemorySizeInBytes() - sizeof(index);
        // Pruner: rotation matrix or flip_masks (DCT mode) + ratios vector (only if owned)
        if (owned_pruner) {
            size += sizeof(*pruner);
            const auto& m = pruner->GetMatrix();
            // matrix heap data (1 x D for DCT sign vector, D x D for full rotation)
            size += static_cast<size_t>(m.rows()) * m.cols() * sizeof(float);
            size += pruner->num_dimensions * sizeof(float); // ratios
            if (m.rows() == 1) {
                size += pruner->num_dimensions * sizeof(uint32_t); // flip_masks
            }
        }
        if (searcher) {
            size += sizeof(*searcher);
        }
        size += row_id_cluster_mapping.SizeInBytes();
        return size;
    }

    size_t GetAllocatedSizeInBytes() const override {
        return index.GetAllocatedSizeInBytes() +
               (owned_pruner ? owned_pruner->GetAllocatedSizeInBytes() : 0);
    }

    void BuildIndex(const float* const embeddings, const size_t num_embeddings) override {
        std::vector<size_t> row_ids(num_embeddings);
        std::iota(row_ids.begin(), row_ids.end(), 0);
        BuildIndex(row_ids.data(), embeddings, num_embeddings);
    }

    void BuildIndex(
        const size_t* const row_ids,
        const float* const embeddings,
        const size_t num_embeddings
    ) {
        config.ValidateNumEmbeddings(num_embeddings);
        skmeans::ParallelSection parallel_section(GetExecutor());
        const IVFUtils ivf_utils(GetExecutor(), &GetAllocator());

        const auto num_dimensions = config.num_dimensions;
        auto num_clusters = config.num_clusters;
        if (num_clusters == 0) {
            num_clusters = ComputeNumberOfClusters(num_embeddings);
        }
        const bool normalize =
            config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);

        assert(num_embeddings > 0);
        assert(pruner);

        skmeans::AllocatedBuffer<float> transformed;
        if (!config.is_data_transformed) {
            transformed = ivf_utils.NormalizeAndRotate(
                embeddings, num_embeddings, num_dimensions, normalize, *pruner
            );
        }
        const float* preprocessed = config.is_data_transformed ? embeddings : transformed.get();

        float quantization_base = 0.0f;
        float quantization_scale = 1.0f;
        if constexpr (Q == PDX::U8) {
            const PDX::ScalarQuantizer<Q> quantizer(num_dimensions, &GetExecutor());
            const auto params = quantizer.ComputeQuantizationParams(
                preprocessed, static_cast<size_t>(num_embeddings) * num_dimensions
            );
            quantization_base = params.quantization_base;
            quantization_scale = params.quantization_scale;
            index = PDX::IVF<Q>(
                num_dimensions,
                num_embeddings,
                num_clusters,
                normalize,
                quantization_scale,
                quantization_base
            );
        } else {
            index = PDX::IVF<Q>(num_dimensions, num_embeddings, num_clusters, normalize);
        }
        index.SetAllocator(&GetAllocator());

        KMeansResult kmeans_result = ivf_utils.ComputeKMeans(
            preprocessed,
            num_embeddings,
            num_dimensions,
            num_clusters,
            config.distance_metric,
            config.seed,
            config.normalize,
            config.sampling_fraction,
            config.kmeans_iters,
            config.hierarchical_indexing
        );
        if (config.max_cluster_size > 0) {
            ivf_utils.SplitClustersAboveMaxClusterSize(
                kmeans_result,
                preprocessed,
                num_dimensions,
                config.max_cluster_size,
                config.distance_metric,
                config.seed,
                config.normalize
            );
            num_clusters = static_cast<uint32_t>(kmeans_result.assignments.size());
            index.num_clusters = num_clusters;
        }
        index.centroids.StoreCentroids(kmeans_result.centroids.data(), num_clusters);

        ivf_utils.PopulateIVFClusters<Q>(
            index,
            kmeans_result,
            preprocessed,
            row_ids,
            num_dimensions,
            num_clusters,
            quantization_base,
            quantization_scale,
            config.max_cluster_size,
            config.expect_appends
        );

        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
        BuildRowIdClusterMapping();
    }

    // ******************************************
    // Maintenance (SPFresh-like): same recipe as PDXTreeIndex minus the meso-cluster layer.
    // The candidates for placing, splitting and reassigning embeddings are all the clusters.
    // ******************************************

    // Concurrent writes must always go through a single writer thread
    void Append(size_t row_id, const float* PDX_RESTRICT embedding) override {
        PDX_PROFILE_SCOPE("Append");
        if (Contains(row_id)) {
            throw std::invalid_argument(
                "Append: row_id " + std::to_string(row_id) + " already exists in the index"
            );
        }
        ReserveClusterSlotIfNeeded();
        skmeans::ParallelSection parallel_section(GetExecutor());

        skmeans::AllocatedBuffer<float> transformed;
        if (!config.is_data_transformed) {
            transformed = IVFUtils(GetExecutor())
                              .NormalizeAndRotate(
                                  embedding, 1, index.num_dimensions, index.is_normalized, *pruner
                              );
        }
        const float* preprocessed = config.is_data_transformed ? embedding : transformed.get();

        // Find nearest centroid for the new embedding
        uint32_t closest_centroid_idx;
        {
            PDX_PROFILE_SCOPE("Append/FindNearestCentroid");
            closest_centroid_idx = FindNearestCentroid(preprocessed);
        }

        auto& cluster = index.clusters[closest_centroid_idx];
        if (!cluster.data) {
            throw std::logic_error("Append: the index was loaded without its clusters' data");
        }

        uint32_t new_index_in_cluster = QuantizeAndAppend<Q>(
            index, searcher->quantizer, cluster, static_cast<uint32_t>(row_id), preprocessed
        );
        SetRowIdMapping(row_id, closest_centroid_idx, new_index_in_cluster);
        index.total_num_embeddings++;
        CheckClusterHealth(cluster);
    }

    void BulkAppend(const size_t* row_ids, const float* embeddings, size_t num_embeddings)
        override {
        PDX_PROFILE_SCOPE("BulkAppend");
        if (num_embeddings == 0) {
            return;
        }
        ValidateBulkAppendRowIds(row_ids, num_embeddings, config.base_row_id);
        if (index.num_clusters == 0) {
            throw std::logic_error("BulkAppend: the index is not built");
        }
        if (searcher->cluster_source) {
            throw std::logic_error("BulkAppend: the index was loaded without its clusters' data");
        }
        skmeans::ParallelSection parallel_section(GetExecutor());
        const IVFUtils ivf_utils(GetExecutor(), &GetAllocator());
        skmeans::AllocatedBuffer<float> transformed;
        if (!config.is_data_transformed) {
            PDX_PROFILE_SCOPE("BulkAppend/Transform");
            transformed = ivf_utils.NormalizeAndRotate(
                embeddings, num_embeddings, index.num_dimensions, index.is_normalized, *pruner
            );
        }
        const float* pending_embeddings =
            config.is_data_transformed ? embeddings : transformed.get();
        auto pending_row_ids = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), num_embeddings);
        for (size_t i = 0; i < num_embeddings; i++) {
            pending_row_ids[i] = static_cast<uint32_t>(row_ids[i]);
        }
        skmeans::AllocatedBuffer<float> held_embeddings;
        while (pending_row_ids.get_deleter().Size() > 0) {
            PDX_PROFILE_SCOPE("BulkAppend/Round");
            held_embeddings = BulkAppendRound(ivf_utils, pending_embeddings, pending_row_ids);
            pending_embeddings = held_embeddings.get();
        }
        index.ComputeClusterOffsets();
    }

    // Concurrent deletes must always go through a single writer thread
    bool Delete(size_t row_id) override {
        PDX_PROFILE_SCOPE("Delete");
        const auto [cluster_id, index_in_cluster] = GetRowIdMapping(row_id);
        if (cluster_id == DELETED_MARKER) {
            return false;
        }
        ReserveClusterSlotIfNeeded();
        skmeans::ParallelSection parallel_section(GetExecutor());
        auto& cluster = index.clusters[cluster_id];
        cluster.DeleteEmbedding(index_in_cluster);
        DeleteRowIdMapping(row_id);
        index.total_num_embeddings--;
        if (cluster.data) {
            CheckClusterHealth(cluster);
        }
        return true;
    }

    void ShrinkToFit() override {
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            auto& cluster = index.clusters[c];
            if (!cluster.data) {
                throw std::logic_error(
                    "ShrinkToFit: the index was loaded without its clusters' data"
                );
            }
            cluster.ShrinkToFit();
        }
        index.centroids.ShrinkToFit();
        index.ComputeClusterOffsets();
    }

  private:
    static constexpr PDXIndexType GetIndexType() {
        if constexpr (Q == F32)
            return PDXIndexType::PDX_F32;
        else
            return PDXIndexType::PDX_U8;
    }

    void SetRowIdMapping(uint32_t row_id, uint32_t cluster_id, uint32_t idx_in_cluster) {
        row_id_cluster_mapping.Set(row_id, cluster_id, idx_in_cluster);
    }

    void DeleteRowIdMapping(uint32_t row_id) { row_id_cluster_mapping.Delete(row_id); }

    void BuildRowIdClusterMapping() {
        row_id_cluster_mapping.Rebuild(index.clusters, index.num_clusters);
    }

    void CompactClusters() {
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            auto moves = index.clusters[c].CompactCluster();
            for (const auto& [row_id, new_idx] : moves) {
                SetRowIdMapping(row_id, c, new_idx);
            }
        }
    }

    PDX::PredicateEvaluator CreatePredicateEvaluator(const std::vector<size_t>& passing_row_ids
    ) const {
        PDX_PROFILE_SCOPE("PredicateEvaluator");
        PDX::PredicateEvaluator evaluator(index.num_clusters, index.total_capacity);
        if (index.num_clusters == 0 || index.total_capacity == 0) {
            return evaluator;
        }
        for (const auto row_id : passing_row_ids) {
            const auto [cluster_id, index_in_cluster] = GetRowIdMapping(row_id);
            if (cluster_id == DELETED_MARKER)
                continue;
            evaluator.n_passing_tuples[cluster_id]++;
            evaluator.total_passing_tuples++;
            evaluator.selection_vector[index.cluster_offsets[cluster_id] + index_in_cluster] = 1;
        }
        return evaluator;
    }

    // Ensure the clusters vector won't reallocate while we hold a reference
    void ReserveClusterSlotIfNeeded() {
        if (index.clusters.size() == index.clusters.capacity()) {
            index.clusters.reserve(index.clusters.capacity() * 2);
        }
    }

    // Scan over all centroids (embedding is already normalized + rotated)
    uint32_t FindNearestCentroid(const float* embedding) const {
        std::unique_ptr<float[]> distances_to_centroids(new float[index.num_clusters]);
        index.centroids.ComputeDistancesToCentroids(embedding, distances_to_centroids.get());
        return static_cast<uint32_t>(
            std::min_element(
                distances_to_centroids.get(), distances_to_centroids.get() + index.num_clusters
            ) -
            distances_to_centroids.get()
        );
    }

    // Ids of the max_neighbors clusters whose centroids are nearest to `centroid`, excluding
    // cluster_id itself. Exact scan over all centroids; the result is unordered.
    std::vector<uint32_t> GetNearestNeighborClusterIds(
        uint32_t cluster_id,
        const float* centroid,
        size_t max_neighbors = SPLIT_MAX_NEIGHBOR_CLUSTERS
    ) const {
        PDX_PROFILE_SCOPE("GetNeighboringClusters");
        std::unique_ptr<float[]> distances_to_centroids(new float[index.num_clusters]);
        index.centroids.ComputeDistancesToCentroids(centroid, distances_to_centroids.get());
        std::vector<std::pair<float, uint32_t>> neighbor_dists;
        neighbor_dists.reserve(index.num_clusters);
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            if (c == cluster_id)
                continue;
            neighbor_dists.emplace_back(distances_to_centroids[c], c);
        }
        if (neighbor_dists.size() > max_neighbors) {
            std::nth_element(
                neighbor_dists.begin(),
                neighbor_dists.begin() + max_neighbors,
                neighbor_dists.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
            neighbor_dists.resize(max_neighbors);
        }
        std::vector<uint32_t> neighbor_ids;
        neighbor_ids.reserve(neighbor_dists.size());
        for (const auto& [dist, id] : neighbor_dists) {
            neighbor_ids.push_back(id);
        }
        return neighbor_ids;
    }

    void CheckClusterHealth(cluster_t& cluster, bool allow_merges = true) {
        if (cluster.used_capacity == cluster.max_capacity) {
            // Its less expensive to compact than to Split
            if (cluster.num_embeddings < cluster.used_capacity) {
                auto moves = cluster.CompactCluster();
                for (const auto& [row_id, new_idx] : moves) {
                    SetRowIdMapping(row_id, cluster.id, new_idx);
                }
                // NOLINTNEXTLINE(bugprone-branch-clone)
            } else if (cluster.max_capacity < config.max_cluster_size) {
                cluster.GrowTo(
                    cluster_t::CapacityFor(cluster.num_embeddings, config.max_cluster_size)
                );
                index.ComputeClusterOffsets();
            } else if (split_depth >= SPLIT_CASCADE_MAX_DEPTH) {
                cluster.Grow();
                index.ComputeClusterOffsets();
            } else {
                SplitCluster(cluster);
            }
        } else if (allow_merges && index.num_clusters > 1 &&
                   cluster.num_embeddings <= cluster.min_capacity) {
            DestroyAndMergeCluster(cluster);
        }
    }

    void DestroyAndMergeCluster(cluster_t& cluster) {
        PDX_PROFILE_SCOPE("Merge");
        cluster.CompactCluster();
        const uint32_t cluster_id = cluster.id;
        const uint32_t n_emb = cluster.num_embeddings;

        auto raw_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();
        auto cluster_indices = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), n_emb);
        std::memcpy(cluster_indices.get(), cluster.indices.get(), n_emb * sizeof(uint32_t));
        auto cluster_embeddings = DequantizeClusterEmbeddings<Q>(
            index, searcher->quantizer, raw_embeddings.get(), n_emb, GetAllocator()
        );

        // Swap-and-pop: move last cluster into the dead slot
        uint32_t last_id = index.num_clusters - 1;
        if (cluster_id != last_id) {
            index.clusters[cluster_id] = std::move(index.clusters[last_id]);
            index.clusters[cluster_id].id = cluster_id;

            auto& moved_cluster = index.clusters[cluster_id];
            for (uint32_t i = 0; i < moved_cluster.used_capacity; i++) {
                if (!moved_cluster.HasTombstone(i)) {
                    SetRowIdMapping(moved_cluster.indices[i], cluster_id, i);
                }
            }
        }

        // Pop the dead cluster and its centroid (ensured to be at the end)
        index.clusters.pop_back();
        index.centroids.RemoveCentroidBySwapWithLast(cluster_id);
        index.num_clusters--;

        index.ComputeClusterOffsets();

        // Fully removed the dying cluster before reassignment
        ReassignEmbeddings(cluster_indices.get(), cluster_embeddings.get(), n_emb, false);
    }

    // Assumes cluster is compacted and has no tombstones
    void SplitCluster(cluster_t& cluster) {
        PDX_PROFILE_SCOPE("Split");
        const uint32_t cluster_id = cluster.id;
        const uint32_t d = index.num_dimensions;

        auto raw_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();
        auto cluster_embeddings = DequantizeClusterEmbeddings<Q>(
            index, searcher->quantizer, raw_embeddings.get(), cluster.num_embeddings, GetAllocator()
        );

        std::unique_ptr<float[]> decoded_centroid_to_split(new float[d]);
        const float* centroid_to_split =
            index.centroids.GetF32CentroidPtr(cluster_id, decoded_centroid_to_split.get());
        auto neighboring_clusters_ids = GetNearestNeighborClusterIds(cluster_id, centroid_to_split);

        // 2-means split: each embedding goes to A, B, or rest (closer to a neighboring cluster)
        auto partition = PartitionClusterForSplit<Q>(
            index,
            cluster_embeddings.get(),
            cluster.num_embeddings,
            centroid_to_split,
            neighboring_clusters_ids,
            config.distance_metric,
            config.seed,
            GetExecutor(),
            GetAllocator()
        );
        auto& centroid_a = partition.centroid_a;
        auto& centroid_b = partition.centroid_b;
        auto& group_a_idx = partition.group_a_idx;
        auto& group_b_idx = partition.group_b_idx;
        auto& group_rest_idx = partition.group_rest_idx;

        // A side is empty (e.g. identical embeddings): grow the cluster instead of splitting off
        // an empty one, which would only send the rest back here and split again
        if (group_a_idx.empty() || group_b_idx.empty()) {
            cluster.Grow();
            index.ComputeClusterOffsets();
            return;
        }

        // Neighbors' rows closer to A or B than to their own centroid, stolen below
        const auto rows_to_steal = FindNeighborRowsToSteal<Q>(
            index,
            neighboring_clusters_ids,
            centroid_a.get(),
            centroid_b.get(),
            GetExecutor(),
            GetAllocator()
        );
        const auto [num_stolen_a, num_stolen_b] = CountRowsToSteal(rows_to_steal);
        const size_t count_a = group_a_idx.size() + num_stolen_a;
        const size_t count_b = group_b_idx.size() + num_stolen_b;

        // Gather embeddings and IDs, accumulate centroid sums
        auto embs_a = skmeans::AllocateBuffer<embedding_storage_t>(GetAllocator(), count_a * d);
        auto embs_b = skmeans::AllocateBuffer<embedding_storage_t>(GetAllocator(), count_b * d);
        auto ids_a = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), count_a);
        auto ids_b = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), count_b);
        auto centroid_sum_a = std::make_unique<float[]>(d); // zero-init needed
        auto centroid_sum_b = std::make_unique<float[]>(d); // zero-init needed
        {
            PDX_PROFILE_SCOPE("Split/GatherEmbeddings");
            GatherGroupEmbeddings<Q>(
                index,
                cluster,
                group_a_idx,
                raw_embeddings.get(),
                cluster_embeddings.get(),
                embs_a.get(),
                ids_a.get(),
                centroid_sum_a.get()
            );
            GatherGroupEmbeddings<Q>(
                index,
                cluster,
                group_b_idx,
                raw_embeddings.get(),
                cluster_embeddings.get(),
                embs_b.get(),
                ids_b.get(),
                centroid_sum_b.get()
            );
        }

        // Gather group_rest NOW, before the cluster is replaced
        auto float_rest = skmeans::AllocateBuffer<float>(GetAllocator(), group_rest_idx.size() * d);
        auto ids_rest = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), group_rest_idx.size());
        for (size_t i = 0; i < group_rest_idx.size(); i++) {
            std::memcpy(
                float_rest.get() + i * d,
                cluster_embeddings.get() + static_cast<size_t>(group_rest_idx[i]) * d,
                d * sizeof(float)
            );
            ids_rest[i] = cluster.indices[group_rest_idx[i]];
        }

        // Steal neighbors closer to A or B than to their own centroid
        StealNeighborEmbeddings<Q>(
            index,
            neighboring_clusters_ids,
            rows_to_steal,
            embs_a.get() + group_a_idx.size() * d,
            ids_a.get() + group_a_idx.size(),
            centroid_sum_a.get(),
            embs_b.get() + group_b_idx.size() * d,
            ids_b.get() + group_b_idx.size(),
            centroid_sum_b.get()
        );

        // Compute true centroids from accumulated sums
        std::unique_ptr<float[]> true_centroid_a(new float[d]);
        std::unique_ptr<float[]> true_centroid_b(new float[d]);
        {
            PDX_PROFILE_SCOPE("Split/ComputeTrueCentroids");
            ComputeCentroidMean(
                d,
                index.is_normalized,
                centroid_sum_a.get(),
                count_a,
                centroid_a.get(),
                true_centroid_a.get()
            );
            ComputeCentroidMean(
                d,
                index.is_normalized,
                centroid_sum_b.get(),
                count_b,
                centroid_b.get(),
                true_centroid_b.get()
            );
        }

        // Create new clusters and update all data structures
        {
            PDX_PROFILE_SCOPE("Split/ConsolidateNewClusters");
            const auto size_a = static_cast<uint32_t>(count_a);
            cluster_t new_cluster_a(
                size_a, cluster_t::CapacityFor(size_a, config.max_cluster_size), d, &GetAllocator()
            );
            new_cluster_a.id = cluster_id;
            if (count_a > 0) {
                std::memcpy(new_cluster_a.indices.get(), ids_a.get(), count_a * sizeof(uint32_t));
                new_cluster_a.StoreEmbeddings(embs_a.get(), count_a);
            }
            uint32_t new_cluster_b_id = index.num_clusters;
            const auto size_b = static_cast<uint32_t>(count_b);
            cluster_t new_cluster_b(
                size_b, cluster_t::CapacityFor(size_b, config.max_cluster_size), d, &GetAllocator()
            );
            new_cluster_b.id = new_cluster_b_id;
            if (count_b > 0) {
                std::memcpy(new_cluster_b.indices.get(), ids_b.get(), count_b * sizeof(uint32_t));
                new_cluster_b.StoreEmbeddings(embs_b.get(), count_b);
            }
            // Replace old cluster with A, append B
            index.clusters[cluster_id] = std::move(new_cluster_a);
            index.clusters.push_back(std::move(new_cluster_b));
            index.num_clusters++;
            // Update centroids
            index.centroids.StoreCentroid(cluster_id, true_centroid_a.get());
            index.centroids.AppendCentroid(true_centroid_b.get());
            // Update row_id_cluster_mapping (includes both original and stolen-neighbor points)
            for (size_t i = 0; i < count_a; i++) {
                SetRowIdMapping(ids_a[i], cluster_id, static_cast<uint32_t>(i));
            }
            for (size_t i = 0; i < count_b; i++) {
                SetRowIdMapping(ids_b[i], new_cluster_b_id, static_cast<uint32_t>(i));
            }
        }

        // Reassign rest group (closer to other centroids than A or B)
        if (!group_rest_idx.empty()) {
            split_depth++;
            struct DepthGuard {
                uint32_t& depth;
                ~DepthGuard() { depth--; }
            } depth_guard{split_depth};
            // The rest group was closer to one of the neighbors than to A and B
            std::vector<uint32_t> rest_candidates = neighboring_clusters_ids;
            rest_candidates.push_back(cluster_id);
            rest_candidates.push_back(index.num_clusters - 1);
            ReassignEmbeddings(
                ids_rest.get(),
                float_rest.get(),
                static_cast<uint32_t>(group_rest_idx.size()),
                false,
                &rest_candidates
            );
        }

        index.ComputeClusterOffsets();
    }

    // One round of BulkAppend: each cluster takes what fits of its rows (one slot stays free), then
    // the clusters that overflowed split. Returns the rows held back, whose ids replace row_ids.
    skmeans::AllocatedBuffer<float> BulkAppendRound(
        const IVFUtils& ivf_utils,
        const float* embeddings,
        skmeans::AllocatedBuffer<uint32_t>& row_ids
    ) {
        const uint32_t d = index.num_dimensions;
        const size_t num_rows = row_ids.get_deleter().Size();
        auto assignments = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), num_rows);
        {
            PDX_PROFILE_SCOPE("BulkAppend/Assign");
            skmeans::AllocatedBuffer<float> decoded_centroids;
            ivf_utils.AssignToNearestCentroids(
                embeddings,
                num_rows,
                index.centroids.GetF32CentroidsPtr(decoded_centroids),
                index.centroids.GetF32CentroidsSquaredNormsPtr(),
                index.num_clusters,
                d,
                assignments.get()
            );
        }
        std::vector<uint32_t> cluster_starts;
        auto rows_by_cluster = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), num_rows);
        IVFUtils::GroupRowsByCluster(
            assignments.get(), num_rows, index.num_clusters, cluster_starts, rows_by_cluster.get()
        );

        std::vector<IVFUtils::ClusterAppendRange> ranges;
        std::vector<std::pair<uint32_t, uint32_t>> overflowed_clusters; // (cluster, trigger row)
        auto held_rows = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), num_rows);
        size_t num_held = 0;
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            const uint32_t first = cluster_starts[c];
            const uint32_t num_cluster_rows = cluster_starts[c + 1] - first;
            if (num_cluster_rows == 0) {
                continue;
            }
            cluster_t& cluster = index.clusters[c];
            if (cluster.GrowToFit(num_cluster_rows, config.max_cluster_size)) {
                index.ComputeClusterOffsets();
            }
            const uint32_t free_slots = cluster.max_capacity - cluster.used_capacity +
                                        static_cast<uint32_t>(cluster.tombstones.size());
            const uint32_t num_fitting = std::min(num_cluster_rows, free_slots - 1);
            if (num_fitting > 0) {
                ranges.push_back({c, first, num_fitting});
            }
            if (num_fitting < num_cluster_rows) {
                overflowed_clusters.emplace_back(c, rows_by_cluster[first + num_fitting]);
                for (uint32_t j = num_fitting + 1; j < num_cluster_rows; j++) {
                    held_rows[num_held++] = rows_by_cluster[first + j];
                }
            }
        }

        auto positions = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), num_rows);
        {
            PDX_PROFILE_SCOPE("BulkAppend/Write");
            ivf_utils.AppendRowsToClusters(
                index, embeddings, row_ids.get(), rows_by_cluster.get(), ranges, positions.get()
            );
        }
        for (const auto& range : ranges) {
            for (uint32_t j = 0; j < range.num_rows; j++) {
                const uint32_t row = rows_by_cluster[range.first_row + j];
                SetRowIdMapping(row_ids[row], range.cluster_id, positions[row]);
            }
            index.total_num_embeddings += range.num_rows;
        }
        {
            PDX_PROFILE_SCOPE("BulkAppend/Split");
            SplitOverflowedClusters(embeddings, row_ids.get(), overflowed_clusters);
        }

        auto held_embeddings = skmeans::AllocateBuffer<float>(GetAllocator(), num_held * d);
        auto held_row_ids = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), num_held);
        for (size_t i = 0; i < num_held; i++) {
            std::memcpy(
                held_embeddings.get() + i * d,
                embeddings + static_cast<size_t>(held_rows[i]) * d,
                d * sizeof(float)
            );
            held_row_ids[i] = row_ids[held_rows[i]];
        }
        row_ids = std::move(held_row_ids);
        return held_embeddings;
    }

    // Each cluster's trigger row fills its last free slot, so the cluster splits as with Append
    void SplitOverflowedClusters(
        const float* embeddings,
        const uint32_t* row_ids,
        const std::vector<std::pair<uint32_t, uint32_t>>& overflowed_clusters
    ) {
        const uint32_t d = index.num_dimensions;
        for (const auto& [cluster_id, trigger_row] : overflowed_clusters) {
            ReserveClusterSlotIfNeeded();
            cluster_t& cluster = index.clusters[cluster_id];
            const uint32_t row_id = row_ids[trigger_row];
            const uint32_t position = QuantizeAndAppend<Q>(
                index,
                searcher->quantizer,
                cluster,
                row_id,
                embeddings + static_cast<size_t>(trigger_row) * d
            );
            SetRowIdMapping(row_id, cluster_id, position);
            index.total_num_embeddings++;
            CheckClusterHealth(cluster, /*allow_merges=*/false);
        }
    }

    // Reassign dequantized (float) embeddings to their closest centroid among candidate_cluster_ids
    // (all clusters when null).
    // allow_merges: passed to CheckClusterHealth — false suppresses merge cascades.
    // TODO(@lkuffo, med): We can optimize reassignments by doing GEMM+PRUNING for assignments
    // TODO(@lkuffo, high): The assignments are computed once, against the centroids at loop start.
    // A split or merge run by CheckClusterHealth inside the loop restructures the clusters (a merge
    // swaps-and-pops, so a precomputed id can point past the end), which is why every caller passes
    // allow_merges=false today and merges of clusters drained by StealNeighborEmbeddings lag until
    // a later Append/Delete touches them. Root fix: bump a structure version in SplitCluster and
    // DestroyAndMergeCluster and recompute the assignments of the remaining points when it changes.
    void ReassignEmbeddings(
        const uint32_t* row_ids,
        const float* embeddings,
        uint32_t num_embeddings,
        bool allow_merges = true,
        const std::vector<uint32_t>* candidate_cluster_ids = nullptr
    ) {
        PDX_PROFILE_SCOPE("Reassign");
        if (num_embeddings == 0) {
            return;
        }
        const uint32_t d = index.num_dimensions;
        const float* centroids = nullptr;
        uint32_t n_clusters = index.num_clusters;
        skmeans::AllocatedBuffer<float> candidate_centroids;
        if (candidate_cluster_ids != nullptr) {
            n_clusters = static_cast<uint32_t>(candidate_cluster_ids->size());
            candidate_centroids =
                skmeans::AllocateBuffer<float>(GetAllocator(), static_cast<size_t>(n_clusters) * d);
            for (size_t i = 0; i < n_clusters; i++) {
                index.centroids.DecodeCentroid(
                    (*candidate_cluster_ids)[i], candidate_centroids.get() + i * d
                );
            }
            centroids = candidate_centroids.get();
        } else {
            centroids = index.centroids.GetF32CentroidsPtr(candidate_centroids);
        }

        auto assignments = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), num_embeddings);
        auto result_distances = skmeans::AllocateBuffer<float>(GetAllocator(), num_embeddings);
        auto tmp_distances_buf = skmeans::AllocateBuffer<float>(
            GetAllocator(), batch_computer::ScratchSize(GetExecutor())
        );

        auto embeddings_norms = skmeans::AllocateBuffer<float>(GetAllocator(), num_embeddings);
        Eigen::Map<const MatrixR> embeddings_matrix(embeddings, num_embeddings, d);
        Eigen::Map<VectorR> v_norms(embeddings_norms.get(), num_embeddings);
        v_norms.noalias() = embeddings_matrix.rowwise().squaredNorm();

        std::vector<float> centroid_norms(n_clusters);
        Eigen::Map<const MatrixR> centroids_matrix(centroids, n_clusters, d);
        Eigen::Map<VectorR> c_norms(centroid_norms.data(), n_clusters);
        c_norms.noalias() = centroids_matrix.rowwise().squaredNorm();

        batch_computer::FindNearestNeighbor(
            GetExecutor(),
            embeddings,
            centroids,
            num_embeddings,
            n_clusters,
            d,
            embeddings_norms.get(),
            centroid_norms.data(),
            assignments.get(),
            result_distances.get(),
            tmp_distances_buf.get()
        );

        for (size_t i = 0; i < num_embeddings; i++) {
            const uint32_t best_cluster = candidate_cluster_ids != nullptr
                                              ? (*candidate_cluster_ids)[assignments[i]]
                                              : assignments[i];
            uint32_t row_id = row_ids[i];
            uint32_t new_pos = QuantizeAndAppend<Q>(
                index, searcher->quantizer, index.clusters[best_cluster], row_id, embeddings + i * d
            );
            SetRowIdMapping(row_id, best_cluster, new_pos);
            ReserveClusterSlotIfNeeded();
            CheckClusterHealth(index.clusters[best_cluster], allow_merges);
        }
    }
};

} // namespace PDX

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
#include <unordered_map>
#include <utility>
#include <vector>

#include "pdx/common.hpp"
#include "pdx/indexes/ivf_utils.hpp"
#include "pdx/profiler.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"
#include "pdx/searcher.hpp"

namespace PDX {

class IPDXIndex {
  public:
    virtual ~IPDXIndex() = default;
    virtual std::vector<KNNCandidate> Search(const float* query_embedding, size_t knn) const = 0;
    virtual std::vector<KNNCandidate> FilteredSearch(
        const float* query_embedding,
        size_t knn,
        const std::vector<size_t>& passing_row_ids
    ) const = 0;
    virtual void BuildIndex(const float* embeddings, size_t num_embeddings) = 0;
    virtual void SetNProbe(uint32_t n_probe) = 0;
    virtual void Save(const std::string& path) = 0;
    virtual void Restore(const std::string& path) = 0;
    virtual uint32_t GetNumDimensions() const = 0;
    virtual uint32_t GetNumClusters() const = 0;
    virtual uint32_t GetClusterSize(uint32_t cluster_id) const = 0;
    virtual std::vector<uint32_t> GetClusterRowIds(uint32_t cluster_id) const = 0;
    virtual size_t GetInMemorySizeInBytes() const = 0;
    // Maintenance (SPFresh-like). Concurrent writes must go through a single writer thread.
    virtual void Append(size_t row_id, const float* embedding) = 0;
    virtual void Delete(size_t row_id) = 0;
    // Resumable search into the caller's TopKHeap (construct it thread_safe when several cursors
    // share it); passing_row_ids to filter (nullptr: unfiltered).
    virtual std::unique_ptr<IIterativeSearch> BeginIterativeSearch(
        const float* query_embedding,
        uint32_t knn,
        TopKHeap& top_k_heap,
        const std::vector<size_t>* passing_row_ids
    ) const = 0;
};

template <PDX::Quantization Q>
class PDXIndex : public IPDXIndex {
  public:
    using embedding_storage_t = PDX::pdx_data_t<Q>;
    using cluster_t = PDX::Cluster<Q>;
    using distance_computer_f32_t = DistanceComputer<DistanceMetric::L2SQ, F32>;
    using batch_computer =
        skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;

  private:
    static constexpr uint32_t DELETED_MARKER = std::numeric_limits<uint32_t>::max();

    PDXIndexConfig config{};
    PDX::IVF<Q> index;
    std::unique_ptr<PDX::ADSamplingPruner> pruner;
    std::unique_ptr<PDX::PDXearch<Q>> searcher;
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> row_id_cluster_mapping;

  public:
    PDXIndex() = default;

    explicit PDXIndex(PDXIndexConfig config) : config(config) {
        config.Validate();
        PDX::g_n_threads = (config.n_threads == 0) ? omp_get_max_threads() : config.n_threads;
        pruner = std::make_unique<PDX::ADSamplingPruner>(config.num_dimensions, config.seed);
    }

    void Save(const std::string& path) override {
        // Compact all clusters before saving
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            auto moves = index.clusters[c].CompactCluster();
            for (const auto& [row_id, new_idx] : moves) {
                SetRowIdMapping(row_id, c, new_idx);
            }
        }

        std::ofstream out(path, std::ios::binary);

        uint8_t type_flag = static_cast<uint8_t>(GetIndexType());
        out.write(reinterpret_cast<const char*>(&type_flag), sizeof(uint8_t));

        // Rotation matrix
        const auto& matrix = pruner->GetMatrix();
        uint32_t matrix_rows = static_cast<uint32_t>(matrix.rows());
        uint32_t matrix_cols = static_cast<uint32_t>(matrix.cols());
        out.write(reinterpret_cast<const char*>(&matrix_rows), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&matrix_cols), sizeof(uint32_t));
        out.write(
            reinterpret_cast<const char*>(matrix.data()), sizeof(float) * matrix_rows * matrix_cols
        );

        // IVF data
        index.Save(out);
    }

    void Restore(const std::string& path) override {
        auto buffer = MmapFile(path);
        char* ptr = buffer.get();

        // Index type flag
        ptr += sizeof(uint8_t);

        // Rotation matrix (ptr may be misaligned after the uint8_t type flag)
        uint32_t matrix_rows, matrix_cols;
        std::memcpy(&matrix_rows, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        std::memcpy(&matrix_cols, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        const size_t matrix_floats = static_cast<size_t>(matrix_rows) * matrix_cols;
        auto aligned_matrix = std::unique_ptr<float[]>(new float[matrix_floats]);
        std::memcpy(aligned_matrix.get(), ptr, sizeof(float) * matrix_floats);
        ptr += sizeof(float) * matrix_floats;

        // Load IVF data
        index.Load(ptr);
        // No PDXIndexConfig is stored on disk: recover what maintenance needs so that an index
        // loaded through LoadPDXIndex() can Append/Delete like a freshly built one.
        config.num_dimensions = index.num_dimensions;
        config.normalize = index.is_normalized;

        // Create pruner and searcher
        pruner =
            std::make_unique<PDX::ADSamplingPruner>(index.num_dimensions, aligned_matrix.get());
        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
        BuildRowIdClusterMapping();
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
        const std::vector<size_t>* passing_row_ids
    ) const override {
        if (!passing_row_ids) {
            return std::make_unique<typename PDXearch<Q>::template IterativeSearch<false>>(
                searcher->BeginIterativeSearch(query_embedding, knn, top_k_heap)
            );
        }
        auto evaluator =
            std::make_unique<PredicateEvaluator>(CreatePredicateEvaluator(*passing_row_ids));
        return std::make_unique<typename PDXearch<Q>::template IterativeSearch<true>>(
            searcher->BeginFilteredIterativeSearch(
                query_embedding, knn, std::move(evaluator), top_k_heap
            )
        );
    }

    void SetNProbe(uint32_t n_probe) override { searcher->SetNProbe(n_probe); }

    const PDX::PDXearch<Q>& GetSearcher() const { return *searcher; }

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
        // Pruner: rotation matrix or flip_masks (DCT mode) + ratios vector
        if (pruner) {
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
        // Row ID to cluster mapping
        size += row_id_cluster_mapping.size() *
                (sizeof(uint32_t) + sizeof(std::pair<uint32_t, uint32_t>));
        return size;
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

        const auto num_dimensions = config.num_dimensions;
        auto num_clusters = config.num_clusters;
        if (num_clusters == 0) {
            num_clusters = ComputeNumberOfClusters(num_embeddings);
        }
        const bool normalize =
            config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);

        assert(num_embeddings > 0);
        assert(pruner);

        auto preprocessed =
            NormalizeAndRotate(embeddings, num_embeddings, num_dimensions, normalize, *pruner);

        float quantization_base = 0.0f;
        float quantization_scale = 1.0f;
        if constexpr (Q == PDX::U8) {
            const auto params = PDX::ScalarQuantizer<Q>::ComputeQuantizationParams(
                preprocessed.get(), static_cast<size_t>(num_embeddings) * num_dimensions
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

        KMeansResult kmeans_result = ComputeKMeans(
            preprocessed.get(),
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
        index.centroids = std::move(kmeans_result.centroids);

        PopulateIVFClusters<Q>(
            index,
            kmeans_result,
            preprocessed.get(),
            row_ids,
            num_dimensions,
            num_clusters,
            quantization_base,
            quantization_scale
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
        const auto [existing_cluster, _] = GetRowIdMapping(row_id);
        if (existing_cluster != DELETED_MARKER) {
            throw std::invalid_argument(
                "Append: row_id " + std::to_string(row_id) + " already exists in the index"
            );
        }
        ReserveClusterSlotIfNeeded();

        auto preprocessed =
            NormalizeAndRotate(embedding, 1, index.num_dimensions, index.is_normalized, *pruner);

        // Find nearest centroid for the new embedding
        uint32_t closest_centroid_idx;
        {
            PDX_PROFILE_SCOPE("Append/FindNearestCentroid");
            closest_centroid_idx = FindNearestCentroid(preprocessed.get());
        }

        auto& cluster = index.clusters[closest_centroid_idx];

        uint32_t new_index_in_cluster = QuantizeAndAppend<Q>(
            index, searcher->quantizer, cluster, static_cast<uint32_t>(row_id), preprocessed.get()
        );
        SetRowIdMapping(row_id, closest_centroid_idx, new_index_in_cluster);
        index.total_num_embeddings++;
        CheckClusterHealth(cluster);
    }

    // Concurrent deletes must always go through a single writer thread
    void Delete(size_t row_id) override {
        PDX_PROFILE_SCOPE("Delete");
        const auto [cluster_id, index_in_cluster] = GetRowIdMapping(row_id);
        if (cluster_id == DELETED_MARKER) {
            throw std::invalid_argument(
                "Delete: row_id " + std::to_string(row_id) + " is not in the index"
            );
        }
        ReserveClusterSlotIfNeeded();
        auto& cluster = index.clusters[cluster_id];
        cluster.DeleteEmbedding(index_in_cluster);
        DeleteRowIdMapping(row_id);
        index.total_num_embeddings--;
        CheckClusterHealth(cluster);
    }

  private:
    static constexpr PDXIndexType GetIndexType() {
        if constexpr (Q == F32)
            return PDXIndexType::PDX_F32;
        else
            return PDXIndexType::PDX_U8;
    }

    void SetRowIdMapping(uint32_t row_id, uint32_t cluster_id, uint32_t idx_in_cluster) {
        row_id_cluster_mapping[row_id] = {cluster_id, idx_in_cluster};
    }

    void DeleteRowIdMapping(uint32_t row_id) {
        row_id_cluster_mapping[row_id] = {DELETED_MARKER, DELETED_MARKER};
    }

    std::pair<uint32_t, uint32_t> GetRowIdMapping(uint32_t row_id) const {
        auto it = row_id_cluster_mapping.find(row_id);
        if (it == row_id_cluster_mapping.end()) {
            return {DELETED_MARKER, DELETED_MARKER};
        }
        return it->second;
    }

    void BuildRowIdClusterMapping() {
        size_t total = 0;
        for (size_t c = 0; c < index.num_clusters; c++) {
            total += index.clusters[c].num_embeddings;
        }
        row_id_cluster_mapping.clear();
        row_id_cluster_mapping.reserve(total);
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            for (uint32_t p = 0; p < index.clusters[c].num_embeddings; p++) {
                SetRowIdMapping(index.clusters[c].indices[p], c, p);
            }
        }
    }

    PDX::PredicateEvaluator CreatePredicateEvaluator(const std::vector<size_t>& passing_row_ids
    ) const {
        PDX_PROFILE_SCOPE("PredicateEvaluator");
        PDX::PredicateEvaluator evaluator(index.num_clusters, index.total_capacity);
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

    // Exact scan over all centroids (embedding is already normalized + rotated)
    uint32_t FindNearestCentroid(const float* embedding) const {
        const uint32_t d = index.num_dimensions;
        uint32_t best_cluster = 0;
        float best_distance = std::numeric_limits<float>::max();
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            float distance = distance_computer_f32_t::Horizontal(
                embedding, index.centroids.data() + static_cast<size_t>(c) * d, d
            );
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = c;
            }
        }
        return best_cluster;
    }

    // Ids of the max_neighbors clusters whose centroids are nearest to `centroid`, excluding
    // cluster_id itself. Exact scan over all centroids; the result is unordered.
    std::vector<uint32_t> GetNearestNeighborClusterIds(
        uint32_t cluster_id,
        const float* centroid,
        size_t max_neighbors = SPLIT_MAX_NEIGHBOR_CLUSTERS
    ) const {
        PDX_PROFILE_SCOPE("GetNeighboringClusters");
        const uint32_t d = index.num_dimensions;
        std::vector<std::pair<float, uint32_t>> neighbor_dists;
        neighbor_dists.reserve(index.num_clusters);
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            if (c == cluster_id)
                continue;
            float dist = distance_computer_f32_t::Horizontal(
                centroid, index.centroids.data() + static_cast<size_t>(c) * d, d
            );
            neighbor_dists.emplace_back(dist, c);
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
        const uint32_t d = index.num_dimensions;

        auto raw_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();
        std::vector<uint32_t> cluster_indices(cluster.indices, cluster.indices + n_emb);
        auto cluster_embeddings =
            DequantizeClusterEmbeddings<Q>(index, searcher->quantizer, raw_embeddings.get(), n_emb);

        // Swap-and-pop: move last cluster into the dead slot
        uint32_t last_id = index.num_clusters - 1;
        if (cluster_id != last_id) {
            index.clusters[cluster_id] = std::move(index.clusters[last_id]);
            index.clusters[cluster_id].id = cluster_id;

            auto& moved_cluster = index.clusters[cluster_id];
            std::memcpy(
                index.centroids.data() + static_cast<size_t>(cluster_id) * d,
                index.centroids.data() + static_cast<size_t>(last_id) * d,
                d * sizeof(float)
            );
            for (uint32_t i = 0; i < moved_cluster.used_capacity; i++) {
                if (!moved_cluster.HasTombstone(i)) {
                    SetRowIdMapping(moved_cluster.indices[i], cluster_id, i);
                }
            }
        }

        // Pop the dead cluster and its centroid (ensured to be at the end)
        index.clusters.pop_back();
        index.centroids.resize(index.centroids.size() - d);
        index.num_clusters--;

        index.ComputeClusterOffsets();

        // Fully removed the dying cluster before reassignment
        ReassignEmbeddings(cluster_indices.data(), cluster_embeddings.get(), n_emb, false);
    }

    // Assumes cluster is compacted and has no tombstones
    void SplitCluster(cluster_t& cluster) {
        PDX_PROFILE_SCOPE("Split");
        const uint32_t cluster_id = cluster.id;
        const uint32_t d = index.num_dimensions;

        auto raw_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();
        auto cluster_embeddings = DequantizeClusterEmbeddings<Q>(
            index, searcher->quantizer, raw_embeddings.get(), cluster.num_embeddings
        );

        auto centroid_to_split = index.centroids.data() + static_cast<size_t>(cluster_id) * d;
        auto neighboring_clusters_ids = GetNearestNeighborClusterIds(cluster_id, centroid_to_split);

        // 2-means split: each embedding goes to A, B, or rest (closer to a neighboring cluster)
        auto partition = PartitionClusterForSplit<Q>(
            index,
            cluster_embeddings.get(),
            cluster.num_embeddings,
            centroid_to_split,
            neighboring_clusters_ids,
            config.distance_metric,
            config.seed
        );
        auto& centroid_a = partition.centroid_a;
        auto& centroid_b = partition.centroid_b;
        auto& group_a_idx = partition.group_a_idx;
        auto& group_b_idx = partition.group_b_idx;
        auto& group_rest_idx = partition.group_rest_idx;

        // Gather embeddings and IDs, accumulate centroid sums
        std::vector<embedding_storage_t> embs_a, embs_b;
        std::vector<uint32_t> ids_a, ids_b;
        embs_a.reserve(group_a_idx.size() * d);
        embs_b.reserve(group_b_idx.size() * d);
        ids_a.reserve(group_a_idx.size());
        ids_b.reserve(group_b_idx.size());
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
                embs_a,
                ids_a,
                centroid_sum_a.get()
            );
            GatherGroupEmbeddings<Q>(
                index,
                cluster,
                group_b_idx,
                raw_embeddings.get(),
                cluster_embeddings.get(),
                embs_b,
                ids_b,
                centroid_sum_b.get()
            );
        }

        // Gather group_rest NOW, before the cluster is replaced
        std::unique_ptr<float[]> float_rest(new float[group_rest_idx.size() * d]);
        std::unique_ptr<uint32_t[]> ids_rest(new uint32_t[group_rest_idx.size()]);
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
            searcher->quantizer,
            neighboring_clusters_ids,
            centroid_a.get(),
            centroid_b.get(),
            embs_a,
            ids_a,
            centroid_sum_a.get(),
            embs_b,
            ids_b,
            centroid_sum_b.get()
        );

        // Compute true centroids from accumulated sums
        size_t count_a = ids_a.size();
        size_t count_b = ids_b.size();
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
            cluster_t new_cluster_a(static_cast<uint32_t>(count_a), d);
            new_cluster_a.id = cluster_id;
            if (count_a > 0) {
                std::memcpy(new_cluster_a.indices, ids_a.data(), count_a * sizeof(uint32_t));
                StoreClusterEmbeddings<Q, embedding_storage_t>(
                    new_cluster_a, index, embs_a.data(), count_a
                );
            }
            uint32_t new_cluster_b_id = index.num_clusters;
            cluster_t new_cluster_b(static_cast<uint32_t>(count_b), d);
            new_cluster_b.id = new_cluster_b_id;
            if (count_b > 0) {
                std::memcpy(new_cluster_b.indices, ids_b.data(), count_b * sizeof(uint32_t));
                StoreClusterEmbeddings<Q, embedding_storage_t>(
                    new_cluster_b, index, embs_b.data(), count_b
                );
            }
            // Replace old cluster with A, append B
            index.clusters[cluster_id] = std::move(new_cluster_a);
            index.clusters.push_back(std::move(new_cluster_b));
            index.num_clusters++;
            // Update centroids
            std::memcpy(
                index.centroids.data() + static_cast<size_t>(cluster_id) * d,
                true_centroid_a.get(),
                d * sizeof(float)
            );
            index.centroids.insert(
                index.centroids.end(), true_centroid_b.get(), true_centroid_b.get() + d
            );
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
            ReassignEmbeddings(
                ids_rest.get(), float_rest.get(), static_cast<uint32_t>(group_rest_idx.size())
            );
        }

        index.ComputeClusterOffsets();
    }

    // Reassign dequantized (float) embeddings to their closest centroid among all clusters.
    // allow_merges: passed to CheckClusterHealth — false suppresses merge cascades.
    // TODO(@lkuffo, med): We can optimize reassignments by doing GEMM+PRUNING for assignments
    void ReassignEmbeddings(
        const uint32_t* row_ids,
        const float* embeddings,
        uint32_t num_embeddings,
        bool allow_merges = true
    ) {
        PDX_PROFILE_SCOPE("Reassign");
        if (num_embeddings == 0) {
            return;
        }
        const uint32_t d = index.num_dimensions;
        const uint32_t n_clusters = index.num_clusters;

        std::unique_ptr<uint32_t[]> assignments(new uint32_t[num_embeddings]);
        std::unique_ptr<float[]> result_distances(new float[num_embeddings]);
        std::unique_ptr<float[]> tmp_distances_buf(
            new float[skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE]
        );

        std::vector<float> embeddings_norms(num_embeddings);
        Eigen::Map<const MatrixR> embeddings_matrix(embeddings, num_embeddings, d);
        Eigen::Map<VectorR> v_norms(embeddings_norms.data(), num_embeddings);
        v_norms.noalias() = embeddings_matrix.rowwise().squaredNorm();

        std::vector<float> centroid_norms(n_clusters);
        Eigen::Map<const MatrixR> centroids_matrix(index.centroids.data(), n_clusters, d);
        Eigen::Map<VectorR> c_norms(centroid_norms.data(), n_clusters);
        c_norms.noalias() = centroids_matrix.rowwise().squaredNorm();

        batch_computer::FindNearestNeighbor(
            embeddings,
            index.centroids.data(),
            num_embeddings,
            n_clusters,
            d,
            embeddings_norms.data(),
            centroid_norms.data(),
            assignments.get(),
            result_distances.get(),
            tmp_distances_buf.get()
        );

        for (size_t i = 0; i < num_embeddings; i++) {
            uint32_t best_cluster = assignments[i];
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

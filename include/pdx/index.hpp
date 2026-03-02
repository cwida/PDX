#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "pdx/clustering.hpp"
#include "pdx/common.hpp"
#include "pdx/ivf_wrapper.hpp"
#include "pdx/layout.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"
#include "pdx/searcher.hpp"
#include "pdx/utils.hpp"
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
#pragma omp parallel for num_threads(PDX::g_n_threads)
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
    virtual void SetNProbe(uint32_t n_probe) const = 0;
    virtual void Save(const std::string& path) = 0;
    virtual void Restore(const std::string& path) = 0;
    virtual uint32_t GetNumDimensions() const = 0;
    virtual uint32_t GetNumClusters() const = 0;
    virtual uint32_t GetClusterSize(uint32_t cluster_id) const = 0;
    virtual std::vector<uint32_t> GetClusterRowIds(uint32_t cluster_id) const = 0;
    virtual size_t GetInMemorySizeInBytes() const = 0;
};

template <PDX::Quantization Q>
class PDXIndex : public IPDXIndex {
  public:
    using embedding_storage_t = PDX::pdx_data_t<Q>;
    using cluster_t = PDX::Cluster<Q>;

  private:
    PDXIndexConfig config{};
    PDX::IVF<Q> index;
    std::unique_ptr<PDX::ADSamplingPruner> pruner;
    std::unique_ptr<PDX::PDXearch<Q>> searcher;
    std::vector<std::pair<uint32_t, uint32_t>> row_id_cluster_mapping;

    static constexpr PDXIndexType GetIndexType() {
        if constexpr (Q == F32)
            return PDXIndexType::PDX_F32;
        else
            return PDXIndexType::PDX_U8;
    }

    // Known bug: When the cluster is compacted, row_id_cluster_mapping is not updated,
    // Which leads to inconsistency between the cluster's internal state and the mapping. 
    // This can cause errors in subsequent deletions or filtered searches that rely on the mapping.
    void BuildRowIdClusterMapping() {
        size_t total = 0;
        for (size_t c = 0; c < index.num_clusters; c++) {
            total += index.clusters[c].num_embeddings;
        }
        row_id_cluster_mapping.resize(total);
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            for (uint32_t p = 0; p < index.clusters[c].num_embeddings; p++) {
                row_id_cluster_mapping[index.clusters[c].indices[p]] = {c, p};
            }
        }
    }

    PDX::PredicateEvaluator CreatePredicateEvaluator(const std::vector<size_t>& passing_row_ids
    ) const {
        PDX::PredicateEvaluator evaluator(index.num_clusters, index.total_capacity);
        for (const auto row_id : passing_row_ids) {
            const auto& [cluster_id, index_in_cluster] = row_id_cluster_mapping[row_id];
            evaluator.n_passing_tuples[cluster_id]++;
            evaluator.selection_vector[index.cluster_offsets[cluster_id] + index_in_cluster] =
                1;
        }
        return evaluator;
    }

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
                row_id_cluster_mapping[row_id] = {c, new_idx};
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
        return searcher->FilteredSearch(query_embedding, knn, evaluator);
    }

    void SetNProbe(uint32_t n_probe) const override { searcher->SetNProbe(n_probe); }

    const PDX::PDXearch<Q>& GetSearcher() const { return *searcher; }

    uint32_t GetNumDimensions() const override { return index.num_dimensions; }

    uint32_t GetNumClusters() const override { return index.num_clusters; }

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
        size += row_id_cluster_mapping.capacity() * sizeof(std::pair<uint32_t, uint32_t>);
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
};

template <PDX::Quantization Q>
class PDXTreeIndex : public IPDXIndex {
  public:
    using embedding_storage_t = PDX::pdx_data_t<Q>;
    using cluster_t = PDX::Cluster<Q>;

  private:
    PDXIndexConfig config{};
    PDX::IVFTree<Q> index;
    std::unique_ptr<PDX::ADSamplingPruner> pruner;
    std::unique_ptr<PDX::PDXearch<Q>> searcher;
    std::unique_ptr<PDX::PDXearch<F32>> top_level_searcher;
    std::vector<std::pair<uint32_t, uint32_t>> row_id_cluster_mapping;

    static constexpr PDXIndexType GetIndexType() {
        if constexpr (Q == F32)
            return PDXIndexType::PDX_TREE_F32;
        else
            return PDXIndexType::PDX_TREE_U8;
    }

    void BuildRowIdClusterMapping() {
        size_t total = 0;
        for (size_t c = 0; c < index.num_clusters; c++) {
            total += index.clusters[c].num_embeddings;
        }
        row_id_cluster_mapping.resize(total);
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            for (uint32_t p = 0; p < index.clusters[c].num_embeddings; p++) {
                row_id_cluster_mapping[index.clusters[c].indices[p]] = {c, p};
            }
        }
    }

    PDX::PredicateEvaluator CreatePredicateEvaluator(const std::vector<size_t>& passing_row_ids
    ) const {
        PDX::PredicateEvaluator evaluator(index.num_clusters, index.total_capacity);
        for (const auto row_id : passing_row_ids) {
            const auto& [cluster_id, index_in_cluster] = row_id_cluster_mapping[row_id];
            evaluator.n_passing_tuples[cluster_id]++;
            evaluator.selection_vector[index.cluster_offsets[cluster_id] + index_in_cluster] =
                1;
        }
        return evaluator;
    }

  public:
    PDXTreeIndex() = default;

    explicit PDXTreeIndex(PDXIndexConfig config) : config(config) {
        config.Validate();
        PDX::g_n_threads = (config.n_threads == 0) ? omp_get_max_threads() : config.n_threads;
        pruner = std::make_unique<PDX::ADSamplingPruner>(config.num_dimensions, config.seed);
    }

    void Save(const std::string& path) override {
        // Compact L1 clusters before saving (update row_id_cluster_mapping from moves)
        for (uint32_t c = 0; c < index.num_clusters; c++) {
            auto moves = index.clusters[c].CompactCluster();
            for (const auto& [row_id, new_idx] : moves) {
                row_id_cluster_mapping[row_id] = {c, new_idx};
            }
        }
        // Compact L0 clusters (no mapping to update for meso-clusters)
        for (uint32_t c = 0; c < index.l0.num_clusters; c++) {
            index.l0.clusters[c].CompactCluster();
        }

        std::ofstream out(path, std::ios::binary);

        // Index type flag
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

        // IVFTree data
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

        // Load IVFTree data
        index.Load(ptr);

        // Create pruner and searchers
        pruner =
            std::make_unique<PDX::ADSamplingPruner>(index.num_dimensions, aligned_matrix.get());
        searcher = std::make_unique<PDX::PDXearch<Q>>(index, *pruner);
        top_level_searcher = std::make_unique<PDX::PDXearch<F32>>(index.l0, *pruner);
        BuildRowIdClusterMapping();
    }

    std::vector<PDX::KNNCandidate> Search(const float* query_embedding, size_t knn) const override {
        auto n_probe_top_level = GetTopLevelNumClusters();
        // We confidently prune half of the search space
        if (searcher->GetNProbe() < GetNumClusters() / 2) {
            n_probe_top_level /= 2;
        }
        top_level_searcher->SetNProbe(n_probe_top_level);
        auto top_level_results = top_level_searcher->Search(query_embedding, searcher->GetNProbe());

        std::vector<uint32_t> top_level_indexes(top_level_results.size());
        for (size_t i = 0; i < top_level_results.size(); i++) {
            top_level_indexes[i] = top_level_results[i].index;
        }
        searcher->SetClusterAccessOrder(top_level_indexes);

        return searcher->Search(query_embedding, knn);
    }

    std::vector<PDX::KNNCandidate> FilteredSearch(
        const float* query_embedding,
        size_t knn,
        const std::vector<size_t>& passing_row_ids
    ) const override {
        auto evaluator = CreatePredicateEvaluator(passing_row_ids);
        return searcher->FilteredSearch(query_embedding, knn, evaluator);
    }

    // TODO: Concurrent writes always go through a single writer thread
    void Append(size_t row_id, const float* PDX_RESTRICT embedding){
        // Row ids must be dense and sequential (next expected = current size of the mapping)
        if (row_id != row_id_cluster_mapping.size()) {
            throw std::invalid_argument(
                "Append: row_id " + std::to_string(row_id) + " is not sequential (expected " +
                std::to_string(row_id_cluster_mapping.size()) + ")"
            );
        }

        const auto num_dimensions = index.num_dimensions;
        const bool normalize =
            config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);

        auto preprocessed = NormalizeAndRotate(
            embedding, 1, num_dimensions, normalize, *pruner
        );

        // Find nearest centroid for the new embedding
        auto n_probe_top_level = GetTopLevelNumClusters();
        // We confidently prune a quarter of the search space
        n_probe_top_level = std::max(1, n_probe_top_level / 4);
        top_level_searcher->SetNProbe(n_probe_top_level);
        std::vector<KNNCandidate> centroid_candidates = top_level_searcher->Search(preprocessed.get(), 1);
        uint32_t closest_centroid_idx = centroid_candidates[0].index;

        // Assign to the corresponding cluster
        auto& cluster = index.clusters[closest_centroid_idx];

        // Lock the cluster for the entire append + health check sequence
        std::lock_guard<std::mutex> lock(cluster.cluster_mutex);
        uint32_t index_in_cluster;
        if constexpr (Q == U8) {
            ScalarQuantizer<Q> quantizer(num_dimensions);
            auto quantized = std::make_unique<embedding_storage_t[]>(num_dimensions);
            quantizer.QuantizeEmbedding(preprocessed.get(), index.quantization_base, index.quantization_scale, quantized.get());
            index_in_cluster = cluster.AppendEmbedding(static_cast<uint32_t>(row_id), quantized.get());
        } else {
            index_in_cluster = cluster.AppendEmbedding(static_cast<uint32_t>(row_id), preprocessed.get());
        }
        row_id_cluster_mapping.emplace_back(closest_centroid_idx, index_in_cluster);
        index.total_num_embeddings++;
        CheckClusterHealth(closest_centroid_idx, cluster);
    }

    // TODO: Concurrent writes always go through a single writer thread
    void Delete(size_t row_id){
        // Find the cluster and index in cluster for the given row_id
        const auto& [cluster_id, index_in_cluster] = row_id_cluster_mapping[row_id];
        auto& cluster = index.clusters[cluster_id];

        // Lock the cluster for the entire delete + health check sequence
        std::lock_guard<std::mutex> lock(cluster.cluster_mutex);

        cluster.DeleteEmbedding(index_in_cluster);
        // As row_ids are assumed to be non-replacable,
        // we don't need to remove the entry from row_id_cluster_mapping
        index.total_num_embeddings--;
        CheckClusterHealth(cluster_id, cluster);
    }

    // Caller must hold cluster.cluster_mutex
    void CheckClusterHealth(uint32_t cluster_id, cluster_t& cluster) {
        if (cluster.used_capacity == cluster.max_capacity) {
            if (cluster.num_embeddings < cluster.used_capacity) {
                // There are tombstone slots we can reclaim
                auto moves = cluster.CompactCluster();
                for (const auto& [row_id, new_idx] : moves) {
                    row_id_cluster_mapping[row_id] = {cluster_id, new_idx};
                }
            } else {
                // Cluster is truly full, split it
                SplitCluster(cluster);
            }
        } else if (cluster.num_embeddings == cluster.min_capacity) {
            DestroyAndMergeCluster(cluster);
        }
    }

    // Caller must hold cluster.cluster_mutex
    void DestroyAndMergeCluster(cluster_t& cluster){
        cluster.CompactCluster();
        std::unique_ptr<float[]> cluster_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();
        ReassignEmbeddings(cluster, cluster_embeddings.get());
        // Remove the cluster from index.clusters
        // Remove the centroid from this cluster from index.centroids
        // For this, we can swap with the last cluster and centroid in the respective arrays and pop_back, then update the row_id_cluster_mapping for the affected clusters

    }

    // Caller must hold cluster.cluster_mutex
    void SplitCluster(cluster_t& cluster){
        // Split the cluster into two, with the new centroids having a small symmetric perturbation
        // Get inspired from: SuperKMeans::SplitCluster in superkmeans.hpp, but we need to adapt it to our data layout and also update the IVF tree structure accordingly
        // Replace cluster and centroid from Clusters array with one of the new ones
        // Add a new cluster to the cluster array
        // For every point in the deleted cluster:
        // Is the distance to one of the new centroids smaller than to the deleted centroid?
        // --> Then this is the best assignment in the mesocluster
        // Most points will go to either of one of the two new centroids. So we just need to compact 
        // the horizontal buffers for each of these centroids, and call StoreClusterEmbeddings to properly create 
        // the cluster's buffer
        // Then, compute the centroids manually (easy, just average every dimension, we have the horizontal layout)
        // And, update the index.l0 structure. The two new clusters are going to the same mesocluster
        // In the remaining points (should be few), compact buffer and call REASSIGN
    }

    void ReassignEmbeddings(cluster_t& cluster, const embedding_storage_t* cluster_embeddings){
        // Cluster embeddings are assumed to be in the horizontal layout
        // Dequantize the cluster_embeddings
        // For each embedding in the cluster, find the closest centroid among all clusters in the same mesocluster and reassign
        // We need to exclude the deleted cluster's centroid from the candidates when reassigning, otherwise all embeddings will be assigned to the same closest centroid (the one from the deleted cluster)
        // This is called after splitting or merging a cluster, so we only need to reassign embeddings in the affected mesocluster
        // If coming from SPLIT:
        // --> Take the closest splitted centroid as current assignment
        // --> Run a GEMM+PRUNING iteration to get the new assignments (you need to call superkmeans API)
        // If coming from MERGE:
        // --> Run a GEMM iteration to get the new assignments
        std::unique_ptr<uint32_t[]> new_assignments(new uint32_t[cluster.num_embeddings]);
        // Assume new_assignments is filled already with the new cluster assignments for each embedding in the cluster
        for (size_t i = 0; i < cluster.num_embeddings; i++) {
            uint32_t new_cluster_id = new_assignments[i];
            uint32_t row_id = cluster.indices[i];
            auto new_index_in_cluster = index.clusters[new_cluster_id].AppendEmbedding(row_id, cluster_embeddings + i * cluster.num_dimensions);
            row_id_cluster_mapping[row_id] = {new_cluster_id, new_index_in_cluster};
        }
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
            index = PDX::IVFTree<Q>(
                num_dimensions,
                num_embeddings,
                num_clusters,
                normalize,
                quantization_scale,
                quantization_base
            );
        } else {
            index = PDX::IVFTree<Q>(num_dimensions, num_embeddings, num_clusters, normalize);
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

        // L0 index (meso-clusters over centroids)
        auto l0_num_clusters = config.num_meso_clusters;
        if (l0_num_clusters == 0) {
            l0_num_clusters = static_cast<uint32_t>(std::sqrt(num_clusters));
        }

        index.l0 = PDX::IVF<F32>(num_dimensions, num_clusters, l0_num_clusters, normalize);
        KMeansResult l0_kmeans_result = ComputeKMeans(
            index.centroids.data(),
            num_clusters,
            num_dimensions,
            l0_num_clusters,
            config.distance_metric,
            config.seed,
            config.normalize,
            1.0f,
            10,
            false // No hierarchical indexing
        );
        index.l0.centroids = std::move(l0_kmeans_result.centroids);

        // L0 row_ids are identity (centroid indices)
        std::vector<size_t> l0_row_ids(num_clusters);
        std::iota(l0_row_ids.begin(), l0_row_ids.end(), 0);
        PopulateIVFClusters<F32>(
            index.l0,
            l0_kmeans_result,
            index.centroids.data(),
            l0_row_ids.data(),
            num_dimensions,
            l0_num_clusters,
            0.0f,
            1.0f
        );

        top_level_searcher = std::make_unique<PDX::PDXearch<F32>>(index.l0, *pruner);
        BuildRowIdClusterMapping();
    }

    void SetNProbe(uint32_t n_probe) const override { searcher->SetNProbe(n_probe); }

    const PDX::PDXearch<Q>& GetSearcher() const { return *searcher; }

    uint32_t GetNumDimensions() const override { return index.num_dimensions; }

    uint32_t GetNumClusters() const override { return index.num_clusters; }

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

    uint32_t GetTopLevelNumClusters() const { return index.l0.num_clusters; }

    size_t GetInMemorySizeInBytes() const override {
        size_t size = sizeof(*this);
        // IVFTree heap allocations (L1 + L0 clusters and centroids)
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
        if (top_level_searcher) {
            size += sizeof(*top_level_searcher);
        }
        // Row ID to cluster mapping
        size += row_id_cluster_mapping.capacity() * sizeof(std::pair<uint32_t, uint32_t>);
        return size;
    }
};

using PDXIndexF32 = PDXIndex<PDX::F32>;
using PDXIndexU8 = PDXIndex<PDX::U8>;
using PDXTreeIndexF32 = PDXTreeIndex<PDX::F32>;
using PDXTreeIndexU8 = PDXTreeIndex<PDX::U8>;

inline std::unique_ptr<IPDXIndex> LoadPDXIndex(const std::string& path) {
    auto buffer = MmapFile(path);
    auto type = static_cast<PDXIndexType>(buffer.get()[0]);
    std::unique_ptr<IPDXIndex> idx;
    switch (type) {
    case PDXIndexType::PDX_F32:
        idx = std::make_unique<PDXIndexF32>();
        break;
    case PDXIndexType::PDX_U8:
        idx = std::make_unique<PDXIndexU8>();
        break;
    case PDXIndexType::PDX_TREE_F32:
        idx = std::make_unique<PDXTreeIndexF32>();
        break;
    case PDXIndexType::PDX_TREE_U8:
        idx = std::make_unique<PDXTreeIndexU8>();
        break;
    default:
        throw std::runtime_error(
            "Unknown PDX index type: " + std::to_string(static_cast<int>(type))
        );
    }
    idx->Restore(path);
    return idx;
}

} // namespace PDX

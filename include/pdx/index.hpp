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
#include "pdx/ivf_wrapper.hpp"
#include "pdx/layout.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"
#include "pdx/searcher.hpp"
#include "pdx/utils.hpp"
#include "pdx/profiler.hpp"
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
        PDX_PROFILE_SCOPE("PredicateEvaluator");
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
    using distance_computer_t = DistanceComputer<DistanceMetric::L2SQ, Q>;
    using distance_computer_f32_t = DistanceComputer<DistanceMetric::L2SQ, F32>;

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
        PDX_PROFILE_SCOPE("PredicateEvaluator");
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
        PDX_PROFILE_SCOPE("Search");
        auto n_probe = searcher->GetNProbe();
        if (n_probe == 0) {
            searcher->SetNProbe(GetNumClusters());
        }
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
        {
            PDX_PROFILE_SCOPE("FilteredSearch");
            return searcher->FilteredSearch(query_embedding, knn, evaluator);
        }
    }

    // Ensure index.clusters has room for at least one more element so that
    // push_back inside SplitCluster / ReassignEmbeddings→CheckClusterHealth
    // will never reallocate.  Must be called BEFORE taking any reference or
    // lock into the clusters vector.
    void EnsureClusterVectorHeadroom() {
        if (index.clusters.size() == index.clusters.capacity()) {
            index.clusters.reserve(index.clusters.capacity() * 2);
        }
    }

    // TODO: Concurrent writes always go through a single writer thread
    void Append(size_t row_id, const float* PDX_RESTRICT embedding){
        PDX_PROFILE_SCOPE("Append");
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
        // Pass the RAW embedding to the L0 search — PDXearch::Search internally
        // normalizes and rotates the query. Passing `preprocessed` would double-rotate.
        auto n_probe_top_level = GetTopLevelNumClusters();
        n_probe_top_level = std::max(1u, n_probe_top_level / 4);
        top_level_searcher->SetNProbe(n_probe_top_level);
        std::vector<KNNCandidate> centroid_candidates = top_level_searcher->Search(embedding, 1);
        uint32_t closest_centroid_idx = centroid_candidates[0].index;

        // Ensure the clusters vector won't reallocate while we hold a reference/lock
        EnsureClusterVectorHeadroom();

        // Assign to the corresponding cluster
        auto& cluster = index.clusters[closest_centroid_idx];

        // Lock only for the mutation; unlock before health check which may
        // trigger SplitCluster → ReassignEmbeddings → EnsureClusterVectorHeadroom,
        // causing vector reallocation that invalidates the cluster reference.
        uint32_t index_in_cluster;
        {
            std::lock_guard<std::mutex> lock(cluster.cluster_mutex);
            if constexpr (Q == U8) {
                ScalarQuantizer<Q> quantizer(num_dimensions);
                auto quantized = std::make_unique<embedding_storage_t[]>(num_dimensions);
                quantizer.QuantizeEmbedding(preprocessed.get(), index.quantization_base, index.quantization_scale, quantized.get());
                index_in_cluster = cluster.AppendEmbedding(static_cast<uint32_t>(row_id), quantized.get());
            } else {
                index_in_cluster = cluster.AppendEmbedding(static_cast<uint32_t>(row_id), preprocessed.get());
            }
        }
        row_id_cluster_mapping.emplace_back(closest_centroid_idx, index_in_cluster);
        index.total_num_embeddings++;
        CheckClusterHealth(cluster);
    }

    // TODO: Concurrent writes always go through a single writer thread
    void Delete(size_t row_id){
        PDX_PROFILE_SCOPE("Delete");
        // Find the cluster and index in cluster for the given row_id
        const auto& [cluster_id, index_in_cluster] = row_id_cluster_mapping[row_id];

        // Ensure the clusters vector won't reallocate while we hold a reference/lock
        // (DestroyAndMergeCluster→ReassignEmbeddings→CheckClusterHealth can split)
        EnsureClusterVectorHeadroom();

        auto& cluster = index.clusters[cluster_id];

        {
            // Lock only for the mutation; unlock before health check which may
            // destroy/replace the cluster (DestroyAndMergeCluster pop_back or
            // SplitCluster move-assignment).  Single-writer is assumed.
            std::lock_guard<std::mutex> lock(cluster.cluster_mutex);
            cluster.DeleteEmbedding(index_in_cluster);
        }
        // As row_ids are assumed to be non-replacable,
        // we don't need to remove the entry from row_id_cluster_mapping
        index.total_num_embeddings--;
        CheckClusterHealth(cluster);
    }

    // Caller must hold cluster.cluster_mutex.
    // allow_merges: when false, underfull clusters are not merged (only compaction
    // and splits are performed).  SPFresh triggers merges lazily from the search
    // path, not during reassignment — suppressing merges here prevents cascading
    // merge ping-pong where two underfull clusters keep destroying each other.
    void CheckClusterHealth(cluster_t& cluster, bool allow_merges = true) {
        if (cluster.used_capacity == cluster.max_capacity) {
            if (cluster.num_embeddings < cluster.used_capacity) {
                // There are tombstone slots we can reclaim
                auto moves = cluster.CompactCluster();
                for (const auto& [row_id, new_idx] : moves) {
                    row_id_cluster_mapping[row_id] = {cluster.id, new_idx};
                }
            } else {
                // Cluster is truly full and compacted, split it
                SplitCluster(cluster);
            }
        } else if (allow_merges && cluster.num_embeddings <= cluster.min_capacity) {
            DestroyAndMergeCluster(cluster);
        }
    }

    // Find the position of an L1 cluster within its known L0 meso-cluster.
    uint32_t FindPositionInMesoCluster(uint32_t l1_cluster_id, uint32_t mesocluster_id) const {
        auto& l0_cluster = index.l0.clusters[mesocluster_id];
        for (uint32_t p = 0; p < l0_cluster.used_capacity; p++) {
            if (!l0_cluster.HasTombstone(p) && l0_cluster.indices[p] == l1_cluster_id) {
                return p;
            }
        }
        throw std::runtime_error(
            "FindPositionInMesoCluster: L1 cluster " + std::to_string(l1_cluster_id) +
            " not found in L0 meso-cluster " + std::to_string(mesocluster_id)
        );
    }

    // Caller must hold cluster.cluster_mutex
    void DestroyAndMergeCluster(cluster_t& cluster) {
        PDX_PROFILE_SCOPE("Merge");
        //std::cout << "Destroy and Merge cluster " << cluster.id << " (num_embeddings=" << cluster.num_embeddings
        //          << ", used_capacity=" << cluster.used_capacity << ")\n";
        const uint32_t cluster_id = cluster.id;
        const uint32_t mc = cluster.mesocluster_id;
        const uint32_t num_dims = index.num_dimensions;
        cluster.CompactCluster();
        const uint32_t n_emb = cluster.num_embeddings;
        auto raw_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();

        // Copy indices and dequantized embeddings into local buffers BEFORE
        // destroying the cluster — after swap-and-pop the cluster object is gone.
        std::vector<uint32_t> local_indices(cluster.indices, cluster.indices + n_emb);
        std::unique_ptr<float[]> cluster_embeddings(new float[n_emb * num_dims]);
        if constexpr (Q == U8) {
            for (size_t i = 0; i < n_emb; i++) {
                searcher->quantizer.DequantizeEmbedding(
                    raw_embeddings.get() + i * num_dims,
                    index.quantization_base,
                    index.quantization_scale,
                    cluster_embeddings.get() + i * num_dims
                );
            }
        } else {
            std::memcpy(
                cluster_embeddings.get(),
                raw_embeddings.get(),
                n_emb * num_dims * sizeof(float)
            );
        }

        // ---- Fully destroy the cluster BEFORE reassignment ----
        // This ensures the dying cluster has no centroid and no slot in
        // index.clusters, so no cascading operation (split→group_rest→reassign)
        // can accidentally assign new points to it or try to merge it again.

        // 1. Remove from L0
        uint32_t pos_del = FindPositionInMesoCluster(cluster_id, mc);
        index.l0.clusters[mc].DeleteEmbedding(pos_del);
        index.l0.total_num_embeddings--;

        // 2. Swap-and-pop: move last cluster into the dead slot
        uint32_t last_id = index.num_clusters - 1;
        if (cluster_id != last_id) {
            index.clusters[cluster_id] = std::move(index.clusters[last_id]);
            index.clusters[cluster_id].id = cluster_id;

            std::memcpy(
                index.centroids.data() + static_cast<size_t>(cluster_id) * num_dims,
                index.centroids.data() + static_cast<size_t>(last_id) * num_dims,
                num_dims * sizeof(float)
            );

            auto& moved = index.clusters[cluster_id];
            for (uint32_t i = 0; i < moved.used_capacity; i++) {
                if (!moved.HasTombstone(i)) {
                    row_id_cluster_mapping[moved.indices[i]] = {cluster_id, i};
                }
            }

            uint32_t pos_last = FindPositionInMesoCluster(last_id, moved.mesocluster_id);
            index.l0.clusters[moved.mesocluster_id].indices[pos_last] = cluster_id;
        }

        // 3. Pop the dead cluster and shrink centroids
        index.clusters.pop_back();
        index.centroids.resize(index.centroids.size() - num_dims);
        index.num_clusters--;

        index.ComputeClusterOffsets();
        index.l0.ComputeClusterOffsets();

        // ---- Now reassign the extracted points ----
        // The dead cluster is fully removed: no slot, no centroid, no L0 entry.
        // No exclude_cluster_id needed — the slot is occupied by the moved cluster.
        ReassignEmbeddings(local_indices.data(), cluster_embeddings.get(), n_emb,
                           /*exclude_cluster_id=*/UINT32_MAX, /*allow_merges=*/false);
    }

    // Caller must hold cluster.cluster_mutex
    // Assumes cluster is compacted and has no tombstones
    void SplitCluster(cluster_t& cluster){
        PDX_PROFILE_SCOPE("Split");
        const uint32_t cluster_id = cluster.id;
        const uint32_t mc = cluster.mesocluster_id;
        const uint32_t num_dims = index.num_dimensions;
        const uint32_t cluster_num_embeddings = cluster.num_embeddings;

        // 1. Extract all embeddings from PDX layout (row-major)
        std::unique_ptr<embedding_storage_t[]> raw_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();
        std::unique_ptr<float[]> cluster_embeddings(new float[cluster.num_embeddings * num_dims]);
        if constexpr (Q == U8) {
            for (size_t i = 0; i < cluster.num_embeddings; i++) {
                searcher->quantizer.DequantizeEmbedding(
                    raw_embeddings.get() + i * num_dims,
                    index.quantization_base,
                    index.quantization_scale,
                    cluster_embeddings.get() + i * num_dims
                );
            }
        } else {
            std::memcpy(
                cluster_embeddings.get(),
                raw_embeddings.get(),
                cluster.num_embeddings * num_dims * sizeof(float)
            );
        }

        // 2. Split using 2-means clustering (balanced split, avoids degenerate perturbation)
        auto centroid_to_split = index.centroids.data() + static_cast<size_t>(cluster_id) * num_dims;
        KMeansResult split_result = ComputeKMeans(
            cluster_embeddings.get(),
            cluster.num_embeddings,
            num_dims,
            2,                          // k=2
            config.distance_metric,
            config.seed,
            true,                      // already preprocessed, don't normalize
            1.0f,                       // use all points (small cluster)
            4,                          // iterations
            false                       // no hierarchical for 2 clusters
        );
        auto centroid_a = std::make_unique<float[]>(num_dims);
        auto centroid_b = std::make_unique<float[]>(num_dims);
        std::memcpy(centroid_a.get(), split_result.centroids.data(), num_dims * sizeof(float));
        std::memcpy(centroid_b.get(), split_result.centroids.data() + num_dims, num_dims * sizeof(float));

        // [Old perturbation-based split — replaced by 2-means above]
        // auto centroid_a = std::make_unique<float[]>(num_dims);
        // auto centroid_b = std::make_unique<float[]>(num_dims);
        // for (size_t j = 0; j < num_dims; j++) {
        //     if (j % 2 == 0) {
        //         centroid_a[j] = centroid_to_split[j] * (1 + CENTROID_PERTURBATION_EPS);
        //         centroid_b[j] = centroid_to_split[j] * (1 - CENTROID_PERTURBATION_EPS);
        //     } else {
        //         centroid_a[j] = centroid_to_split[j] * (1 - CENTROID_PERTURBATION_EPS);
        //         centroid_b[j] = centroid_to_split[j] * (1 + CENTROID_PERTURBATION_EPS);
        //     }
        // }

        // 3. Partition embeddings using KMeans centroids + SPFresh Condition 1.
        //    If A or B is closer than the original centroid → assign directly.
        //    Otherwise the original was closest but is being deleted — check if some
        //    OTHER existing centroid is even closer → group_rest for global reassignment.
        std::vector<uint32_t> group_a_idx, group_b_idx, group_rest_idx;
        group_a_idx.reserve(split_result.assignments[0].size());
        group_b_idx.reserve(split_result.assignments[1].size());
        // std::cout << "Split cluster " << cluster_id << " into A(num=" << split_result.assignments[0].size()
        //           << ") and B(num=" << split_result.assignments[1].size() << ")\n";
        for (size_t i = 0; i < cluster.num_embeddings; i++) {
            const float* embedding_ptr = cluster_embeddings.get() + i * num_dims;
            float distance_to_centroid_to_split = distance_computer_f32_t::Horizontal(
                embedding_ptr, centroid_to_split, num_dims
            );
            float distance_to_a = distance_computer_f32_t::Horizontal(
                embedding_ptr, centroid_a.get(), num_dims
            );
            float distance_to_b = distance_computer_f32_t::Horizontal(
                embedding_ptr, centroid_b.get(), num_dims
            );
            float min_ab = std::min(distance_to_a, distance_to_b);

            // if (min_ab <= distance_to_centroid_to_split) {
                // A or B is at least as close as the original — just pick the closer one
                if (distance_to_a <= distance_to_b) {
                    group_a_idx.push_back(i);
                } else {
                    group_b_idx.push_back(i);
                }
            // } else {
            //     // Original was closer than both A and B, but it's being deleted.
            //     // Check if some other existing centroid is even closer.
            //     bool closer_elsewhere = false;
            //     for (uint32_t c = 0; c < index.num_clusters; c++) {
            //         if (c == cluster_id) continue;
            //         float d = distance_computer_f32_t::Horizontal(
            //             embedding_ptr,
            //             index.centroids.data() + static_cast<size_t>(c) * num_dims, num_dims
            //         );
            //         if (d < min_ab) {
            //             closer_elsewhere = true;
            //             break;
            //         }
            //     }
            //     if (closer_elsewhere) {
            //         group_rest_idx.push_back(i);
            //     } else if (distance_to_a <= distance_to_b) {
            //         group_a_idx.push_back(i);
            //     } else {
            //         group_b_idx.push_back(i);
            //     }
            // }
        }

        // 4. Gather embeddings and IDs into growable vectors + accumulate centroid sums.
        //    These vectors will also receive stolen neighbor embeddings before clusters are created.
        std::vector<embedding_storage_t> embs_a, embs_b;
        std::vector<uint32_t> ids_a, ids_b;
        embs_a.reserve(group_a_idx.size() * num_dims);
        embs_b.reserve(group_b_idx.size() * num_dims);
        ids_a.reserve(group_a_idx.size());
        ids_b.reserve(group_b_idx.size());

        auto centroid_sum_a = std::make_unique<float[]>(num_dims);
        auto centroid_sum_b = std::make_unique<float[]>(num_dims);
        std::memset(centroid_sum_a.get(), 0, num_dims * sizeof(float));
        std::memset(centroid_sum_b.get(), 0, num_dims * sizeof(float));

        for (uint32_t idx : group_a_idx) {
            embs_a.insert(embs_a.end(),
                raw_embeddings.get() + static_cast<size_t>(idx) * num_dims,
                raw_embeddings.get() + (static_cast<size_t>(idx) + 1) * num_dims
            );
            ids_a.push_back(cluster.indices[idx]);
            const float* emb_f = cluster_embeddings.get() + static_cast<size_t>(idx) * num_dims;
            for (size_t d = 0; d < num_dims; d++) {
                centroid_sum_a[d] += emb_f[d];
            }
        }
        for (uint32_t idx : group_b_idx) {
            embs_b.insert(embs_b.end(),
                raw_embeddings.get() + static_cast<size_t>(idx) * num_dims,
                raw_embeddings.get() + (static_cast<size_t>(idx) + 1) * num_dims
            );
            ids_b.push_back(cluster.indices[idx]);
            const float* emb_f = cluster_embeddings.get() + static_cast<size_t>(idx) * num_dims;
            for (size_t d = 0; d < num_dims; d++) {
                centroid_sum_b[d] += emb_f[d];
            }
        }

        // 5. Gather group_rest float embeddings and IDs NOW, before the cluster is replaced
        auto float_rest = std::make_unique<float[]>(group_rest_idx.size() * num_dims);
        auto ids_rest = std::make_unique<uint32_t[]>(group_rest_idx.size());
        for (size_t i = 0; i < group_rest_idx.size(); i++) {
            std::memcpy(
                float_rest.get() + i * num_dims,
                cluster_embeddings.get() + static_cast<size_t>(group_rest_idx[i]) * num_dims,
                num_dims * sizeof(float)
            );
            ids_rest[i] = cluster.indices[group_rest_idx[i]];
        }

        // 6. SPFresh neighbor reassignment using new centroids.
        //    Clusters A and B don't exist yet — they're just buffers — so there's no
        //    risk of overflow, cascading splits/merges, or dangling references.
        {
            PDX_PROFILE_SCOPE("Split/NeighborReassign");
            //std::cout << "Split reassign" << "\n";
            auto& l0_mc = index.l0.clusters[mc];
            std::vector<uint32_t> neighbor_ids;
            for (uint32_t p = 0; p < l0_mc.used_capacity; p++) {
                if (l0_mc.HasTombstone(p)) continue;
                uint32_t nid = l0_mc.indices[p];
                if (nid == cluster_id) continue; // cluster B doesn't exist yet
                neighbor_ids.push_back(nid);
            }

            for (uint32_t neighbor_id : neighbor_ids) {
                auto& neighbor = index.clusters[neighbor_id];
                const float* neighbor_centroid = index.centroids.data() +
                    static_cast<size_t>(neighbor_id) * num_dims;

                //std::cout << "Neighbor " << neighbor_id << "\n";
                // Single-pass: scan embeddings, identify candidates, and immediately
                // delete + buffer. DeleteEmbedding only tombstones so data stays readable.
                for (uint32_t p = 0; p < neighbor.used_capacity; p++) {
                    if (neighbor.HasTombstone(p)) continue;

                    auto raw_emb = neighbor.GetHorizontalEmbeddingFromPDXBuffer(p);
                    const float* emb_ptr;
                    std::unique_ptr<float[]> emb_f32;
                    if constexpr (Q == U8) {
                        emb_f32 = std::make_unique<float[]>(num_dims);
                        searcher->quantizer.DequantizeEmbedding(
                            raw_emb.get(), index.quantization_base,
                            index.quantization_scale, emb_f32.get()
                        );
                        emb_ptr = emb_f32.get();
                    } else {
                        emb_ptr = raw_emb.get();
                    }

                    float dist_own = distance_computer_f32_t::Horizontal(
                        emb_ptr, neighbor_centroid, num_dims
                    );
                    float dist_a = distance_computer_f32_t::Horizontal(
                        emb_ptr, centroid_a.get(), num_dims
                    );
                    float dist_b = distance_computer_f32_t::Horizontal(
                        emb_ptr, centroid_b.get(), num_dims
                    );

                    if (dist_a < dist_own && dist_a <= dist_b) {
                        uint32_t row_id = neighbor.indices[p];
                        neighbor.DeleteEmbedding(p);
                        embs_a.insert(embs_a.end(), raw_emb.get(), raw_emb.get() + num_dims);
                        ids_a.push_back(row_id);
                        for (size_t d = 0; d < num_dims; d++) {
                            centroid_sum_a[d] += emb_ptr[d];
                        }
                    } else if (dist_b < dist_own && dist_b < dist_a) {
                        uint32_t row_id = neighbor.indices[p];
                        neighbor.DeleteEmbedding(p);
                        embs_b.insert(embs_b.end(), raw_emb.get(), raw_emb.get() + num_dims);
                        ids_b.push_back(row_id);
                        for (size_t d = 0; d < num_dims; d++) {
                            centroid_sum_b[d] += emb_ptr[d];
                        }
                    }
                }
            }
        }

        // 7. Compute true centroids from accumulated sums (includes stolen neighbors)
        size_t count_a = ids_a.size();
        size_t count_b = ids_b.size();
        auto true_centroid_a = std::make_unique<float[]>(num_dims);
        auto true_centroid_b = std::make_unique<float[]>(num_dims);

        if (count_a == 0) {
            std::memcpy(true_centroid_a.get(), centroid_a.get(), num_dims * sizeof(float));
        } else {
            float inv = 1.0f / static_cast<float>(count_a);
#pragma clang loop vectorize(enable)
            for (size_t d = 0; d < num_dims; d++) {
                true_centroid_a[d] = centroid_sum_a[d] * inv;
            }
        }
        if (count_b == 0) {
            std::memcpy(true_centroid_b.get(), centroid_b.get(), num_dims * sizeof(float));
        } else {
            float inv = 1.0f / static_cast<float>(count_b);
#pragma clang loop vectorize(enable)
            for (size_t d = 0; d < num_dims; d++) {
                true_centroid_b[d] = centroid_sum_b[d] * inv;
            }
        }

        if (config.normalize){
            Quantizer quantizer(index.num_dimensions);
            quantizer.NormalizeQuery(
                true_centroid_a.get(), true_centroid_a.get()
            );
            quantizer.NormalizeQuery(
                true_centroid_b.get(), true_centroid_b.get()
            );
        }

        // 8. Create new cluster A (replaces old cluster at cluster_id)
        // std::cout << "Split cluster " << cluster_id << " (" << cluster_num_embeddings << " embeddings) into A (count=" << count_a << ") and B (count=" << count_b
        //           << "), stealing " << (count_a + count_b - split_result.assignments[0].size() - split_result.assignments[1].size())
        //           << " neighbors\n";
        cluster_t new_cluster_a(static_cast<uint32_t>(count_a), num_dims);
        new_cluster_a.id = cluster_id;
        new_cluster_a.mesocluster_id = mc;
        if (count_a > 0) {
            std::memcpy(new_cluster_a.indices, ids_a.data(), count_a * sizeof(uint32_t));
            StoreClusterEmbeddings<Q, embedding_storage_t>(new_cluster_a, index, embs_a.data(), count_a);
        }

        // 9. Create new cluster B (will be appended)
        uint32_t new_cluster_b_id = index.num_clusters;
        cluster_t new_cluster_b(static_cast<uint32_t>(count_b), num_dims);
        new_cluster_b.id = new_cluster_b_id;
        new_cluster_b.mesocluster_id = mc;
        if (count_b > 0) {
            std::memcpy(new_cluster_b.indices, ids_b.data(), count_b * sizeof(uint32_t));
            StoreClusterEmbeddings<Q, embedding_storage_t>(new_cluster_b, index, embs_b.data(), count_b);
        }

        // 10. Replace old cluster with A, append B.
        //     The caller must have reserved capacity before taking a reference/lock
        //     (see EnsureClusterVectorHeadroom), so push_back will not reallocate.
        index.clusters[cluster_id] = std::move(new_cluster_a);
        index.clusters.push_back(std::move(new_cluster_b));
        index.num_clusters++;

        // 11. Update centroids: replace old centroid with A's, append B's
        std::memcpy(
            index.centroids.data() + static_cast<size_t>(cluster_id) * num_dims,
            true_centroid_a.get(),
            num_dims * sizeof(float)
        );
        index.centroids.insert(
            index.centroids.end(),
            true_centroid_b.get(),
            true_centroid_b.get() + num_dims
        );

        // 12. Update row_id_cluster_mapping (includes both original and stolen-neighbor points)
        for (size_t i = 0; i < count_a; i++) {
            row_id_cluster_mapping[ids_a[i]] = {cluster_id, static_cast<uint32_t>(i)};
        }
        for (size_t i = 0; i < count_b; i++) {
            row_id_cluster_mapping[ids_b[i]] = {new_cluster_b_id, static_cast<uint32_t>(i)};
        }

        // 13. Update L0: remove old centroid, add both new centroids
        uint32_t pos = FindPositionInMesoCluster(cluster_id, mc);
        auto& l0_cluster = index.l0.clusters[mc];
        l0_cluster.DeleteEmbedding(pos);
        l0_cluster.CompactCluster();
        if (l0_cluster.used_capacity + 2 > l0_cluster.max_capacity) {
            l0_cluster.GrowCluster(std::max(l0_cluster.max_capacity * 2, l0_cluster.used_capacity + 2));
        }
        l0_cluster.AppendEmbedding(cluster_id, true_centroid_a.get());
        l0_cluster.AppendEmbedding(new_cluster_b_id, true_centroid_b.get());
        index.l0.total_num_embeddings++;

        // std::cout << "Size of rest: " << group_rest_idx.size() << "\n";

        // 14. Reassign undecided points (equidistant from both new centroids)
        if (!group_rest_idx.empty()) {
            ReassignEmbeddings(ids_rest.get(), float_rest.get(), static_cast<uint32_t>(group_rest_idx.size()));
        }

        index.ComputeClusterOffsets();
        index.l0.ComputeClusterOffsets();
    }

    // Reassign dequantized (float) embeddings to their closest centroid.
    // exclude_cluster_id: skip this cluster during nearest-centroid search (use UINT32_MAX for no exclusion).
    // allow_merges: passed to CheckClusterHealth — false suppresses merge cascades.
    void ReassignEmbeddings(
        uint32_t* row_ids,
        const float* embeddings,
        uint32_t num_embeddings,
        uint32_t exclude_cluster_id = UINT32_MAX,
        bool allow_merges = true
    ) {
        PDX_PROFILE_SCOPE("Reassign");
        const uint32_t num_dims = index.num_dimensions;

        for (size_t i = 0; i < num_embeddings; i++) {
            const float* emb = embeddings + i * num_dims;

            // Find closest centroid
            uint32_t best_cluster = UINT32_MAX;
            float best_dist = std::numeric_limits<float>::max();
            for (uint32_t c = 0; c < index.num_clusters; c++) {
                if (c == exclude_cluster_id) {
                    continue;
                }
                float dist = distance_computer_f32_t::Horizontal(
                    emb, index.centroids.data() + static_cast<size_t>(c) * num_dims, num_dims
                );
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }

            // Quantize back to storage type if needed, then append
            uint32_t row_id = row_ids[i];
            uint32_t new_idx;
            if constexpr (Q == U8) {
                ScalarQuantizer<Q> quantizer(num_dims);
                auto quantized = std::make_unique<embedding_storage_t[]>(num_dims);
                quantizer.QuantizeEmbedding(
                    emb, index.quantization_base, index.quantization_scale, quantized.get()
                );
                new_idx = index.clusters[best_cluster].AppendEmbedding(row_id, quantized.get());
            } else {
                new_idx = index.clusters[best_cluster].AppendEmbedding(row_id, emb);
            }
            row_id_cluster_mapping[row_id] = {best_cluster, new_idx};
            // Health check may split → push_back, so ensure headroom each iteration
            EnsureClusterVectorHeadroom();
            CheckClusterHealth(index.clusters[best_cluster], allow_merges);
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

        // Set mesocluster_id on each L1 cluster from L0 kmeans assignments
        for (uint32_t mc = 0; mc < l0_num_clusters; mc++) {
            for (uint32_t l1_id : l0_kmeans_result.assignments[mc]) {
                index.clusters[l1_id].mesocluster_id = mc;
            }
        }

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

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
#include "pdx/profiler.hpp"
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
    virtual void Append(size_t /*row_id*/, const float* /*embedding*/) {
        throw std::runtime_error("Append is not supported by this index type. Use PDXTreeIndex.");
    }
    virtual void Delete(size_t /*row_id*/) {
        throw std::runtime_error("Delete is not supported by this index type. Use PDXTreeIndex.");
    }
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

    void Append(size_t /*row_id*/, const float* /*embedding*/) override {
        throw std::runtime_error("Append is not implemented in PDXIndex. Use PDXTreeIndex instead."
        );
    }

    void Delete(size_t /*row_id*/) override {
        throw std::runtime_error("Delete is not implemented in PDXIndex. Use PDXTreeIndex instead."
        );
    }

  private:
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
            evaluator.selection_vector[index.cluster_offsets[cluster_id] + index_in_cluster] = 1;
        }
        return evaluator;
    }
};

template <PDX::Quantization Q>
class PDXTreeIndex : public IPDXIndex {
  public:
    using embedding_storage_t = PDX::pdx_data_t<Q>;
    using cluster_t = PDX::Cluster<Q>;
    using distance_computer_t = DistanceComputer<DistanceMetric::L2SQ, Q>;
    using distance_computer_f32_t = DistanceComputer<DistanceMetric::L2SQ, F32>;
    using batch_computer =
        skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;

  private:
    static constexpr uint32_t DELETED_MARKER = std::numeric_limits<uint32_t>::max();

    PDXIndexConfig config{};
    uint32_t d = 0;
    PDX::IVFTree<Q> index;
    std::unique_ptr<PDX::ADSamplingPruner> pruner;
    std::unique_ptr<PDX::PDXearch<Q>> searcher;
    std::unique_ptr<PDX::PDXearch<F32>> top_level_searcher;
    ScalarQuantizer<Q> quantizer{0};
    std::vector<std::pair<uint32_t, uint32_t>> row_id_cluster_mapping;

  public:
    PDXTreeIndex() = default;

    explicit PDXTreeIndex(PDXIndexConfig config)
        : config(config), d(config.num_dimensions), quantizer(config.num_dimensions) {
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
        d = index.num_dimensions;

        // Create pruner and searchers
        pruner = std::make_unique<PDX::ADSamplingPruner>(d, aligned_matrix.get());
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

    // Concurrent writes must always go through a single writer thread
    void Append(size_t row_id, const float* PDX_RESTRICT embedding) override {
        PDX_PROFILE_SCOPE("Append");
        if (row_id != row_id_cluster_mapping.size()) {
            throw std::invalid_argument(
                "Append: row_id " + std::to_string(row_id) + " is not sequential (expected " +
                std::to_string(row_id_cluster_mapping.size()) + ")"
            );
        }
        ReserveClusterSlotIfNeeded();

        const bool normalize =
            config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);

        auto preprocessed = NormalizeAndRotate(embedding, 1, d, normalize, *pruner);

        // Find nearest centroid for the new embedding
        uint32_t closest_centroid_idx;
        {
            PDX_PROFILE_SCOPE("Append/FindNearestCentroid");
            auto n_probe_top_level = GetTopLevelNumClusters();
            // We confidently prune 1/8 of the search space
            n_probe_top_level = std::max(1u, n_probe_top_level / 8);
            top_level_searcher->SetNProbe(n_probe_top_level);
            // We must pass the raw embedding to the top-level searcher
            // TODO: We should be able to avoid this by changing the search interface
            std::vector<KNNCandidate> centroid_candidates =
                top_level_searcher->Search(preprocessed.get(), 1, true);
            closest_centroid_idx = centroid_candidates[0].index;
        }

        auto& cluster = index.clusters[closest_centroid_idx];

        uint32_t new_index_in_cluster =
            QuantizeAndAppend(cluster, static_cast<uint32_t>(row_id), preprocessed.get());
        row_id_cluster_mapping.emplace_back(closest_centroid_idx, new_index_in_cluster);
        index.total_num_embeddings++;
        CheckClusterHealth(cluster);
    }

    // Concurrent deletes must always go through a single writer thread
    void Delete(size_t row_id) override {
        PDX_PROFILE_SCOPE("Delete");
        if (row_id >= row_id_cluster_mapping.size()) {
            throw std::invalid_argument(
                "Delete: row_id " + std::to_string(row_id) + " is not in the index"
            );
        }
        const auto& [cluster_id, index_in_cluster] = row_id_cluster_mapping[row_id];
        if (cluster_id == DELETED_MARKER) {
            throw std::invalid_argument(
                "Delete: row_id " + std::to_string(row_id) + " was already deleted"
            );
        }
        ReserveClusterSlotIfNeeded();
        auto& cluster = index.clusters[cluster_id];
        cluster.DeleteEmbedding(index_in_cluster);
        row_id_cluster_mapping[row_id] = {DELETED_MARKER, DELETED_MARKER};
        index.total_num_embeddings--;
        CheckClusterHealth(cluster);
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

    uint32_t GetNumDimensions() const override { return d; }

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

  private:
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
            if (cluster_id == DELETED_MARKER)
                continue;
            evaluator.n_passing_tuples[cluster_id]++;
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

    // Dequantize raw (Q-type) embeddings to float. For F32 this is a memcpy.
    std::unique_ptr<float[]> DequantizeClusterEmbeddings(
        const embedding_storage_t* raw_embeddings,
        uint32_t n_emb
    ) const {
        std::unique_ptr<float[]> result(new float[static_cast<size_t>(n_emb) * d]);
        if constexpr (Q == U8) {
            for (size_t i = 0; i < n_emb; i++) {
                searcher->quantizer.DequantizeEmbedding(
                    raw_embeddings + i * d,
                    index.quantization_base,
                    index.quantization_scale,
                    result.get() + i * d
                );
            }
        } else {
            std::memcpy(
                result.get(), raw_embeddings, static_cast<size_t>(n_emb) * d * sizeof(float)
            );
        }
        return result;
    }

    // Quantize (if U8) and append a float embedding to a cluster.
    uint32_t QuantizeAndAppend(cluster_t& cluster, uint32_t row_id, const float* embedding) {
        if constexpr (Q == U8) {
            std::unique_ptr<embedding_storage_t[]> quantized(new embedding_storage_t[d]);
            quantizer.QuantizeEmbedding(
                embedding, index.quantization_base, index.quantization_scale, quantized.get()
            );
            return cluster.AppendEmbedding(row_id, quantized.get());
        } else {
            return cluster.AppendEmbedding(row_id, embedding);
        }
    }

    // Gather raw embeddings, row IDs, and accumulate centroid sum for a group of indices.
    void GatherGroupEmbeddings(
        const std::vector<uint32_t>& group_idx,
        const embedding_storage_t* raw_embeddings,
        const float* float_embeddings,
        const cluster_t& cluster,
        std::vector<embedding_storage_t>& embs_out,
        std::vector<uint32_t>& ids_out,
        float* centroid_sum
    ) const {
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

    // Compute mean centroid from accumulated sum. Falls back to fallback if count == 0.
    void ComputeCentroidMean(
        const float* centroid_sum,
        size_t count,
        const float* fallback,
        float* output
    ) const {
        if (count == 0) {
            std::memcpy(output, fallback, d * sizeof(float));
        } else {
            float inv = 1.0f / static_cast<float>(count);
#pragma clang loop vectorize(enable)
            for (size_t j = 0; j < d; j++) {
                output[j] = centroid_sum[j] * inv;
            }
        }
        if (config.normalize) {
            Quantizer q(d);
            q.NormalizeQuery(output, output);
        }
    }

    // Get neighboring cluster IDs from the same meso-cluster, limited to max_neighbors nearest.
    std::vector<uint32_t> GetNearestNeighborClusterIds(
        uint32_t cluster_id,
        uint32_t mesocluster_id,
        const float* centroid,
        size_t max_neighbors = 32
    ) const {
        PDX_PROFILE_SCOPE("GetNeighboringClusters");
        std::vector<uint32_t> neighbor_ids;
        auto& mesocluster = index.l0.clusters[mesocluster_id];
        for (uint32_t pos = 0; pos < mesocluster.used_capacity; pos++) {
            if (mesocluster.HasTombstone(pos))
                continue;
            uint32_t nid = mesocluster.indices[pos];
            if (nid == cluster_id)
                continue;
            neighbor_ids.push_back(nid);
        }
        if (neighbor_ids.size() > max_neighbors) {
            std::vector<std::pair<float, uint32_t>> neighbor_dists;
            neighbor_dists.reserve(neighbor_ids.size());
            for (uint32_t nid : neighbor_ids) {
                float dist = distance_computer_f32_t::Horizontal(
                    centroid, index.centroids.data() + static_cast<size_t>(nid) * d, d
                );
                neighbor_dists.push_back({dist, nid});
            }
            std::nth_element(
                neighbor_dists.begin(),
                neighbor_dists.begin() + max_neighbors,
                neighbor_dists.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
            neighbor_ids.clear();
            for (size_t i = 0; i < max_neighbors; ++i) {
                neighbor_ids.push_back(neighbor_dists[i].second);
            }
        }
        return neighbor_ids;
    }

    uint32_t FindPositionInMesoCluster(uint32_t l1_cluster_id, uint32_t mesocluster_id) const {
        PDX_PROFILE_SCOPE("FindPositionInMesoCluster");
        auto& l0_cluster = index.l0.clusters[mesocluster_id];
        for (uint32_t idx = 0; idx < l0_cluster.used_capacity; idx++) {
            if (!l0_cluster.HasTombstone(idx) && l0_cluster.indices[idx] == l1_cluster_id) {
                return idx;
            }
        }
        throw std::runtime_error(
            "FindPositionInMesoCluster: L1 cluster " + std::to_string(l1_cluster_id) +
            " not found in L0 meso-cluster " + std::to_string(mesocluster_id)
        );
    }

    void CheckClusterHealth(cluster_t& cluster, bool allow_merges = true) {
        if (cluster.used_capacity == cluster.max_capacity) {
            // Its less expensive to compact than to Split
            if (cluster.num_embeddings < cluster.used_capacity) {
                auto moves = cluster.CompactCluster();
                for (const auto& [row_id, new_idx] : moves) {
                    row_id_cluster_mapping[row_id] = {cluster.id, new_idx};
                }
            } else {
                SplitCluster(cluster);
            }
        } else if (allow_merges && cluster.num_embeddings <= cluster.min_capacity) {
            DestroyAndMergeCluster(cluster);
        }
    }

    void DestroyAndMergeCluster(cluster_t& cluster) {
        PDX_PROFILE_SCOPE("Merge");
        cluster.CompactCluster();
        const uint32_t cluster_id = cluster.id;
        const uint32_t mesocluster_id = cluster.mesocluster_id;
        const uint32_t n_emb = cluster.num_embeddings;

        auto raw_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();
        std::vector<uint32_t> cluster_indices(cluster.indices, cluster.indices + n_emb);
        auto cluster_embeddings = DequantizeClusterEmbeddings(raw_embeddings.get(), n_emb);

        // Remove from L0
        uint32_t position_in_mesocluster = FindPositionInMesoCluster(cluster_id, mesocluster_id);
        index.l0.clusters[mesocluster_id].DeleteEmbedding(position_in_mesocluster);
        index.l0.total_num_embeddings--;

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
                    row_id_cluster_mapping[moved_cluster.indices[i]] = {cluster_id, i};
                }
            }

            uint32_t l0_moved_cluster_position =
                FindPositionInMesoCluster(last_id, moved_cluster.mesocluster_id);
            index.l0.clusters[moved_cluster.mesocluster_id].indices[l0_moved_cluster_position] =
                cluster_id;
        }

        // Pop the dead cluster and its centroid (ensured to be at the end)
        index.clusters.pop_back();
        index.centroids.resize(index.centroids.size() - d);
        index.num_clusters--;

        index.ComputeClusterOffsets();
        index.l0.ComputeClusterOffsets();

        // Fully removed the dying cluster before reassignment
        ReassignEmbeddings(
            cluster_indices.data(), cluster_embeddings.get(), n_emb, mesocluster_id, false
        );
    }

    // Assumes cluster is compacted and has no tombstones
    void SplitCluster(cluster_t& cluster) {
        PDX_PROFILE_SCOPE("Split");
        const uint32_t cluster_id = cluster.id;
        const uint32_t mesocluster_id = cluster.mesocluster_id;

        auto raw_embeddings = cluster.GetHorizontalEmbeddingsFromPDXBuffer();
        auto cluster_embeddings =
            DequantizeClusterEmbeddings(raw_embeddings.get(), cluster.num_embeddings);

        auto centroid_to_split = index.centroids.data() + static_cast<size_t>(cluster_id) * d;
        auto neighboring_clusters_ids =
            GetNearestNeighborClusterIds(cluster_id, mesocluster_id, centroid_to_split);

        // 2-means split
        std::unique_ptr<float[]> centroid_a(new float[d]);
        std::unique_ptr<float[]> centroid_b(new float[d]);
        std::vector<uint32_t> group_a_idx, group_b_idx, group_rest_idx;
        {
            PDX_PROFILE_SCOPE("Split/KMeans");
            KMeansResult split_result = ComputeKMeans(
                cluster_embeddings.get(),
                cluster.num_embeddings,
                d,
                2,
                config.distance_metric,
                config.seed,
                true,
                1.0f,
                4,
                false,
                1
            );
            std::memcpy(centroid_a.get(), split_result.centroids.data(), d * sizeof(float));
            std::memcpy(centroid_b.get(), split_result.centroids.data() + d, d * sizeof(float));
            group_a_idx.reserve(split_result.assignments[0].size());
            group_b_idx.reserve(split_result.assignments[1].size());
        }

        // Assign each embedding to A, B, or rest (closer elsewhere)
        {
            PDX_PROFILE_SCOPE("Split/Partition");
            for (size_t i = 0; i < cluster.num_embeddings; i++) {
                const float* emb = cluster_embeddings.get() + i * d;
                float dist_old = distance_computer_f32_t::Horizontal(emb, centroid_to_split, d);
                // TODO(@lkuffo, high): We can avoid this since we have the distance
                // from k-means, we just need to bring it here!
                float dist_a = distance_computer_f32_t::Horizontal(emb, centroid_a.get(), d);
                float dist_b = distance_computer_f32_t::Horizontal(emb, centroid_b.get(), d);
                float min_ab = std::min(dist_a, dist_b);

                if (min_ab <= dist_old) {
                    (dist_a <= dist_b ? group_a_idx : group_b_idx).push_back(i);
                } else {
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
                        group_rest_idx.push_back(i);
                    } else {
                        (dist_a <= dist_b ? group_a_idx : group_b_idx).push_back(i);
                    }
                }
            }
        }

        // Gather embeddings and IDs, accumulate centroid sums
        std::vector<embedding_storage_t> embs_a, embs_b;
        std::vector<uint32_t> ids_a, ids_b;
        embs_a.reserve(group_a_idx.size() * d);
        embs_b.reserve(group_b_idx.size() * d);
        ids_a.reserve(group_a_idx.size());
        ids_b.reserve(group_b_idx.size());
        auto centroid_sum_a = std::make_unique<float[]>(d);
        auto centroid_sum_b = std::make_unique<float[]>(d);
        {
            PDX_PROFILE_SCOPE("Split/GatherEmbeddings");
            GatherGroupEmbeddings(
                group_a_idx,
                raw_embeddings.get(),
                cluster_embeddings.get(),
                cluster,
                embs_a,
                ids_a,
                centroid_sum_a.get()
            );
            GatherGroupEmbeddings(
                group_b_idx,
                raw_embeddings.get(),
                cluster_embeddings.get(),
                cluster,
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
        {
            PDX_PROFILE_SCOPE("Split/NeighborReassign");
            for (uint32_t neighbor_id : neighboring_clusters_ids) {
                auto& neighbor = index.clusters[neighbor_id];
                const float* neighbor_centroid =
                    index.centroids.data() + static_cast<size_t>(neighbor_id) * d;

                for (uint32_t p = 0; p < neighbor.used_capacity; p++) {
                    if (neighbor.HasTombstone(p))
                        continue;

                    auto raw_emb = neighbor.GetHorizontalEmbeddingFromPDXBuffer(p);
                    const float* emb_ptr;
                    std::unique_ptr<float[]> emb_f32;
                    if constexpr (Q == U8) {
                        emb_f32.reset(new float[d]);
                        searcher->quantizer.DequantizeEmbedding(
                            raw_emb.get(),
                            index.quantization_base,
                            index.quantization_scale,
                            emb_f32.get()
                        );
                        emb_ptr = emb_f32.get();
                    } else {
                        emb_ptr = raw_emb.get();
                    }

                    float dist_own =
                        distance_computer_f32_t::Horizontal(emb_ptr, neighbor_centroid, d);
                    float dist_a =
                        distance_computer_f32_t::Horizontal(emb_ptr, centroid_a.get(), d);
                    float dist_b =
                        distance_computer_f32_t::Horizontal(emb_ptr, centroid_b.get(), d);

                    if (dist_a < dist_own && dist_a <= dist_b) {
                        uint32_t row_id = neighbor.indices[p];
                        neighbor.DeleteEmbedding(p);
                        embs_a.insert(embs_a.end(), raw_emb.get(), raw_emb.get() + d);
                        ids_a.push_back(row_id);
                        for (size_t j = 0; j < d; j++)
                            centroid_sum_a[j] += emb_ptr[j];
                    } else if (dist_b < dist_own && dist_b < dist_a) {
                        uint32_t row_id = neighbor.indices[p];
                        neighbor.DeleteEmbedding(p);
                        embs_b.insert(embs_b.end(), raw_emb.get(), raw_emb.get() + d);
                        ids_b.push_back(row_id);
                        for (size_t j = 0; j < d; j++)
                            centroid_sum_b[j] += emb_ptr[j];
                    }
                }
            }
        }

        // Compute true centroids from accumulated sums
        size_t count_a = ids_a.size();
        size_t count_b = ids_b.size();
        std::unique_ptr<float[]> true_centroid_a(new float[d]);
        std::unique_ptr<float[]> true_centroid_b(new float[d]);
        {
            PDX_PROFILE_SCOPE("Split/ComputeTrueCentroids");
            ComputeCentroidMean(
                centroid_sum_a.get(), count_a, centroid_a.get(), true_centroid_a.get()
            );
            ComputeCentroidMean(
                centroid_sum_b.get(), count_b, centroid_b.get(), true_centroid_b.get()
            );
        }

        // Create new clusters and update all data structures
        {
            PDX_PROFILE_SCOPE("Split/ConsolidateNewClusters");
            cluster_t new_cluster_a(static_cast<uint32_t>(count_a), d);
            new_cluster_a.id = cluster_id;
            new_cluster_a.mesocluster_id = mesocluster_id;
            if (count_a > 0) {
                std::memcpy(new_cluster_a.indices, ids_a.data(), count_a * sizeof(uint32_t));
                StoreClusterEmbeddings<Q, embedding_storage_t>(
                    new_cluster_a, index, embs_a.data(), count_a
                );
            }
            uint32_t new_cluster_b_id = index.num_clusters;
            cluster_t new_cluster_b(static_cast<uint32_t>(count_b), d);
            new_cluster_b.id = new_cluster_b_id;
            new_cluster_b.mesocluster_id = mesocluster_id;
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
                row_id_cluster_mapping[ids_a[i]] = {cluster_id, static_cast<uint32_t>(i)};
            }
            for (size_t i = 0; i < count_b; i++) {
                row_id_cluster_mapping[ids_b[i]] = {new_cluster_b_id, static_cast<uint32_t>(i)};
            }
            // Update L0: remove old centroid, add both new centroids
            uint32_t pos = FindPositionInMesoCluster(cluster_id, mesocluster_id);
            auto& l0_cluster = index.l0.clusters[mesocluster_id];
            l0_cluster.DeleteEmbedding(pos);
            l0_cluster.CompactCluster();
            if (l0_cluster.used_capacity + 2 > l0_cluster.max_capacity) {
                l0_cluster.GrowCluster(
                    std::max(l0_cluster.max_capacity * 2, l0_cluster.used_capacity + 2)
                );
            }
            l0_cluster.AppendEmbedding(cluster_id, true_centroid_a.get());
            l0_cluster.AppendEmbedding(new_cluster_b_id, true_centroid_b.get());
            index.l0.total_num_embeddings++;
        }

        // Reassign rest group (closer to other centroids than A or B)
        if (!group_rest_idx.empty()) {
            ReassignEmbeddings(
                ids_rest.get(),
                float_rest.get(),
                static_cast<uint32_t>(group_rest_idx.size()),
                mesocluster_id
            );
        }

        index.ComputeClusterOffsets();
        index.l0.ComputeClusterOffsets();
    }

    // Reassign dequantized (float) embeddings to their closest centroid
    // within the given mesocluster.
    // allow_merges: passed to CheckClusterHealth — false suppresses merge cascades.
    // TODO: We can optimize reassignments by doing GEMM+PRUNING for assignments
    void ReassignEmbeddings(
        uint32_t* row_ids,
        const float* embeddings,
        uint32_t num_embeddings,
        uint32_t mesocluster_id,
        bool allow_merges = true
    ) {
        PDX_PROFILE_SCOPE("Reassign");

        // Gather cluster IDs and centroids from the mesocluster
        auto& meso = index.l0.clusters[mesocluster_id];
        std::vector<uint32_t> candidate_ids;
        candidate_ids.reserve(meso.used_capacity);
        for (uint32_t p = 0; p < meso.used_capacity; p++) {
            if (!meso.HasTombstone(p)) {
                candidate_ids.push_back(meso.indices[p]);
            }
        }
        const uint32_t n_candidates = static_cast<uint32_t>(candidate_ids.size());

        std::vector<float> candidate_centroids(static_cast<size_t>(n_candidates) * d);
        for (size_t i = 0; i < n_candidates; i++) {
            std::memcpy(
                candidate_centroids.data() + i * d,
                index.centroids.data() + static_cast<size_t>(candidate_ids[i]) * d,
                d * sizeof(float)
            );
        }

        std::unique_ptr<uint32_t[]> assignments(new uint32_t[num_embeddings]);
        std::unique_ptr<float[]> result_distances(new float[num_embeddings]);
        std::unique_ptr<float[]> tmp_distances_buf(
            new float[skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE]
        );

        std::vector<float> embeddings_norms(num_embeddings);
        Eigen::Map<const MatrixR> embeddings_matrix(embeddings, num_embeddings, d);
        Eigen::Map<VectorR> v_norms(embeddings_norms.data(), num_embeddings);
        v_norms.noalias() = embeddings_matrix.rowwise().squaredNorm();

        std::vector<float> centroid_norms(n_candidates);
        Eigen::Map<const MatrixR> centroids_matrix(candidate_centroids.data(), n_candidates, d);
        Eigen::Map<VectorR> c_norms(centroid_norms.data(), n_candidates);
        c_norms.noalias() = centroids_matrix.rowwise().squaredNorm();

        batch_computer::FindNearestNeighbor(
            embeddings,
            candidate_centroids.data(),
            num_embeddings,
            n_candidates,
            d,
            embeddings_norms.data(),
            centroid_norms.data(),
            assignments.get(),
            result_distances.get(),
            tmp_distances_buf.get()
        );

        // assignments[i] is an index into candidate_ids, map back to actual cluster ID
        for (size_t i = 0; i < num_embeddings; i++) {
            uint32_t best_cluster = candidate_ids[assignments[i]];
            uint32_t row_id = row_ids[i];
            uint32_t new_pos =
                QuantizeAndAppend(index.clusters[best_cluster], row_id, embeddings + i * d);
            row_id_cluster_mapping[row_id] = {best_cluster, new_pos};
            ReserveClusterSlotIfNeeded();
            CheckClusterHealth(index.clusters[best_cluster], allow_merges);
        }
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

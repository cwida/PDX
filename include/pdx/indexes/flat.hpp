#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pdx/common.hpp"
#include "pdx/flat_searcher.hpp"
#include "pdx/indexes/flat_core.hpp"
#include "pdx/indexes/ivf_utils.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/pruners/adsampling.hpp"

namespace PDX {

// Exact search over a Flat block, for sets too small to cluster. Same maintenance API as
// PDXIndex; GetRowIds()/GetEmbeddings() feed a PDXIndex built with is_data_transformed.
class FlatIndex : public IPDXIndex {
  private:
    PDXIndexConfig config{};
    Flat index;
    std::unique_ptr<ADSamplingPruner> owned_pruner;
    ADSamplingPruner* pruner = nullptr;
    std::unique_ptr<FlatSearcher> searcher;
    RowIdClusterMapping row_id_cluster_mapping;

  public:
    FlatIndex() = default;

    explicit FlatIndex(PDXIndexConfig config)
        : config(config), index(config.num_dimensions, Normalize(config)) {
        config.Validate();
        PDX::g_n_threads = (config.n_threads == 0) ? omp_get_max_threads() : config.n_threads;
        owned_pruner = std::make_unique<ADSamplingPruner>(config.num_dimensions, config.seed);
        pruner = owned_pruner.get();
        searcher = std::make_unique<FlatSearcher>(index, *pruner);
        row_id_cluster_mapping.base_row_id = config.base_row_id;
    }

    FlatIndex(PDXIndexConfig config, ADSamplingPruner& external_pruner)
        : config(config), index(config.num_dimensions, Normalize(config)),
          pruner(&external_pruner) {
        config.Validate();
        PDX::g_n_threads = (config.n_threads == 0) ? omp_get_max_threads() : config.n_threads;
        searcher = std::make_unique<FlatSearcher>(index, *pruner);
        row_id_cluster_mapping.base_row_id = config.base_row_id;
    }

    void BuildIndex(const float* embeddings, size_t num_embeddings) override {
        std::vector<size_t> row_ids(num_embeddings);
        std::iota(row_ids.begin(), row_ids.end(), 0);
        BuildIndex(row_ids.data(), embeddings, num_embeddings);
    }

    void BuildIndex(const size_t* row_ids, const float* embeddings, size_t num_embeddings) {
        std::unique_ptr<float[]> transformed;
        if (!config.is_data_transformed) {
            transformed = NormalizeAndRotate(
                embeddings, num_embeddings, index.num_dimensions, index.is_normalized, *pruner
            );
        }
        const float* preprocessed = config.is_data_transformed ? embeddings : transformed.get();
        index.Clear();
        row_id_cluster_mapping.entries.clear();
        for (size_t i = 0; i < num_embeddings; i++) {
            const auto position = index.AppendEmbedding(
                static_cast<uint32_t>(row_ids[i]), preprocessed + i * index.num_dimensions
            );
            row_id_cluster_mapping.Set(row_ids[i], 0, position);
        }
    }

    void Append(size_t row_id, const float* embedding) override {
        if (Contains(row_id)) {
            throw std::invalid_argument(
                "Append: row_id " + std::to_string(row_id) + " already exists in the index"
            );
        }
        std::unique_ptr<float[]> transformed;
        if (!config.is_data_transformed) {
            transformed = NormalizeAndRotate(
                embedding, 1, index.num_dimensions, index.is_normalized, *pruner
            );
        }
        const float* preprocessed = config.is_data_transformed ? embedding : transformed.get();
        const auto position = index.AppendEmbedding(static_cast<uint32_t>(row_id), preprocessed);
        row_id_cluster_mapping.Set(row_id, 0, position);
    }

    bool Delete(size_t row_id) override {
        const auto [cluster_id, position] = GetRowIdMapping(row_id);
        if (cluster_id == DELETED_MARKER) {
            return false;
        }
        index.DeleteEmbedding(position);
        row_id_cluster_mapping.Delete(row_id);
        return true;
    }

    std::pair<uint32_t, uint32_t> GetRowIdMapping(size_t row_id) const override {
        return row_id_cluster_mapping.Get(row_id);
    }

    std::vector<KNNCandidate> Search(const float* query_embedding, size_t knn) const override {
        return searcher->Search(query_embedding, knn);
    }

    std::vector<KNNCandidate> FilteredSearch(
        const float* query_embedding,
        size_t knn,
        const std::vector<size_t>& passing_row_ids
    ) const override {
        return searcher->FilteredSearch(query_embedding, knn, PassingPositions(passing_row_ids));
    }

    std::unique_ptr<IIterativeSearch> BeginIterativeSearch(
        const float* query_embedding,
        uint32_t knn,
        TopKHeap& top_k_heap,
        const std::vector<size_t>* passing_row_ids,
        bool is_query_transformed = false,
        const std::vector<uint32_t>* /*clusters_access_order*/ = nullptr
    ) const override {
        if (!passing_row_ids) {
            return std::make_unique<FlatSearcher::IterativeSearch>(searcher->BeginIterativeSearch(
                query_embedding, knn, top_k_heap, is_query_transformed
            ));
        }
        return std::make_unique<FlatSearcher::IterativeSearch>(
            searcher->BeginFilteredIterativeSearch(
                query_embedding,
                knn,
                PassingPositions(*passing_row_ids),
                top_k_heap,
                is_query_transformed
            )
        );
    }

    // One cluster: the selection vector is indexed by position.
    std::unique_ptr<PredicateEvaluator> CreateSharedPredicateEvaluator(
        const std::vector<size_t>& passing_row_ids
    ) const override {
        auto evaluator = std::make_unique<PredicateEvaluator>(1, index.UsedCapacity());
        const auto positions = PassingPositions(passing_row_ids);
        for (const uint32_t position : *positions) {
            evaluator->selection_vector[position] = 1;
            evaluator->n_passing_tuples[0]++;
            evaluator->total_passing_tuples++;
        }
        return evaluator;
    }

    // The positions are rebuilt from the selection vector per search (a Flat index is small).
    std::unique_ptr<IIterativeSearch> BeginIterativeSearchWithSharedEvaluator(
        const float* query_embedding,
        uint32_t knn,
        TopKHeap& top_k_heap,
        const PredicateEvaluator& evaluator,
        bool is_query_transformed = false,
        const std::vector<uint32_t>* /*clusters_access_order*/ = nullptr
    ) const override {
        auto positions = std::make_unique<std::vector<uint32_t>>();
        positions->reserve(evaluator.total_passing_tuples);
        for (uint32_t position = 0; position < index.UsedCapacity(); position++) {
            if (evaluator.selection_vector[position]) {
                positions->push_back(position);
            }
        }
        return std::make_unique<FlatSearcher::IterativeSearch>(
            searcher->BeginFilteredIterativeSearch(
                query_embedding, knn, std::move(positions), top_k_heap, is_query_transformed
            )
        );
    }

    void GetEmbeddingsFromIndexByRowIds(const std::vector<size_t>& row_ids, float* out)
        const override {
        const size_t d = config.num_dimensions;
        for (size_t i = 0; i < row_ids.size(); i++) {
            const auto [cluster_id, position] = GetRowIdMapping(row_ids[i]);
            if (cluster_id == DELETED_MARKER) {
                throw std::invalid_argument(
                    "GetEmbeddingsFromIndexByRowIds: a row id is not in the index"
                );
            }
            std::copy(
                index.GetEmbeddingPtrAtPosition(position),
                index.GetEmbeddingPtrAtPosition(position) + d,
                out + i * d
            );
        }
    }

    void GetDistancesToCentroids(
        const float* query_embedding,
        bool is_query_transformed,
        float* out
    ) const override {
        searcher->GetDistancesToCentroids(query_embedding, is_query_transformed, out);
    }

    // Its one cluster.
    std::vector<uint32_t> GetClustersAccessOrder(const float*, bool = false) const override {
        return {0};
    }

    void SetNProbe(uint32_t) override {}

    void Save(const std::string& path) override {
        std::ofstream out(path, std::ios::binary);
        WriteValue(out, static_cast<uint8_t>(PDXIndexType::PDX_FLAT));
        WriteRotationMatrix(out, *pruner);
        index.Save(out);
    }

    void Restore(const std::string& path) override {
        auto buffer = MmapFile(path);
        char* ptr = buffer.get();
        BufferReader reader{ptr};

        // Index type flag
        ReadValue<uint8_t>(reader);
        const auto matrix = ReadRotationMatrix(reader);
        index.Load(reader);
        config.num_dimensions = index.num_dimensions;
        config.normalize = index.is_normalized;

        owned_pruner = std::make_unique<ADSamplingPruner>(index.num_dimensions, matrix.get());
        pruner = owned_pruner.get();
        searcher = std::make_unique<FlatSearcher>(index, *pruner);
        BuildRowIdClusterMapping();
    }

    void SaveToStream(std::ostream& out) override {
        WriteStreamHeader(out, PDXIndexType::PDX_FLAT, config);
        index.Save(out);
    }

    void LoadFromStream(std::istream& in) override {
        StreamReader reader{in};
        index.Load(reader);
        BuildRowIdClusterMapping();
    }

    uint32_t GetNumDimensions() const override { return index.num_dimensions; }

    uint32_t GetNumClusters() const override { return 1; }

    uint32_t GetClusterSize(uint32_t) const override {
        return static_cast<uint32_t>(index.num_embeddings);
    }

    std::vector<uint32_t> GetClusterRowIds(uint32_t) const override {
        std::vector<uint32_t> row_ids;
        row_ids.reserve(index.num_embeddings);
        for (uint32_t position = 0; position < index.UsedCapacity(); position++) {
            if (!index.HasTombstone(position)) {
                row_ids.push_back(index.indices[position]);
            }
        }
        return row_ids;
    }

    std::vector<size_t> GetRowIds() const {
        std::vector<size_t> row_ids;
        row_ids.reserve(index.num_embeddings);
        for (uint32_t position = 0; position < index.UsedCapacity(); position++) {
            if (!index.HasTombstone(position)) {
                row_ids.push_back(index.indices[position]);
            }
        }
        return row_ids;
    }

    std::unique_ptr<float[]> GetEmbeddings() const {
        const size_t d = index.num_dimensions;
        std::unique_ptr<float[]> embeddings(new float[index.num_embeddings * d]);
        size_t n_copied = 0;
        for (uint32_t position = 0; position < index.UsedCapacity(); position++) {
            if (!index.HasTombstone(position)) {
                std::copy(
                    index.GetEmbeddingPtrAtPosition(position),
                    index.GetEmbeddingPtrAtPosition(position) + d,
                    embeddings.get() + n_copied * d
                );
                n_copied++;
            }
        }
        return embeddings;
    }

    size_t GetInMemorySizeInBytes() const override {
        size_t size = sizeof(*this);
        size += index.GetInMemorySizeInBytes() - sizeof(index);
        if (owned_pruner) {
            size += sizeof(*owned_pruner);
            const auto& m = owned_pruner->GetMatrix();
            size += static_cast<size_t>(m.rows()) * m.cols() * sizeof(float);
        }
        if (searcher) {
            size += sizeof(*searcher);
        }
        size += row_id_cluster_mapping.SizeInBytes();
        return size;
    }

  private:
    static bool Normalize(const PDXIndexConfig& config) {
        return config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);
    }

    void BuildRowIdClusterMapping() {
        row_id_cluster_mapping.entries.clear();
        for (size_t position = 0; position < index.UsedCapacity(); position++) {
            row_id_cluster_mapping.Set(index.indices[position], 0, static_cast<uint32_t>(position));
        }
    }

    std::unique_ptr<std::vector<uint32_t>> PassingPositions(
        const std::vector<size_t>& passing_row_ids
    ) const {
        auto positions = std::make_unique<std::vector<uint32_t>>();
        positions->reserve(passing_row_ids.size());
        for (const size_t row_id : passing_row_ids) {
            const auto [cluster_id, position] = GetRowIdMapping(row_id);
            if (cluster_id != DELETED_MARKER) {
                positions->push_back(position);
            }
        }
        return positions;
    }
};

} // namespace PDX

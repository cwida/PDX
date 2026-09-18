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
    explicit FlatIndex(PDXIndexConfig config)
        : config(config), index(config.num_dimensions, Normalize(config)) {
        config.Validate();
        PDX::g_n_threads = (config.n_threads == 0) ? omp_get_max_threads() : config.n_threads;
        owned_pruner = std::make_unique<ADSamplingPruner>(config.num_dimensions, config.seed);
        pruner = owned_pruner.get();
        searcher = std::make_unique<FlatSearcher>(index, *pruner);
    }

    FlatIndex(PDXIndexConfig config, ADSamplingPruner& external_pruner)
        : config(config), index(config.num_dimensions, Normalize(config)),
          pruner(&external_pruner) {
        config.Validate();
        PDX::g_n_threads = (config.n_threads == 0) ? omp_get_max_threads() : config.n_threads;
        searcher = std::make_unique<FlatSearcher>(index, *pruner);
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
        if (GetRowIdMapping(row_id).first != DELETED_MARKER) {
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

    void Delete(size_t row_id) override {
        const auto [cluster_id, position] = GetRowIdMapping(row_id);
        if (cluster_id == DELETED_MARKER) {
            return;
        }
        index.DeleteEmbedding(position);
        row_id_cluster_mapping.Delete(row_id);
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
        bool is_query_transformed = false
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

    void SetNProbe(uint32_t) override {}

    void Save(const std::string&) override {
        throw std::logic_error("FlatIndex does not support Save");
    }

    void Restore(const std::string&) override {
        throw std::logic_error("FlatIndex does not support Restore");
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
                    index.GetEmbedding(position),
                    index.GetEmbedding(position) + d,
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

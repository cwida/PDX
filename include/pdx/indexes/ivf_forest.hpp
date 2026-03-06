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
#include "pdx/indexes/ivf_core.hpp"
#include "pdx/indexes/ivf_utils.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/profiler.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"
#include "pdx/searcher.hpp"
#include "pdx/utils.hpp"
#include <omp.h>

namespace PDX {
template <PDX::Quantization Q>
class PDXForestIndex : public IPDXIndex {

    constexpr static uint32_t NUM_EMBEDDINGS_PER_TREE = 128000;
    constexpr static uint32_t NUM_CLUSTERS_PER_TREE = 1280;

    constexpr static uint32_t MIN_EMBEDDINGS_TO_CREATE_A_TREE = 2048;
    constexpr static uint32_t NUM_CLUSTERS_FOR_NEW_TREE = 8;

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
    PDXIndexConfig config{};
    uint32_t d = 0;
    std::vector<std::pair<uint32_t, uint32_t>> row_id_tree_offset;
    std::vector<std::unique_ptr<PDXTreeIndex<Q>>> forest;

    uint32_t forest_n_probe = 16;

    std::unique_ptr<embedding_storage_t[]> new_tree_buffer;
    std::unique_ptr<size_t[]> new_tree_buffer_row_ids;
    size_t embeddings_in_new_tree_buffer = 0;
    bool appending_to_new_tree_buffer = false;
    std::mutex new_tree_buffer_mutex;

    // PRUNER SHOULD BE THE SAME FOR ALL tree IN forest
    std::unique_ptr<PDX::ADSamplingPruner> pruner;
    ScalarQuantizer<Q> quantizer{0};

  public:
    PDXForestIndex() = default;

    explicit PDXForestIndex(PDXIndexConfig config)
        : config(config), d(config.num_dimensions), quantizer(config.num_dimensions) {
        config.Validate();
        PDX::g_n_threads = (config.n_threads == 0) ? omp_get_max_threads() : config.n_threads;
        pruner = std::make_unique<PDX::ADSamplingPruner>(config.num_dimensions, config.seed);
    }

    void SetNProbe(uint32_t n_probe) override { forest_n_probe = n_probe; }

    uint32_t GetNumDimensions() const override { return d; }

    uint32_t GetNumTrees() const { return forest.size(); }

    uint32_t GetNumClusters() const override {
        uint32_t total = 0;
        for (const auto& tree : forest) {
            total += tree->GetNumClusters();
        }
        return total;
    }

    uint32_t GetForestNClusters() const {
        uint32_t total = 0;
        for (const auto& tree : forest) {
            total += tree->GetNumClusters();
        }
        return total;
    }

    uint32_t GetForestNL0Clusters() const {
        uint32_t total = 0;
        for (const auto& tree : forest) {
            total += tree->GetTopLevelNumClusters();
        }
        return total;
    }

    size_t GetNumVectorsAccessed() const {
        size_t total = 0;
        for (const auto& tree : forest) {
            total += tree->GetNumVectorsAccessed();
        }
        return total;
    }

    uint32_t GetClusterSize(uint32_t /*cluster_id*/) const override {
        throw std::runtime_error("GetClusterSize is not supported by PDXForestIndex.");
    }

    std::vector<uint32_t> GetClusterRowIds(uint32_t /*cluster_id*/) const override {
        throw std::runtime_error("GetClusterRowIds is not supported by PDXForestIndex.");
    }

    size_t GetInMemorySizeInBytes() const override {
        size_t size = sizeof(*this);
        for (const auto& tree : forest) {
            size += tree->GetInMemorySizeInBytes();
        }
        size += row_id_tree_offset.capacity() * sizeof(std::pair<uint32_t, uint32_t>);
        if (pruner) {
            size += sizeof(*pruner);
            const auto& m = pruner->GetMatrix();
            size += static_cast<size_t>(m.rows()) * m.cols() * sizeof(float);
            size += pruner->num_dimensions * sizeof(float);
            if (m.rows() == 1) {
                size += pruner->num_dimensions * sizeof(uint32_t);
            }
        }
        return size;
    }

    void Save(const std::string& /*path*/) override {
        throw std::runtime_error("Save is not yet implemented for PDXForestIndex.");
    }

    void Restore(const std::string& /*path*/) override {
        throw std::runtime_error("Restore is not yet implemented for PDXForestIndex.");
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
        assert(num_embeddings > 0);
        config.ValidateNumEmbeddings(num_embeddings);

        size_t num_trees = (num_embeddings + NUM_EMBEDDINGS_PER_TREE - 1) / NUM_EMBEDDINGS_PER_TREE;

        forest.clear();
        forest.reserve(num_trees);
        row_id_tree_offset.resize(num_embeddings);

        for (size_t t = 0; t < num_trees; t++) {
            size_t start = t * NUM_EMBEDDINGS_PER_TREE;
            size_t count = std::min(
                static_cast<size_t>(NUM_EMBEDDINGS_PER_TREE), num_embeddings - start
            );

            uint32_t tree_num_clusters = NUM_CLUSTERS_PER_TREE;
            if (t == num_trees - 1 && count < NUM_EMBEDDINGS_PER_TREE) {
                tree_num_clusters = std::max(1u, static_cast<uint32_t>(count / 256));
            }

            PDXIndexConfig tree_config = config;
            tree_config.num_clusters = tree_num_clusters;
            tree_config.hierarchical_indexing = false;
            tree_config.sampling_fraction = 1.0f;

            auto tree = std::make_unique<PDXTreeIndex<Q>>(tree_config, *pruner);
            tree->BuildIndex(row_ids + start, embeddings + start * d, count);
            forest.push_back(std::move(tree));

            // Map global row_id -> (tree_index, local_row_id)
            for (size_t i = 0; i < count; i++) {
                row_id_tree_offset[row_ids[start + i]] = {
                    static_cast<uint32_t>(t),
                    static_cast<uint32_t>(i)
                };
            }
        }
    }

    void Append(size_t row_id, const float* PDX_RESTRICT embedding) override {
        // Always append to the last tree in the forest.
        // If the last tree is full (tree.num_embeddings == NUM_EMBEDDINGS_PER_TREE), 
        //  -> start appending embeddings to the new_tree_buffer
        // When embeddings_in_new_tree_buffer reaches MIN_EMBEDDINGS_TO_CREATE_A_TREE, 
        //  -> create a new tree with the embeddings in new_tree_buffer with NUM_CLUSTERS_FOR_NEW_TREE
        //  -> push_back it to the forest
        // The following appends would go to this new tree until it reaches NUM_EMBEDDINGS_PER_TREE, and everything repeats
    }

    void Delete(size_t row_id) override {
        // Determine in which Tree the row_id is located
        // Then, transform the row_id to the local row_id in that Tree and call Delete on that Tree
    }

    std::vector<PDX::KNNCandidate> Search(const float* query_embedding, size_t knn) const override {
        PDX_PROFILE_SCOPE("ForestSearch");
        const bool normalize =
            config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);
        auto preprocessed = NormalizeAndRotate(query_embedding, 1, d, normalize, *pruner);
        Heap forest_heap{};
        // std::mutex *forest_heap_mutex = nullptr;
        // Transform query embedding once before searching in the trees
        for (const auto& tree : forest) {
            if (forest_n_probe == 0) {
                tree->searcher->SetNProbe(tree->GetNumClusters());
            } else {
                tree->searcher->SetNProbe(forest_n_probe);
            }
            auto n_probe_top_level = tree->GetTopLevelNumClusters();
            // We confidently prune half of the search space
            if (tree->searcher->GetNProbe() < tree->GetNumClusters() / 2) {
                n_probe_top_level /= 2;
            }

            {
                PDX_PROFILE_SCOPE("ForestL0Search");
                tree->top_level_searcher->SetNProbe(n_probe_top_level);
                auto top_level_results = tree->top_level_searcher->Search(preprocessed.get(), tree->searcher->GetNProbe(), true);

                std::vector<uint32_t> top_level_indexes(top_level_results.size());
                for (size_t i = 0; i < top_level_results.size(); i++) {
                    top_level_indexes[i] = top_level_results[i].index;
                }
                tree->searcher->SetClusterAccessOrder(top_level_indexes);
            }

            // TODO(@lkuffo, mid): Each search will still Quantize the embedding
            // We could add a new parameter to the search to tell it that the embedding is quantized
            // It would require more work as Search always expects to receive a float
            {
                PDX_PROFILE_SCOPE("ForestL1Search");
                tree->searcher->Search(preprocessed.get(), knn, true, &forest_heap);
            }
        }
        return PDXearch<Q>::BuildResultSetFromHeap(knn, forest_heap);
    }

    std::vector<PDX::KNNCandidate> FilteredSearch(
        const float* query_embedding,
        size_t knn,
        const std::vector<size_t>& passing_row_ids
    ) const override {
        PDX_PROFILE_SCOPE("ForestFilteredSearch");
        const bool normalize =
            config.normalize || DistanceMetricRequiresNormalization(config.distance_metric);
        auto preprocessed = NormalizeAndRotate(query_embedding, 1, d, normalize, *pruner);
        Heap forest_heap{};
        for (const auto& tree : forest) {
            if (forest_n_probe == 0) {
                tree->searcher->SetNProbe(tree->GetNumClusters());
            } else {
                tree->searcher->SetNProbe(forest_n_probe);
            }
            auto evaluator = tree->CreatePredicateEvaluator(passing_row_ids);
            tree->searcher->FilteredSearch(preprocessed.get(), knn, evaluator, true, &forest_heap);
        }
        return PDXearch<Q>::BuildResultSetFromHeap(knn, forest_heap);
    }

};


using PDXForestIndexF32 = PDXForestIndex<PDX::F32>;
using PDXForestIndexU8 = PDXForestIndex<PDX::U8>;

} // namespace PDX
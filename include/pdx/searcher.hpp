#pragma once

#include "pdx/common.hpp"
#include "pdx/db_mock/predicate_evaluator.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/indexes/cluster.hpp"
#include "pdx/indexes/ivf_core.hpp"
#include "pdx/profiler.hpp"
#include "pdx/pruners/adsampling.hpp"
#include "pdx/quantizers/scalar.hpp"
#include "pdx/utils.hpp"
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <vector>

namespace PDX {

[[nodiscard]] inline std::vector<KNNCandidate> BuildResultSetFromHeap(uint32_t k, Heap& heap) {
    size_t result_set_size = std::min(heap.size(), static_cast<size_t>(k));
    std::vector<KNNCandidate> result;
    result.resize(result_set_size);
    for (size_t i = result_set_size; i > 0; --i) {
        result[i - 1] = heap.top();
        heap.pop();
    }
    return result;
}

// Resumable search over one query. Next() probes the next clusters (nearest first) into the
// caller's heap; Done() is true once no cluster that can still contribute remains.
class IIterativeSearch {
  public:
    virtual ~IIterativeSearch() = default;
    virtual size_t Next(size_t n_clusters) = 0;
    [[nodiscard]] virtual bool Done() const = 0;
    [[nodiscard]] virtual size_t ClustersRemaining() const = 0;
};

struct TopKHeap {
    explicit TopKHeap(bool thread_safe = false) : thread_safe(thread_safe) {}

    [[nodiscard]] std::unique_lock<std::mutex> GetLock() {
        return thread_safe ? std::unique_lock<std::mutex>(mutex) : std::unique_lock<std::mutex>();
    }

    Heap heap;
    std::mutex mutex;
    bool thread_safe;
};

template <
    Quantization Q = F32,
    class Index = IVF<Q>,
    class Quantizer = ScalarQuantizer<Q>,
    DistanceMetric alpha = DistanceMetric::L2SQ,
    class Pruner = ADSamplingPruner>
class PDXearch {
  public:
    using distance_t = pdx_distance_t<Q>;
    using data_t = pdx_data_t<Q>;
    using quantized_embedding_t = pdx_quantized_embedding_t<Q>;
    using index_t = Index;
    using cluster_t = Cluster<Q>;
    using tombstones_t = typename cluster_t::tombstones_t;
    using distance_computer_t = DistanceComputer<alpha, Q>;

    Quantizer quantizer;
    Pruner& pruner;
    index_t& pdx_data;

    PDXearch(index_t& data_index, Pruner& pruner)
        : quantizer(data_index.num_dimensions), pruner(pruner), pdx_data(data_index) {}

    void SetNProbe(size_t nprobe) { ivf_nprobe = nprobe; }

    size_t GetNProbe() const { return ivf_nprobe; }

    void SetClusterAccessOrder(const std::vector<uint32_t>& cluster_indexes) {
        cluster_access_order_size = cluster_indexes.size();
        cluster_indices_in_access_order.reset(new uint32_t[cluster_indexes.size()]);
        std::copy(
            cluster_indexes.begin(), cluster_indexes.end(), cluster_indices_in_access_order.get()
        );
    }

  protected:
    float selectivity_threshold = 0.80;
    size_t ivf_nprobe = 0;

    // Prioritized list of indices of the clusters to probe. E.g., [0, 2, 1].
    std::unique_ptr<uint32_t[]> cluster_indices_in_access_order;
    size_t cluster_access_order_size = 0;

    size_t ClustersToVisit() const {
        return (ivf_nprobe == 0 || ivf_nprobe > pdx_data.num_clusters) ? pdx_data.num_clusters
                                                                       : ivf_nprobe;
    }

    void ResetPruningDistances(size_t n_vectors, distance_t* pruning_distances) {
        std::fill(pruning_distances, pruning_distances + n_vectors, distance_t{0});
    }

    // The pruning threshold by default is the top of the heap
    void GetPruningThreshold(
        uint32_t k,
        TopKHeap& top_k_heap,
        distance_t& pruning_threshold,
        uint32_t current_dimension_idx
    ) {
        float float_threshold;
        {
            auto lock = top_k_heap.GetLock();
            // Fewer than k candidates: nothing real can be pruned, only masked slots
            if (top_k_heap.heap.size() < k) {
                pruning_threshold = std::numeric_limits<distance_t>::max() / 2;
                return;
            }
            float_threshold = pruner.GetPruningThreshold(k, top_k_heap.heap, current_dimension_idx);
        }
        if constexpr (Q == U8) {
            // We need to avoid undefined behaviour when overflow happens
            const float scaled = float_threshold * pdx_data.quantization_scale_squared;
            pruning_threshold = scaled >= static_cast<float>(std::numeric_limits<distance_t>::max())
                                    ? std::numeric_limits<distance_t>::max()
                                    : static_cast<distance_t>(scaled);
        } else {
            pruning_threshold = float_threshold;
        }
    };

    void EvaluatePruningPredicateScalar(
        uint32_t& n_pruned,
        size_t n_vectors,
        distance_t* pruning_distances,
        const distance_t pruning_threshold
    ) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
        }
    };

    void EvaluatePruningPredicateOnPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        distance_t pruning_threshold,
        distance_t* pruning_distances
    ) {
        // PDX_PROFILE_SCOPE("Search/EvaluatePruningPredicate");
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
            n_vectors_not_pruned +=
                pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
        }
    };

    template <bool IS_FILTERED = false>
    void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        distance_t pruning_threshold,
        distance_t* pruning_distances,
        const uint8_t* selection_vector = nullptr
    ) {
        // PDX_PROFILE_SCOPE("Search/InitPositionsArray");
        n_vectors_not_pruned = 0;
        if constexpr (IS_FILTERED) {
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                pruning_positions[n_vectors_not_pruned] = vector_idx;
                n_vectors_not_pruned += (pruning_distances[vector_idx] < pruning_threshold) &&
                                        (selection_vector[vector_idx] == 1);
            }
        } else {
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                pruning_positions[n_vectors_not_pruned] = vector_idx;
                n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
            }
        }
    };

    void InitPositionsArrayFromSelectionVector(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        const uint8_t* selection_vector
    ) {
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += selection_vector[vector_idx] == 1;
        }
    };

    void MaskDistancesWithSelectionVector(
        size_t n_vectors,
        distance_t* pruning_distances,
        const uint8_t* selection_vector
    ) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            if (selection_vector[vector_idx] == 0) {
                // Why max()/2? To prevent overflow if distances are still added to these
                pruning_distances[vector_idx] = std::numeric_limits<distance_t>::max() / 2;
            }
        }
    };

    void MaskDistancesWithTombstones(
        const typename cluster_t::tombstones_t& tombstones,
        distance_t* pruning_distances
    ) {
        if (tombstones.empty())
            return;
        const distance_t mask = std::numeric_limits<distance_t>::max() / 2;
        for (uint32_t idx : tombstones) {
            pruning_distances[idx] = mask;
        }
    }

    static void GetClustersAccessOrderIVF(
        const float* PDX_RESTRICT query,
        const index_t& data,
        size_t nprobe,
        uint32_t* clusters_indices
    ) {
        PDX_PROFILE_SCOPE("Search/GetClustersAccessOrderIVF");
        std::unique_ptr<float[]> distances_to_centroids(new float[data.num_clusters]);
        for (size_t cluster_idx = 0; cluster_idx < data.num_clusters; cluster_idx++) {
            distances_to_centroids[cluster_idx] =
                DistanceComputer<DistanceMetric::L2SQ, F32>::Horizontal(
                    query,
                    data.centroids.data() + cluster_idx * data.num_dimensions,
                    data.num_dimensions
                );
        }
        std::iota(clusters_indices, clusters_indices + data.num_clusters, static_cast<uint32_t>(0));
        if (nprobe >= data.num_clusters) {
            std::sort(
                clusters_indices,
                clusters_indices + data.num_clusters,
                [&distances_to_centroids](size_t i1, size_t i2) {
                    return distances_to_centroids[i1] < distances_to_centroids[i2];
                }
            );
        } else {
            std::partial_sort(
                clusters_indices,
                clusters_indices + static_cast<int64_t>(nprobe),
                clusters_indices + data.num_clusters,
                [&distances_to_centroids](size_t i1, size_t i2) {
                    return distances_to_centroids[i1] < distances_to_centroids[i2];
                }
            );
        }
    }

    // On the first bucket, we do a full scan (we do not prune vectors)
    void Start(
        const quantized_embedding_t* PDX_RESTRICT query,
        const data_t* data,
        const size_t n_vectors,
        const size_t buffer_stride,
        uint32_t k,
        const uint32_t* vector_indices,
        uint32_t* pruning_positions,
        distance_t* pruning_distances,
        TopKHeap& top_k_heap,
        const tombstones_t& tombstones
    ) {
        Heap& heap = top_k_heap.heap;
        ResetPruningDistances(n_vectors, pruning_distances);
        distance_computer_t::Vertical(
            query,
            data,
            n_vectors,
            buffer_stride,
            0,
            pdx_data.num_vertical_dimensions,
            pruning_distances,
            pruning_positions
        );
        for (size_t horizontal_dimension = 0;
             horizontal_dimension < pdx_data.num_horizontal_dimensions;
             horizontal_dimension += H_DIM_SIZE) {
            for (size_t vector_idx = 0; vector_idx < n_vectors; vector_idx++) {
                size_t data_pos = (pdx_data.num_vertical_dimensions * buffer_stride) +
                                  (horizontal_dimension * buffer_stride) +
                                  (vector_idx * H_DIM_SIZE);
                pruning_distances[vector_idx] += distance_computer_t::Horizontal(
                    query + pdx_data.num_vertical_dimensions + horizontal_dimension,
                    data + data_pos,
                    H_DIM_SIZE
                );
            }
        }
        MaskDistancesWithTombstones(tombstones, pruning_distances);
        size_t max_possible_k = std::min(
            static_cast<size_t>(k) - heap.size(),
            n_vectors
        ); // Note: Start() should not be called if heap.size() >= k
        std::unique_ptr<size_t[]> indices_sorted(new size_t[n_vectors]);
        std::iota(indices_sorted.get(), indices_sorted.get() + n_vectors, static_cast<size_t>(0));
        std::partial_sort(
            indices_sorted.get(),
            indices_sorted.get() + static_cast<int64_t>(max_possible_k),
            indices_sorted.get() + n_vectors,
            [pruning_distances](size_t i1, size_t i2) {
                return pruning_distances[i1] < pruning_distances[i2];
            }
        );
        // insert first k results into the heap
        for (size_t idx = 0; idx < max_possible_k; ++idx) {
            auto embedding = KNNCandidate{};
            size_t index = indices_sorted[idx];
            embedding.index = vector_indices[index];
            embedding.distance = static_cast<float>(pruning_distances[index]);
            if constexpr (Q == U8) {
                embedding.distance *= pdx_data.inverse_quantization_scale_squared;
            }
            heap.push(embedding);
        }
    }

    // On the first bucket, we do a full scan (we do not prune vectors)
    void FilteredStart(
        const quantized_embedding_t* PDX_RESTRICT query,
        const data_t* data,
        const size_t n_vectors,
        const size_t buffer_stride,
        uint32_t k,
        const uint32_t* vector_indices,
        uint32_t* pruning_positions,
        distance_t* pruning_distances,
        TopKHeap& top_k_heap,
        uint8_t* selection_vector,
        uint32_t passing_tuples,
        const tombstones_t& tombstones
    ) {
        // PDX_PROFILE_SCOPE("Search/FilteredStart");
        Heap& heap = top_k_heap.heap;
        ResetPruningDistances(n_vectors, pruning_distances);
        size_t n_vectors_not_pruned = 0;
        float selection_percentage =
            (static_cast<float>(passing_tuples) / static_cast<float>(n_vectors));
        InitPositionsArrayFromSelectionVector(
            n_vectors, n_vectors_not_pruned, pruning_positions, selection_vector
        );
        // Always start with horizontal block, regardless of selectivity
        for (size_t horizontal_dimension = 0;
             horizontal_dimension < pdx_data.num_horizontal_dimensions;
             horizontal_dimension += H_DIM_SIZE) {
            size_t offset_data = (pdx_data.num_vertical_dimensions * buffer_stride) +
                                 (horizontal_dimension * buffer_stride);
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                pruning_distances[v_idx] += distance_computer_t::Horizontal(
                    query + pdx_data.num_vertical_dimensions + horizontal_dimension,
                    data + data_pos,
                    H_DIM_SIZE
                );
            }
        }
        if (selection_percentage > (1 - selectivity_threshold)) {
            // It is then faster to do the full scan (thanks to SIMD)
            distance_computer_t::Vertical(
                query,
                data,
                n_vectors,
                buffer_stride,
                0,
                pdx_data.num_vertical_dimensions,
                pruning_distances,
                pruning_positions
            );
        } else {
            // We access individual values
            distance_computer_t::VerticalPruning(
                query,
                data,
                n_vectors_not_pruned,
                buffer_stride,
                0,
                pdx_data.num_vertical_dimensions,
                pruning_distances,
                pruning_positions
            );
        }
        // TODO: Everything down from here is a bottleneck when selection % is ultra low
        size_t max_possible_k =
            std::min(static_cast<size_t>(k) - heap.size(), static_cast<size_t>(passing_tuples));
        MaskDistancesWithSelectionVector(n_vectors, pruning_distances, selection_vector);
        MaskDistancesWithTombstones(tombstones, pruning_distances);
        std::unique_ptr<size_t[]> indices_sorted(new size_t[n_vectors]);
        std::iota(indices_sorted.get(), indices_sorted.get() + n_vectors, static_cast<size_t>(0));
        std::partial_sort(
            indices_sorted.get(),
            indices_sorted.get() + static_cast<int64_t>(max_possible_k),
            indices_sorted.get() + n_vectors,
            [pruning_distances](size_t i1, size_t i2) {
                return pruning_distances[i1] < pruning_distances[i2];
            }
        );
        // insert first k results into the heap
        for (size_t idx = 0; idx < max_possible_k; ++idx) {
            auto embedding = KNNCandidate{};
            size_t index = indices_sorted[idx];
            embedding.index = vector_indices[index];
            embedding.distance = static_cast<float>(pruning_distances[index]);
            if constexpr (Q == U8) {
                embedding.distance *= pdx_data.inverse_quantization_scale_squared;
            }
            heap.push(embedding);
        }
    }

    // On the warmup phase, we keep scanning dimensions until the amount of not-yet pruned vectors
    // is low
    template <bool FILTERED = false>
    void Warmup(
        const quantized_embedding_t* PDX_RESTRICT query,
        const data_t* PDX_RESTRICT data,
        const size_t n_vectors,
        const size_t buffer_stride,
        uint32_t k,
        float tuples_threshold,
        uint32_t* pruning_positions,
        distance_t* pruning_distances,
        distance_t& pruning_threshold,
        TopKHeap& top_k_heap,
        uint32_t& current_dimension_idx,
        size_t& n_vectors_not_pruned,
        const tombstones_t& tombstones,
        uint32_t passing_tuples = 0,
        uint8_t* selection_vector = nullptr
    ) {
        // PDX_PROFILE_SCOPE("Search/Warmup");
        current_dimension_idx = 0;
        size_t cur_subgrouping_size_idx = 0;
        size_t tuples_needed_to_exit =
            static_cast<size_t>(std::ceil(tuples_threshold * static_cast<float>(n_vectors)));
        ResetPruningDistances(n_vectors, pruning_distances);
        MaskDistancesWithTombstones(tombstones, pruning_distances);
        uint32_t n_tuples_to_prune = 0;
        if constexpr (FILTERED) {
            float selection_percentage =
                (static_cast<float>(passing_tuples) / static_cast<float>(n_vectors));
            MaskDistancesWithSelectionVector(n_vectors, pruning_distances, selection_vector);
            if (selection_percentage < (1 - tuples_threshold)) {
                // Go directly to the PRUNE phase for direct tuples access in the Horizontal block
                return;
            }
        }
        GetPruningThreshold(k, top_k_heap, pruning_threshold, current_dimension_idx);
        while (n_tuples_to_prune < tuples_needed_to_exit &&
               current_dimension_idx < pdx_data.num_vertical_dimensions) {
            size_t last_dimension_to_fetch = std::min(
                current_dimension_idx + DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
                pdx_data.num_vertical_dimensions
            );
            distance_computer_t::Vertical(
                query,
                data,
                n_vectors,
                buffer_stride,
                current_dimension_idx,
                last_dimension_to_fetch,
                pruning_distances,
                pruning_positions
            );
            current_dimension_idx = last_dimension_to_fetch;
            cur_subgrouping_size_idx += 1;
            GetPruningThreshold(k, top_k_heap, pruning_threshold, current_dimension_idx);
            n_tuples_to_prune = 0;
            EvaluatePruningPredicateScalar(
                n_tuples_to_prune, n_vectors, pruning_distances, pruning_threshold
            );
        }
    }

    // We scan only the not-yet pruned vectors
    template <bool FILTERED = false>
    void Prune(
        const quantized_embedding_t* PDX_RESTRICT query,
        const data_t* PDX_RESTRICT data,
        const size_t n_vectors,
        const size_t buffer_stride,
        uint32_t k,
        uint32_t* pruning_positions,
        distance_t* pruning_distances,
        distance_t& pruning_threshold,
        TopKHeap& top_k_heap,
        uint32_t& current_dimension_idx,
        size_t& n_vectors_not_pruned,
        const tombstones_t& tombstones,
        const uint8_t* selection_vector = nullptr
    ) {
        // PDX_PROFILE_SCOPE("Search/Prune");
        GetPruningThreshold(k, top_k_heap, pruning_threshold, current_dimension_idx);
        MaskDistancesWithTombstones(tombstones, pruning_distances);
        InitPositionsArray<FILTERED>(
            n_vectors,
            n_vectors_not_pruned,
            pruning_positions,
            pruning_threshold,
            pruning_distances,
            selection_vector
        );
        size_t cur_n_vectors_not_pruned = 0;
        size_t current_vertical_dimension = current_dimension_idx;
        size_t current_horizontal_dimension = 0;
        while (pdx_data.num_horizontal_dimensions && n_vectors_not_pruned &&
               current_horizontal_dimension < pdx_data.num_horizontal_dimensions) {
            // PDX_PROFILE_SCOPE("Search/PruneHorizontal");
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            size_t offset_data = (pdx_data.num_vertical_dimensions * buffer_stride) +
                                 (current_horizontal_dimension * buffer_stride);
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                __builtin_prefetch(data + data_pos, 0, 3);
            }
            size_t offset_query = pdx_data.num_vertical_dimensions + current_horizontal_dimension;
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                pruning_distances[v_idx] += distance_computer_t::Horizontal(
                    query + offset_query, data + data_pos, H_DIM_SIZE
                );
            }
            // end of clipping
            current_horizontal_dimension += H_DIM_SIZE;
            current_dimension_idx += H_DIM_SIZE;
            GetPruningThreshold(k, top_k_heap, pruning_threshold, current_dimension_idx);
            assert(
                current_dimension_idx == current_vertical_dimension + current_horizontal_dimension
            );
            EvaluatePruningPredicateOnPositionsArray(
                cur_n_vectors_not_pruned,
                n_vectors_not_pruned,
                pruning_positions,
                pruning_threshold,
                pruning_distances
            );
        }
        // GO THROUGH THE REST IN THE VERTICAL
        while (n_vectors_not_pruned && current_vertical_dimension < pdx_data.num_vertical_dimensions
        ) {
            // PDX_PROFILE_SCOPE("Search/PruneVertical");
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            size_t last_dimension_to_test_idx = std::min(
                current_vertical_dimension + H_DIM_SIZE,
                static_cast<size_t>(pdx_data.num_vertical_dimensions)
            );
            distance_computer_t::VerticalPruning(
                query,
                data,
                cur_n_vectors_not_pruned,
                buffer_stride,
                current_vertical_dimension,
                last_dimension_to_test_idx,
                pruning_distances,
                pruning_positions
            );
            current_dimension_idx = std::min(
                current_dimension_idx + H_DIM_SIZE, static_cast<size_t>(pdx_data.num_dimensions)
            );
            current_vertical_dimension = std::min(
                static_cast<uint32_t>(current_vertical_dimension + H_DIM_SIZE),
                pdx_data.num_vertical_dimensions
            );
            assert(
                current_dimension_idx == current_vertical_dimension + current_horizontal_dimension
            );
            GetPruningThreshold(k, top_k_heap, pruning_threshold, current_dimension_idx);
            EvaluatePruningPredicateOnPositionsArray(
                cur_n_vectors_not_pruned,
                n_vectors_not_pruned,
                pruning_positions,
                pruning_threshold,
                pruning_distances
            );
            if (current_dimension_idx == pdx_data.num_dimensions) {
                break;
            }
        }
    }

    void MergeIntoHeap(
        const uint32_t* vector_indices,
        const size_t n_vectors,
        const uint32_t k,
        const uint32_t* pruning_positions,
        const distance_t* pruning_distances,
        TopKHeap& top_k_heap
    ) {
        Heap& heap = top_k_heap.heap;
        for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
            const size_t index = pruning_positions[position_idx];
            float current_distance = static_cast<float>(pruning_distances[index]);
            if constexpr (Q == U8) {
                current_distance *= pdx_data.inverse_quantization_scale_squared;
            }
            if (heap.size() < k || current_distance < heap.top().distance) {
                KNNCandidate embedding{};
                embedding.distance = current_distance;
                embedding.index = vector_indices[index];
                if (heap.size() >= k) {
                    heap.pop();
                }
                heap.push(embedding);
            }
        }
    }

    void GetClustersAccessOrderRandom() {
        std::iota(
            cluster_indices_in_access_order.get(),
            cluster_indices_in_access_order.get() + pdx_data.num_clusters,
            0
        );
    }

  public:
    [[nodiscard]] static std::vector<KNNCandidate> BuildResultSetFromHeap(uint32_t k, Heap& heap) {
        return ::PDX::BuildResultSetFromHeap(k, heap);
    }

    // Cursor of a resumable search. Owns all per-query state, so any number of cursors can run
    // concurrently on the same searcher, and several may share one thread_safe TopKHeap.
    template <bool FILTERED>
    class IterativeSearch final : public IIterativeSearch {
      public:
        IterativeSearch(IterativeSearch&&) noexcept = default;

        size_t Next(size_t n_clusters) override {
            size_t n_probed_this_call = 0;
            for (; n_probed_this_call < n_clusters && n_probed_clusters < queued_clusters_size;
                 ++n_probed_this_call, ++n_probed_clusters) {
                ProbeCluster(queued_clusters[n_probed_clusters]);
            }
            return n_probed_this_call;
        }

        [[nodiscard]] bool Done() const override {
            return n_probed_clusters >= queued_clusters_size;
        }

        [[nodiscard]] size_t ClustersRemaining() const override {
            return queued_clusters_size - n_probed_clusters;
        }

      private:
        friend class PDXearch;

        IterativeSearch(
            PDXearch& searcher,
            uint32_t k,
            TopKHeap& top_k_heap,
            const PredicateEvaluator* evaluator
        )
            : searcher(&searcher), top_k_heap(&top_k_heap), k(k), evaluator(evaluator) {}

        void ProbeCluster(uint32_t cluster_id) {
            PDXearch& s = *searcher;
            cluster_t& cluster = s.pdx_data.clusters[cluster_id];
            cluster.n_accessed.fetch_add(1, std::memory_order_relaxed);
            Heap& heap = top_k_heap->heap;
            uint8_t* selection_vector = nullptr;
            uint32_t passing_tuples = 0;
            if constexpr (FILTERED) {
                auto selection = evaluator->GetSelectionVector(
                    cluster_id, s.pdx_data.cluster_offsets[cluster_id]
                );
                selection_vector = selection.first;
                passing_tuples = selection.second;
            }
            {
                auto lock = top_k_heap->GetLock();
                if (heap.size() < k) {
                    // NOLINTNEXTLINE(bugprone-branch-clone)
                    if constexpr (FILTERED) {
                        s.FilteredStart(
                            prepared_query,
                            cluster.data,
                            cluster.used_capacity,
                            cluster.max_capacity,
                            k,
                            cluster.indices,
                            pruning_positions.get(),
                            pruning_distances.get(),
                            *top_k_heap,
                            selection_vector,
                            passing_tuples,
                            cluster.tombstones
                        );
                    } else {
                        s.Start(
                            prepared_query,
                            cluster.data,
                            cluster.used_capacity,
                            cluster.max_capacity,
                            k,
                            cluster.indices,
                            pruning_positions.get(),
                            pruning_distances.get(),
                            *top_k_heap,
                            cluster.tombstones
                        );
                    }
                    return;
                }
            }
            distance_t pruning_threshold = std::numeric_limits<distance_t>::max();
            uint32_t current_dimension_idx = 0;
            size_t n_vectors_not_pruned = 0;
            s.template Warmup<FILTERED>(
                prepared_query,
                cluster.data,
                cluster.used_capacity,
                cluster.max_capacity,
                k,
                s.selectivity_threshold,
                pruning_positions.get(),
                pruning_distances.get(),
                pruning_threshold,
                *top_k_heap,
                current_dimension_idx,
                n_vectors_not_pruned,
                cluster.tombstones,
                passing_tuples,
                selection_vector
            );
            s.template Prune<FILTERED>(
                prepared_query,
                cluster.data,
                cluster.used_capacity,
                cluster.max_capacity,
                k,
                pruning_positions.get(),
                pruning_distances.get(),
                pruning_threshold,
                *top_k_heap,
                current_dimension_idx,
                n_vectors_not_pruned,
                cluster.tombstones,
                selection_vector
            );
            if (n_vectors_not_pruned) {
                auto lock = top_k_heap->GetLock();
                s.MergeIntoHeap(
                    cluster.indices,
                    n_vectors_not_pruned,
                    k,
                    pruning_positions.get(),
                    pruning_distances.get(),
                    *top_k_heap
                );
            }
        }

        PDXearch* searcher;
        TopKHeap* top_k_heap;
        uint32_t k;
        const PredicateEvaluator* evaluator;
        // To keep the evaluator alive as long as the cursor is alive
        std::unique_ptr<PredicateEvaluator> owned_evaluator;
        std::unique_ptr<float[]> query;
        std::unique_ptr<quantized_embedding_t[]> quantized_query;
        const quantized_embedding_t* prepared_query = nullptr;
        std::unique_ptr<uint32_t[]> queued_clusters;
        size_t queued_clusters_size = 0;
        size_t n_probed_clusters = 0;
        std::unique_ptr<distance_t[]> pruning_distances;
        std::unique_ptr<uint32_t[]> pruning_positions;
    };

    [[nodiscard]] IterativeSearch<false> BeginIterativeSearch(
        const float* PDX_RESTRICT raw_query,
        uint32_t k,
        TopKHeap& top_k_heap,
        bool is_query_transformed = false
    ) {
        IterativeSearch<false> search_cursor(*this, k, top_k_heap, nullptr);
        InitializeSearchCursor(
            search_cursor, raw_query, is_query_transformed, pdx_data.num_clusters
        );
        return search_cursor;
    }

    [[nodiscard]] IterativeSearch<true> BeginFilteredIterativeSearch(
        const float* PDX_RESTRICT raw_query,
        uint32_t k,
        std::unique_ptr<PredicateEvaluator> evaluator,
        TopKHeap& top_k_heap,
        bool is_query_transformed = false
    ) {
        assert(evaluator);
        IterativeSearch<true> search_cursor(*this, k, top_k_heap, evaluator.get());
        search_cursor.owned_evaluator = std::move(evaluator);
        InitializeSearchCursor(
            search_cursor, raw_query, is_query_transformed, pdx_data.num_clusters
        );
        return search_cursor;
    }

  protected:
    template <bool FILTERED>
    void InitializeSearchCursor(
        IterativeSearch<FILTERED>& search_cursor,
        const float* PDX_RESTRICT raw_query,
        bool is_query_transformed,
        size_t n_clusters_to_rank,
        const uint32_t* preset_clusters_access_order = nullptr
    ) {
        const size_t d = pdx_data.num_dimensions;
        search_cursor.query.reset(new float[d]);
        if (is_query_transformed) {
            std::copy(raw_query, raw_query + d, search_cursor.query.get());
        } else if (!pdx_data.is_normalized) {
            pruner.PreprocessQuery(raw_query, search_cursor.query.get());
        } else {
            std::unique_ptr<float[]> normalized_query(new float[d]);
            quantizer.NormalizeQuery(raw_query, normalized_query.get());
            pruner.PreprocessQuery(normalized_query.get(), search_cursor.query.get());
        }
        if constexpr (Q == U8) {
            search_cursor.quantized_query.reset(new quantized_embedding_t[d]);
            quantizer.QuantizeEmbedding(
                search_cursor.query.get(),
                pdx_data.quantization_base,
                pdx_data.quantization_scale,
                search_cursor.quantized_query.get()
            );
            search_cursor.prepared_query = search_cursor.quantized_query.get();
        } else {
            search_cursor.prepared_query = search_cursor.query.get();
        }

        // Rank (or copy) the clusters to probe, then drop in place the ones that cannot contribute
        search_cursor.queued_clusters.reset(new uint32_t[pdx_data.num_clusters]);
        // IVFTree path: Meso-clusters help determine access order to leaf clusters
        if (preset_clusters_access_order) {
            std::copy(
                preset_clusters_access_order,
                preset_clusters_access_order + n_clusters_to_rank,
                search_cursor.queued_clusters.get()
            );
        } else {
            GetClustersAccessOrderIVF(
                search_cursor.query.get(),
                pdx_data,
                n_clusters_to_rank,
                search_cursor.queued_clusters.get()
            );
        }
        search_cursor.queued_clusters_size = 0;
        for (size_t i = 0; i < n_clusters_to_rank; i++) {
            const uint32_t cluster_id = search_cursor.queued_clusters[i];
            if (pdx_data.clusters[cluster_id].num_embeddings == 0) {
                continue;
            }
            if constexpr (FILTERED) {
                if (search_cursor.evaluator->n_passing_tuples[cluster_id] == 0) {
                    continue;
                }
            }
            search_cursor.queued_clusters[search_cursor.queued_clusters_size++] = cluster_id;
        }

        search_cursor.pruning_distances.reset(new distance_t[pdx_data.max_cluster_capacity]);
        search_cursor.pruning_positions.reset(new uint32_t[pdx_data.max_cluster_capacity]);
    }

  public:
    std::vector<KNNCandidate> Search(
        const float* PDX_RESTRICT const raw_query,
        const uint32_t k,
        const bool is_query_transformed = false
    ) {
        TopKHeap top_k_heap{};
        IterativeSearch<false> search_cursor(*this, k, top_k_heap, nullptr);
        size_t clusters_to_visit = ClustersToVisit();
        const uint32_t* preset_clusters_access_order = nullptr;
        if (cluster_indices_in_access_order) {
            clusters_to_visit = std::min(clusters_to_visit, cluster_access_order_size);
            preset_clusters_access_order = cluster_indices_in_access_order.get();
        }
        InitializeSearchCursor(
            search_cursor,
            raw_query,
            is_query_transformed,
            clusters_to_visit,
            preset_clusters_access_order
        );
        search_cursor.Next(search_cursor.ClustersRemaining());
        return BuildResultSetFromHeap(k, top_k_heap.heap);
    }

    // TODO(@lkuffo, high): FastPath
    // If the number of passing tuples is lower than the cost of accesing the index
    // Then access the tuples directly and compute distances on the fly, without accessing the
    // index.
    std::vector<KNNCandidate> FilteredSearch(
        const float* PDX_RESTRICT const raw_query,
        const uint32_t k,
        const PredicateEvaluator& predicate_evaluator,
        const bool is_query_transformed = false
    ) {
        TopKHeap top_k_heap{};
        IterativeSearch<true> search_cursor(*this, k, top_k_heap, &predicate_evaluator);
        InitializeSearchCursor(search_cursor, raw_query, is_query_transformed, ClustersToVisit());
        search_cursor.Next(search_cursor.ClustersRemaining());
        return BuildResultSetFromHeap(k, top_k_heap.heap);
    }
};

} // namespace PDX

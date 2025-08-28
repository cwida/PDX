#ifndef PDX_PDXEARCH_HPP
#define PDX_PDXEARCH_HPP

#include <queue>
#include <cassert>
#include <algorithm>
#include <unordered_set>
#include "common.hpp"
#include "db_mock/predicate_evaluator.hpp"
#include "utils/tictoc.hpp"
#include "distance_computers/base_computers.hpp"
#include "quantizers/global.h"
#include "index_base/pdx_ivf.hpp"
#include "index_base/pdx_ivf2.hpp"
#include "pruners/adsampling.hpp"
#include "pruners/bond.hpp"

namespace PDX {

/******************************************************************
 * PDXearch
 * Implements our algorithm for vertical pruning
 ******************************************************************/
template<
    Quantization q=F32,
    class Index=IndexPDXIVF<q>,
    class Quantizer=Global8Quantizer<q>,
    DistanceFunction alpha=L2,
    class Pruner=ADSamplingPruner<q>
>
class PDXearch {
public:
    using DISTANCES_TYPE = DistanceType_t<q>;
    using QUANTIZED_VECTOR_TYPE = QuantizedVectorType_t<q>;
    using DATA_TYPE = DataType_t<q>;
    using INDEX_TYPE = Index;
    using CLUSTER_TYPE = Cluster<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;

    Quantizer quantizer;
    Pruner pruner;
    INDEX_TYPE &pdx_data;
    uint32_t current_dimension_idx {0};

    PDXearch(
            INDEX_TYPE &data_index,
            Pruner &pruner,
            int position_prune_distance,
            DimensionsOrder dimension_order
    ) : pdx_data(data_index),
        pruner(pruner),
        is_positional_pruning(position_prune_distance),
            dimension_order(dimension_order){
        indices_dimensions.resize(pdx_data.num_dimensions);
        clusters_indices.resize(pdx_data.num_clusters);
        cluster_offsets.resize(pdx_data.num_clusters);
        for (size_t i = 0; i < pdx_data.num_clusters; ++i){
            cluster_offsets[i] = total_embeddings;
            total_embeddings += pdx_data.clusters[i].num_embeddings;
        }
        if constexpr(std::is_same_v<Pruner, BondPruner<q>>) {
            pdx_data.num_horizontal_dimensions = 0;
            pdx_data.num_vertical_dimensions = pdx_data.num_dimensions;
        }
        quantizer.SetD(pdx_data.num_dimensions);
    }

    void SetNProbe(size_t nprobe){
        ivf_nprobe = nprobe;
    }

    TicToc end_to_end_clock = TicToc();

    void ResetClocks(){
        end_to_end_clock.Reset();
    }

protected:
    float selectivity_threshold = 0.80;
    size_t ivf_nprobe = 0;
    int is_positional_pruning = false;
    size_t current_cluster = 0;

    DimensionsOrder dimension_order = SEQUENTIAL;
    // Evaluating the pruning threshold is so fast that we can allow smaller fetching sizes
    // to avoid more data access. Super useful in architectures with low bandwidth at L3/DRAM like Intel SPR
    static constexpr uint32_t DIMENSIONS_FETCHING_SIZES[24] = {
            4, 4, 8, 8, 8, 16, 16, 32, 32, 32, 32,
            64, 64, 64, 64, 128, 128, 128, 128, 256,
            256, 512, 1024, 2048
    };

    size_t H_DIM_SIZE = 64;

    size_t cur_subgrouping_size_idx {0};
    size_t total_embeddings {0};

    std::vector<uint32_t> indices_dimensions;
    std::vector<uint32_t> clusters_indices;
    std::vector<uint32_t> clusters_indices_l0;
    std::vector<size_t> cluster_offsets;

    size_t n_vectors_not_pruned = 0;

    DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
    DistanceType_t<F32> pruning_threshold_l0 = std::numeric_limits<DistanceType_t<F32>>::max();

    // For pruning we do not use tight loops of 64. We know that tight loops bring benefits
    // to the distance kernels (40% faster), however doing so + PRUNING in the tight block of 64
    // slightly reduces the performance of PDXearch. We are still investigating why.
    static constexpr uint16_t PDX_VECTOR_SIZE = 64;
    alignas(64) inline static DISTANCES_TYPE distances[PDX_VECTOR_SIZE]; // Used in full scans (no pruning)
    alignas(64) inline static DISTANCES_TYPE pruning_distances[10240]; // TODO: Use dynamic arrays. Buckets with more than 10k vectors (rare) overflow
    alignas(64) inline static uint32_t pruning_positions[10240];
    std::priority_queue<KNNCandidate<q>, std::vector<KNNCandidate<q>>, VectorComparator<q>> best_k{};

    alignas(64) inline static DistanceType_t<F32> centroids_distances[PDX_VECTOR_SIZE];
    alignas(64) inline static DistanceType_t<F32> pruning_distances_l0[10240];
    alignas(64) inline static uint32_t pruning_positions_l0[10240];
    std::priority_queue<KNNCandidate<F32>, std::vector<KNNCandidate<F32>>, VectorComparator<F32>> best_k_centroids{};

    void ResetDistancesScalar(size_t n_vectors){
        memset((void*) distances, 0, n_vectors * sizeof(DISTANCES_TYPE));
    }

    template<Quantization Q=q>
    void ResetPruningDistances(
        size_t n_vectors,
        DistanceType_t<Q> *pruning_distances
    ){
        memset((void*) pruning_distances, 0, n_vectors * sizeof(DistanceType_t<Q>));
    }

    template<Quantization Q=q>
    void ResetDistancesVectorized(
        DistanceType_t<Q> *distances
    ){
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(DistanceType_t<Q>));
    }

    // The pruning threshold by default is the top of the heap
    template<Quantization Q=q>
    void GetPruningThreshold(
        uint32_t k, std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
        DistanceType_t<Q> &pruning_threshold
    ){
        pruning_threshold = pruner.template GetPruningThreshold<Q>(k, heap, current_dimension_idx);
    };

    template<Quantization Q=q>
    void EvaluatePruningPredicateScalar(
        uint32_t &n_pruned,
        size_t n_vectors,
        DistanceType_t<Q> *pruning_distances,
        const DistanceType_t<Q> pruning_threshold
    ) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
        }
    };

    template<Quantization Q=q>
    void EvaluatePruningPredicateOnPositionsArray(
        size_t n_vectors,
        size_t &n_vectors_not_pruned,
        uint32_t * pruning_positions,
        DistanceType_t<Q> pruning_threshold,
        DistanceType_t<Q> * pruning_distances
    ){
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
            n_vectors_not_pruned += pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
        }
    };

    template<Quantization Q=q>
    void EvaluatePruningPredicateVectorized(
        uint32_t &n_pruned,
        DistanceType_t<Q> pruning_threshold,
        DistanceType_t<Q> * pruning_distances
    ) {
        for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
        }
    };

    template<Quantization Q=q>
    void InitPositionsArray(
        size_t n_vectors,
        size_t &n_vectors_not_pruned,
        uint32_t * pruning_positions,
        DistanceType_t<Q> pruning_threshold,
        DistanceType_t<Q> * pruning_distances
    ){
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    };

    template<Quantization Q=q>
    void InitPositionsArrayFromSelectionVector(
        size_t n_vectors,
        size_t &n_vectors_not_pruned,
        uint32_t * pruning_positions,
        DistanceType_t<Q> pruning_threshold,
        DistanceType_t<Q> * pruning_distances,
        const uint8_t * selection_vector
    ) {
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += selection_vector[vector_idx] == 1;
        }
    };

    template<Quantization Q=q>
    void MaskDistancesWithSelectionVector(
        size_t n_vectors,
        size_t &n_vectors_not_pruned,
        uint32_t * pruning_positions,
        DistanceType_t<Q> pruning_threshold,
        DistanceType_t<Q> * pruning_distances,
        const uint8_t * selection_vector
    ) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            if (selection_vector[vector_idx] == 0) {
                // Why max()/2? To prevent overflow if distances are still added to these
                pruning_distances[vector_idx] = std::numeric_limits<DistanceType_t<Q>>::max() / 2;
            }
        }
    };

    void GetDimensionsAccessOrder(const float *__restrict query, const float *__restrict means) {
        std::iota(indices_dimensions.begin(), indices_dimensions.end(), 0);
        if (dimension_order == DISTANCE_TO_MEANS) {
            std::sort(indices_dimensions.begin(), indices_dimensions.end(),
                      [&query, &means](size_t i1, size_t i2) {
                          return std::abs(query[i1] - means[i1]) > std::abs(query[i2] - means[i2]);
                      });
        } else if (dimension_order == DISTANCE_TO_MEANS_IMPROVED) {
            // Improves Cache performance
            auto const top_perc = static_cast<size_t>(std::floor(indices_dimensions.size() / 4));
            std::partial_sort(indices_dimensions.begin(), indices_dimensions.begin() + top_perc, indices_dimensions.end(),
                              [&query, &means](size_t i1, size_t i2) {
                                  return std::abs(query[i1] - means[i1]) > std::abs(query[i2] - means[i2]);
                              });
            // By taking the top 25% dimensions and sorting them ascendingly by index
            std::sort(indices_dimensions.begin(), indices_dimensions.begin() + top_perc);
            // Then sorting the rest of the dimensions ascendingly by index
            std::sort(indices_dimensions.begin() + top_perc, indices_dimensions.end());
        } else if (dimension_order == DIMENSION_ZONES){
            uint16_t dimensions = pdx_data.num_dimensions;
            size_t estimated_embeddings_per_vg = total_embeddings / pdx_data.num_clusters;
            size_t n_dimensions_per_zone = 8192 / estimated_embeddings_per_vg;
            size_t total_zones = dimensions / n_dimensions_per_zone;
            std::vector<std::pair<uint16_t, uint16_t>> zones;
            std::vector<float> zone_ranking;
            std::vector<size_t> zones_indexes;
            zones.resize(total_zones);
            zones_indexes.resize(total_zones);
            zone_ranking.resize(total_zones);
            std::iota(zones_indexes.begin(), zones_indexes.end(), 0);
            for (size_t i = 0; i < total_zones; i++){
                uint16_t start = i * n_dimensions_per_zone;
                uint16_t end = i * n_dimensions_per_zone + n_dimensions_per_zone;
                if (end > dimensions - 1 ){
                    end = dimensions - 1;
                }
                //end = std::min(end, (uint16_t) dimensions - 1);
                zones[i] = std::pair<uint16_t, uint16_t>(start, end);
                zone_ranking[i] = 0.0;
            }
            for (size_t i = 0; i < total_zones; i++){
                uint16_t zone_start = zones[i].first;
                uint16_t zone_end = zones[i].second;
                for (size_t d = zone_start; d < zone_end; d++){
                    zone_ranking[i] += std::abs(query[d] - means[d]);
                }
                // Normalizing
                zone_ranking[i] = zone_ranking[i] / (1.0 * (zone_end - zone_start));
            }
            auto const top_perc = static_cast<size_t>(std::ceil(zones.size() / 8));

            std::partial_sort(zones_indexes.begin(), zones_indexes.begin() + top_perc, zones_indexes.end(),
                              [&zone_ranking](size_t i1, size_t i2) {
                                  return zone_ranking[i1] > zone_ranking[i2];
                              });
            // We also prioritize them
            std::sort(zones_indexes.begin(), zones_indexes.begin() + top_perc);
            // The rest we resort to access them sequentially by zone
            std::sort(zones_indexes.begin() + top_perc, zones_indexes.end());
            size_t offset_tmp = 0;
            for (size_t i = 0; i < total_zones; i++){
                std::pair<uint16_t, uint16_t> priority_zone = zones[zones_indexes[i]];
                for (size_t d = priority_zone.first; d < priority_zone.second; d++){
                    indices_dimensions[offset_tmp] = d;
                    offset_tmp += 1;
                }
            }
        }
    }

    static void GetClustersAccessOrderIVF(const float *__restrict query, const INDEX_TYPE &data, size_t nprobe, std::vector<uint32_t> &clusters_indices) {
        std::vector<float> distances_to_centroids;
        distances_to_centroids.resize(data.num_clusters);
        for (size_t cluster_idx = 0; cluster_idx < data.num_clusters; cluster_idx++) {
            distances_to_centroids[cluster_idx] =
                    DistanceComputer<L2, F32>::Horizontal(query,
                                                          data.centroids +
                                                          cluster_idx *
                                                          data.num_dimensions,
                                                          data.num_dimensions,
                                                          nullptr);
        }
        clusters_indices.resize(data.num_clusters);
        std::iota(clusters_indices.begin(), clusters_indices.end(), 0);
        std::partial_sort(clusters_indices.begin(), clusters_indices.begin() + nprobe, clusters_indices.end(),
                          [&distances_to_centroids](size_t i1, size_t i2) {
                              return distances_to_centroids[i1] < distances_to_centroids[i2];
                          }
        );
    }

    // On the first bucket, we do a full scan (we do not prune vectors)
    template<Quantization Q=q>
    void Start(
        const QuantizedVectorType_t<Q> *__restrict query,
        const DataType_t<Q> * data,
        const size_t n_vectors,
        uint32_t k,
        const uint32_t * vector_indices,
        uint32_t * pruning_positions,
        DistanceType_t<Q> * pruning_distances,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap
    ) {
        ResetPruningDistances<Q>(n_vectors, pruning_distances);
        DistanceComputer<alpha, Q>::Vertical(
            query, data, n_vectors, n_vectors, 0, pdx_data.num_vertical_dimensions,
            pruning_distances, pruning_positions, indices_dimensions.data(), quantizer.dim_clip_value,
            nullptr);
        for (size_t horizontal_dimension = 0; horizontal_dimension < pdx_data.num_horizontal_dimensions; horizontal_dimension+=H_DIM_SIZE) {
            for (size_t vector_idx = 0; vector_idx < n_vectors; vector_idx++) {
                size_t data_pos = (pdx_data.num_vertical_dimensions * n_vectors) +
                                (horizontal_dimension * n_vectors) +
                                  (vector_idx * H_DIM_SIZE);
                pruning_distances[vector_idx] += DistanceComputer<alpha, Q>::Horizontal(
                    query + pdx_data.num_vertical_dimensions + horizontal_dimension,
                    data + data_pos,
                    H_DIM_SIZE,
                    nullptr
                );
            }
        }
        size_t max_possible_k = std::min((size_t) k - heap.size(), n_vectors);  // Note: Start() should not be called if heap.size() >= k
        std::vector<size_t> indices_sorted;
        indices_sorted.resize(n_vectors);
        std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
        std::partial_sort(indices_sorted.begin(), indices_sorted.begin() + max_possible_k, indices_sorted.end(),
            [pruning_distances](size_t i1, size_t i2) {
                return pruning_distances[i1] < pruning_distances[i2];
        });
        // insert first k results into the heap
        for (size_t idx = 0; idx < max_possible_k; ++idx) {
            auto embedding = KNNCandidate<Q>{};
            size_t index = indices_sorted[idx];
            embedding.index = vector_indices[index];
            embedding.distance = pruning_distances[index];
            heap.push(embedding);
        }
    }

    // On the first bucket, we do a full scan (we do not prune vectors)
    template<Quantization Q=q>
    void FilteredStart(
        const QuantizedVectorType_t<Q> *__restrict query,
        const DataType_t<Q> * data,
        const size_t n_vectors,
        uint32_t k,
        const uint32_t * vector_indices,
        uint32_t * pruning_positions,
        DistanceType_t<Q> * pruning_distances,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
        uint8_t * selection_vector,
        uint32_t passing_tuples
    ) {
        ResetPruningDistances<Q>(n_vectors, pruning_distances);
        n_vectors_not_pruned = 0;
        float selection_percentage = (passing_tuples / (float) n_vectors);
        InitPositionsArrayFromSelectionVector<Q>(
            n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances, selection_vector
        );
        // Always start with horizontal block, regardless of selectivity
        for (size_t horizontal_dimension = 0; horizontal_dimension < pdx_data.num_horizontal_dimensions; horizontal_dimension+=H_DIM_SIZE) {
            size_t offset_data = (pdx_data.num_vertical_dimensions * n_vectors) +
                                (horizontal_dimension * n_vectors);
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                pruning_distances[v_idx] += DistanceComputer<alpha, Q>::Horizontal(
                    query + pdx_data.num_vertical_dimensions + horizontal_dimension,
                    data + data_pos,
                    H_DIM_SIZE,
                    nullptr
                );
            }
        }
        if (selection_percentage > 0.20) { // TODO: 0.20 comes from the `selectivity_threshold`
            // It is then faster to do the full scan (thanks to SIMD)
            DistanceComputer<alpha, Q>::Vertical(
                query, data, n_vectors, n_vectors, 0, pdx_data.num_vertical_dimensions,
                pruning_distances, pruning_positions, indices_dimensions.data(), quantizer.dim_clip_value,
                nullptr);
        } else {
            // We access individual values
            DistanceComputer<alpha, Q>::VerticalPruning(
                query, data, n_vectors_not_pruned,
                n_vectors, 0, pdx_data.num_vertical_dimensions, pruning_distances,
                pruning_positions, indices_dimensions.data(), quantizer.dim_clip_value,
                nullptr);
        }
        // TODO: Everything down from here is a bottleneck when selection % is ultra low
        size_t max_possible_k = std::min((size_t) k - heap.size(), (size_t) passing_tuples);
        MaskDistancesWithSelectionVector(
            n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances, selection_vector
        );
        std::vector<size_t> indices_sorted;
        indices_sorted.resize(n_vectors);
        std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
        std::partial_sort(indices_sorted.begin(), indices_sorted.begin() + max_possible_k, indices_sorted.end(),
            [pruning_distances](size_t i1, size_t i2) {
                return pruning_distances[i1] < pruning_distances[i2];
        });
        // insert first k results into the heap
        for (size_t idx = 0; idx < max_possible_k; ++idx) {
            auto embedding = KNNCandidate<Q>{};
            size_t index = indices_sorted[idx];
            embedding.index = vector_indices[index];
            embedding.distance = pruning_distances[index];
            heap.push(embedding);
        }
    }

    // On the warmup phase, we keep scanning dimensions until the amount of not-yet pruned vectors is low
    template<Quantization Q=q, bool FILTERED=false>
    void Warmup(
        const QuantizedVectorType_t<Q> *__restrict query,
        const DataType_t<Q> *__restrict data,
        const size_t n_vectors,
        uint32_t k,
        float tuples_threshold,
        uint32_t * pruning_positions,
        DistanceType_t<Q> * pruning_distances,
        DistanceType_t<Q> &pruning_threshold,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
        uint32_t passing_tuples = 0,
        uint8_t * selection_vector = nullptr
    ) {
        current_dimension_idx = 0;
        cur_subgrouping_size_idx = 0;
        size_t tuples_needed_to_exit = std::ceil(1.0 * tuples_threshold * n_vectors);
        ResetPruningDistances<Q>(n_vectors, pruning_distances);
        uint32_t n_tuples_to_prune = 0;
        if constexpr (FILTERED) {
            float selection_percentage = (passing_tuples / (float) n_vectors);
            MaskDistancesWithSelectionVector(
                n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances, selection_vector
            );
            if (selection_percentage < 0.20) {
                // Go directly to the PRUNE phase for direct tuples access in the Horizontal block
                return;
            }
        }
        if (!is_positional_pruning) GetPruningThreshold<Q>(k, heap, pruning_threshold);
        while (
                1.0 * n_tuples_to_prune < tuples_needed_to_exit &&
                current_dimension_idx < pdx_data.num_vertical_dimensions) {
            size_t last_dimension_to_fetch = std::min(current_dimension_idx + DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
                                                      pdx_data.num_vertical_dimensions);
            if (dimension_order == SEQUENTIAL){
                    DistanceComputer<alpha, Q>::Vertical(query, data, n_vectors, n_vectors, current_dimension_idx,
                                                     last_dimension_to_fetch, pruning_distances,
                                                     pruning_positions, indices_dimensions.data(), quantizer.dim_clip_value,
                                                     nullptr);
            } else {
                DistanceComputer<alpha, Q>::VerticalReordered(query, data, n_vectors, n_vectors, current_dimension_idx,
                                                    last_dimension_to_fetch, pruning_distances,
                                                    pruning_positions, indices_dimensions.data(), quantizer.dim_clip_value,
                                                    nullptr);
            }
            current_dimension_idx = last_dimension_to_fetch;
            cur_subgrouping_size_idx += 1;
            if (is_positional_pruning) GetPruningThreshold<Q>(k, heap, pruning_threshold);
            n_tuples_to_prune = 0;
            EvaluatePruningPredicateScalar<Q>(n_tuples_to_prune, n_vectors, pruning_distances, pruning_threshold);
        }
    }

    // We scan only the not-yet pruned vectors
    template<Quantization Q=q>
    void Prune(
        const QuantizedVectorType_t<Q> *__restrict query,
        const DataType_t<Q> *__restrict data,
        const size_t n_vectors,
        uint32_t k,
        uint32_t * pruning_positions,
        DistanceType_t<Q> *pruning_distances,
        DistanceType_t<Q> &pruning_threshold,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap
    ) {
        GetPruningThreshold<Q>(k, heap, pruning_threshold);
        InitPositionsArray<Q>(n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances);
        size_t cur_n_vectors_not_pruned = 0;
        size_t current_vertical_dimension = current_dimension_idx;
        size_t current_horizontal_dimension = 0;
        while (
                pdx_data.num_horizontal_dimensions &&
                n_vectors_not_pruned &&
                        current_horizontal_dimension < pdx_data.num_horizontal_dimensions
        ) {
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            size_t offset_data = (pdx_data.num_vertical_dimensions * n_vectors) +
                                 (current_horizontal_dimension * n_vectors);
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                __builtin_prefetch(data + data_pos, 0, 3);
            }
            size_t offset_query = pdx_data.num_vertical_dimensions + current_horizontal_dimension;
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                pruning_distances[v_idx] += DistanceComputer<alpha, Q>::Horizontal(
                        query + offset_query,
                        data + data_pos,
                        H_DIM_SIZE,
                        nullptr
                );
            }
            // end of clipping
            current_horizontal_dimension += H_DIM_SIZE;
            current_dimension_idx += H_DIM_SIZE;
            if (is_positional_pruning) GetPruningThreshold<Q>(k, heap, pruning_threshold);
            assert(current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
            EvaluatePruningPredicateOnPositionsArray<Q>(cur_n_vectors_not_pruned, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances);
        }
        // GO THROUGH THE REST IN THE VERTICAL
        while (
                n_vectors_not_pruned &&
                current_vertical_dimension < pdx_data.num_vertical_dimensions
                ) {
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            size_t last_dimension_to_test_idx = std::min(current_vertical_dimension + H_DIM_SIZE,
                                                         (size_t)pdx_data.num_vertical_dimensions);
            if (dimension_order == SEQUENTIAL){
                DistanceComputer<alpha, Q>::VerticalPruning(
                    query, data, cur_n_vectors_not_pruned,
                    n_vectors, current_vertical_dimension,
                    last_dimension_to_test_idx, pruning_distances,
                    pruning_positions, indices_dimensions.data(), quantizer.dim_clip_value,
                    nullptr);
            } else {
                DistanceComputer<alpha, Q>::VerticalReorderedPruning(
                    query, data, cur_n_vectors_not_pruned,
                    n_vectors, current_vertical_dimension,
                    last_dimension_to_test_idx, pruning_distances,
                    pruning_positions, indices_dimensions.data(), quantizer.dim_clip_value,
                    nullptr);
            }
            current_dimension_idx = std::min(current_dimension_idx+H_DIM_SIZE, (size_t)pdx_data.num_dimensions);
            current_vertical_dimension = std::min((uint32_t)(current_vertical_dimension+H_DIM_SIZE), pdx_data.num_vertical_dimensions);
            assert(current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
            if (is_positional_pruning) GetPruningThreshold<Q>(k, heap, pruning_threshold);
            EvaluatePruningPredicateOnPositionsArray<Q>(cur_n_vectors_not_pruned, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances);
            if (current_dimension_idx == pdx_data.num_dimensions) break;
        }
    }

    // TODO: Manage the heap elsewhere
    template <bool IS_PRUNING=false, Quantization Q=q>
    void MergeIntoHeap(
        const uint32_t * vector_indices,
        size_t n_vectors,
        uint32_t k,
        const uint32_t * pruning_positions,
        DistanceType_t<Q> *pruning_distances,
        DistanceType_t<Q> *distances,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap
    ) {
        for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
            size_t index = position_idx;
            //DISTANCES_TYPE current_distance;
            float current_distance;
            if constexpr (IS_PRUNING){
                index = pruning_positions[position_idx];
                current_distance = pruning_distances[index];
            } else {
                current_distance = distances[index];
            }
            if (heap.size() < k || current_distance < heap.top().distance) {
                KNNCandidate<Q> embedding{};
                embedding.distance = current_distance;
                embedding.index = vector_indices[index];
                if (heap.size() >= k) {
                    heap.pop();
                }
                heap.push(embedding);
            }
        }
    }

    std::vector<KNNCandidate_t> BuildResultSet(uint32_t k){
        size_t result_set_size = std::min(best_k.size(), (size_t) k);
        std::vector<KNNCandidate_t> result;
        result.resize(result_set_size);
        // We return distances in the original domain (do we need it?)
        float inverse_scale_factor;
        if constexpr (q == U8) {
            inverse_scale_factor = 1.0f / pdx_data.scale_factor;
            inverse_scale_factor = inverse_scale_factor * inverse_scale_factor;
        }
        for (int i = result_set_size - 1; i >= 0; --i) {
            const KNNCandidate_t& embedding = best_k.top();
            if constexpr (q == U8) {
                result[i].distance = embedding.distance * inverse_scale_factor;
            } else if constexpr (q == F32){
                result[i].distance = embedding.distance;
            }
            result[i].index = embedding.index;
            best_k.pop();
        }
        return result;
    }

    void BuildResultSetCentroids(uint32_t k){
        for (int i = k - 1; i >= 0; --i) {
            const KNNCandidate<F32>& embedding = best_k_centroids.top();
            clusters_indices[i] = embedding.index;
            best_k_centroids.pop();
        }
    }

    // We store centroids using PDX in tight blocks of 64
    void GetClustersAccessOrderIVFPDX(const float *__restrict query) {
        best_k_centroids = std::priority_queue<KNNCandidate<F32>, std::vector<KNNCandidate<F32>>, VectorComparator<F32>>{};
        clusters_indices.resize(pdx_data.num_clusters);
        std::iota(clusters_indices.begin(), clusters_indices.end(), 0);
        float * tmp_centroids_pdx = pdx_data.centroids_pdx;
        uint32_t * tmp_cluster_indices = clusters_indices.data();
        size_t SKIPPING_SIZE = PDX_VECTOR_SIZE * pdx_data.num_dimensions;
        size_t remainder_block_size = pdx_data.num_clusters % PDX_VECTOR_SIZE;
        size_t full_blocks = std::floor(1.0 * pdx_data.num_clusters / PDX_VECTOR_SIZE);
        for (size_t centroid_idx = 0; centroid_idx < full_blocks; ++centroid_idx) {
            // TODO: Use another distances array for the centroids so I can use ResetDistancesVectorized() instead of memset
            memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
            DistanceComputer<L2, F32>::VerticalBlock(query, tmp_centroids_pdx, 0, pdx_data.num_dimensions, distances, nullptr);
            MergeIntoHeap<false, F32>(tmp_cluster_indices, PDX_VECTOR_SIZE, ivf_nprobe, pruning_positions, pruning_distances, distances, best_k_centroids);
            tmp_cluster_indices += PDX_VECTOR_SIZE;
            tmp_centroids_pdx += SKIPPING_SIZE;
        }
        if (remainder_block_size){
            memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
            DistanceComputer<L2, F32>::Vertical(query, tmp_centroids_pdx, remainder_block_size, remainder_block_size, 0, pdx_data.num_dimensions, distances, nullptr, nullptr, nullptr, nullptr);
            MergeIntoHeap<false, F32>(tmp_cluster_indices, remainder_block_size, ivf_nprobe, pruning_positions, pruning_distances, distances, best_k_centroids);
        }
        for (size_t i = 0; i < ivf_nprobe; ++i){
            const KNNCandidate<F32> & c = best_k_centroids.top();
            clusters_indices[ivf_nprobe - i - 1] = c.index; // I need to inverse the allocation
            best_k_centroids.pop();
        }
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
    }

    // We store centroids using PDX in tight blocks of 64
    // TODO: Always assumes multiple of 64
    void GetL0ClustersAccessOrderPDX(const float *__restrict query) {
        best_k_centroids = std::priority_queue<KNNCandidate<F32>, std::vector<KNNCandidate<F32>>, VectorComparator<F32>> {};
        clusters_indices_l0.resize(pdx_data.num_clusters_l0);
        std::iota(clusters_indices_l0.begin(), clusters_indices_l0.end(), 0);
        float * tmp_centroids_pdx = pdx_data.centroids_pdx;
        uint32_t * tmp_cluster_indices = clusters_indices_l0.data();
        size_t SKIPPING_SIZE = PDX_VECTOR_SIZE * pdx_data.num_dimensions;
        size_t full_blocks = std::floor(1.0 * pdx_data.num_clusters_l0 / PDX_VECTOR_SIZE);
        for (size_t centroid_idx = 0; centroid_idx < full_blocks; ++centroid_idx) {
            memset((void*) centroids_distances, 0, PDX_VECTOR_SIZE * sizeof(float));
            DistanceComputer<alpha, F32>::VerticalBlock(query, tmp_centroids_pdx, 0, pdx_data.num_dimensions, centroids_distances, nullptr);
            tmp_cluster_indices += PDX_VECTOR_SIZE;
            tmp_centroids_pdx += SKIPPING_SIZE;
        }
        std::vector<size_t> indices_sorted;
        indices_sorted.resize(pdx_data.num_clusters_l0);
        std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
        std::partial_sort(indices_sorted.begin(), indices_sorted.begin() + 64, indices_sorted.end(),
                          [](size_t i1, size_t i2) {
                              return centroids_distances[i1] < centroids_distances[i2];
                          });
        // Sort the distance of the first N centroids to determine access order
        for (size_t idx = 0; idx < pdx_data.num_clusters_l0; ++idx) {
            clusters_indices_l0[idx] = indices_sorted[idx];
        }
    }

    void GetL1ClustersAccessOrderPDX(
        const float *__restrict query,
        size_t n_buckets,
        bool safe_to_prune_space = true
    ){
        size_t clusters_to_visit = pdx_data.num_clusters_l0;
        if ((n_buckets < pdx_data.num_clusters / 2) && safe_to_prune_space) {
            // We prune half of the super-clusters only if the user wants to
            // visit less than half of the available clusters
            clusters_to_visit = pdx_data.num_clusters_l0 / 2;
        }
        current_dimension_idx = 0;
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_cluster = clusters_indices_l0[cluster_idx];
            Cluster<F32>& cluster = pdx_data.clusters_l0[current_cluster];
            if (best_k_centroids.size() < n_buckets) {
                // The heap may not be filled with the first super-cluster
                // if the number of clusters probed is high or if the
                // number of vectors per super-cluster is small.
                Start<F32>(
                    query, cluster.data, cluster.num_embeddings, n_buckets, cluster.indices,
                    pruning_positions_l0, pruning_distances_l0, best_k_centroids
                );
                continue;
            }
            Warmup<F32>(query, cluster.data, cluster.num_embeddings, n_buckets, selectivity_threshold, pruning_positions_l0, pruning_distances_l0, pruning_threshold_l0, best_k_centroids);
            Prune<F32>(query, cluster.data, cluster.num_embeddings, n_buckets, pruning_positions_l0, pruning_distances_l0, pruning_threshold_l0, best_k_centroids);
            if (n_vectors_not_pruned){
                MergeIntoHeap<true, F32>(cluster.indices, n_vectors_not_pruned, n_buckets, pruning_positions_l0, pruning_distances_l0, nullptr, best_k_centroids);
            }
        }
        // Rare case in which half of the space is not enough to fill the best_k_centroids heap
        if (best_k_centroids.size() < n_buckets) {
            // From 32 to 64 (in the default case)
            for (size_t cluster_idx = clusters_to_visit; cluster_idx < pdx_data.num_clusters_l0; ++cluster_idx) {
                current_cluster = clusters_indices_l0[cluster_idx];
                Cluster<F32>& cluster = pdx_data.clusters_l0[current_cluster];
                Start<F32>(
                    query, cluster.data, cluster.num_embeddings, n_buckets, cluster.indices,
                    pruning_positions_l0, pruning_distances_l0, best_k_centroids
                );
                if (best_k_centroids.size() == n_buckets) {
                    break;
                }
            }
        }
        BuildResultSetCentroids(n_buckets);
    }

    void GetClustersAccessOrderRandom() {
        std::iota(clusters_indices.begin(), clusters_indices.end(), 0);
    }

public:
    /******************************************************************
     * Search methods
     ******************************************************************/
    std::vector<KNNCandidate_t> Search(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        alignas(64) float query[pdx_data.num_dimensions];
        if (!pdx_data.is_normalized) {
            pruner.PreprocessQuery(raw_query, query);
        } else {
            alignas(64) float normalized_query[pdx_data.num_dimensions];
            quantizer.NormalizeQuery(raw_query, normalized_query);
            pruner.PreprocessQuery(normalized_query, query);
        }
        GetDimensionsAccessOrder(query, pdx_data.means);
        size_t clusters_to_visit = (ivf_nprobe == 0 || ivf_nprobe > pdx_data.num_clusters)
                    ? pdx_data.num_clusters
                    : ivf_nprobe;
        if constexpr (std::is_same_v<Index, IndexPDXIVF2<q>>) {
            // Multilevel access
            GetL0ClustersAccessOrderPDX(query);
            GetL1ClustersAccessOrderPDX(query, clusters_to_visit);
        } else {
            if (pdx_data.is_ivf) {
                // TODO: Incorporate this to U8 PDX (no IVF2)
                // GetClustersAccessOrderIVFPDX(query);
                GetClustersAccessOrderIVF(query, pdx_data, clusters_to_visit, clusters_indices);
            } else {
                // If there is no index, we just access the clusters in order
                GetClustersAccessOrderRandom();
            }
        }
        // PDXearch core
        current_dimension_idx = 0;
        QUANTIZED_VECTOR_TYPE *prepared_query;
        if constexpr (q == U8) {
            quantizer.PrepareQuery(query, pdx_data.for_base, pdx_data.scale_factor);
            prepared_query = quantizer.quantized_query;
        } else {
            prepared_query = query;
        }
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_cluster = clusters_indices[cluster_idx];
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            if (best_k.size() < k) {
                // We cannot prune until we fill the heap
                Start(
                    prepared_query, cluster.data, cluster.num_embeddings, k, cluster.indices,
                    pruning_positions, pruning_distances, best_k
                );
                continue;
            }
            Warmup(prepared_query, cluster.data, cluster.num_embeddings, k, selectivity_threshold, pruning_positions, pruning_distances, pruning_threshold, best_k);
            Prune(prepared_query, cluster.data, cluster.num_embeddings, k, pruning_positions, pruning_distances, pruning_threshold, best_k);
            if (n_vectors_not_pruned){
                MergeIntoHeap<true>(cluster.indices, n_vectors_not_pruned, k, pruning_positions, pruning_distances, nullptr, best_k);
            }
        }
        std::vector<KNNCandidate_t> result = BuildResultSet(k);
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return result;
    }

    std::vector<KNNCandidate_t> FilteredSearch(float *__restrict raw_query, uint32_t k, const PredicateEvaluator& predicate_evaluator) {
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        alignas(64) float query[pdx_data.num_dimensions];
        if (!pdx_data.is_normalized) {
            pruner.PreprocessQuery(raw_query, query);
        } else {
            alignas(64) float normalized_query[pdx_data.num_dimensions];
            quantizer.NormalizeQuery(raw_query, normalized_query);
            pruner.PreprocessQuery(normalized_query, query);
        }
        GetDimensionsAccessOrder(query, pdx_data.means);
        size_t clusters_to_visit = (ivf_nprobe == 0 || ivf_nprobe > pdx_data.num_clusters)
                    ? pdx_data.num_clusters
                    : ivf_nprobe;
        if constexpr (std::is_same_v<Index, IndexPDXIVF2<q>>) {
            // Multilevel access
            GetL0ClustersAccessOrderPDX(query);
            GetL1ClustersAccessOrderPDX(query, clusters_to_visit, false);
        } else {
            if (pdx_data.is_ivf) {
                // GetClustersAccessOrderIVFPDX(query);
                GetClustersAccessOrderIVF(query, pdx_data, clusters_to_visit, clusters_indices);
            } else {
                // If there is no index, we just access the clusters in order
                GetClustersAccessOrderRandom();
            }
        }
        // PDXearch core
        current_dimension_idx = 0;
        QUANTIZED_VECTOR_TYPE *prepared_query;
        if constexpr (q == U8) {
            quantizer.PrepareQuery(query, pdx_data.for_base, pdx_data.scale_factor);
            prepared_query = quantizer.quantized_query;
        } else {
            prepared_query = query;
        }
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_cluster = clusters_indices[cluster_idx];
            auto [selection_vector, passing_tuples] = predicate_evaluator.GetSelectionVector(current_cluster, cluster_offsets[current_cluster]);
            if (passing_tuples == 0) {
                continue;
            }
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            if (best_k.size() < k) {
                // We cannot prune until we fill the heap
                FilteredStart(
                    prepared_query, cluster.data, cluster.num_embeddings, k, cluster.indices,
                    pruning_positions, pruning_distances, best_k, selection_vector, passing_tuples
                );
                continue;
            }
            Warmup<q, true>(prepared_query, cluster.data, cluster.num_embeddings, k, selectivity_threshold, pruning_positions, pruning_distances, pruning_threshold, best_k, passing_tuples, selection_vector);
            Prune(prepared_query, cluster.data, cluster.num_embeddings, k, pruning_positions, pruning_distances, pruning_threshold, best_k);
            if (n_vectors_not_pruned){
                MergeIntoHeap<true>(cluster.indices, n_vectors_not_pruned, k, pruning_positions, pruning_distances, nullptr, best_k);
            }
        }
        std::vector<KNNCandidate_t> result = BuildResultSet(k);
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return result;
    }


    template<Quantization Q=q, std::enable_if_t<Q==U8, int> = 0>
    std::vector<KNNCandidate_t> LinearScan(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        std::vector<float> dummy_for_bases(4096, 0);
        std::vector<float> dummy_scale_factors(4096, 1);
        alignas(64) float query[pdx_data.num_dimensions];
        if (!pdx_data.is_normalized) {
        } else {
            quantizer.NormalizeQuery(raw_query, query);
        }
        quantizer.PrepareQuery(query, dummy_for_bases.data(), dummy_scale_factors);
        best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        size_t clusters_to_visit = pdx_data.num_clusters;
        GetClustersAccessOrderRandom();
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_cluster = clusters_indices[cluster_idx];
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            if (cluster.num_embeddings == PDX_VECTOR_SIZE){
                ResetDistancesVectorized<Q>(distances);
                DistanceComputer<alpha, Q>::VerticalBlock(query, cluster.data, 0, pdx_data.num_dimensions, distances, nullptr);
                MergeIntoHeap<false>(cluster.indices, PDX_VECTOR_SIZE, k, nullptr, nullptr, distances, best_k);
            } else if (cluster.num_embeddings < PDX_VECTOR_SIZE) {
                ResetDistancesVectorized<Q>(distances);
                DistanceComputer<alpha, Q>::Vertical(query, cluster.data, cluster.num_embeddings, cluster.num_embeddings,  0, pdx_data.num_dimensions, distances,
                                            nullptr, nullptr, nullptr, nullptr);
                MergeIntoHeap<false>(cluster.indices, cluster.num_embeddings, k, nullptr, nullptr, distances, best_k);
            }
        }
        std::vector<KNNCandidate_t> result = BuildResultSet(k);
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return result;
    }

    // Full Linear Scans that do not prune vectors
    template<Quantization Q=q, std::enable_if_t<Q==F32, int> = 0>
    std::vector<KNNCandidate_t> LinearScan(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        float query[pdx_data.num_dimensions];
        pruner.PreprocessQuery(raw_query, query);
        best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        size_t clusters_to_visit = pdx_data.num_clusters;
        GetClustersAccessOrderRandom();
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_cluster = clusters_indices[cluster_idx];
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            if (cluster.num_embeddings == PDX_VECTOR_SIZE){
                ResetDistancesVectorized<Q>(distances);
                DistanceComputer<alpha, Q>::VerticalBlock(query, cluster.data, 0, pdx_data.num_dimensions, distances, nullptr);
                MergeIntoHeap<false>(cluster.indices, PDX_VECTOR_SIZE, k, nullptr, nullptr, distances,  best_k);
            } else if (cluster.num_embeddings < PDX_VECTOR_SIZE) {
                ResetDistancesVectorized<Q>(distances);
                DistanceComputer<alpha, Q>::Vertical(query, cluster.data, cluster.num_embeddings, cluster.num_embeddings, 0, pdx_data.num_dimensions, distances, nullptr, nullptr, nullptr, nullptr);
                MergeIntoHeap<false>(cluster.indices, cluster.num_embeddings, k, nullptr, nullptr, distances, best_k);
            }
        }
        std::vector<KNNCandidate_t> result = BuildResultSet(k);
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return result;
    }

};

} // namespace PDX

#endif //PDX_PDXEARCH_HPP

#ifndef EMBEDDINGSEARCH_PDXEARCH_U6x8_HPP
#define EMBEDDINGSEARCH_PDXEARCH_U6x8_HPP

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif

#include <queue>
#include <cassert>
#include <algorithm>
#include <array>
#include "pdx/index_base/pdx_ivf.hpp"
#include "pdx/bitpacker/unpacker_neon.h"
#include "vector_searcher.hpp"
#include "utils/tictoc.hpp"

namespace PDX {

enum PDXearchDimensionsOrder {
    SEQUENTIAL,
    DISTANCE_TO_MEANS,
    DECREASING,
    DISTANCE_TO_MEANS_IMPROVED,
    DECREASING_IMPROVED,
    DIMENSION_ZONES
};

enum DistanceFunction {
    L2,
    IP,
    L1
};

/******************************************************************
 * PDXearch
 * Implements our algorithm for vertical pruning
 * TODO: Move search methods that do not prune dimensions to another class
 * TODO: Centralize PDX distance kernels to another class
 * TODO: Probably having the distance metric as a template parameter was not the smartest idea
 ******************************************************************/
template<DistanceFunction ALPHA=L2>
class PDXearchU6x8: public VectorSearcher {
public:
    PDXearchU6x8(IndexPDXIVFFlatU6x8 &data_index, size_t ivf_nprobe, int position_prune_distance,
             PDXearchDimensionsOrder dimension_order) :
            pdx_data(data_index),
            ivf_nprobe(ivf_nprobe),
            is_positional_pruning(position_prune_distance),
            dimension_order(dimension_order){
        indices_dimensions.resize(pdx_data.num_dimensions);
        vectorgroups_indices.resize(pdx_data.num_vectorgroups);
        // TODO: We should not need this
        for (size_t i = 0; i < pdx_data.num_vectorgroups; ++i){
            total_embeddings += pdx_data.vectorgroups[i].num_embeddings;
        }
    }

    IndexPDXIVFFlatU6x8 &pdx_data;
    uint32_t current_dimension_idx {0};


    void SetNProbe(size_t nprobe){
        ivf_nprobe = nprobe;
    }

    void SetExponent(int exponent){
        lep_exponent = exponent;
    }

protected:
    float selectivity_threshold = 0.80;
    size_t ivf_nprobe = 0;
    int is_positional_pruning = false;
    size_t current_vectorgroup = 0;
    int lep_exponent = 0;

    PDXearchDimensionsOrder dimension_order = SEQUENTIAL;
    // Evaluating the pruning threshold is so fast that we can allow smaller fetching sizes
    // to avoid more data access. Super useful in architectures with low bandwidth at L3/DRAM like Intel SPR
    static constexpr uint32_t DIMENSIONS_FETCHING_SIZES[24] = {
            4, 8, 8, 12, 16, 16, 32, 32, 32, 32,
            64, 64, 64, 64, 128, 128, 128, 128, 256,
            256, 512, 1024, 2048, 4096
    };

    size_t cur_subgrouping_size_idx {0};
    size_t total_embeddings {0};

    std::vector<uint32_t> indices_dimensions;
    std::vector<uint32_t> vectorgroups_indices;

    size_t n_vectors_not_pruned = 0;
    size_t clipped = 0;

    uint32_t pruning_threshold = std::numeric_limits<uint32_t>::max();

    static constexpr uint16_t PDX_VECTOR_SIZE = 64;
    alignas(64) inline static uint32_t distances[PDX_VECTOR_SIZE]; // Used in full scans (no pruning)

    alignas(64) inline static uint32_t pruning_distances[10240]; // TODO: Use dynamic arrays. Buckets with more than 10k vectors (rare) overflow
    alignas(64) inline static uint32_t pruning_positions[10240];
    alignas(64) inline static uint32_t pruning_distances_tmp[10240];
    alignas(64) inline static bool dim_clip[4096];
    alignas(64) inline static int32_t dim_clip_value[4096];

    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k_centroids;

    virtual void PreprocessQuery(float * raw_query, float *query){};

    inline void ResetDistancesScalar(size_t n_vectors){
        memset((void*) distances, 0, n_vectors * sizeof(uint32_t));
    }

    inline virtual void ResetPruningDistances(size_t n_vectors){
        memset((void*) pruning_distances, 0, n_vectors * sizeof(uint32_t));
    }

    virtual void ResetDistancesVectorized(){
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(uint32_t));
    }

    // The pruning threshold by default is the top of the heap
    virtual void GetPruningThreshold(uint32_t k, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap){
        pruning_threshold = heap.size() == k ? heap.top().distance : std::numeric_limits<uint32_t>::max();
    };

    virtual void EvaluatePruningPredicateScalar(uint32_t &n_pruned, size_t n_vectors) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
        }
    };

    virtual void EvaluatePruningPredicateOnPositionsArray(size_t n_vectors){
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
            n_vectors_not_pruned += pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
        }
    };

    inline virtual void InitPositionsArray(size_t n_vectors){
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    };

    void GetDimensionsAccessOrder(const float *__restrict query, const float *__restrict means) {
        std::iota(indices_dimensions.begin(), indices_dimensions.end(), 0);
        if (dimension_order == DISTANCE_TO_MEANS) {
            std::sort(indices_dimensions.begin(), indices_dimensions.end(),
                      [&query, &means](size_t i1, size_t i2) {
                          return std::abs(query[i1] - means[i1]) > std::abs(query[i2] - means[i2]);
                      });
        } else if (dimension_order == DECREASING) {
            std::sort(indices_dimensions.begin(), indices_dimensions.end(),
                      [&query](size_t i1, size_t i2) {
                          return query[i1] > query[i2];
                      });
        } else if (dimension_order == DISTANCE_TO_MEANS_IMPROVED) { // Improves Cache performance
            auto const top_perc = static_cast<size_t>(std::floor(indices_dimensions.size() / 4));
            std::partial_sort(indices_dimensions.begin(), indices_dimensions.begin() + top_perc, indices_dimensions.end(),
                              [&query, &means](size_t i1, size_t i2) {
                                  return std::abs(query[i1] - means[i1]) > std::abs(query[i2] - means[i2]);
                              });
            // By taking the top 25% dimensions and sorting them ascendingly by index
            std::sort(indices_dimensions.begin(), indices_dimensions.begin() + top_perc);
            // Then sorting the rest of the dimensions ascendingly by index
            std::sort(indices_dimensions.begin() + top_perc, indices_dimensions.end());
        } else if (dimension_order == DECREASING_IMPROVED){
            auto const top_perc = static_cast<size_t>(std::floor(indices_dimensions.size() / 4));
            std::partial_sort(indices_dimensions.begin(), indices_dimensions.begin() + top_perc, indices_dimensions.end(),
                              [&query](size_t i1, size_t i2) {
                                  return query[i1] > query[i2];
                              });
            // By taking the top 25% dimensions and sorting them ascendingly by index
            std::sort(indices_dimensions.begin(), indices_dimensions.begin() + top_perc);
            // Then sorting the rest of the dimensions ascendingly by index
            std::sort(indices_dimensions.begin() + top_perc, indices_dimensions.end());
        } else if (dimension_order == DIMENSION_ZONES){
            uint16_t dimensions = pdx_data.num_dimensions;
            size_t estimated_embeddings_per_vg = total_embeddings / pdx_data.num_vectorgroups;
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

    /******************************************************************
     * Distance computers. We have FOUR versions of distance calculations:
     * CalculateVerticalDistancesScalar: Using non-tight loops of any length
     * CalculateVerticalDistancesForPruning: Using non-tight loops of any length accumulating distances for pruning
     * CalculateVerticalDistancesVectorized: Using tight loops of 64 vectors
     * CalculateVerticalDistancesOnPositionsArray: Using non-tight loops on the array of not-yet pruned vectors
     ******************************************************************/
    template<bool USE_DIMENSIONS_REORDER, bool SKIP_PRUNED, DistanceFunction L_ALPHA=ALPHA>
    void CalculateVerticalDistancesScalar(
            const uint8_t *__restrict query,
            const uint8_t *__restrict data,
            size_t n_vectors,
            size_t total_vectors,
            size_t start_dimension,
            size_t end_dimension,
            uint32_t * distances_p
    ){
#ifdef false
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            size_t offset_to_dimension_start = true_dimension_idx * n_vectors;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                if constexpr (L_ALPHA == L2){
                    int to_multiply = query[true_dimension_idx] - data[offset_to_dimension_start + vector_idx];
                    distances[vector_idx] += to_multiply * to_multiply;
                }
            }
        }
// SIMD Dot Product (G4): 2.81 (uint8) vs 7.86 (float32) !!! 2.8x faster; not ideal but still nice
// Perhaps now that we have less data we can afford to blockify vectors and prune
#elif defined(__ARM_NEON) && true
        // Compute L2
        // TODO: Handle tail in dimension length, for now im not going to worry on that
        // as all the datasets are divisible by 4
        // Todo: template this 4 parameter
        // TODO: Get this distance functions out of here to another folder / file structure

        // TODO: BIT UNPACKING should happen here

        /////////////////

        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            // TODO: Readd dimension reordering efficiently
            //            if constexpr (USE_DIMENSIONS_REORDER){
            //                true_dimension_idx = indices_dimensions[dimension_idx];
            //            }
            uint8x8_t vals = vld1_u8(&query[dimension_idx]);
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
            // TODO: RE ADD
//            if constexpr (!SKIP_PRUNED){
//                uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
//                uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
//                for (; i <= n_vectors - 4; i+=4) {
//                    // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
//                    uint32x4_t res = vld1q_u32(&distances_p[i]);
//                    uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 4]);
//                    uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
//                    vst1q_u32(&distances_p[i], vdotq_u32(res, diff_u8, diff_u8));
//                }
//            }
            // n_vectors % 4 (rest)
//#ifdef BENCHMARK_TIME
//            end_to_end_clock.Tic();
//#endif
            uint8x8_t idx = {0, 1, 2, 3, 0, 1, 2, 3};
            uint8x8_t vec1_u8 = vtbl1_u8(vals, idx);


            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED){
                    vector_idx = pruning_positions[vector_idx];
                }
                /*
                uint32x2_t res = vdup_n_s32(0);
                // Not needed
                //result = vld1_lane_s32(&distances_p[vector_idx], result, 0);
                uint8x8_t vec2_u8 = vld1_u8(&data[offset_to_dimension_start + (vector_idx * 4)]);
                uint8x8_t diff_u8 = vabd_u8(vec1_u8, vec2_u8);
                res = vdot_u32(res, diff_u8, diff_u8);
                distances_p[vector_idx] += vget_lane_u32(res, 0);
                */

                // I am sure I will have 4 dims
                // L2
                int to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (to_multiply_a * to_multiply_a) +
                        (to_multiply_b * to_multiply_b) +
                        (to_multiply_c * to_multiply_c) +
                        (to_multiply_d * to_multiply_d);

                // IP
//                int to_multiply_a = query[dimension_idx] * data[offset_to_dimension_start + (vector_idx * 4)];
//                int to_multiply_b = query[dimension_idx + 1] * data[offset_to_dimension_start + (vector_idx * 4) + 1];
//                int to_multiply_c = query[dimension_idx + 2] * data[offset_to_dimension_start + (vector_idx * 4) + 2];
//                int to_multiply_d = query[dimension_idx + 3] * data[offset_to_dimension_start + (vector_idx * 4) + 3];
//                distances_p[vector_idx] += to_multiply_a +
//                                           to_multiply_b +
//                                           to_multiply_c +
//                                           to_multiply_d;

            }
//#ifdef BENCHMARK_TIME
//            end_to_end_clock.Toc();
//#endif
        }
        size_t group = start_dimension;
        size_t loop_c = 0;
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; ++dim_idx) {
            // TODO: Do something more clean
            if (loop_c == 4) {
                group += 4;
                loop_c = 0;
            }
            // Todo: detect clip up, but this is super rare
            if (dim_clip_value[dim_idx] < 0) {
                for (size_t j=0; j < n_vectors; ++j) {
                    size_t vector_idx = j;
                    size_t offset_to_dimension_start = group * total_vectors;
                    if constexpr (SKIP_PRUNED){
                        vector_idx = pruning_positions[vector_idx];
                    }
                    // L2
                    // TODO: Altho this clipping fixing is meaningless to the total runtime
                    // We can pushup this term to the end of the calculation, and only add it on vectors
                    // when we are merging to the heap.
                    // If I remove it completely, and not push it up before adding it to the queue,
                    // it will only work if the same query dimension always clips in all the buckets
                    // otherwise I could get some errors
                    // ACTUALLY THIS ALWAYS WORK BECAUSE dim_clip_value[dim_idx] IS ALWAYS NEGATIVE
                    // IF THE CLIPPING IS POSITIVE, I COULD BE PRUNING VECTORS INCORRECTLY
                    // (in other context, here it works because it is an instant correction)
                    // In fact TODO: Check that dim_clip_value[dim_idx] is always negative
                     distances_p[vector_idx] -= 2 * data[offset_to_dimension_start + (vector_idx * 4) + (dim_idx % 4)] * dim_clip_value[dim_idx];
                     distances_p[vector_idx] += dim_clip_value[dim_idx] * dim_clip_value[dim_idx];
                    clipped += 1;
                    // IP
                    // distances_p[vector_idx] += data[offset_to_dimension_start + (vector_idx * 4) + (dim_idx % 4)] * dim_clip_value[dim_idx];
                }
            }
            loop_c += 1;
        }
#endif
    }

    template<DistanceFunction L_ALPHA=ALPHA>
    void CalculateVerticalDistancesVectorized(
        const uint8_t *__restrict query, const uint8_t *__restrict data, size_t start_dimension, size_t end_dimension, uint32_t * distances_p){
#if false
//Auto-vectorized M1: 3.24 (uint8) vs 4.13 (float32)
//Auto-vectorized G4: 4.18 (uint8) vs 7.81 (float32)
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            size_t dimension_idx = dim_idx;
            if constexpr (USE_DIMENSIONS_REORDER) {
                dimension_idx = indices_dimensions[dim_idx];
            }
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
                if constexpr (L_ALPHA == L2) {
                    int to_multiply = query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
                    distances[vector_idx] += to_multiply * to_multiply;
                }
            }
        }
// SIMD Dot Product (M1): XXXX (uint8) vs 4.17 (float32)
// SIMD Dot Product (G4): 1.94 (uint8) vs 8.10 (float32) !!! 4.17x faster; perfect scaling!
#elif defined(__ARM_NEON) && true
        uint32x4_t res[16];
        // Load initial values
        for (size_t i = 0; i < 16; ++i) {
            res[i] = vdupq_n_u32(0);
        }
        // Compute L2
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            // TODO: Readd dimension reordering efficiently
            //            if constexpr (USE_DIMENSIONS_REORDER){
            //                true_dimension_idx = indices_dimensions[dimension_idx];
            //            }
            uint8x8_t vals = vld1_u8(&query[dimension_idx]);
            uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
            uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (int i = 0; i < 16; ++i) { // total: 64 vectors * 4 dimensions each (at 1 byte per value = 2048-bits)
                // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
                uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 16]);
                uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
                // TODO: RE ADD
                //res[i] = vdotq_u32(res[i], diff_u8, diff_u8);
            }
        }
        // Store results back
        for (int i = 0; i < 16; ++i) {
            vst1q_u32(&distances_p[i * 4], res[i]);
        }
#endif
    }

    // On the first bucket, we do a full scan (we do not prune vectors)
    template <DistanceFunction L_ALPHA=ALPHA>
    void Start(const uint8_t *__restrict query, const uint8_t * data, const size_t n_vectors, uint32_t k, const uint32_t * vector_indices) {
        ResetPruningDistances(n_vectors);
        CalculateVerticalDistancesScalar<false, false, L_ALPHA>(query, data, n_vectors, n_vectors, 0, pdx_data.num_dimensions, pruning_distances);
        size_t max_possible_k = std::min((size_t) k, n_vectors);
        std::vector<size_t> indices_sorted;
        indices_sorted.resize(n_vectors);
        std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
        std::partial_sort(indices_sorted.begin(), indices_sorted.begin() + max_possible_k, indices_sorted.end(),
                          [](size_t i1, size_t i2) {
                              return pruning_distances[i1] < pruning_distances[i2];
                          });
        // insert first k results into the heap
        for (size_t idx = 0; idx < max_possible_k; ++idx) {
            auto embedding = KNNCandidate{};
            size_t index = indices_sorted[idx];
            embedding.index = vector_indices[index];
            embedding.distance = pruning_distances[index];
            best_k.push(embedding);
        }
    }

    // On the warmup phase, we keep scanning dimensions until the amount of not-yet pruned vectors is low
    template <DistanceFunction L_ALPHA=ALPHA>
    void Warmup(const uint8_t *__restrict query, const uint8_t * data, const size_t n_vectors, uint32_t k, float tuples_threshold, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap) {
        current_dimension_idx = 0;
        cur_subgrouping_size_idx = 0;
        size_t tuples_needed_to_exit = std::ceil(1.0 * tuples_threshold * n_vectors);
        ResetPruningDistances(n_vectors);
        uint32_t n_tuples_to_prune = 0;
        if (!is_positional_pruning) GetPruningThreshold(k, heap);

        while (
                1.0 * n_tuples_to_prune < tuples_needed_to_exit &&
                current_dimension_idx < pdx_data.num_dimensions) {
            size_t last_dimension_to_fetch = std::min(current_dimension_idx + DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
                                                      pdx_data.num_dimensions);
            if (dimension_order == SEQUENTIAL){
                CalculateVerticalDistancesScalar<false, false, L_ALPHA>(query, data, n_vectors, n_vectors, current_dimension_idx,
                                                                     last_dimension_to_fetch, pruning_distances);
            } else {
                CalculateVerticalDistancesScalar<true, false, L_ALPHA>(query, data, n_vectors, n_vectors, current_dimension_idx,
                                                                    last_dimension_to_fetch, pruning_distances);
            }
            current_dimension_idx = last_dimension_to_fetch;
            cur_subgrouping_size_idx += 1;
            if (is_positional_pruning) GetPruningThreshold(k, heap);
            n_tuples_to_prune = 0;
            EvaluatePruningPredicateScalar(n_tuples_to_prune, n_vectors);
        }
    }

    // We scan only the not-yet pruned vectors
    template <DistanceFunction L_ALPHA=ALPHA>
    void Prune(const uint8_t *__restrict query, const uint8_t *__restrict data, const size_t n_vectors, uint32_t k, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap) {
        GetPruningThreshold(k, heap);
        InitPositionsArray(n_vectors);
        size_t cur_n_vectors_not_pruned = 0;
        while ( // We try to prune until the end
                n_vectors_not_pruned
                ) {
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            size_t last_dimension_to_test_idx = std::min(current_dimension_idx + DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
                                                         pdx_data.num_dimensions);
            if (dimension_order == SEQUENTIAL){
                CalculateVerticalDistancesScalar<false, true, L_ALPHA>(query, data, cur_n_vectors_not_pruned,
                                                                           n_vectors, current_dimension_idx,
                                                                           last_dimension_to_test_idx, pruning_distances);
            } else {
                CalculateVerticalDistancesScalar<true, true, L_ALPHA>(query, data, cur_n_vectors_not_pruned,
                                                                          n_vectors, current_dimension_idx,
                                                                          last_dimension_to_test_idx, pruning_distances);
            }

            current_dimension_idx = last_dimension_to_test_idx;
            if (is_positional_pruning) GetPruningThreshold(k, heap);
            EvaluatePruningPredicateOnPositionsArray(cur_n_vectors_not_pruned);
            if (current_dimension_idx == pdx_data.num_dimensions) break;
        }
    }

    template <bool IS_PRUNING=false>
    void MergeIntoHeap(const uint32_t * vector_indices, size_t n_vectors, uint32_t k, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap) {
        for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
            size_t index = position_idx;
            uint32_t current_distance;
            if constexpr (IS_PRUNING){
                index = pruning_positions[position_idx];
                current_distance = pruning_distances[index];
            } else {
                current_distance = distances[index];
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

    std::vector<KNNCandidate> BuildResultSet(uint32_t k){
        std::vector<KNNCandidate> result;
        result.resize(k);
        for (int i = k - 1; i >= 0; --i) {
            const KNNCandidate& embedding = best_k.top();
            result[i].distance = embedding.distance;
            result[i].index = embedding.index;
            best_k.pop();
        }
        return result;
    }

    void GetVectorgroupsAccessOrderRandom() {
        std::iota(vectorgroups_indices.begin(), vectorgroups_indices.end(), 0);
    }

    void Decompress(uint8_t * data, uint8_t * out, size_t length){
//        uint8_t * out = new uint8_t[(n_vectors * total_dims) + 64];
        uint8_t * out_ptr = out;
        uint8_t * data_ptr = data;
        // TODO: Handle remain properly, for now I will let it overflow slightly at the end by aligning n_vectors to 32
        for (uint32_t pos = 0; pos < IndexPDXIVFFlatU6x8::AlignValue<uint32_t, 1024>(length); pos+=1024){
            //Unpacker::unpackblock6_notoptimal(&data, &out_ptr);
            size_t data_pos = (size_t)(pos * 0.75); // To skip correctly
            Unpacker::unpack_6bw_8ow_128crw_8uf(data_ptr + data_pos, out_ptr+pos);
        }
    }

    void NormalizeQuery(const float * src, float * dst) {
        float sum = 0.0f;
        for (size_t i = 0; i < pdx_data.num_dimensions; ++i) {
            sum += src[i] * src[i];
        }
        float norm = std::sqrt(sum);
        for (size_t i = 0; i < pdx_data.num_dimensions; ++i) {
            dst[i] = src[i] / norm;
        }
    }

    void ScaleQuery(const float * src, int32_t * dst) {
        for (size_t i = 0; i < pdx_data.num_dimensions; ++i) {
            // Fast round: + 12582912.0 - 12582912.0;
            dst[i] = static_cast<int>(std::round(src[i] * lep_exponent * 0.25));
            //dst[i] = (src[i] * 316.227766017f); // gist 10^2.5
            // dst[i] = (src[i] * 1258.92541179f); // arXiv to slighlly increase recall 10^3.1
        }
        //std::cout << +dst[0] << "\n";
    }

    // TODO: There are several ways to optimize this
    void PrepareQuery(const int32_t * scaled_query, uint8_t *query, const int32_t *for_bases){
        for (size_t i = 0; i < pdx_data.num_dimensions; i += 16) {
            // Load 8 int32 values in two NEON registers
            int32x4_t sub_a = vld1q_s32(scaled_query + i);
            int32x4_t sub_b = vld1q_s32(scaled_query + i + 4);
            int32x4_t sub_c = vld1q_s32(scaled_query + i + 8);
            int32x4_t sub_d = vld1q_s32(scaled_query + i + 12);
            int32x4_t for_a = vld1q_s32(for_bases + i);
            int32x4_t for_b = vld1q_s32(for_bases + i + 4);
            int32x4_t for_c = vld1q_s32(for_bases + i + 8);
            int32x4_t for_d = vld1q_s32(for_bases + i + 12);

            int32x4_t input_low_1 = vsubq_s32(sub_a, for_a);
            int32x4_t input_high_1 = vsubq_s32(sub_b, for_b);
            int32x4_t input_low_2 = vsubq_s32(sub_c, for_c);
            int32x4_t input_high_2 = vsubq_s32(sub_d, for_d);

            vst1q_s32(dim_clip_value + i, input_low_1);
            vst1q_s32(dim_clip_value + i + 4, input_high_1);
            vst1q_s32(dim_clip_value + i + 8, input_low_2);
            vst1q_s32(dim_clip_value + i + 12, input_high_2);

            // Narrow from int32 to int16 (saturating)
            int16x4_t narrowed_low_1 = vqmovn_s32(input_low_1);
            int16x4_t narrowed_high_1 = vqmovn_s32( input_high_1);
            int16x4_t narrowed_low_2 = vqmovn_s32(input_low_2);
            int16x4_t narrowed_high_2 = vqmovn_s32(input_high_2);

            // Combine into a single int16x8_t vector
            int16x8_t combined_1 = vcombine_s16(narrowed_low_1, narrowed_high_1);
            int16x8_t combined_2 = vcombine_s16(narrowed_low_2, narrowed_high_2);

            // Narrow from int16 to uint8 (saturating)
            uint8x8_t result_1 = vqmovun_s16(combined_1);
            uint8x8_t result_2 = vqmovun_s16(combined_2);

            // Mask out values that were clamped to 255 (set them to 0)
            // TODO: 255 is going to be clamped to 0
            uint8x8_t mask_1 = vceq_u8(result_1, vdup_n_u8(255)); // Create mask where result == 255
            uint8x8_t mask_2 = vceq_u8(result_2, vdup_n_u8(255)); // Create mask where result == 255
            result_1 = vbic_u8(result_1, mask_1); // Zero out those values
            result_2 = vbic_u8(result_2, mask_2);

            //uint8x16_t combined_result = vcombine_u8(result_1, result_2);
            //vst1q_u8(query + i, combined_result);

            // Store result
            vst1_u8(query + i, result_1);
            vst1_u8(query + i + 8, result_2);
        }
    };

public:
    /******************************************************************
     * Search methods
     ******************************************************************/
    // PDXearch: PDX + Pruning
    std::vector<KNNCandidate> Search(float *__restrict raw_query, uint32_t k) {
        size_t max_length = 4096 * pdx_data.num_dimensions;
        uint8_t * out = new uint8_t[max_length];
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        clipped = 0;
        size_t all_values = 0;
        alignas(64) uint8_t query[pdx_data.num_dimensions];
        alignas(64) int32_t scaled_query[4096]; // TODO
        alignas(64) float transformed_raw_query[pdx_data.num_dimensions];
        alignas(64) float normalized_query[pdx_data.num_dimensions];
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        NormalizeQuery(raw_query, normalized_query);
        PreprocessQuery(normalized_query, transformed_raw_query);
        GetDimensionsAccessOrder(transformed_raw_query, pdx_data.means);
        // TODO: This should probably not be evaluated here
        if (pdx_data.is_ivf) {
            if (ivf_nprobe == 0){
                vectorgroups_to_visit = pdx_data.num_vectorgroups;
            } else {
                vectorgroups_to_visit = ivf_nprobe;
            }
#ifdef BENCHMARK_TIME
            end_to_end_clock.Toc();
#endif
            //GetVectorgroupsAccessOrderIVFPDX(query, vectorgroups_to_visit, vectorgroups_indices);
            //GetVectorgroupsAccessOrderRandom();
            GetVectorgroupsAccessOrderIVF(transformed_raw_query, pdx_data, ivf_nprobe, vectorgroups_indices);
#ifdef BENCHMARK_TIME
            end_to_end_clock.Tic();
#endif
        } else {
            // If there is no index, we just access the vectorgroups in order
            GetVectorgroupsAccessOrderRandom();
        }
        // PDXearch core
        current_dimension_idx = 0;
        current_vectorgroup = vectorgroups_indices[0];
        VectorgroupU6x8& first_vectorgroup = pdx_data.vectorgroups[vectorgroups_indices[0]];

        ScaleQuery(transformed_raw_query, scaled_query);
        PrepareQuery(scaled_query, query, first_vectorgroup.for_bases);
        all_values += first_vectorgroup.num_embeddings * pdx_data.num_dimensions;
        size_t length = first_vectorgroup.num_embeddings * pdx_data.num_dimensions;
//        uint8_t * out = new uint8_t[length + 1024];
        Decompress(first_vectorgroup.data, out, length);
        Start(query, out, first_vectorgroup.num_embeddings, k, first_vectorgroup.indices);
        //delete [] out;
        for (size_t vectorgroup_idx = 1; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VectorgroupU6x8& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            length = vectorgroup.num_embeddings * pdx_data.num_dimensions;
//            out = new uint8_t[length + 1024];
//#ifdef BENCHMARK_TIME
//            end_to_end_clock.Toc();
//#endif
            Decompress(vectorgroup.data, out, length);
//#ifdef BENCHMARK_TIME
//            end_to_end_clock.Tic();
//#endif
            PrepareQuery(scaled_query, query, vectorgroup.for_bases);
            Warmup(query, out, vectorgroup.num_embeddings, k, selectivity_threshold, best_k);
//#ifdef BENCHMARK_TIME
//            end_to_end_clock.Toc();
//#endif
            Prune(query, out, vectorgroup.num_embeddings, k, best_k);
            all_values += vectorgroup.num_embeddings * pdx_data.num_dimensions;
//#ifdef BENCHMARK_TIME
//            end_to_end_clock.Tic();
//#endif
            if (n_vectors_not_pruned){
                MergeIntoHeap<true>(vectorgroup.indices, n_vectors_not_pruned, k, best_k);
            }
            //delete [] out;
        }
        //std::cout << "Clipped dimensions: " <<  clipped << " on " << ivf_nprobe << " clusters (" << (float)(clipped)/(ivf_nprobe*pdx_data.num_dimensions) * 100.0 << " )\n";
        //std::cout << "Clipped values: " <<  clipped << " on " << ivf_nprobe << " clusters (" << (double)(clipped)/(all_values) * 100.0 << " )\n";
#ifdef BENCHMARK_TIME
        end_to_end_clock.Toc();
#endif
        delete [] out;
        return BuildResultSet(k);
    }


    // Full Linear Scans that do not prune vectors
    std::vector<KNNCandidate> LinearScan(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        uint8_t transformed_raw_query[pdx_data.num_dimensions];
        uint8_t query[pdx_data.num_dimensions];
        PreprocessQuery(raw_query, transformed_raw_query);
        PrepareQuery(transformed_raw_query, query);
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        GetVectorgroupsAccessOrderRandom();
        for (size_t vectorgroup_idx = 0; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VectorgroupU6x8& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            if (vectorgroup.num_embeddings == PDX_VECTOR_SIZE){
                ResetDistancesVectorized();
                CalculateVerticalDistancesVectorized<ALPHA>(query, vectorgroup.data, 0, pdx_data.num_dimensions, distances);
                MergeIntoHeap<false>(vectorgroup.indices, PDX_VECTOR_SIZE, k, best_k);
            } else if (vectorgroup.num_embeddings < PDX_VECTOR_SIZE) {
                ResetDistancesVectorized();
                CalculateVerticalDistancesScalar<false, false, ALPHA>(query, vectorgroup.data, vectorgroup.num_embeddings, vectorgroup.num_embeddings,  0, pdx_data.num_dimensions, distances);
                MergeIntoHeap<false>(vectorgroup.indices, vectorgroup.num_embeddings, k, best_k);
            }
        }
#ifdef BENCHMARK_TIME
        end_to_end_clock.Toc();
#endif
        return BuildResultSet(k);
    }

};

} // namespace PDX

#endif //EMBEDDINGSEARCH_PDXEARCH_HPP

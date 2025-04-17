#ifndef PDX_PDXEARCH64_HPP
#define PDX_PDXEARCH64_HPP

#include <queue>
#include <cassert>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/quantizers/quantizers.h"
#include "pdx/index_base/pdx_ivf.hpp"
#include "vector_searcher.hpp"
#include "utils/tictoc.hpp"

namespace PDX {

/******************************************************************
 * PDXearch
 * Implements our algorithm for vertical pruning
 ******************************************************************/
template<Quantization q=F32, class quantizer=LEPQuantizer<q>, class distance_computer=DistanceComputer<L2, q>>
class PDXearch64: public ::VectorSearcher<q> {
public:
    using DISTANCES_TYPE = DistanceType_t<q>;
    using QUANTIZED_VECTOR_TYPE = QuantizedVectorType_t<q>;
    using DATA_TYPE = DataType_t<q>;
    using INDEX_TYPE = IndexPDXIVF<q>;
    using VECTORGROUP_TYPE = Vectorgroup<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;

    quantizer quant;

    PDXearch64(
            INDEX_TYPE &data_index,
            size_t ivf_nprobe,
            int position_prune_distance,
            DimensionsOrder dimension_order
    ) : pdx_data(data_index), ivf_nprobe(ivf_nprobe),
            is_positional_pruning(position_prune_distance),
            dimension_order(dimension_order){
        indices_dimensions.resize(pdx_data.num_dimensions);
        vectorgroups_indices.resize(pdx_data.num_vectorgroups);
        for (size_t i = 0; i < pdx_data.num_vectorgroups; ++i){
            total_embeddings += pdx_data.vectorgroups[i].num_embeddings;
        }
        quant.SetD(pdx_data.num_dimensions);
    }

    INDEX_TYPE &pdx_data;
    uint32_t current_dimension_idx {0};

    void SetNProbe(size_t nprobe){
        ivf_nprobe = nprobe;
    }

    void SetExponent(int exponent){
        quant.lep_exponent = exponent;
    }

protected:
    float selectivity_threshold = 0.80;
    size_t ivf_nprobe = 0;
    int is_positional_pruning = false;
    size_t current_vectorgroup = 0;

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
    //inline static std::unordered_map<size_t, size_t> when_is_pruned{};
    //inline static std::unordered_set<size_t> when_is_pruned_exist{};

    // Debugging
    size_t warmup_bytes = 0;
    size_t prune_bytes = 0;
    size_t start_bytes = 0;
    size_t processed_bytes = 0;
    size_t total_bytes = 0;

    std::vector<uint32_t> indices_dimensions;
    std::vector<uint32_t> vectorgroups_indices;

    size_t n_vectors_not_pruned = 0;

    DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();

    // For pruning we do not use tight loops of 64. We know that tight loops bring benefits
    // to the distance kernels (40% faster), however doing so + PRUNING in the tight block of 64
    // slightly reduces the performance of PDXearch. We are still investigating why.
    static constexpr uint16_t PDX_VECTOR_SIZE = 64;
    alignas(64) inline static DISTANCES_TYPE distances[PDX_VECTOR_SIZE]; // Used in full scans (no pruning)
    alignas(64) inline static DISTANCES_TYPE pruning_distances[64]; // TODO: Use dynamic arrays. Buckets with more than 10k vectors (rare) overflow

    alignas(64) inline static uint32_t pruning_positions[64];

    std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t> best_k_centroids{};

    virtual void PreprocessQuery(float * raw_query, float *query){};

    inline void ResetDistancesScalar(size_t n_vectors){
        memset((void*) distances, 0, n_vectors * sizeof(DISTANCES_TYPE));
    }

    inline virtual void ResetPruningDistances(size_t n_vectors){
        memset((void*) pruning_distances, 0, n_vectors * sizeof(DISTANCES_TYPE));
    }

    virtual void ResetDistancesVectorized(){
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(DISTANCES_TYPE));
    }

    // The pruning threshold by default is the top of the heap
    virtual void GetPruningThreshold(uint32_t k, std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t> &heap){
        pruning_threshold = heap.size() == k ? heap.top().distance : std::numeric_limits<DISTANCES_TYPE>::max();
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
            //if (pruning_distances[pruning_positions[vector_idx]] >= pruning_threshold && (when_is_pruned_exist.find(pruning_positions[vector_idx]) == when_is_pruned_exist.end())){
            //    when_is_pruned[current_dimension_idx]++;
            //    when_is_pruned_exist.insert(pruning_positions[vector_idx]);
            //}
        }
    };

    virtual void EvaluatePruningPredicateVectorized(uint32_t &n_pruned) {
        for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
        }
    };

    inline virtual void InitPositionsArray(size_t n_vectors){
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
            //if (pruning_distances[vector_idx] >= pruning_threshold && (when_is_pruned_exist.find(vector_idx) == when_is_pruned_exist.end())){
            //    when_is_pruned[current_dimension_idx]++;
            //    when_is_pruned_exist.insert(vector_idx);
            //}
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

    // On the first bucket, we do a full scan (we do not prune vectors)
    template<bool FULL_BLOCK=false>
    void Start(const QUANTIZED_VECTOR_TYPE *__restrict query, const DATA_TYPE * data, size_t n_vectors, uint32_t k, const uint32_t * vector_indices) {
        //processed_bytes += n_vectors * pdx_data.num_dimensions;
        //start_bytes += n_vectors * pdx_data.num_dimensions;
        ResetPruningDistances(n_vectors);
        // Vertical part
        if constexpr (FULL_BLOCK){
            n_vectors = PDX_VECTOR_SIZE;
            distance_computer::VerticalBlock(query, data, 0, pdx_data.num_vertical_dimensions, pruning_distances);
        } else {
            distance_computer::Vertical(query, data, n_vectors, n_vectors, 0, pdx_data.num_vertical_dimensions, pruning_distances, pruning_positions, indices_dimensions.data(), quant.dim_clip_value);
        }
        // Horizontal part
        for (size_t horizontal_dimension = 0; horizontal_dimension < pdx_data.num_horizontal_dimensions; horizontal_dimension+=H_DIM_SIZE) {
            for (size_t vector_idx = 0; vector_idx < n_vectors; vector_idx++) {
                size_t data_pos = (pdx_data.num_vertical_dimensions * n_vectors) +
                                (horizontal_dimension * n_vectors) +
                                  (vector_idx * H_DIM_SIZE);
                pruning_distances[vector_idx] += distance_computer::Horizontal(
                        query + pdx_data.num_vertical_dimensions + horizontal_dimension,
                        data + data_pos,
                        H_DIM_SIZE
                );
            }
        }
        // Clipping (TODO: This looks horrible)
        for (size_t horizontal_dimension = 0; horizontal_dimension < pdx_data.num_horizontal_dimensions; horizontal_dimension++) {
            if (quant.dim_clip_value[pdx_data.num_vertical_dimensions + horizontal_dimension] < 0){
                for (size_t vector_idx = 0; vector_idx < n_vectors; vector_idx++) {
                    size_t data_pos = (pdx_data.num_vertical_dimensions * n_vectors) +
                                      ((horizontal_dimension / H_DIM_SIZE) * H_DIM_SIZE * n_vectors) +
                                      (vector_idx * H_DIM_SIZE) + (horizontal_dimension % H_DIM_SIZE);
                    pruning_distances[vector_idx] -= 2 * data[data_pos] * quant.dim_clip_value[pdx_data.num_vertical_dimensions + horizontal_dimension];
                    pruning_distances[vector_idx] += quant.dim_clip_value[pdx_data.num_vertical_dimensions + horizontal_dimension] * quant.dim_clip_value[pdx_data.num_vertical_dimensions + horizontal_dimension];
                }
            }
        }
        // end of horizontal part
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
            auto embedding = KNNCandidate_t{};
            size_t index = indices_sorted[idx];
            embedding.index = vector_indices[index];
            embedding.distance = pruning_distances[index];
            if (this->best_k.size() < k || embedding.distance < this->best_k.top().distance) {
                if (this->best_k.size() >= k) {
                    this->best_k.pop();
                }
                this->best_k.push(embedding);
            }
            // this->best_k.push(embedding);
        }
    }

    // On the warmup phase, we keep scanning dimensions until the amount of not-yet pruned vectors is low
    template<bool FULL_BLOCK=false>
    void Warmup(const QUANTIZED_VECTOR_TYPE *__restrict query, const DATA_TYPE * data, size_t n_vectors, uint32_t k, float tuples_threshold, std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t> &heap) {
        if constexpr(FULL_BLOCK){
            n_vectors = PDX_VECTOR_SIZE;
        }
        current_dimension_idx = 0;
        cur_subgrouping_size_idx = 0;
        size_t tuples_needed_to_exit = std::ceil(1.0 * tuples_threshold * n_vectors);
        ResetPruningDistances(n_vectors);
        uint32_t n_tuples_to_prune = 0;
        if (!is_positional_pruning) GetPruningThreshold(k, heap);

        while (
                1.0 * n_tuples_to_prune < tuples_needed_to_exit &&
                current_dimension_idx < pdx_data.num_vertical_dimensions) {
            size_t last_dimension_to_fetch = std::min(current_dimension_idx + DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
                                                      pdx_data.num_vertical_dimensions);
                                                      //pdx_data.num_dimensions);
            //warmup_bytes += (last_dimension_to_fetch - current_dimension_idx) * n_vectors;
            //processed_bytes += (last_dimension_to_fetch - current_dimension_idx) * n_vectors;
            if (dimension_order == SEQUENTIAL){
                if constexpr (FULL_BLOCK){
                    distance_computer::VerticalBlock(query, data, current_dimension_idx,
                                                last_dimension_to_fetch, pruning_distances);
                } else {
                    distance_computer::Vertical(query, data, n_vectors, n_vectors, current_dimension_idx,
                                                last_dimension_to_fetch, pruning_distances,
                                                pruning_positions, indices_dimensions.data(), quant.dim_clip_value);
                }
            } else {
                distance_computer::VerticalReordered(query, data, n_vectors, n_vectors, current_dimension_idx,
                                                                    last_dimension_to_fetch, pruning_distances,
                                                                    pruning_positions, indices_dimensions.data(), quant.dim_clip_value);
            }
            current_dimension_idx = last_dimension_to_fetch;
            cur_subgrouping_size_idx += 1;
            if (is_positional_pruning) GetPruningThreshold(k, heap);
            n_tuples_to_prune = 0;
            EvaluatePruningPredicateScalar(n_tuples_to_prune, n_vectors);
        }
    }

    // We scan only the not-yet pruned vectors
    void Prune(const QUANTIZED_VECTOR_TYPE *__restrict query, const DATA_TYPE *__restrict data, const size_t n_vectors, uint32_t k, std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t> &heap) {
        //when_is_pruned_exist.clear();
        GetPruningThreshold(k, heap);
        InitPositionsArray(n_vectors);
        size_t cur_n_vectors_not_pruned = 0;
        // To count all bytes that went to the PRUNE phase
        //prune_bytes += (pdx_data.num_dimensions - current_dimension_idx) * n_vectors_not_pruned;
        // GO THROUGH THE HORIZONTAL DIMENSIONS (possibly) AT THE MIDDLE OF THE VERTICAL ONES
        size_t current_vertical_dimension = current_dimension_idx;
        size_t current_horizontal_dimension = 0;
        while (
                pdx_data.num_horizontal_dimensions &&
                n_vectors_not_pruned &&
                        current_horizontal_dimension < pdx_data.num_horizontal_dimensions
        ) {
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            //prune_bytes += H_DIM_SIZE * cur_n_vectors_not_pruned;
            //processed_bytes += H_DIM_SIZE * cur_n_vectors_not_pruned;
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
                pruning_distances[v_idx] += distance_computer::Horizontal(
                        query + offset_query,
                        data + data_pos,
                        H_DIM_SIZE
                );
            }
            // Clipping (TODO: This looks horrible)
            for (size_t horizontal_dimension = current_horizontal_dimension; horizontal_dimension < current_horizontal_dimension + H_DIM_SIZE; horizontal_dimension++) {
                if (quant.dim_clip_value[pdx_data.num_vertical_dimensions + horizontal_dimension] < 0){
                    for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                        size_t v_idx = pruning_positions[vector_idx];
                        size_t data_pos = (pdx_data.num_vertical_dimensions * n_vectors) +
                                          (current_horizontal_dimension * n_vectors) +
                                          (v_idx * H_DIM_SIZE) + (horizontal_dimension % H_DIM_SIZE);
                        pruning_distances[v_idx] -= 2 * data[data_pos] * quant.dim_clip_value[pdx_data.num_vertical_dimensions + horizontal_dimension];
                        pruning_distances[v_idx] += quant.dim_clip_value[pdx_data.num_vertical_dimensions + horizontal_dimension] * quant.dim_clip_value[pdx_data.num_vertical_dimensions + horizontal_dimension];
                    }
                }
            }
            // end of clipping
            current_horizontal_dimension += H_DIM_SIZE;
            current_dimension_idx += H_DIM_SIZE;
            if (is_positional_pruning) GetPruningThreshold(k, heap);
            assert(current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
            EvaluatePruningPredicateOnPositionsArray(cur_n_vectors_not_pruned);
        }
        // GO THROUGH THE REST IN THE VERTICAL
        while (
                n_vectors_not_pruned &&
                current_vertical_dimension < pdx_data.num_vertical_dimensions
                ) {
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            // TODO: ADD variable fetching size back in instead of H_DIM_SIZE
            size_t last_dimension_to_test_idx = std::min(current_vertical_dimension + H_DIM_SIZE,
                                                         (size_t)pdx_data.num_vertical_dimensions);
//            size_t last_dimension_to_test_idx = std::min(current_dimension_idx + 4, pdx_data.num_dimensions);
            //prune_bytes += (last_dimension_to_test_idx - current_vertical_dimension) * cur_n_vectors_not_pruned;
            //processed_bytes += (last_dimension_to_test_idx - current_vertical_dimension) * cur_n_vectors_not_pruned;
            if (dimension_order == SEQUENTIAL){
                distance_computer::VerticalPruning(query, data, cur_n_vectors_not_pruned,
                                                                           n_vectors, current_vertical_dimension,
                                                                           last_dimension_to_test_idx, pruning_distances,
                                                                           pruning_positions, indices_dimensions.data(), quant.dim_clip_value);
            } else {
                distance_computer::VerticalReorderedPruning(query, data, cur_n_vectors_not_pruned,
                                                                          n_vectors, current_vertical_dimension,
                                                                          last_dimension_to_test_idx, pruning_distances,
                                                                          pruning_positions, indices_dimensions.data(), quant.dim_clip_value);
            }
            current_dimension_idx = std::min(current_dimension_idx+H_DIM_SIZE, (size_t)pdx_data.num_dimensions);
            current_vertical_dimension = std::min((uint32_t)(current_vertical_dimension+H_DIM_SIZE), pdx_data.num_vertical_dimensions);
            assert(current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
            if (is_positional_pruning) GetPruningThreshold(k, heap);
            EvaluatePruningPredicateOnPositionsArray(cur_n_vectors_not_pruned);
            if (current_dimension_idx == pdx_data.num_dimensions) break;
        }
    }

    // TODO: Manage the heap elsewhere
    template <bool IS_PRUNING=false>
    void MergeIntoHeap(const uint32_t * vector_indices, size_t n_vectors, uint32_t k, std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t> &heap) {
        for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
            size_t index = position_idx;
            DISTANCES_TYPE current_distance;
            if constexpr (IS_PRUNING){
                index = pruning_positions[position_idx];
                current_distance = pruning_distances[index];
            } else {
                current_distance = distances[index];
            }
            //when_is_pruned[current_dimension_idx]++;
            if (heap.size() < k || current_distance < heap.top().distance) {
                KNNCandidate_t embedding{};
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
        std::vector<KNNCandidate_t> result;
        result.resize(k);
        for (int i = k - 1; i >= 0; --i) {
            const KNNCandidate_t& embedding = this->best_k.top();
            result[i].distance = embedding.distance;
            result[i].index = embedding.index;
            this->best_k.pop();
        }
        return result;
    }

    // EXPERIMENTAL (not used in the publication)
    // In some datasets, we can apply the pruning idea to the IVF centroids.
    // The recall reduction is OK, but the speedup only happens if the number of buckets probed is around 20
    // Otherwise, algorithms have a hard time trying to prune.
    // This effect is partially explained in a recent paper (https://dl.acm.org/doi/pdf/10.1145/3709743)
//    void GetVectorgroupsAccessOrderIVFPDXearch(const float *__restrict query, size_t ivf_nprobe, std::vector<uint32_t> &vectorgroups_indices) {
//        best_k_centroids = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
//        vectorgroups_indices.resize(pdx_data.num_vectorgroups);
//        std::iota(vectorgroups_indices.begin(), vectorgroups_indices.end(), 0);
//        float * tmp_centroids_pdx = pdx_data.centroids_pdx;
//        float * tmp_centroids = pdx_data.centroids;
//        uint32_t * tmp_vectorgroup_indices = vectorgroups_indices.data();
//        size_t SKIPPING_SIZE = PDX_VECTOR_SIZE * pdx_data.num_dimensions;
//        size_t centroids_visited = 0;
//
//        size_t remainder_block_size = pdx_data.num_vectorgroups % PDX_VECTOR_SIZE;
//        size_t full_blocks = std::floor(1.0 * pdx_data.num_vectorgroups / PDX_VECTOR_SIZE);
//
//        const float TIGHT_SELECTIVITY_THRESHOLD = 0.8; // We push warmup to 90% due to dual-storage being present
//        while (best_k_centroids.size() != ivf_nprobe){ // This while works since we cannot call an IVF index with more IVF_NPROBE than N_BUCKETS
//            // TODO: Use another distances array for the centroids so I can use ResetDistancesVectorized() instead of memset
//            memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
//            CalculateVerticalDistancesVectorized<false, L2>(query, tmp_centroids_pdx, 0, pdx_data.num_dimensions);
//            MergeIntoHeap<false>(tmp_vectorgroup_indices, PDX_VECTOR_SIZE, ivf_nprobe, best_k_centroids);
//            tmp_vectorgroup_indices += PDX_VECTOR_SIZE;
//            tmp_centroids_pdx += SKIPPING_SIZE;
//            tmp_centroids += SKIPPING_SIZE;
//            centroids_visited += PDX_VECTOR_SIZE;
//            full_blocks -= 1;
//        }
//        if (centroids_visited < pdx_data.num_vectorgroups){
//            for (size_t block_idx = 0; block_idx < full_blocks; ++block_idx){
//                Warmup<L2>(query, tmp_centroids_pdx, PDX_VECTOR_SIZE, ivf_nprobe, TIGHT_SELECTIVITY_THRESHOLD, best_k_centroids);
//                // Instead of trying to Prune, since we have dual-storage we just go directly to PlateauHorizontal
//                GetPruningThreshold(ivf_nprobe, best_k_centroids);
//                InitPositionsArray(PDX_VECTOR_SIZE);
//                if (n_vectors_not_pruned){
//                    CalculateDistancesNary(query, tmp_centroids);
//                    MergeIntoHeap<true>(tmp_vectorgroup_indices, n_vectors_not_pruned, ivf_nprobe, best_k_centroids);
//                }
//                tmp_centroids_pdx += SKIPPING_SIZE;
//                tmp_centroids += SKIPPING_SIZE;
//                tmp_vectorgroup_indices += PDX_VECTOR_SIZE;
//                centroids_visited += PDX_VECTOR_SIZE;
//            }
//            if (remainder_block_size){
//                Warmup<L2>(query, tmp_centroids_pdx, remainder_block_size, ivf_nprobe, TIGHT_SELECTIVITY_THRESHOLD, best_k_centroids);
//                GetPruningThreshold(ivf_nprobe, best_k_centroids);
//                InitPositionsArray(remainder_block_size);
//                if (n_vectors_not_pruned){
//                    CalculateDistancesNary(query, tmp_centroids);
//                    MergeIntoHeap<true>(tmp_vectorgroup_indices, n_vectors_not_pruned, ivf_nprobe, best_k_centroids);
//                }
//            }
//        }
//        for (size_t i = 0; i < ivf_nprobe; ++i){
//            const KNNCandidate& c = best_k_centroids.top();
//            vectorgroups_indices[ivf_nprobe - i - 1] = c.index; // I need to inverse the allocation
//            best_k_centroids.pop();
//        }
//        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
//        n_vectors_not_pruned = 0;
//    }

    // We store centroids using PDX in tight blocks of 64
    void GetVectorgroupsAccessOrderIVFPDX(const float *__restrict query) {
        best_k_centroids = std::priority_queue<KNNCandidate<F32>, std::vector<KNNCandidate<F32>>, VectorComparator<F32>>{};
        vectorgroups_indices.resize(pdx_data.num_vectorgroups);
        std::iota(vectorgroups_indices.begin(), vectorgroups_indices.end(), 0);
        float * tmp_centroids_pdx = pdx_data.centroids_pdx;
        uint32_t * tmp_vectorgroup_indices = vectorgroups_indices.data();
        size_t SKIPPING_SIZE = PDX_VECTOR_SIZE * pdx_data.num_dimensions;
        size_t remainder_block_size = pdx_data.num_vectorgroups % PDX_VECTOR_SIZE;
        size_t full_blocks = std::floor(1.0 * pdx_data.num_vectorgroups / PDX_VECTOR_SIZE);
        for (size_t centroid_idx = 0; centroid_idx < full_blocks; ++centroid_idx) {
            // TODO: Use another distances array for the centroids so I can use ResetDistancesVectorized() instead of memset
            memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
            DistanceComputer<L2, F32>::VerticalBlock(query, tmp_centroids_pdx, 0, pdx_data.num_dimensions, distances);
            // CalculateVerticalDistancesVectorized<false, L2>(query, tmp_centroids_pdx, 0, pdx_data.num_dimensions);
            MergeIntoHeap<false>(tmp_vectorgroup_indices, PDX_VECTOR_SIZE, ivf_nprobe, best_k_centroids);
            tmp_vectorgroup_indices += PDX_VECTOR_SIZE;
            tmp_centroids_pdx += SKIPPING_SIZE;
        }
        if (remainder_block_size){
            memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
            DistanceComputer<L2, F32>::Vertical(query, tmp_centroids_pdx, remainder_block_size, remainder_block_size, 0, pdx_data.num_dimensions, distances, nullptr, nullptr, nullptr);
            // CalculateVerticalDistancesScalar<false, L2>(query, tmp_centroids_pdx, remainder_block_size, 0, pdx_data.num_dimensions);
            MergeIntoHeap<false>(tmp_vectorgroup_indices, remainder_block_size, ivf_nprobe, best_k_centroids);
        }
        for (size_t i = 0; i < ivf_nprobe; ++i){
            const KNNCandidate<F32> & c = best_k_centroids.top();
            vectorgroups_indices[ivf_nprobe - i - 1] = c.index; // I need to inverse the allocation
            best_k_centroids.pop();
        }
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
    }

    void GetVectorgroupsAccessOrderRandom() {
        std::iota(vectorgroups_indices.begin(), vectorgroups_indices.end(), 0);
    }

public:
    /******************************************************************
     * Search methods
     ******************************************************************/
    // PDXearch: PDX + Pruning
    template<Quantization Q=q, std::enable_if_t<Q==U8, int> = 0>
    std::vector<KNNCandidate_t> Search(float *__restrict raw_query, uint32_t k) {
        processed_bytes = 0;
        total_bytes = 0;
        start_bytes = 0;
        prune_bytes = 0;
        warmup_bytes = 0;
        this->best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        quant.NormalizeQuery(raw_query);
        PreprocessQuery(quant.normalized_query, quant.transformed_raw_query);
        // For now I will not take into account the preprocessing query time
        // because RaBitQ cheats a bit by doing all the queries at once
        GetDimensionsAccessOrder(quant.transformed_raw_query, pdx_data.means);
        // TODO: This should probably not be evaluated here
        if (pdx_data.is_ivf) {
            if (ivf_nprobe == 0){
                vectorgroups_to_visit = pdx_data.num_vectorgroups;
            } else {
                vectorgroups_to_visit = ivf_nprobe;
            }
//#ifdef BENCHMARK_TIME
//            this->end_to_end_clock.Toc();
//#endif
            //this->GetVectorgroupsAccessOrderIVFPDX(query, vectorgroups_to_visit, vectorgroups_indices);
            this->GetVectorgroupsAccessOrderIVF(quant.transformed_raw_query, pdx_data, ivf_nprobe, vectorgroups_indices);
//#ifdef BENCHMARK_TIME
//            this->end_to_end_clock.Tic();
//#endif
        } else {
            // If there is no index, we just access the vectorgroups in order
            GetVectorgroupsAccessOrderRandom();
        }
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        // PDXearch core
        current_dimension_idx = 0;
        size_t skip_block_size = PDX_VECTOR_SIZE * pdx_data.num_dimensions;
        current_vectorgroup = vectorgroups_indices[0];
        VECTORGROUP_TYPE& first_vectorgroup = pdx_data.vectorgroups[vectorgroups_indices[0]];
        quant.ScaleQuery(quant.transformed_raw_query);
        quant.PrepareQuery(first_vectorgroup.for_bases);
        // START
        size_t fv_num_complete_blocks = std::floor(1.0 * first_vectorgroup.num_embeddings / PDX_VECTOR_SIZE);
        size_t fv_remainder = first_vectorgroup.num_embeddings % PDX_VECTOR_SIZE;
        uint8_t * fv_vg_data_p = first_vectorgroup.data;
        uint32_t * fv_vg_indices_p = first_vectorgroup.indices;
        for (size_t block_id = 0; block_id < fv_num_complete_blocks; block_id++){
            Start<true>(quant.quantized_query, fv_vg_data_p, PDX_VECTOR_SIZE, k, fv_vg_indices_p);
            fv_vg_data_p += skip_block_size;
            fv_vg_indices_p += PDX_VECTOR_SIZE;
        }
        if (fv_remainder){
            Start<false>(quant.quantized_query, fv_vg_data_p, fv_remainder, k, fv_vg_indices_p);
        }
        total_bytes += first_vectorgroup.num_embeddings * pdx_data.num_dimensions;
        for (size_t vectorgroup_idx = 1; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VECTORGROUP_TYPE& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            total_bytes += vectorgroup.num_embeddings * pdx_data.num_dimensions;
            quant.PrepareQuery(vectorgroup.for_bases);
            size_t num_complete_blocks = std::floor(1.0 * vectorgroup.num_embeddings / PDX_VECTOR_SIZE);
            size_t remainder = vectorgroup.num_embeddings % PDX_VECTOR_SIZE;
            uint8_t * vg_data_p = vectorgroup.data;
            uint32_t * vg_indices_p = vectorgroup.indices;
            for (size_t block_id = 0; block_id < num_complete_blocks; block_id++){
                Warmup<true>(quant.quantized_query, vg_data_p, PDX_VECTOR_SIZE, k, selectivity_threshold, this->best_k);
                Prune(quant.quantized_query, vg_data_p, PDX_VECTOR_SIZE, k, this->best_k);
                if (n_vectors_not_pruned){
                    MergeIntoHeap<true>(vg_indices_p, n_vectors_not_pruned, k, this->best_k);
                }
                vg_data_p += skip_block_size;
                vg_indices_p += PDX_VECTOR_SIZE;
            }
            if (remainder){
                Warmup(quant.quantized_query, vg_data_p, remainder, k, selectivity_threshold, this->best_k);
                Prune(quant.quantized_query, vg_data_p, remainder, k, this->best_k);
                if (n_vectors_not_pruned){
                    MergeIntoHeap<true>(vg_indices_p, n_vectors_not_pruned, k, this->best_k);
                }
            }
        }
        //std::cout << total_bytes << "," << processed_bytes << "," << start_bytes << "," << warmup_bytes << "," << prune_bytes << "\n";
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        //std::ofstream outfile("./pruning-histogram-u8-k10.txt");
        //for (const auto& [key, value] : when_is_pruned) {
        //    outfile << key << "," << value << "\n";
        //}
        return BuildResultSet(k);
    }


    template<Quantization Q=q, std::enable_if_t<Q==F32, int> = 0>
    std::vector<KNNCandidate_t> Search(float *__restrict raw_query, uint32_t k) {
        processed_bytes = 0;
        total_bytes = 0;
        start_bytes = 0;
        prune_bytes = 0;
        warmup_bytes = 0;
        alignas(64) float query[pdx_data.num_dimensions];
        // TODO: The query transformer should be given somewhere
        // Here I am doing this to support BSA, and now I am ignoring PDX bond actually
        // But this depends on the dataset and query. So it should be a parameter
        // This would also be the case in other calls of Search
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        if constexpr (std::is_same_v<distance_computer, DistanceComputer<NEGATIVE_L2, F32>>){
            PreprocessQuery(raw_query, query);
        } else {
            quant.NormalizeQuery(raw_query);
            PreprocessQuery(quant.normalized_query, query);
        }
        this->best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        GetDimensionsAccessOrder(query, pdx_data.means);
        // TODO: This should probably not be evaluated here
        if (pdx_data.is_ivf) {
            if (ivf_nprobe == 0){
                vectorgroups_to_visit = pdx_data.num_vectorgroups;
            } else {
                vectorgroups_to_visit = ivf_nprobe;
            }
//#ifdef BENCHMARK_TIME
//            this->end_to_end_clock.Toc();
//#endif
            //this->GetVectorgroupsAccessOrderIVFPDX(query);
//#ifdef BENCHMARK_TIME
//            this->end_to_end_clock.Tic();
//#endif
            this->GetVectorgroupsAccessOrderIVF(query, pdx_data, ivf_nprobe, vectorgroups_indices);
        } else {
            // If there is no index, we just access the vectorgroups in order
            GetVectorgroupsAccessOrderRandom();
        }
        // PDXearch core
        current_dimension_idx = 0;
        current_vectorgroup = vectorgroups_indices[0];
        VECTORGROUP_TYPE& first_vectorgroup = pdx_data.vectorgroups[vectorgroups_indices[0]];
        Start(query, first_vectorgroup.data, first_vectorgroup.num_embeddings, k, first_vectorgroup.indices);
        //total_bytes += first_vectorgroup.num_embeddings * pdx_data.num_dimensions;
        for (size_t vectorgroup_idx = 1; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VECTORGROUP_TYPE& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            //total_bytes += vectorgroup.num_embeddings * pdx_data.num_dimensions;
            Warmup(query, vectorgroup.data, vectorgroup.num_embeddings, k, selectivity_threshold, this->best_k);
            Prune(query, vectorgroup.data, vectorgroup.num_embeddings, k, this->best_k);
            if (n_vectors_not_pruned){
                MergeIntoHeap<true>(vectorgroup.indices, n_vectors_not_pruned, k, this->best_k);
            }
        }
        //std::cout << total_bytes * 4 << "," << processed_bytes * 4 << "," << start_bytes * 4 << "," << warmup_bytes * 4 << "," << prune_bytes * 4 << "\n";

#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return BuildResultSet(k);
    }


    template<Quantization Q=q, std::enable_if_t<Q==U8, int> = 0>
    std::vector<KNNCandidate_t> LinearScan(float *__restrict raw_query, uint32_t k) {
        std::vector<int> dummy_for_bases(4096, 0);
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        // TODO: Not all queries will need this step. What if raw data is uint8?
        quant.NormalizeQuery(raw_query);
        quant.ScaleQuery(quant.normalized_query);
        quant.PrepareQuery(dummy_for_bases.data());
        // TODO: How to quantize query?
        this->best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        GetVectorgroupsAccessOrderRandom();
        for (size_t vectorgroup_idx = 0; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VECTORGROUP_TYPE& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            if (vectorgroup.num_embeddings == PDX_VECTOR_SIZE){
                ResetDistancesVectorized();
                distance_computer::VerticalBlock(quant.quantized_query, vectorgroup.data, 0, pdx_data.num_dimensions, distances);
                MergeIntoHeap<false>(vectorgroup.indices, PDX_VECTOR_SIZE, k, this->best_k);
            } else if (vectorgroup.num_embeddings < PDX_VECTOR_SIZE) {
                ResetDistancesVectorized();
                distance_computer::Vertical(quant.quantized_query, vectorgroup.data, vectorgroup.num_embeddings, vectorgroup.num_embeddings,  0, pdx_data.num_dimensions, distances,
                                            nullptr, nullptr, nullptr);
                MergeIntoHeap<false>(vectorgroup.indices, vectorgroup.num_embeddings, k, this->best_k);
            }
        }
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return BuildResultSet(k);
    }

    // Full Linear Scans that do not prune vectors
    template<Quantization Q=q, std::enable_if_t<Q==F32, int> = 0>
    std::vector<KNNCandidate_t> LinearScan(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        float query[pdx_data.num_dimensions];
        PreprocessQuery(raw_query, query);
        this->best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        GetVectorgroupsAccessOrderRandom();
        for (size_t vectorgroup_idx = 0; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VECTORGROUP_TYPE& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            if (vectorgroup.num_embeddings == PDX_VECTOR_SIZE){
                ResetDistancesVectorized();
                distance_computer::VerticalBlock(query, vectorgroup.data, 0, pdx_data.num_dimensions, distances);
                MergeIntoHeap<false>(vectorgroup.indices, PDX_VECTOR_SIZE, k, this->best_k);
            } else if (vectorgroup.num_embeddings < PDX_VECTOR_SIZE) {
                ResetDistancesVectorized();
                distance_computer::Vertical(query, vectorgroup.data, vectorgroup.num_embeddings, vectorgroup.num_embeddings, 0, pdx_data.num_dimensions, distances, nullptr, nullptr, nullptr);
                MergeIntoHeap<false>(vectorgroup.indices, vectorgroup.num_embeddings, k, this->best_k);
            }
        }
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return BuildResultSet(k);
    }


    // PDXearch: PDX + Pruning
    template<Quantization Q=q, std::enable_if_t<(Q==U4 || Q==U6), int> = 0>
    std::vector<KNNCandidate_t> Search(float *__restrict raw_query, uint32_t k) {
        auto * out = new uint8_t[4096 * pdx_data.num_dimensions];
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        this->best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        quant.NormalizeQuery(raw_query);
        PreprocessQuery(quant.normalized_query, quant.transformed_raw_query);
        GetDimensionsAccessOrder(quant.transformed_raw_query, pdx_data.means);
        // TODO: This should probably not be evaluated here
        if (pdx_data.is_ivf) {
            if (ivf_nprobe == 0){
                vectorgroups_to_visit = pdx_data.num_vectorgroups;
            } else {
                vectorgroups_to_visit = ivf_nprobe;
            }
#ifdef BENCHMARK_TIME
            this->end_to_end_clock.Toc();
#endif
            //this->GetVectorgroupsAccessOrderIVFPDX(query, vectorgroups_to_visit, vectorgroups_indices);
            //this->GetVectorgroupsAccessOrderRandom();
            this->GetVectorgroupsAccessOrderIVF(quant.transformed_raw_query, pdx_data, ivf_nprobe, vectorgroups_indices);
#ifdef BENCHMARK_TIME
            this->end_to_end_clock.Tic();
#endif
        } else {
            // If there is no index, we just access the vectorgroups in order
            GetVectorgroupsAccessOrderRandom();
        }
        // PDXearch core
        current_dimension_idx = 0;
        current_vectorgroup = vectorgroups_indices[0];
        VECTORGROUP_TYPE & first_vectorgroup = pdx_data.vectorgroups[vectorgroups_indices[0]];

        quant.ScaleQuery(quant.transformed_raw_query);
        quant.PrepareQuery(first_vectorgroup.for_bases);
        size_t length = first_vectorgroup.num_embeddings * pdx_data.num_dimensions;
        quant.Decompress(first_vectorgroup.data, out, length);
        Start(quant.quantized_query, out, first_vectorgroup.num_embeddings, k, first_vectorgroup.indices);
        for (size_t vectorgroup_idx = 1; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VECTORGROUP_TYPE& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            length = vectorgroup.num_embeddings * pdx_data.num_dimensions;
            quant.Decompress(vectorgroup.data, out, length);
            quant.PrepareQuery(vectorgroup.for_bases);
            Warmup(quant.quantized_query, out, vectorgroup.num_embeddings, k, selectivity_threshold, this->best_k);
            Prune(quant.quantized_query, out, vectorgroup.num_embeddings, k, this->best_k);
            if (n_vectors_not_pruned){
                MergeIntoHeap<true>(vectorgroup.indices, n_vectors_not_pruned, k, this->best_k);
            }
        }
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        delete [] out;
        return BuildResultSet(k);
    }

    // PDXearch: PDX + Pruning
    template<Quantization Q=q, std::enable_if_t<Q==U4, int> = 0>
    std::vector<KNNCandidate_t> SearchSymmetric(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        this->best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        quant.NormalizeQuery(raw_query);
        PreprocessQuery(quant.normalized_query, quant.transformed_raw_query);
        GetDimensionsAccessOrder(quant.transformed_raw_query, pdx_data.means);
        if (pdx_data.is_ivf) {
            if (ivf_nprobe == 0){
                vectorgroups_to_visit = pdx_data.num_vectorgroups;
            } else {
                vectorgroups_to_visit = ivf_nprobe;
            }
#ifdef BENCHMARK_TIME
            this->end_to_end_clock.Toc();
#endif
            //GetVectorgroupsAccessOrderIVFPDX(query, vectorgroups_to_visit, vectorgroups_indices);
            this->GetVectorgroupsAccessOrderIVF(quant.transformed_raw_query, pdx_data, ivf_nprobe, vectorgroups_indices);
#ifdef BENCHMARK_TIME
            this->end_to_end_clock.Tic();
#endif
        } else {
            // If there is no index, we just access the vectorgroups in order
            GetVectorgroupsAccessOrderRandom();
        }
        // PDXearch core
        current_dimension_idx = 0;
        current_vectorgroup = vectorgroups_indices[0];
        VECTORGROUP_TYPE& first_vectorgroup = pdx_data.vectorgroups[vectorgroups_indices[0]];
        quant.ScaleQuery(quant.transformed_raw_query);
        quant.PrepareQuery(first_vectorgroup.for_bases);
        Start(quant.quantized_query, first_vectorgroup.data, first_vectorgroup.num_embeddings, k, first_vectorgroup.indices);
        for (size_t vectorgroup_idx = 1; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VECTORGROUP_TYPE& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            quant.PrepareQuery(vectorgroup.for_bases);
            Warmup(quant.quantized_query, vectorgroup.data, vectorgroup.num_embeddings, k, selectivity_threshold, this->best_k);
            Prune(quant.quantized_query, vectorgroup.data, vectorgroup.num_embeddings, k, this->best_k);
            if (n_vectors_not_pruned){
                MergeIntoHeap<true>(vectorgroup.indices, n_vectors_not_pruned, k, this->best_k);
            }
        }
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return BuildResultSet(k);
    }


    // TODO: Full Linear Scans that do not prune vectors
//    template<Quantization Q=q, std::enable_if_t<Q==U4, int> = 0>
//    std::vector<KNNCandidate_t> LinearScan(float *__restrict raw_query, uint32_t k) {
//#ifdef BENCHMARK_TIME
//        this->ResetClocks();
//        this->end_to_end_clock.Tic();
//#endif
//        PreprocessQuery(raw_query, quant.transformed_raw_query);
//        PrepareQuery(transformed_raw_query, query);
//        this->best_k = std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
//        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
//        GetVectorgroupsAccessOrderRandom();
//        for (size_t vectorgroup_idx = 0; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
//            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
//            VECTORGROUP_TYPE & vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
//            if (vectorgroup.num_embeddings == PDX_VECTOR_SIZE){
//                ResetDistancesVectorized();
//                distance_computer::VerticalBlock(quant.quantized_query, vectorgroup.data, 0, pdx_data.num_dimensions, distances);
//                MergeIntoHeap<false>(vectorgroup.indices, PDX_VECTOR_SIZE, k, this->best_k);
//            } else if (vectorgroup.num_embeddings < PDX_VECTOR_SIZE) {
//                ResetDistancesVectorized();
//                distance_computer::Vertical(quant.quantized_query, vectorgroup.data, vectorgroup.num_embeddings, vectorgroup.num_embeddings,  0, pdx_data.num_dimensions, distances);
//                MergeIntoHeap<false>(vectorgroup.indices, vectorgroup.num_embeddings, k, this->best_k);
//            }
//        }
//#ifdef BENCHMARK_TIME
//        this->end_to_end_clock.Toc();
//#endif
//        return BuildResultSet(k);
//    }

};

} // namespace PDX

#endif //PDX_PDXEARCH_HPP

#ifndef EMBEDDINGSEARCH_PDXEARCH_HPP
#define EMBEDDINGSEARCH_PDXEARCH_HPP

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <queue>
#include <cassert>
#include <array>
#include "pdx/index_base/pdx_ivf.hpp"
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
class PDXearch: public VectorSearcher {
public:
    PDXearch(IndexPDXIVFFlat &data_index, float selectivity_threshold, size_t ivf_nprobe, int position_prune_distance,
             PDXearchDimensionsOrder dimension_order) :
            pdx_data(data_index),
            selectivity_threshold(selectivity_threshold),
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

    void SetSelectivityThreshold(float selectivityThreshold) {
        selectivity_threshold = selectivityThreshold;
    }

    void SetNProbe(size_t nprobe){
        ivf_nprobe = nprobe;
    }

    IndexPDXIVFFlat &pdx_data;
    uint32_t current_dimension_idx {0};

protected:
    float selectivity_threshold = 0.80;
    size_t ivf_nprobe = 0;
    int is_positional_pruning = false;
    size_t current_vectorgroup = 0;

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

    float pruning_threshold = std::numeric_limits<float>::max();

    std::vector<uint32_t> indices_dimensions;
    std::vector<uint32_t> vectorgroups_indices;

    size_t n_vectors_not_pruned = 0;

    static constexpr uint16_t PDX_VECTOR_SIZE = 64;
    alignas(64) inline static float distances[PDX_VECTOR_SIZE]; // Used in full scans (no pruning)

    // For pruning we do not use tight loops of 64. We know that tight loops bring benefits
    // to the distance kernels (40% faster), however doing so + PRUNING in the tight block of 64
    // slightly reduces the performance of PDXearch. We are still investigating why.
    alignas(64) inline static float pruning_distances[10240]; // TODO: Use dynamic arrays. Buckets with more than 10k vectors (rare) overflow
    alignas(64) inline static uint32_t pruning_positions[10240];
    alignas(64) inline static float pruning_distances_tmp[10240];

    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k_centroids;

    inline void ResetDistancesScalar(size_t n_vectors){
        memset((void*) distances, 0, n_vectors * sizeof(float));
    }

    inline virtual void ResetPruningDistances(size_t n_vectors){
        memset((void*) pruning_distances, 0, n_vectors * sizeof(float));
    }

    virtual void ResetDistancesVectorized(){
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
    }

    // We store centroids using PDX in tight blocks of 64
    void GetVectorgroupsAccessOrderIVFPDX(const float *__restrict query, size_t ivf_nprobe, std::vector<uint32_t> &vectorgroups_indices) {
        best_k_centroids = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
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
            CalculateVerticalDistancesVectorized<false, L2>(query, tmp_centroids_pdx, 0, pdx_data.num_dimensions);
            MergeIntoHeap<false>(tmp_vectorgroup_indices, PDX_VECTOR_SIZE, ivf_nprobe, best_k_centroids);
            tmp_vectorgroup_indices += PDX_VECTOR_SIZE;
            tmp_centroids_pdx += SKIPPING_SIZE;
        }
        if (remainder_block_size){
            memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
            CalculateVerticalDistancesScalar<false, L2>(query, tmp_centroids_pdx, remainder_block_size, 0, pdx_data.num_dimensions);
            MergeIntoHeap<false>(tmp_vectorgroup_indices, remainder_block_size, ivf_nprobe, best_k_centroids);
        }
        for (size_t i = 0; i < ivf_nprobe; ++i){
            const KNNCandidate& c = best_k_centroids.top();
            vectorgroups_indices[ivf_nprobe - i - 1] = c.index; // I need to inverse the allocation
            best_k_centroids.pop();
        }
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
    }

    // EXPERIMENTAL (not used in the publication)
    // In some datasets, we can apply the pruning idea to the IVF centroids.
    // The recall reduction is OK, but the speedup only happens if the number of buckets probed is around 20
    // Otherwise, algorithms have a hard time trying to prune.
    // This effect is partially explained in a recent paper (https://dl.acm.org/doi/pdf/10.1145/3709743)
    void GetVectorgroupsAccessOrderIVFPDXearch(const float *__restrict query, size_t ivf_nprobe, std::vector<uint32_t> &vectorgroups_indices) {
        best_k_centroids = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        vectorgroups_indices.resize(pdx_data.num_vectorgroups);
        std::iota(vectorgroups_indices.begin(), vectorgroups_indices.end(), 0);
        float * tmp_centroids_pdx = pdx_data.centroids_pdx;
        float * tmp_centroids = pdx_data.centroids;
        uint32_t * tmp_vectorgroup_indices = vectorgroups_indices.data();
        size_t SKIPPING_SIZE = PDX_VECTOR_SIZE * pdx_data.num_dimensions;
        size_t centroids_visited = 0;

        size_t remainder_block_size = pdx_data.num_vectorgroups % PDX_VECTOR_SIZE;
        size_t full_blocks = std::floor(1.0 * pdx_data.num_vectorgroups / PDX_VECTOR_SIZE); 

        const float TIGHT_SELECTIVITY_THRESHOLD = 0.8; // We push warmup to 90% due to dual-storage being present
        while (best_k_centroids.size() != ivf_nprobe){ // This while works since we cannot call an IVF index with more IVF_NPROBE than N_BUCKETS
            // TODO: Use another distances array for the centroids so I can use ResetDistancesVectorized() instead of memset
            memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
            CalculateVerticalDistancesVectorized<false, L2>(query, tmp_centroids_pdx, 0, pdx_data.num_dimensions);
            MergeIntoHeap<false>(tmp_vectorgroup_indices, PDX_VECTOR_SIZE, ivf_nprobe, best_k_centroids);
            tmp_vectorgroup_indices += PDX_VECTOR_SIZE;
            tmp_centroids_pdx += SKIPPING_SIZE;
            tmp_centroids += SKIPPING_SIZE;
            centroids_visited += PDX_VECTOR_SIZE;
            full_blocks -= 1;
        }
        if (centroids_visited < pdx_data.num_vectorgroups){
            for (size_t block_idx = 0; block_idx < full_blocks; ++block_idx){
                Warmup<L2>(query, tmp_centroids_pdx, PDX_VECTOR_SIZE, ivf_nprobe, TIGHT_SELECTIVITY_THRESHOLD, best_k_centroids);
                // Instead of trying to Prune, since we have dual-storage we just go directly to PlateauHorizontal
                GetPruningThreshold(ivf_nprobe, best_k_centroids);
                InitPositionsArray(PDX_VECTOR_SIZE);
                if (n_vectors_not_pruned){
                    CalculateDistancesNary(query, tmp_centroids);
                    MergeIntoHeap<true>(tmp_vectorgroup_indices, n_vectors_not_pruned, ivf_nprobe, best_k_centroids);
                }
                tmp_centroids_pdx += SKIPPING_SIZE;
                tmp_centroids += SKIPPING_SIZE;
                tmp_vectorgroup_indices += PDX_VECTOR_SIZE;
                centroids_visited += PDX_VECTOR_SIZE;
            }
            if (remainder_block_size){
                Warmup<L2>(query, tmp_centroids_pdx, remainder_block_size, ivf_nprobe, TIGHT_SELECTIVITY_THRESHOLD, best_k_centroids);
                GetPruningThreshold(ivf_nprobe, best_k_centroids);
                InitPositionsArray(remainder_block_size);
                if (n_vectors_not_pruned){
                    CalculateDistancesNary(query, tmp_centroids);
                    MergeIntoHeap<true>(tmp_vectorgroup_indices, n_vectors_not_pruned, ivf_nprobe, best_k_centroids);
                }
            }
        }
        for (size_t i = 0; i < ivf_nprobe; ++i){
            const KNNCandidate& c = best_k_centroids.top();
            vectorgroups_indices[ivf_nprobe - i - 1] = c.index; // I need to inverse the allocation
            best_k_centroids.pop();
        }
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
        n_vectors_not_pruned = 0;
    }

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

    void GetVectorgroupsAccessOrderRandom() {
        std::iota(vectorgroups_indices.begin(), vectorgroups_indices.end(), 0);
    }

    /******************************************************************
     * Distance computers. We have FOUR versions of distance calculations:
     * CalculateVerticalDistancesScalar: Using non-tight loops of any length
     * CalculateVerticalDistancesForPruning: Using non-tight loops of any length accumulating distances for pruning
     * CalculateVerticalDistancesVectorized: Using tight loops of 64 vectors
     * CalculateVerticalDistancesOnPositionsArray: Using non-tight loops on the array of not-yet pruned vectors
     ******************************************************************/
    template<bool USE_DIMENSIONS_REORDER, DistanceFunction L_ALPHA=ALPHA>
    void CalculateVerticalDistancesScalar(const float *__restrict query, const float *__restrict data, size_t n_vectors, size_t start_dimension, size_t end_dimension){
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            size_t offset_to_dimension_start = true_dimension_idx * n_vectors;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                if constexpr (L_ALPHA == L2){
                    float to_multiply = query[true_dimension_idx] - data[offset_to_dimension_start + vector_idx];
                    distances[vector_idx] += to_multiply * to_multiply;
                }
                if constexpr (L_ALPHA == IP){ // TODO: This is a special IP for BSA
                    distances[vector_idx] -= 2 * query[true_dimension_idx] * data[offset_to_dimension_start + vector_idx];
                }
            }
        }
    }

    // For pruning we do not use tight loops of PDX_VECTOR_SIZE
    template<bool USE_DIMENSIONS_REORDER, DistanceFunction L_ALPHA=ALPHA>
    void CalculateVerticalDistancesForPruning(const float *__restrict query, const float *__restrict data, size_t n_vectors, size_t total_vectors, size_t start_dimension, size_t end_dimension){
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                if constexpr (L_ALPHA == L2){
                    float to_multiply = query[true_dimension_idx] - data[offset_to_dimension_start + vector_idx];
                    pruning_distances[vector_idx] += to_multiply * to_multiply;
                }
                if constexpr (L_ALPHA == IP){ // TODO: This is a special IP for BSA
                    pruning_distances[vector_idx] -= 2 * query[true_dimension_idx] * data[offset_to_dimension_start + vector_idx];
                }
            }
        }
    }

    template<bool USE_DIMENSIONS_REORDER, DistanceFunction L_ALPHA=ALPHA>
    void CalculateVerticalDistancesVectorized(
        const float *__restrict query, const float *__restrict data, size_t start_dimension, size_t end_dimension){
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            size_t dimension_idx = dim_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                dimension_idx = indices_dimensions[dim_idx];
            }
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
                if constexpr (L_ALPHA == L2){
                    float to_multiply = query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
                    distances[vector_idx] += to_multiply * to_multiply;
                }
                if constexpr (L_ALPHA == IP){ // TODO: This is a special IP for BSA
                    distances[vector_idx] -= 2 * query[dimension_idx] * data[offset_to_dimension_start + vector_idx];
                }
            }
        }
    }

#if defined(__AVX512F__)
    void GatherDistances(size_t n_vectors){
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                auto true_vector_idx = pruning_positions[vector_idx];
                pruning_distances_tmp[vector_idx] = pruning_distances[true_vector_idx];
            }
    }

    template <bool USE_DIMENSIONS_REORDER, DistanceFunction L_ALPHA=L2>
    void GatherBasedKernel(
            const float *__restrict data, const float *__restrict query,
            size_t n_vectors, size_t total_vectors, size_t start_dimension, size_t end_dimension){
        GatherDistances(n_vectors);
        __m512 data_vec, d_vec, cur_dist_vec;
        __m256 data_vec_m256, d_vec_m256, cur_dist_vec_m256;
        // Then we move data to be sequential
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER) {
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            __m512 query_vec;
            if constexpr (L_ALPHA == L2) {
                query_vec = _mm512_set1_ps(query[true_dimension_idx]);
            }
            if constexpr (L_ALPHA == IP){
                query_vec = _mm512_set1_ps(-2 * query[true_dimension_idx]);
            }
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            const float * tmp_data = data + offset_to_dimension_start;
            // Now we do the sequential distance calculation loop which would use SIMD
            // Up to 16
            size_t i = 0;
            for (; i + 16 < n_vectors; i+=16) {
                cur_dist_vec = _mm512_load_ps(&pruning_distances_tmp[i]);
                data_vec = _mm512_i32gather_ps(
                        _mm512_load_epi32(&pruning_positions[i]),
                        tmp_data, sizeof(float)
                );
                if constexpr (L_ALPHA == L2) {
                    d_vec = _mm512_sub_ps(data_vec, query_vec);
                    cur_dist_vec = _mm512_fmadd_ps(d_vec, d_vec, cur_dist_vec);
                }
                if constexpr (L_ALPHA == IP) { // TODO: This is a special IP for BSA
                    cur_dist_vec = _mm512_fmadd_ps(data_vec, query_vec, cur_dist_vec);
                }
                _mm512_store_ps(&pruning_distances_tmp[i], cur_dist_vec);
            }
            __m256 query_vec_m256;
            if constexpr (L_ALPHA == L2) {
                query_vec_m256 = _mm256_set1_ps(query[true_dimension_idx]);
            }
            if constexpr (L_ALPHA == IP){
                query_vec_m256 = _mm256_set1_ps(-2 * query[true_dimension_idx]);
            }
            // Up to 8
            for (; i + 8 < n_vectors; i+=8) {
                cur_dist_vec_m256 = _mm256_load_ps(&pruning_distances_tmp[i]);
                data_vec_m256 = _mm256_i32gather_ps(
                        tmp_data, _mm256_load_epi32(&pruning_positions[i]),
                        sizeof(float)
                        );
                if constexpr (L_ALPHA == L2) {
                    d_vec_m256 = _mm256_sub_ps(data_vec_m256, query_vec_m256);
                    cur_dist_vec_m256 = _mm256_fmadd_ps(d_vec_m256, d_vec_m256, cur_dist_vec_m256);
                }
                if constexpr (L_ALPHA == IP) { // TODO: This is a special IP for BSA
                    cur_dist_vec_m256 = _mm256_fmadd_ps(data_vec_m256, query_vec_m256, cur_dist_vec_m256);
                }
                _mm256_store_ps(&pruning_distances_tmp[i], cur_dist_vec_m256);
            }
            // TODO: Up to 4
            // Tail
            for (; i < n_vectors; i++){
                if constexpr (L_ALPHA == L2) {
                    float to_multiply = query[true_dimension_idx] - tmp_data[pruning_positions[i]];
                    pruning_distances_tmp[i] += to_multiply * to_multiply;
                }
                if constexpr (L_ALPHA == IP) { // TODO: This is a special IP for BSA
                    pruning_distances_tmp[i] -= 2 * query[true_dimension_idx] * tmp_data[pruning_positions[i]];
                }
            }
        }
        // We now move distances back
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            auto true_vector_idx = pruning_positions[vector_idx];
            pruning_distances[true_vector_idx] = pruning_distances_tmp[vector_idx];
        }
    }
#endif

    template<bool USE_DIMENSIONS_REORDER, DistanceFunction L_ALPHA=ALPHA>
    void CalculateVerticalDistancesOnPositionsArray(const float *__restrict query, const float *__restrict data, size_t n_vectors, size_t total_vectors, size_t start_dimension, size_t end_dimension){
#if defined(__AVX512F__) && defined(__AVX512FP16__)
        // SIMD is less efficient when looping on the array of not-yet pruned vectors
        // A way to improve the performance by ~20% is using a GATHER intrinsic. However this only works on Intel microarchs.
        // In AMD (Zen 4, Zen 3) using a GATHER is shooting ourselves in the foot (~80 uops)
	    // __AVX512FP16__ macro let us detect Intel architectures (from Sapphire Rapids onwards)
        if (n_vectors >= 8) {
            GatherBasedKernel<USE_DIMENSIONS_REORDER, L_ALPHA>(data, query, n_vectors, total_vectors, start_dimension, end_dimension);
            return;
        }
#endif
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                auto true_vector_idx = pruning_positions[vector_idx];
                if constexpr (L_ALPHA == L2) {
                    float to_multiply =
                            query[true_dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
                    pruning_distances[true_vector_idx] += to_multiply * to_multiply;
                }
                if constexpr (L_ALPHA == IP) { // TODO: This is a special IP for BSA
                    pruning_distances[true_vector_idx] -=
                            2 * query[true_dimension_idx] * data[offset_to_dimension_start + true_vector_idx];
                }
            }
        }
    }

    virtual void EvaluatePruningPredicateOnPositionsArray(size_t n_vectors){
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
            n_vectors_not_pruned += pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
        }
    };

    virtual void EvaluatePruningPredicateScalar(uint32_t &n_pruned, size_t n_vectors) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
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
        }
    };

    // On the first bucket, we do a full scan (we do not prune vectors)
    void Start(const float *__restrict query, const float * data, const size_t n_vectors, uint32_t k, const uint32_t * vector_indices) {
        ResetPruningDistances(n_vectors);
        CalculateVerticalDistancesForPruning<false>(query, data, n_vectors, n_vectors, 0, pdx_data.num_dimensions); 
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
    void Warmup(const float *__restrict query, const float * data, const size_t n_vectors, uint32_t k, float tuples_threshold, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap) {
        current_dimension_idx = 0;
        cur_subgrouping_size_idx = 0;
        size_t tuples_needed_to_exit = std::ceil(1.0 * tuples_threshold * n_vectors);
        ResetPruningDistances(n_vectors);
        uint32_t n_tuples_to_prune = 0;
        if (!is_positional_pruning) GetPruningThreshold(k, heap);

        while (1.0 * n_tuples_to_prune < tuples_needed_to_exit && current_dimension_idx < pdx_data.num_dimensions) {
            size_t last_dimension_to_fetch = std::min(current_dimension_idx + DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
                                                     pdx_data.num_dimensions);
            if (dimension_order == SEQUENTIAL){
                CalculateVerticalDistancesForPruning<false, L_ALPHA>(query, data, n_vectors, n_vectors, current_dimension_idx,
                                                    last_dimension_to_fetch);
            } else {
                CalculateVerticalDistancesForPruning<true, L_ALPHA>(query, data, n_vectors, n_vectors, current_dimension_idx,
                                                    last_dimension_to_fetch);
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
    void Prune(const float *__restrict query, const float *__restrict data, const size_t n_vectors, uint32_t k, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap) {
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
                CalculateVerticalDistancesOnPositionsArray<false, L_ALPHA>(query, data, cur_n_vectors_not_pruned,
                                                    n_vectors, current_dimension_idx,
                                                    last_dimension_to_test_idx);
            } else {
                CalculateVerticalDistancesOnPositionsArray<true, L_ALPHA>(query, data, cur_n_vectors_not_pruned,
                                                    n_vectors, current_dimension_idx,
                                                    last_dimension_to_test_idx);                
            }

            current_dimension_idx = last_dimension_to_test_idx;
            if (is_positional_pruning) GetPruningThreshold(k, heap);
            EvaluatePruningPredicateOnPositionsArray(cur_n_vectors_not_pruned);
            if (current_dimension_idx == pdx_data.num_dimensions) break;
        }
    }

    // EXPERIMENTAL (not used in the publication)
    // If a dual-storage (both PDX and Nary) is present, then one can optimize the PRUNE phase by using the Nary
    // storage, in which SIMD can be leveraged more efficiently on this phase.
    void CalculateDistancesNary(const float *__restrict query, const float * data) {
        size_t starting_dimension;
        if (dimension_order == SEQUENTIAL) {
            starting_dimension = current_dimension_idx;
        } else {
            starting_dimension = 0;
            ResetPruningDistances(PDX_VECTOR_SIZE); // TODO: This can be further optimized
        }
        size_t num_dimensions = pdx_data.num_dimensions - starting_dimension;
        for (size_t position_idx = 0; position_idx < n_vectors_not_pruned; ++position_idx) {
            size_t embedding_idx = pruning_positions[position_idx];
            // TODO: This should be changed to implement the distance metric
            pruning_distances[embedding_idx] += CalculateDistanceL2(query, data + (embedding_idx * pdx_data.num_dimensions), num_dimensions);
        }
    }

    template <bool IS_PRUNING=false>
    void MergeIntoHeap(const uint32_t * vector_indices, size_t n_vectors, uint32_t k, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap) {
        for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
            size_t index = position_idx;
            float current_distance;
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

    // The pruning threshold by default is the top of the heap
    virtual void GetPruningThreshold(uint32_t k, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap){
        pruning_threshold = heap.size() == k ? heap.top().distance : std::numeric_limits<float>::max();
    };

    virtual void PreprocessQuery(float * raw_query, float *query){};

public:
    /******************************************************************
     * Search methods
     ******************************************************************/

    // PDXearch: PDX + Pruning
    std::vector<KNNCandidate> Search(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        alignas(64) float query[pdx_data.num_dimensions];
        PreprocessQuery(raw_query, query);
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        GetDimensionsAccessOrder(query, pdx_data.means);
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
            GetVectorgroupsAccessOrderIVFPDX(query, vectorgroups_to_visit, vectorgroups_indices);
#ifdef BENCHMARK_TIME
            end_to_end_clock.Tic();
#endif
            //GetVectorgroupsAccessOrderIVF(query, pdx_data, ivf_nprobe, vectorgroups_indices);
        } else {
            // If there is no index, we just access the vectorgroups in order
            GetVectorgroupsAccessOrderRandom();
        }
        // PDXearch core
        current_dimension_idx = 0;
        current_vectorgroup = vectorgroups_indices[0];
        Vectorgroup& first_vectorgroup = pdx_data.vectorgroups[vectorgroups_indices[0]];
        Start(query, first_vectorgroup.data, first_vectorgroup.num_embeddings, k, first_vectorgroup.indices);
        for (size_t vectorgroup_idx = 1; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            Vectorgroup& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
            Warmup(query, vectorgroup.data, vectorgroup.num_embeddings, k, selectivity_threshold, best_k);
            Prune(query, vectorgroup.data, vectorgroup.num_embeddings, k, best_k);
            if (n_vectors_not_pruned){
                MergeIntoHeap<true>(vectorgroup.indices, n_vectors_not_pruned, k, best_k);
            }
        }
#ifdef BENCHMARK_TIME
        end_to_end_clock.Toc();
#endif
        return BuildResultSet(k);
    }

    // Full Linear Scans that do not prune vectors
    std::vector<KNNCandidate> LinearScan(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        float query[pdx_data.num_dimensions];
        PreprocessQuery(raw_query, query);
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        GetVectorgroupsAccessOrderRandom();
        for (size_t vectorgroup_idx = 0; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            Vectorgroup& vectorgroup = pdx_data.vectorgroups[current_vectorgroup]; 
            if (vectorgroup.num_embeddings == PDX_VECTOR_SIZE){
                ResetDistancesVectorized();
                CalculateVerticalDistancesVectorized<false, ALPHA>(query, vectorgroup.data, 0, pdx_data.num_dimensions);
                MergeIntoHeap<false>(vectorgroup.indices, PDX_VECTOR_SIZE, k, best_k);
            } else if (vectorgroup.num_embeddings < PDX_VECTOR_SIZE) {
                ResetDistancesVectorized();
                CalculateVerticalDistancesScalar<false, ALPHA>(query, vectorgroup.data, vectorgroup.num_embeddings,  0, pdx_data.num_dimensions);
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

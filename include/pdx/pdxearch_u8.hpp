#ifndef EMBEDDINGSEARCH_PDXEARCH_U8_HPP
#define EMBEDDINGSEARCH_PDXEARCH_U8_HPP

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include "arm_neon.h"
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
class PDXearchU8: public VectorSearcher {
public:
    PDXearchU8(IndexPDXIVFFlatU8 &data_index, size_t ivf_nprobe, int position_prune_distance,
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

    IndexPDXIVFFlatU8 &pdx_data;
    uint32_t current_dimension_idx {0};

protected:
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

    std::vector<uint32_t> indices_dimensions;
    std::vector<uint32_t> vectorgroups_indices;

    size_t n_vectors_not_pruned = 0;

    static constexpr uint16_t PDX_VECTOR_SIZE = 64;
    alignas(64) inline static uint32_t distances[PDX_VECTOR_SIZE]; // Used in full scans (no pruning)


    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k_centroids;

    inline void ResetDistancesScalar(size_t n_vectors){
        memset((void*) distances, 0, n_vectors * sizeof(uint32_t));
    }

    virtual void ResetDistancesVectorized(){
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(uint32_t));
    }

    /******************************************************************
     * Distance computers. We have FOUR versions of distance calculations:
     * CalculateVerticalDistancesScalar: Using non-tight loops of any length
     * CalculateVerticalDistancesForPruning: Using non-tight loops of any length accumulating distances for pruning
     * CalculateVerticalDistancesVectorized: Using tight loops of 64 vectors
     * CalculateVerticalDistancesOnPositionsArray: Using non-tight loops on the array of not-yet pruned vectors
     ******************************************************************/
    template<bool USE_DIMENSIONS_REORDER, DistanceFunction L_ALPHA=ALPHA>
    void CalculateVerticalDistancesScalar(const uint8_t *__restrict query, const uint8_t *__restrict data, size_t n_vectors, size_t start_dimension, size_t end_dimension){
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
    }

    template<bool USE_DIMENSIONS_REORDER, DistanceFunction L_ALPHA=ALPHA>
    void CalculateVerticalDistancesVectorized(
        const uint8_t *__restrict query, const uint8_t *__restrict data, size_t start_dimension, size_t end_dimension){
#if false
        //Auto-vectorized: 3.24 (uint8) vs 4.13 (float32)
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
// SIMD Upcast: 5.66819 (uint8) vs 4.17 (float32)
#elif defined(__ARM_NEON) && false
        uint32x4_t res[16];
        // Load initial values
        for (size_t i = 0; i < 16; ++i) {
            res[i] = vld1q_u32(&distances[i * 4]);
        }
        // Compute squared differences
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            uint32_t dimension_idx = dim_idx;
            uint8x16_t vec1_u8 = vdupq_n_u8(query[dimension_idx]);
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (int i = 0; i < 16; i+=4) {
                // Read 16 bytes of data (16 values)
                uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 4]);
                uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);

                // Upcast to uint16_t
                uint16x8_t diff_low = vmovl_u8(vget_low_u8(diff_u8));
                uint16x8_t diff_high = vmovl_u8(vget_high_u8(diff_u8));

                // Square the differences
                uint16x8_t diff_low_sq = vmulq_u16(diff_low, diff_low);
                uint16x8_t diff_high_sq = vmulq_u16(diff_high, diff_high);

                // Upcast the 16 bits to 32 bits
                uint32x4_t m_low_low = vmovl_u16(vget_low_u16(diff_low_sq));
                uint32x4_t m_low_high = vmovl_u16(vget_high_u16(diff_low_sq));

                uint32x4_t m_high_low = vmovl_u16(vget_low_u16(diff_high_sq));
                uint32x4_t m_high_high = vmovl_u16(vget_high_u16(diff_high_sq));

                // Since I am reading 16 values at a time, I can fill 4 registers in one go
                // Each register
                res[i] = vaddq_u32(res[i], m_low_low);
                res[i+1] = vaddq_u32(res[i+1], m_low_high);
                res[i+2] = vaddq_u32(res[i+2], m_high_low);
                res[i+3] = vaddq_u32(res[i+3], m_high_high);
                //std::cout << i << "\n";
            }
        }
        // Store results back
        for (int i = 0; i < 16; ++i) {
            vst1q_u32(&distances[i * 4], res[i]);
        }
// SIMD Dot Product: 5.66819 (uint8) vs 4.17 (float32)
#elif defined(__ARM_NEON) && true
        uint32x4_t res[16];
        // Load initial values
        for (size_t i = 0; i < 16; ++i) {
            res[i] = vdupq_n_u32(0);
        }
        // Compute squared differences
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            uint32_t dimension_idx = dim_idx;
            uint8x16_t vec1_u8 = vdupq_n_u8(query[dimension_idx]);
            vec1_u8 = vsetq_lane_u8(query[dimension_idx] + 1, vec1_u8, 1);
            vec1_u8 = vsetq_lane_u8(query[dimension_idx] + 2, vec1_u8, 2);
            vec1_u8 = vsetq_lane_u8(query[dimension_idx] + 3, vec1_u8, 3);
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (int i = 0; i < 16; ++i) {
                // Read 16 bytes of data (16 values)
                uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 16]);
                uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
                res[i] = vdotq_u32(res[i], diff_u8, diff_u8);
            }
        }
        // Store results back
        for (int i = 0; i < 16; ++i) {
            vst1q_u32(&distances[i * 4], res[i]);
        }
#endif
    }

    template <bool IS_PRUNING=false>
    void MergeIntoHeap(const uint32_t * vector_indices, size_t n_vectors, uint32_t k, std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap) {
        for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
            size_t index = position_idx;
            uint32_t current_distance;
            current_distance = distances[index];
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

    void PreprocessQuery(float * raw_query, uint8_t *query){
        for (size_t i = 0; i < pdx_data.num_dimensions; i++){
            query[i] = static_cast<uint8_t>(raw_query[i]);
        }
    };

public:
    /******************************************************************
     * Search methods
     ******************************************************************/

    // Full Linear Scans that do not prune vectors
    std::vector<KNNCandidate> LinearScan(float *__restrict raw_query, uint32_t k) {
#ifdef BENCHMARK_TIME
        ResetClocks();
        end_to_end_clock.Tic();
#endif
        uint8_t query[pdx_data.num_dimensions];
        PreprocessQuery(raw_query, query);
        best_k = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>{};
        size_t vectorgroups_to_visit = pdx_data.num_vectorgroups;
        GetVectorgroupsAccessOrderRandom();
        for (size_t vectorgroup_idx = 0; vectorgroup_idx < vectorgroups_to_visit; ++vectorgroup_idx) {
            current_vectorgroup = vectorgroups_indices[vectorgroup_idx];
            VectorgroupU8& vectorgroup = pdx_data.vectorgroups[current_vectorgroup];
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

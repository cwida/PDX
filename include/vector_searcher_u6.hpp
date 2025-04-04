#ifndef EMBEDDINGSEARCH_VECTOR_SEARCHER_U6_HPP
#define EMBEDDINGSEARCH_VECTOR_SEARCHER_U6_HPP

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <vector>
#include <cinttypes>
#include "pdx/index_base/pdx_ivf.hpp"
#include "utils/tictoc.hpp"

//struct KNNCandidate {
//    uint32_t index;
//    float distance;
//};
//
//struct VectorComparator {
//    bool operator() (const KNNCandidate& a, const KNNCandidate& b) {
//        return a.distance < b.distance;
//    }
//};

/******************************************************************
 * Base vector searcher class.
 * Contains basic distance calculations for L2 (ADSampling and BOND) and IP (for BSA)
 * in NEON, AVX512 and AVX2. All in the Nary layout.
 * Also contains a method to find the nearest bucket on IVF indexes
 * TODO: Support SVE
 ******************************************************************/
class VectorSearcherU6 {
public:
    TicToc end_to_end_clock = TicToc();

    void ResetClocks(){
        end_to_end_clock.Reset();
    }


protected:
    // These functions are not used on the Python bindings with the PDX indexes
    // However, I need to be careful to not use them as PDX_USE_EXPLICIT_SIMD is not defined within the bindings
    static uint32_t CalculateDistanceL2(const uint8_t *__restrict vector1, const uint8_t *__restrict vector2, size_t num_dimensions) {
#if defined(__ARM_NEON)
        float32x4_t sum_vec = vdupq_n_f32(0);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            uint8x16_t a_vec = vld1q_u8(vector1 + i);
            uint8x16_t b_vec = vld1q_u8(vector2 + i);
            uint8x16_t d_vec = vabdq_u8(a_vec, b_vec);
            sum_vec = vdotq_u32(sum_vec, d_vec, d_vec);
        }
        uint32_t distance = vaddvq_u32(sum_vec);
        for (; i < num_dimensions; ++i) {
            int n = (int)vector1[i] - vector2[i];
            distance += n * n;
        }
        return distance;
#elif defined(PDX_USE_EXPLICIT_SIMD) && defined(__AVX512F__)

#elif defined(PDX_USE_EXPLICIT_SIMD) && defined(__AVX2__)

#else
        uint32_t distance = 0.0;
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            int to_multiply = (int)vector1[dimension_idx] - vector2[dimension_idx];
            distance += to_multiply * to_multiply;
        }
        return distance;
#endif
    }

    // These functions are not used on the Python bindings with the PDX indexes
    // However, I need to be careful to not use them as PDX_USE_EXPLICIT_SIMD is not defined within the bindings
    static uint32_t CalculateDistanceIP(const uint8_t *__restrict vector1, const uint8_t *__restrict vector2, size_t num_dimensions){
#if defined(PDX_USE_EXPLICIT_SIMD) && defined(__ARM_NEON)

#elif defined(PDX_USE_EXPLICIT_SIMD) && defined(__AVX512F__)

#elif defined(PDX_USE_EXPLICIT_SIMD) && defined(__AVX2__)

#else
        float distance = 0.0;
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance += vector1[dimension_idx] * vector2[dimension_idx];
        }
        return distance;
#endif
    }

    /*
    static void GetVectorgroupsAccessOrderIVF(const float *__restrict query, const PDX::IndexPDXIVFFlat &data, size_t ivf_nprobe, std::vector<uint32_t> &vectorgroups_indices) {
        std::vector<float> distances_to_centroids;
        distances_to_centroids.resize(data.num_vectorgroups);
        for (size_t vectorgroup_idx = 0; vectorgroup_idx < data.num_vectorgroups; vectorgroup_idx++) {
            // TODO: From query to centroids we only support L2
            distances_to_centroids[vectorgroup_idx] = CalculateDistanceL2(query,
                                                                          data.centroids +
                                                                          vectorgroup_idx *
                                                                          data.num_dimensions,
                                                                          data.num_dimensions);
        }
        vectorgroups_indices.resize(data.num_vectorgroups);
        std::iota(vectorgroups_indices.begin(), vectorgroups_indices.end(), 0);
        std::partial_sort(vectorgroups_indices.begin(), vectorgroups_indices.begin() + ivf_nprobe, vectorgroups_indices.end(),
                          [&distances_to_centroids](size_t i1, size_t i2) {
                              return distances_to_centroids[i1] < distances_to_centroids[i2];
                          }
        );
    }
     */

};

#endif //EMBEDDINGSEARCH_VECTOR_SEARCHER_HPP

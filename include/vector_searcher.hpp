#ifndef EMBEDDINGSEARCH_VECTOR_SEARCHER_HPP
#define EMBEDDINGSEARCH_VECTOR_SEARCHER_HPP

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

struct KNNCandidate {
    uint32_t index;
    float distance;
};

struct VectorComparator {
    bool operator() (const KNNCandidate& a, const KNNCandidate& b) {
        return a.distance < b.distance;
    }
};

/******************************************************************
 * Base vector searcher class.
 * Contains basic distance calculations for L2 (ADSampling and BOND) and IP (for BSA)
 * in NEON, AVX512 and AVX2. All in the Nary layout.
 * Also contains a method to find the nearest bucket on IVF indexes
 * TODO: Support SVE
 ******************************************************************/
class VectorSearcher {
public:
    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k;
    TicToc end_to_end_clock = TicToc();

    void ResetClocks(){
        end_to_end_clock.Reset();
    }

protected:
    // These functions are not used on the Python bindings with the PDX indexes
    // However, I need to be careful to not use them as PDX_USE_EXPLICIT_SIMD is not defined within the bindings
    static float CalculateDistanceL2(const float *__restrict vector1, const float *__restrict vector2, size_t num_dimensions) {
#if defined(PDX_USE_EXPLICIT_SIMD) && defined(__ARM_NEON)
        float32x4_t sum_vec = vdupq_n_f32(0);
        size_t i = 0;
        for (; i + 4 <= num_dimensions; i += 4) {
            float32x4_t a_vec = vld1q_f32(vector1 + i);
            float32x4_t b_vec = vld1q_f32(vector2 + i);
            float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
            sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
        }
        float distance = vaddvq_f32(sum_vec);
        for (; i < num_dimensions; ++i) {
            float diff = vector1[i] - vector2[i];
            distance += diff * diff;
        }
        return distance;
#elif defined(PDX_USE_EXPLICIT_SIMD) && defined(__AVX512F__)
        __m512 d2_vec = _mm512_setzero();
        __m512 a_vec, b_vec;
        simsimd_l2sq_f32_skylake_cycle:
            if (num_dimensions < 16) {
                __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, num_dimensions);
                a_vec = _mm512_maskz_loadu_ps(mask, vector1);
                b_vec = _mm512_maskz_loadu_ps(mask, vector2);
                num_dimensions = 0;
            } else {
                a_vec = _mm512_loadu_ps(vector1);
                b_vec = _mm512_loadu_ps(vector2);
                vector1 += 16, vector2 += 16, num_dimensions -= 16;
            }
            __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
            d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
        if (num_dimensions)
            goto simsimd_l2sq_f32_skylake_cycle;
        
        // _simsimd_reduce_f32x16_skylake
        __m512 x = _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
#elif defined(PDX_USE_EXPLICIT_SIMD) && defined(__AVX2__)
        __m256 d2_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= num_dimensions; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(vector1 + i);
            __m256 b_vec = _mm256_loadu_ps(vector2 + i);
            __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
            d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
        }

        // _simsimd_reduce_f32x8_haswell
        // Convert the lower and higher 128-bit lanes of the input vector to double precision
        __m128 low_f32 = _mm256_castps256_ps128(d2_vec);
        __m128 high_f32 = _mm256_extractf128_ps(d2_vec, 1);

        // Convert single-precision (float) vectors to double-precision (double) vectors
        __m256d low_f64 = _mm256_cvtps_pd(low_f32);
        __m256d high_f64 = _mm256_cvtps_pd(high_f32);

        // Perform the addition in double-precision
        __m256d sum = _mm256_add_pd(low_f64, high_f64);

        // Reduce the double-precision vector to a scalar
        // Horizontal add the first and second double-precision values, and third and fourth
        __m128d sum_low = _mm256_castpd256_pd128(sum);
        __m128d sum_high = _mm256_extractf128_pd(sum, 1);
        __m128d sum128 = _mm_add_pd(sum_low, sum_high);

        // Horizontal add again to accumulate all four values into one
        sum128 = _mm_hadd_pd(sum128, sum128);

        // Convert the final sum to a scalar double-precision value and return
        double d2 = _mm_cvtsd_f64(sum128);

        for (; i < num_dimensions; ++i) {
            float d = vector1[i] - vector2[i];
            d2 += d * d;
        }

        return static_cast<float>(d2);        
#else
        float distance = 0.0;
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            float to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
            distance += to_multiply * to_multiply;
        }
        return distance;
#endif
    }

    // These functions are not used on the Python bindings with the PDX indexes
    // However, I need to be careful to not use them as PDX_USE_EXPLICIT_SIMD is not defined within the bindings
    static float CalculateDistanceIP(const float *__restrict vector1, const float *__restrict vector2, size_t num_dimensions){
#if defined(PDX_USE_EXPLICIT_SIMD) && defined(__ARM_NEON)
        float32x4_t sum_vec = vdupq_n_f32(0);
        size_t i = 0;
        for (; i + 4 <= num_dimensions; i += 4) {
            float32x4_t a_vec = vld1q_f32(vector1 + i);
            float32x4_t b_vec = vld1q_f32(vector2 + i);
            sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
        }
        float distance = vaddvq_f32(sum_vec);
        for (; i < num_dimensions; ++i) {
            distance += vector1[i] * vector2[i];
        }
        return distance;
#elif defined(PDX_USE_EXPLICIT_SIMD) && defined(__AVX512F__)
        __m512 d2_vec = _mm512_setzero();
        __m512 a_vec, b_vec;

simsimd_ip_f32_skylake_cycle:
            if (num_dimensions < 16) {
                __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, num_dimensions);
                a_vec = _mm512_maskz_loadu_ps(mask, vector1);
                b_vec = _mm512_maskz_loadu_ps(mask, vector2);
                num_dimensions = 0;
            } else {
                a_vec = _mm512_loadu_ps(vector1);
                b_vec = _mm512_loadu_ps(vector2);
                vector1 += 16, vector2 += 16, num_dimensions -= 16;
            }
            d2_vec = _mm512_fmadd_ps(a_vec, b_vec, d2_vec);
        if (num_dimensions)
            goto simsimd_ip_f32_skylake_cycle;

        // _simsimd_reduce_f32x16_skylake
        __m512 x = _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
#elif defined(PDX_USE_EXPLICIT_SIMD) && defined(__AVX2__)
        __m256 d2_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= num_dimensions; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(vector1 + i);
            __m256 b_vec = _mm256_loadu_ps(vector2 + i);
            d2_vec = _mm256_fmadd_ps(a_vec, b_vec, d2_vec);
        }

        // _simsimd_reduce_f32x8_haswell
        // Convert the lower and higher 128-bit lanes of the input vector to double precision
        __m128 low_f32 = _mm256_castps256_ps128(d2_vec);
        __m128 high_f32 = _mm256_extractf128_ps(d2_vec, 1);

        // Convert single-precision (float) vectors to double-precision (double) vectors
        __m256d low_f64 = _mm256_cvtps_pd(low_f32);
        __m256d high_f64 = _mm256_cvtps_pd(high_f32);

        // Perform the addition in double-precision
        __m256d sum = _mm256_add_pd(low_f64, high_f64);

        // Reduce the double-precision vector to a scalar
        // Horizontal add the first and second double-precision values, and third and fourth
        __m128d sum_low = _mm256_castpd256_pd128(sum);
        __m128d sum_high = _mm256_extractf128_pd(sum, 1);
        __m128d sum128 = _mm_add_pd(sum_low, sum_high);

        // Horizontal add again to accumulate all four values into one
        sum128 = _mm_hadd_pd(sum128, sum128);

        // Convert the final sum to a scalar double-precision value and return
        double d2 = _mm_cvtsd_f64(sum128);

        for (; i < num_dimensions; ++i) {
            d2 += vector1[i] * vector2[i];
        }
        return static_cast<float>(d2);
#else
        float distance = 0.0;
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance += vector1[dimension_idx] * vector2[dimension_idx];
        }
        return distance;
#endif
    }

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

    // U8 override: todo: template
    static void GetVectorgroupsAccessOrderIVF(const float *__restrict query, const PDX::IndexPDXIVFFlatU8 &data, size_t ivf_nprobe, std::vector<uint32_t> &vectorgroups_indices) {
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

};

#endif //EMBEDDINGSEARCH_VECTOR_SEARCHER_HPP

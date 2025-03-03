#ifdef __ARM_NEON
#include "arm_neon.h"
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <iostream>
#include <string>
#include "utils/file_reader.hpp"
#include "pdx/bond.hpp"
#include "utils/benchmark_utils.hpp"

/******************************************************************
 * Standalone scanners for float32 L2, IP and L1
 * NOT used for pruned PDXearch but on the linear scans experiment
 * PDXScanner: Vertical distance kernels that auto-vectorize efficiently for float32
 *      Accepts the block size of PDX as a template parameter
 * SIMDScanner: Horizontal distance calculations with SIMD kernels
 *      Taken from USearch and FAISS
 * ScalarScanner: Scalar kernels without SIMD
 * ImpreciseScanner: Scalar loops forcing auto-vectorization with pragmas
 * TODO: Support SVE. Initially we did not implement it as for other workloads NEON > SVE.
 ******************************************************************/
template<PDX::DistanceFunction ALPHA=PDX::L2, size_t BLOCK_SIZE=64>
class PDXScanner {
public:
    static constexpr uint16_t PDX_VECTOR_SIZE = 64;

    alignas(64) inline static float distances[BLOCK_SIZE];

    static void ResetDistances(){
        memset((void*) distances, 0, BLOCK_SIZE * sizeof(float));
    }

    static void CalculateVerticalDistancesVectorized(
            const float *__restrict query, const float *__restrict data, size_t end_dimension){
        for (size_t dim_idx = 0; dim_idx < end_dimension; dim_idx++) {
            size_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * BLOCK_SIZE;
            for (size_t vector_idx = 0; vector_idx < BLOCK_SIZE; ++vector_idx) {
                if constexpr (ALPHA == PDX::L2){
                    float to_multiply = query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
                    distances[vector_idx] += to_multiply * to_multiply;
                }
                if constexpr (ALPHA == PDX::IP){
                    distances[vector_idx] += query[dimension_idx] * data[offset_to_dimension_start + vector_idx];
                }
                if constexpr (ALPHA == PDX::L1){
                    float to_abs = query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
                    distances[vector_idx] += std::fabs(to_abs);
                }
            }
        }
    }
};

template<PDX::DistanceFunction ALPHA=PDX::L2>
class SIMDScanner {

public:

#if defined(__AVX2__) || defined(__AVX512F__)
    // https://github.com/facebookresearch/faiss/blob/697b6ddf558ef4ecb60e72e828c25a69723639c1/faiss/utils/distances_simd.cpp#L309
    static inline __m128 masked_read(int d, const float* x) {
        assert(0 <= d && d < 4);
        alignas(16) float buf[4] = {0, 0, 0, 0};
        switch (d) {
            case 3:
                buf[2] = x[2];
                [[fallthrough]];
            case 2:
                buf[1] = x[1];
                [[fallthrough]];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps(buf);
        // cannot use AVX2 _mm_mask_set1_epi32
    }
#elif defined(__ARM_NEON)
    // https://github.com/facebookresearch/faiss/blob/697b6ddf558ef4ecb60e72e828c25a69723639c1/faiss/utils/distances_simd.cpp#L309
    static inline float32x4_t masked_read(int d, const float* x) {
        assert(0 <= d && d < 4);
        alignas(16) float buf[4] = {0, 0, 0, 0};
        switch (d) {
            case 3:
                buf[2] = x[2];
                [[fallthrough]];
            case 2:
                buf[1] = x[1];
                [[fallthrough]];
            case 1:
                buf[0] = x[0];
        }
        return vld1q_f32(buf);
    }
#endif

    static float CalculateDistance(const float *__restrict vector1, const float *__restrict vector2, size_t num_dimensions) {
#if defined(__ARM_NEON)
        if constexpr(ALPHA == PDX::L2){
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
        }
        if constexpr(ALPHA == PDX::IP){ // simsimd_dot_f32_neon
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
        }
        if constexpr(ALPHA == PDX::L1){
            size_t d = num_dimensions;
            float32x4_t sum_vec = vdupq_n_f32(0);

            // Process 4 elements at a time
            while (d >= 4) {
                float32x4_t mx = vld1q_f32(vector1);
                vector1 += 4;
                float32x4_t my = vld1q_f32(vector2);
                vector2 += 4;
                sum_vec = vaddq_f32(sum_vec, vabsq_f32(vsubq_f32(mx, my)));
                d -= 4;
            }

            // Tail management as in FAISS
            if (d > 0) {
                float32x4_t mx = masked_read(d, vector1);
                float32x4_t my = masked_read(d, vector2);
                sum_vec = vaddq_f32(sum_vec, vabsq_f32(vsubq_f32(mx, my)));
            }
            float distance = vaddvq_f32(sum_vec);
            return distance;
        }
#elif defined(__AVX512F__)
        if constexpr(ALPHA == PDX::L2){
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
    }
    if constexpr(ALPHA == PDX::IP){ // simsimd_dot_f32_skylake
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
    }
    if constexpr(ALPHA == PDX::L1){ // https://github.com/facebookresearch/faiss/blob/697b6ddf558ef4ecb60e72e828c25a69723639c1/faiss/utils/distances_simd.cpp#L2590
        size_t d = num_dimensions;
        __m512 msum1 = _mm512_setzero_ps();
        //const __m512 signmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffffUL));

        while (d >= 16) {
            __m512 mx = _mm512_loadu_ps(vector1);
            vector1 += 16;
            __m512 my = _mm512_loadu_ps(vector2);
            vector2 += 16;
            const __m512 a_m_b = _mm512_sub_ps(mx, my);
            // msum1 = _mm512_add_ps(msum1, _mm512_and_ps(signmask, a_m_b));
            msum1 = _mm512_add_ps(msum1, _mm512_abs_ps(a_m_b));
            d -= 16;
        }

        if (d > 0) {
            __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, d);
            __m512 mx = _mm512_maskz_loadu_ps(mask, vector1);
            __m512 my = _mm512_maskz_loadu_ps(mask, vector2);

            __m512 a_m_b = _mm512_sub_ps(mx, my);
            // msum1 = _mm512_add_ps(msum1, _mm512_and_ps(signmask, a_m_b));
            msum1 = _mm512_add_ps(msum1, _mm512_abs_ps(a_m_b));
        }

        // Final horizontal sum of the accumulated values
        __m512 x = _mm512_add_ps(msum1, _mm512_shuffle_f32x4(msum1, msum1, _MM_SHUFFLE(0, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
    }
#elif defined(__AVX2__)
    if constexpr(ALPHA == PDX::L2){ // simsimd_dot_f32_haswell
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
    }
    if constexpr(ALPHA == PDX::IP){
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
    }
    if constexpr(ALPHA == PDX::L1){ // // https://github.com/facebookresearch/faiss/blob/697b6ddf558ef4ecb60e72e828c25a69723639c1/faiss/utils/distances_simd.cpp#L2590
        size_t d = num_dimensions;
        __m256 msum1 = _mm256_setzero_ps();
        __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffUL));

        while (d >= 8) {
            __m256 mx = _mm256_loadu_ps(vector1);
            vector1 += 8;
            __m256 my = _mm256_loadu_ps(vector2);
            vector2 += 8;
            const __m256 a_m_b = _mm256_sub_ps(mx, my);
            msum1 = _mm256_add_ps(msum1, _mm256_and_ps(signmask, a_m_b));
            d -= 8;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));
        __m128 signmask2 = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffUL));

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps(vector1);
            vector1 += 4;
            __m128 my = _mm_loadu_ps(vector2);
            vector2 += 4;
            const __m128 a_m_b = _mm_sub_ps(mx, my);
            msum2 = _mm_add_ps(msum2, _mm_and_ps(signmask2, a_m_b));
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read(d, vector1);
            __m128 my = masked_read(d, vector2);
            __m128 a_m_b = _mm_sub_ps(mx, my);
            msum2 = _mm_add_ps(msum2, _mm_and_ps(signmask2, a_m_b));
        }

        msum2 = _mm_hadd_ps(msum2, msum2);
        msum2 = _mm_hadd_ps(msum2, msum2);
        return _mm_cvtss_f32(msum2);
    }
#else
    if constexpr(ALPHA == PDX::L2){
        float distance = 0.0;
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            float to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
            distance += to_multiply * to_multiply;
        }
        return distance;
    }
    if constexpr(ALPHA == PDX::IP){
        float distance = 0.0;
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance += vector1[dimension_idx] * vector2[dimension_idx];
        }
        return distance;
    }
    if constexpr(ALPHA == PDX::L1){
        float res = 0;
        for (size_t dimension_idx = 0; i < num_dimensions; ++dimension_idx) {
            float tmp = vector1[dimension_idx] - vector2[dimension_idx];
            res += std::fabs(tmp);
        }
        return res;
    }
#endif
    }
};

template<PDX::DistanceFunction ALPHA=PDX::L2>
class ScalarScanner {
public:
    static float CalculateDistance(const float *__restrict vector1, const float *__restrict vector2, size_t num_dimensions) {
        if constexpr(ALPHA == PDX::L2){
            float distance = 0.0;
#pragma clang loop vectorize(disable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                float to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
                distance += to_multiply * to_multiply;
            }
            return distance;
        }
        if constexpr(ALPHA == PDX::IP){
            float distance = 0.0;
#pragma clang loop vectorize(disable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                distance += vector1[dimension_idx] * vector2[dimension_idx];
            }
            return distance;
        }
        if constexpr(ALPHA == PDX::L1){
            float res = 0;
#pragma clang loop vectorize(disable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                float tmp = vector1[dimension_idx] - vector2[dimension_idx];
                res += std::fabs(tmp);
            }
            return res;
        }
    }

    static double CalculatePreciseDistance(const float *__restrict vector1, const float *__restrict vector2, size_t num_dimensions) {
        if constexpr(ALPHA == PDX::L2){
            double distance = 0.0;
#pragma clang loop vectorize(disable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                double to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
                distance += to_multiply * to_multiply;
            }
            return distance;
        }
        if constexpr(ALPHA == PDX::IP){
            double distance = 0.0;
#pragma clang loop vectorize(disable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                distance += vector1[dimension_idx] * vector2[dimension_idx];
            }
            return distance;
        }
        if constexpr(ALPHA == PDX::L1){
            double res = 0.0;
#pragma clang loop vectorize(disable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                double tmp = static_cast<double>(vector1[dimension_idx] - vector2[dimension_idx]);
                res += std::fabs(tmp);
            }
            return res;
        }
    }

};

template<PDX::DistanceFunction ALPHA=PDX::L2>
class ImpreciseScanner {
public:
    static float CalculateDistance(const float *__restrict vector1, const float *__restrict vector2, size_t num_dimensions) {
        if constexpr(ALPHA == PDX::L2){
            float distance = 0.0;
#pragma clang loop vectorize(enable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                float to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
                distance += to_multiply * to_multiply;
            }
            return distance;
        }
        if constexpr(ALPHA == PDX::IP){
            float distance = 0.0;
#pragma clang loop vectorize(enable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                distance += vector1[dimension_idx] * vector2[dimension_idx];
            }
            return distance;
        }
        if constexpr(ALPHA == PDX::L1){
            float res = 0;
#pragma clang loop vectorize(enable)
            for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
                float tmp = vector1[dimension_idx] - vector2[dimension_idx];
                res += std::fabs(tmp);
            }
            return res;
        }
    }
};

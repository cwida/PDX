#ifndef PDX_AVX2_COMPUTERS_HPP
#define PDX_AVX2_COMPUTERS_HPP

#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include "pdx/common.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"

namespace PDX {

template<DistanceFunction alpha, Quantization q>
class SIMDComputer {
};

template<>
class SIMDComputer<L2, Quantization::U8> {

};


template<>
class SIMDComputer<L2, Quantization::F32> {
public:
    using DISTANCE_TYPE = DistanceType_t<F32>;
    using QUERY_TYPE = QuantizedVectorType_t<F32>;
    using DATA_TYPE = DataType_t<F32>;
    using scalar_computer = ScalarComputer<L2, Quantization::F32>;

    alignas(64) static DISTANCE_TYPE pruning_distances_tmp[4096];

    // Defer to the scalar kernel
    template<bool USE_DIMENSIONS_REORDER, bool SKIP_PRUNED>
    static void VerticalPruning(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t n_vectors,
            size_t total_vectors,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE *distances_p,
            const uint32_t *pruning_positions = nullptr,
            const uint32_t *indices_dimensions = nullptr,
            const int32_t *dim_clip_value = nullptr
    ) {
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER) {
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                auto true_vector_idx = vector_idx;
                if constexpr (SKIP_PRUNED) {
                    true_vector_idx = pruning_positions[vector_idx];
                }
                float to_multiply = query[true_dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
                distances_p[true_vector_idx] += to_multiply * to_multiply;
            }
        }
    }

    // Defer to the scalar kernel
    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE *distances_p
    ) {
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            size_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
                float to_multiply = query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
                distances_p[vector_idx] += to_multiply * to_multiply;
            }
        }
    }

    static DISTANCE_TYPE Horizontal(
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_dimensions
    ) {
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

        return static_cast<DISTANCE_TYPE>(d2);
    };

};

template <>
class SIMDComputer<IP, Quantization::F32>{
public:
    using DISTANCE_TYPE = DistanceType_t<F32>;
    using QUERY_TYPE = QuantizedVectorType_t<F32>;
    using DATA_TYPE = DataType_t<F32>;

    // Defer to the scalar kernel
    template<bool USE_DIMENSIONS_REORDER, bool SKIP_PRUNED>
    static void VerticalPruning(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t n_vectors,
            size_t total_vectors,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p,
            const uint32_t * pruning_positions = nullptr,
            const uint32_t * indices_dimensions = nullptr,
            const int32_t * dim_clip_value = nullptr
    ){
        // TODO
    }

    // Defer to the scalar kernel
    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p
    ){
        // TODO
    }

    static DISTANCE_TYPE Horizontal(
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_dimensions
    ){
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
        return static_cast<DISTANCE_TYPE>(d2);
    };

};

}

#endif //PDX_AVX2_COMPUTERS_HPP

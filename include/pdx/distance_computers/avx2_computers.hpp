#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"
#include <cstdint>
#include <immintrin.h>

namespace PDX {

template <DistanceMetric alpha, Quantization Q>
class SIMDComputer {};

template <>
class SIMDComputer<DistanceMetric::L2SQ, Quantization::F32> {
  public:
    using distance_t = pdx_distance_t<F32>;
    using query_t = pdx_quantized_embedding_t<F32>;
    using data_t = pdx_data_t<F32>;
    using scalar_computer = ScalarComputer<DistanceMetric::L2SQ, Quantization::F32>;

    template <bool SKIP_PRUNED>
    static void Vertical(
        const query_t* PDX_RESTRICT query,
        const data_t* PDX_RESTRICT data,
        size_t n_vectors,
        size_t total_vectors,
        size_t start_dimension,
        size_t end_dimension,
        distance_t* distances_p,
        const uint32_t* pruning_positions = nullptr
    ) {
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension;
             ++dimension_idx) {
            size_t offset_to_dimension_start = dimension_idx * dimensions_jump_factor;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                auto true_vector_idx = vector_idx;
                if constexpr (SKIP_PRUNED) {
                    true_vector_idx = pruning_positions[vector_idx];
                }
                distance_t to_multiply =
                    query[dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
                distances_p[true_vector_idx] += to_multiply * to_multiply;
            }
        }
    }

    static distance_t Horizontal(
        const query_t* PDX_RESTRICT vector1,
        const data_t* PDX_RESTRICT vector2,
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
            distance_t d = vector1[i] - vector2[i];
            d2 += d * d;
        }

        return static_cast<distance_t>(d2);
    };

    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 vec = _mm256_loadu_ps(data + j);
            __m256i mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(masks + j));
            __m256i vec_i = _mm256_castps_si256(vec);
            vec_i = _mm256_xor_si256(vec_i, mask);
            _mm256_storeu_ps(out + j, _mm256_castsi256_ps(vec_i));
        }
        auto data_bits = reinterpret_cast<const uint32_t*>(data);
        auto out_bits = reinterpret_cast<uint32_t*>(out);
        for (; j < d; ++j) {
            out_bits[j] = data_bits[j] ^ masks[j];
        }
    }
};

template <>
class SIMDComputer<DistanceMetric::L2SQ, Quantization::U8> {
  public:
    using distance_t = pdx_distance_t<U8>;
    using query_t = pdx_quantized_embedding_t<U8>;
    using data_t = pdx_data_t<U8>;

    template <bool SKIP_PRUNED>
    static void Vertical(
        const query_t* PDX_RESTRICT query,
        const data_t* PDX_RESTRICT data,
        size_t n_vectors,
        size_t total_vectors,
        size_t start_dimension,
        size_t end_dimension,
        distance_t* distances_p,
        const uint32_t* pruning_positions = nullptr
    ) {
        auto* query_grouped = reinterpret_cast<const uint32_t*>(query);
        size_t dim_idx = start_dimension;
        for (; dim_idx + 4 <= end_dimension; dim_idx += 4) {
            uint32_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
            if constexpr (!SKIP_PRUNED) {
                uint32_t query_value = query_grouped[dimension_idx / 4];
                __m256i vec1_u8 = _mm256_set1_epi32(query_value);
                __m256i zeros = _mm256_setzero_si256();
                for (; i + 8 <= n_vectors; i += 8) {
                    // Load 8 accumulated distances and 32 bytes of data (4 dims x 8 vectors).
                    __m256i res =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&distances_p[i]));
                    __m256i vec2_u8 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(&data[offset_to_dimension_start + i * 4])
                    );
                    __m256i diff = _mm256_or_si256(
                        _mm256_subs_epu8(vec1_u8, vec2_u8), _mm256_subs_epu8(vec2_u8, vec1_u8)
                    );
                    // Zero-extend bytes to 16-bit, square, and horizontal-add to 32-bit.
                    __m256i lo16 = _mm256_unpacklo_epi8(diff, zeros);
                    __m256i hi16 = _mm256_unpackhi_epi8(diff, zeros);
                    __m256i sq_lo = _mm256_madd_epi16(lo16, lo16);
                    __m256i sq_hi = _mm256_madd_epi16(hi16, hi16);
                    // hadd combines partial sums: [v0, v1, v2, v3, v4, v5, v6, v7].
                    __m256i sums = _mm256_hadd_epi32(sq_lo, sq_hi);
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(&distances_p[i]), _mm256_add_epi32(res, sums)
                    );
                }
            }
            // Scalar tail (vectors).
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED) {
                    vector_idx = pruning_positions[vector_idx];
                }
                int da = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int db = query[dimension_idx + 1] -
                         data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int dc = query[dimension_idx + 2] -
                         data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int dd = query[dimension_idx + 3] -
                         data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (da * da) + (db * db) + (dc * dc) + (dd * dd);
            }
        }
        if (dim_idx < end_dimension) {
            auto remaining = static_cast<uint32_t>(end_dimension - dim_idx);
            size_t offset = dim_idx * total_vectors;
            for (size_t i = 0; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED) {
                    vector_idx = pruning_positions[vector_idx];
                }
                for (uint32_t k = 0; k < remaining; ++k) {
                    int diff = query[dim_idx + k] - data[offset + vector_idx * remaining + k];
                    distances_p[vector_idx] += diff * diff;
                }
            }
        }
    }

    static distance_t Horizontal(
        const query_t* PDX_RESTRICT vector1,
        const data_t* PDX_RESTRICT vector2,
        size_t num_dimensions
    ) {
        __m256i d2_vec = _mm256_setzero_si256();
        __m256i zeros = _mm256_setzero_si256();
        size_t i = 0;
        for (; i + 32 <= num_dimensions; i += 32) {
            __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vector1 + i));
            __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vector2 + i));
            __m256i diff =
                _mm256_or_si256(_mm256_subs_epu8(a_vec, b_vec), _mm256_subs_epu8(b_vec, a_vec));
            __m256i lo16 = _mm256_unpacklo_epi8(diff, zeros);
            __m256i hi16 = _mm256_unpackhi_epi8(diff, zeros);
            d2_vec = _mm256_add_epi32(d2_vec, _mm256_madd_epi16(lo16, lo16));
            d2_vec = _mm256_add_epi32(d2_vec, _mm256_madd_epi16(hi16, hi16));
        }
        // Reduce 8 x i32 to scalar (simsimd_reduce_i32x8_haswell)
        __m128i lo = _mm256_castsi256_si128(d2_vec);
        __m128i hi = _mm256_extracti128_si256(d2_vec, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        distance_t distance = _mm_cvtsi128_si32(sum128);
        // Scalar tail.
        for (; i < num_dimensions; ++i) {
            int n = static_cast<int>(vector1[i]) - static_cast<int>(vector2[i]);
            distance += n * n;
        }
        return distance;
    };
};

} // namespace PDX

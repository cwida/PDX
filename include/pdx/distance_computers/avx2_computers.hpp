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

        return static_cast<distance_t>(d2); // NOLINT(bugprone-narrowing-conversions)
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

    static float HorizontalDotWithU8(
        const query_t* PDX_RESTRICT query,
        const uint8_t* PDX_RESTRICT codes,
        size_t num_dimensions
    ) {
        __m256 dot_vec_0 = _mm256_setzero_ps();
        __m256 dot_vec_1 = _mm256_setzero_ps();
        __m256 dot_vec_2 = _mm256_setzero_ps();
        __m256 dot_vec_3 = _mm256_setzero_ps();
        const auto codes_as_floats = [codes](size_t offset) {
            return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + offset))
            ));
        };
        size_t i = 0;
        for (; i + 32 <= num_dimensions; i += 32) {
            dot_vec_0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + i), codes_as_floats(i), dot_vec_0);
            dot_vec_1 =
                _mm256_fmadd_ps(_mm256_loadu_ps(query + i + 8), codes_as_floats(i + 8), dot_vec_1);
            dot_vec_2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(query + i + 16), codes_as_floats(i + 16), dot_vec_2
            );
            dot_vec_3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(query + i + 24), codes_as_floats(i + 24), dot_vec_3
            );
        }
        for (; i + 8 <= num_dimensions; i += 8) {
            dot_vec_0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + i), codes_as_floats(i), dot_vec_0);
        }
        const __m256 dot_vec =
            _mm256_add_ps(_mm256_add_ps(dot_vec_0, dot_vec_1), _mm256_add_ps(dot_vec_2, dot_vec_3));
        __m128 dot_halves =
            _mm_add_ps(_mm256_castps256_ps128(dot_vec), _mm256_extractf128_ps(dot_vec, 1));
        dot_halves = _mm_hadd_ps(dot_halves, dot_halves);
        float dot = _mm_cvtss_f32(_mm_hadd_ps(dot_halves, dot_halves));
        for (; i < num_dimensions; ++i) {
            dot += query[i] * static_cast<float>(codes[i]);
        }
        return dot;
    }

    static float HorizontalDotWithU4(
        const query_t* PDX_RESTRICT query,
        const uint8_t* PDX_RESTRICT codes,
        size_t num_dimensions
    ) {
        __m256 dot_vec_0 = _mm256_setzero_ps();
        __m256 dot_vec_1 = _mm256_setzero_ps();
        __m256 dot_vec_2 = _mm256_setzero_ps();
        __m256 dot_vec_3 = _mm256_setzero_ps();
        const __m128i nibble_mask = _mm_set1_epi8(0x0F);
        const auto accumulate_16_dimensions = [&](size_t offset, __m256& dot_low, __m256& dot_high
                                              ) {
            const __m128i code_pairs =
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + offset / 2));
            const __m128i low_nibbles = _mm_and_si128(code_pairs, nibble_mask);
            const __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(code_pairs, 4), nibble_mask);
            const __m128i ordered_codes = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
            const __m256 codes_low_vec = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(ordered_codes));
            const __m256 codes_high_vec =
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(ordered_codes, 8)));
            dot_low = _mm256_fmadd_ps(_mm256_loadu_ps(query + offset), codes_low_vec, dot_low);
            dot_high =
                _mm256_fmadd_ps(_mm256_loadu_ps(query + offset + 8), codes_high_vec, dot_high);
        };
        size_t i = 0;
        for (; i + 32 <= num_dimensions; i += 32) {
            accumulate_16_dimensions(i, dot_vec_0, dot_vec_1);
            accumulate_16_dimensions(i + 16, dot_vec_2, dot_vec_3);
        }
        for (; i + 16 <= num_dimensions; i += 16) {
            accumulate_16_dimensions(i, dot_vec_0, dot_vec_1);
        }
        const __m256 dot_vec =
            _mm256_add_ps(_mm256_add_ps(dot_vec_0, dot_vec_1), _mm256_add_ps(dot_vec_2, dot_vec_3));
        __m128 dot_halves =
            _mm_add_ps(_mm256_castps256_ps128(dot_vec), _mm256_extractf128_ps(dot_vec, 1));
        dot_halves = _mm_hadd_ps(dot_halves, dot_halves);
        float dot = _mm_cvtss_f32(_mm_hadd_ps(dot_halves, dot_halves));
        for (; i + 2 <= num_dimensions; i += 2) {
            const uint8_t code_pair = codes[i / 2];
            dot += query[i] * static_cast<float>(code_pair & 0x0F) +
                   query[i + 1] * static_cast<float>(code_pair >> 4);
        }
        if (i < num_dimensions) {
            dot += query[i] * static_cast<float>(codes[i / 2] & 0x0F);
        }
        return dot;
    }

    static void MinMax(
        const query_t* PDX_RESTRICT values,
        size_t num_dimensions,
        float& lower,
        float& upper
    ) {
        __m256 lower_vec = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        __m256 upper_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        size_t i = 0;
        for (; i + 8 <= num_dimensions; i += 8) {
            const __m256 values_vec = _mm256_loadu_ps(values + i);
            lower_vec = _mm256_min_ps(lower_vec, values_vec);
            upper_vec = _mm256_max_ps(upper_vec, values_vec);
        }
        alignas(32) float lowers[8];
        alignas(32) float uppers[8];
        _mm256_store_ps(lowers, lower_vec);
        _mm256_store_ps(uppers, upper_vec);
        lower = *std::min_element(lowers, lowers + 8);
        upper = *std::max_element(uppers, uppers + 8);
        for (; i < num_dimensions; ++i) {
            lower = std::min(lower, values[i]);
            upper = std::max(upper, values[i]);
        }
    }

    // Returns the squared norm of the dequantized values
    static float QuantizeToU8(
        const query_t* PDX_RESTRICT values,
        size_t num_dimensions,
        float lower,
        float step,
        uint8_t* PDX_RESTRICT codes
    ) {
        const __m256 lower_vec = _mm256_set1_ps(lower);
        const __m256 inverse_step_vec = _mm256_set1_ps(1.0f / step);
        const __m256 step_vec = _mm256_set1_ps(step);
        __m256 squared_norm_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(codes + i),
                Quantize16<255>(values + i, lower_vec, inverse_step_vec, step_vec, squared_norm_vec)
            );
        }
        return ReduceAdd(squared_norm_vec) +
               scalar_computer::QuantizeToU8(
                   values + i, num_dimensions - i, lower, step, codes + i
               );
    }

    // Dimension 2i in the low nibble of byte i. Returns the squared norm of the dequantized values
    static float QuantizeToU4(
        const query_t* PDX_RESTRICT values,
        size_t num_dimensions,
        float lower,
        float step,
        uint8_t* PDX_RESTRICT codes
    ) {
        const __m256 lower_vec = _mm256_set1_ps(lower);
        const __m256 inverse_step_vec = _mm256_set1_ps(1.0f / step);
        const __m256 step_vec = _mm256_set1_ps(step);
        // maddubs with these weights packs each pair of codes as even | odd << 4
        const __m128i code_pair_weights = _mm_set1_epi16(0x1001);
        __m256 squared_norm_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            const __m128i codes_vec =
                Quantize16<15>(values + i, lower_vec, inverse_step_vec, step_vec, squared_norm_vec);
            const __m128i code_pairs = _mm_maddubs_epi16(codes_vec, code_pair_weights);
            _mm_storel_epi64(
                reinterpret_cast<__m128i*>(codes + i / 2), _mm_packus_epi16(code_pairs, code_pairs)
            );
        }
        return ReduceAdd(squared_norm_vec) +
               scalar_computer::QuantizeToU4(
                   values + i, num_dimensions - i, lower, step, codes + i / 2
               );
    }

    static void DequantizeFromU4(
        const uint8_t* PDX_RESTRICT codes,
        size_t num_dimensions,
        float lower,
        float step,
        float* PDX_RESTRICT out
    ) {
        const __m256 lower_vec = _mm256_set1_ps(lower);
        const __m256 step_vec = _mm256_set1_ps(step);
        const __m128i nibble_mask = _mm_set1_epi8(0x0F);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            const __m128i code_pairs =
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i / 2));
            const __m128i ordered_codes = _mm_unpacklo_epi8(
                _mm_and_si128(code_pairs, nibble_mask),
                _mm_and_si128(_mm_srli_epi16(code_pairs, 4), nibble_mask)
            );
            _mm256_storeu_ps(
                out + i,
                _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(ordered_codes)), step_vec, lower_vec
                )
            );
            _mm256_storeu_ps(
                out + i + 8,
                _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(ordered_codes, 8))),
                    step_vec,
                    lower_vec
                )
            );
        }
        scalar_computer::DequantizeFromU4(codes + i / 2, num_dimensions - i, lower, step, out + i);
    }

    static void QuantizeToInt8(
        const query_t* PDX_RESTRICT values,
        size_t num_dimensions,
        float inverse_scale,
        int8_t* PDX_RESTRICT out
    ) {
        const __m256 inverse_scale_vec = _mm256_set1_ps(inverse_scale);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            const __m256i rounded_0 =
                _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(values + i), inverse_scale_vec));
            const __m256i rounded_1 =
                _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(values + i + 8), inverse_scale_vec)
                );
            // packs_epi32 interleaves the 128-bit lanes; permute4x64 puts the 16 values back in
            // order
            const __m256i rounded_s16 =
                _mm256_permute4x64_epi64(_mm256_packs_epi32(rounded_0, rounded_1), 0xD8);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(out + i),
                _mm_packs_epi16(
                    _mm256_castsi256_si128(rounded_s16), _mm256_extracti128_si256(rounded_s16, 1)
                )
            );
        }
        scalar_computer::QuantizeToInt8(values + i, num_dimensions - i, inverse_scale, out + i);
    }

    // TODO(@lkuffo, low): stored vertically, in blocks of 16 centroids whose 64-byte groups hold 4
    // dimensions of each, the LVQ8 codes would take two _mm256_dpbusd_avx_epi32 per group (against
    // the 4 query bytes broadcast by _mm256_set1_epi32) for the dots of all 16 centroids, with no
    // horizontal sums. Measured 1.35x over this kernel on Zen 5 with AVX-VNNI (geomean of 56
    // cells). It needs a blocked Centroids layout and a new CentroidQuantization nibble. Reads
    // padded_size (a multiple of LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES) bytes of query and codes.
    // query_codes_sum: the sum of the query's codes (the NEON kernel needs it)
    static int32_t HorizontalDotInt8WithU8(
        const int8_t* PDX_RESTRICT query,
        const uint8_t* PDX_RESTRICT codes,
        size_t padded_size,
        int32_t /*query_codes_sum*/
    ) {
        __m256i dot_vec_0 = _mm256_setzero_si256();
        __m256i dot_vec_1 = _mm256_setzero_si256();
        for (size_t i = 0; i < padded_size; i += 64) {
#if defined(__AVXVNNI__)
            dot_vec_0 =
                _mm256_dpbusd_avx_epi32(dot_vec_0, LoadBytes32(codes + i), LoadBytes32(query + i));
            dot_vec_1 = _mm256_dpbusd_avx_epi32(
                dot_vec_1, LoadBytes32(codes + i + 32), LoadBytes32(query + i + 32)
            );
#else
            // Without VNNI: widen to s16, and madd_epi16 sums each pair of products into s32
            for (size_t offset = 0; offset < 64; offset += 32) {
                dot_vec_0 = _mm256_add_epi32(
                    dot_vec_0,
                    _mm256_madd_epi16(
                        _mm256_cvtepu8_epi16(LoadBytes16(codes + i + offset)),
                        _mm256_cvtepi8_epi16(LoadBytes16(query + i + offset))
                    )
                );
                dot_vec_1 = _mm256_add_epi32(
                    dot_vec_1,
                    _mm256_madd_epi16(
                        _mm256_cvtepu8_epi16(LoadBytes16(codes + i + offset + 16)),
                        _mm256_cvtepi8_epi16(LoadBytes16(query + i + offset + 16))
                    )
                );
            }
#endif
        }
        return ReduceAdd(_mm256_add_epi32(dot_vec_0, dot_vec_1));
    }

    // Byte i of codes holds dimension 2i (low nibble) and 2i+1 (high), so even_query[i] and
    // odd_query[i] are those dimensions of the query. Reads padded_size bytes of each
    static int32_t HorizontalDotInt8WithU4(
        const int8_t* PDX_RESTRICT even_query,
        const int8_t* PDX_RESTRICT odd_query,
        const uint8_t* PDX_RESTRICT codes,
        size_t padded_size
    ) {
        __m256i dot_vec_0 = _mm256_setzero_si256();
        __m256i dot_vec_1 = _mm256_setzero_si256();
        const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
#if !defined(__AVXVNNI__)
        const __m256i ones_s16 = _mm256_set1_epi16(1);
#endif
        for (size_t i = 0; i < padded_size; i += 32) {
            const __m256i code_pairs = LoadBytes32(codes + i);
            const __m256i low_nibbles = _mm256_and_si256(code_pairs, nibble_mask);
            const __m256i high_nibbles =
                _mm256_and_si256(_mm256_srli_epi16(code_pairs, 4), nibble_mask);
#if defined(__AVXVNNI__)
            dot_vec_0 =
                _mm256_dpbusd_avx_epi32(dot_vec_0, low_nibbles, LoadBytes32(even_query + i));
            dot_vec_1 =
                _mm256_dpbusd_avx_epi32(dot_vec_1, high_nibbles, LoadBytes32(odd_query + i));
#else
            // Without VNNI: maddubs pairs fit in s16 (15 * 128 * 2 < 2^15), madd with ones widens
            // them
            dot_vec_0 = _mm256_add_epi32(
                dot_vec_0,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(low_nibbles, LoadBytes32(even_query + i)), ones_s16
                )
            );
            dot_vec_1 = _mm256_add_epi32(
                dot_vec_1,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(high_nibbles, LoadBytes32(odd_query + i)), ones_s16
                )
            );
#endif
        }
        return ReduceAdd(_mm256_add_epi32(dot_vec_0, dot_vec_1));
    }

  private:
    static float ReduceAdd(const __m256 vec) {
        __m128 halves = _mm_add_ps(_mm256_castps256_ps128(vec), _mm256_extractf128_ps(vec, 1));
        halves = _mm_hadd_ps(halves, halves);
        return _mm_cvtss_f32(_mm_hadd_ps(halves, halves));
    }

    static int32_t ReduceAdd(const __m256i vec) {
        __m128i halves =
            _mm_add_epi32(_mm256_castsi256_si128(vec), _mm256_extracti128_si256(vec, 1));
        halves = _mm_hadd_epi32(halves, halves);
        return _mm_cvtsi128_si32(_mm_hadd_epi32(halves, halves));
    }

    static __m256i LoadBytes32(const void* address) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(address));
    }

    static __m128i LoadBytes16(const void* address) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(address));
    }

    // The codes of 16 values (round to nearest even, clamped to [0, MAX_CODE]); adds the squares of
    // their dequantized values to squared_norm_vec
    template <int MAX_CODE>
    static __m128i Quantize16(
        const query_t* PDX_RESTRICT values,
        const __m256 lower_vec,
        const __m256 inverse_step_vec,
        const __m256 step_vec,
        __m256& squared_norm_vec
    ) {
        __m256i codes_vec[2];
        for (size_t half = 0; half < 2; ++half) {
            const __m256 scaled = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_loadu_ps(values + 8 * half), lower_vec), inverse_step_vec
            );
            codes_vec[half] = _mm256_min_epi32(
                _mm256_max_epi32(_mm256_cvtps_epi32(scaled), _mm256_setzero_si256()),
                _mm256_set1_epi32(MAX_CODE)
            );
            const __m256 dequantized =
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes_vec[half]), step_vec, lower_vec);
            squared_norm_vec = _mm256_fmadd_ps(dequantized, dequantized, squared_norm_vec);
        }
        // packus_epi32 interleaves the 128-bit lanes; permute4x64 puts the 16 codes back in order
        const __m256i codes_u16 =
            _mm256_permute4x64_epi64(_mm256_packus_epi32(codes_vec[0], codes_vec[1]), 0xD8);
        return _mm_packus_epi16(
            _mm256_castsi256_si128(codes_u16), _mm256_extracti128_si256(codes_u16, 1)
        );
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

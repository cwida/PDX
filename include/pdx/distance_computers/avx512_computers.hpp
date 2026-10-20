#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"
#include <cassert>
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
        __m512 d2_vec = _mm512_setzero();
        __m512 a_vec, b_vec;
    simsimd_l2sq_f32_skylake_cycle:
        if (num_dimensions < 16) {
            __mmask16 mask = static_cast<__mmask16>(_bzhi_u32(0xFFFFFFFF, num_dimensions));
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
        if (num_dimensions) {
            goto simsimd_l2sq_f32_skylake_cycle;
        }

        // _simsimd_reduce_f32x16_skylake
        __m512 x =
            _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(
            _mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1)))
        );
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
    };

    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        size_t j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 vec = _mm512_loadu_ps(data + j);
            __m512i mask = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(masks + j));
            __m512i vec_i = _mm512_castps_si512(vec);
            vec_i = _mm512_xor_si512(vec_i, mask);
            _mm512_storeu_ps(out + j, _mm512_castsi512_ps(vec_i));
        }
        for (; j + 8 <= d; j += 8) {
            __m256 vec = _mm256_loadu_ps(data + j);
            __m256i mask_avx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(masks + j));
            __m256i vec_i = _mm256_castps_si256(vec);
            vec_i = _mm256_xor_si256(vec_i, mask_avx);
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
        __m512 dot_vec_0 = _mm512_setzero_ps();
        __m512 dot_vec_1 = _mm512_setzero_ps();
        __m512 dot_vec_2 = _mm512_setzero_ps();
        __m512 dot_vec_3 = _mm512_setzero_ps();
        const auto codes_as_floats = [codes](size_t offset) {
            return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + offset))
            ));
        };
        size_t i = 0;
        for (; i + 64 <= num_dimensions; i += 64) {
            dot_vec_0 = _mm512_fmadd_ps(_mm512_loadu_ps(query + i), codes_as_floats(i), dot_vec_0);
            dot_vec_1 = _mm512_fmadd_ps(
                _mm512_loadu_ps(query + i + 16), codes_as_floats(i + 16), dot_vec_1
            );
            dot_vec_2 = _mm512_fmadd_ps(
                _mm512_loadu_ps(query + i + 32), codes_as_floats(i + 32), dot_vec_2
            );
            dot_vec_3 = _mm512_fmadd_ps(
                _mm512_loadu_ps(query + i + 48), codes_as_floats(i + 48), dot_vec_3
            );
        }
        for (; i < num_dimensions; i += 16) {
            const auto mask = static_cast<__mmask16>(
                _bzhi_u32(0xFFFFFFFF, static_cast<uint32_t>(num_dimensions - i))
            );
            const __m512 codes_vec =
                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, codes + i)));
            dot_vec_0 =
                _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, query + i), codes_vec, dot_vec_0);
        }
        return _mm512_reduce_add_ps(
            _mm512_add_ps(_mm512_add_ps(dot_vec_0, dot_vec_1), _mm512_add_ps(dot_vec_2, dot_vec_3))
        );
    }

    static float HorizontalDotWithU4(
        const query_t* PDX_RESTRICT query,
        const uint8_t* PDX_RESTRICT codes,
        size_t num_dimensions
    ) {
        __m512 dot_vec_0 = _mm512_setzero_ps();
        __m512 dot_vec_1 = _mm512_setzero_ps();
        __m512 dot_vec_2 = _mm512_setzero_ps();
        __m512 dot_vec_3 = _mm512_setzero_ps();
        const __m128i nibble_mask = _mm_set1_epi8(0x0F);
        const auto accumulate_32_dimensions = [&](size_t offset, __m512& dot_low, __m512& dot_high
                                              ) {
            const __m128i code_pairs =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + offset / 2));
            const __m128i low_nibbles = _mm_and_si128(code_pairs, nibble_mask);
            const __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(code_pairs, 4), nibble_mask);
            const __m512 codes_low_vec =
                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(low_nibbles, high_nibbles)
                ));
            const __m512 codes_high_vec =
                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(low_nibbles, high_nibbles)
                ));
            dot_low = _mm512_fmadd_ps(_mm512_loadu_ps(query + offset), codes_low_vec, dot_low);
            dot_high =
                _mm512_fmadd_ps(_mm512_loadu_ps(query + offset + 16), codes_high_vec, dot_high);
        };
        size_t i = 0;
        for (; i + 64 <= num_dimensions; i += 64) {
            accumulate_32_dimensions(i, dot_vec_0, dot_vec_1);
            accumulate_32_dimensions(i + 32, dot_vec_2, dot_vec_3);
        }
        for (; i + 32 <= num_dimensions; i += 32) {
            accumulate_32_dimensions(i, dot_vec_0, dot_vec_1);
        }
        float dot = _mm512_reduce_add_ps(
            _mm512_add_ps(_mm512_add_ps(dot_vec_0, dot_vec_1), _mm512_add_ps(dot_vec_2, dot_vec_3))
        );
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
        __m512 lower_vec = _mm512_set1_ps(std::numeric_limits<float>::infinity());
        __m512 upper_vec = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < num_dimensions; i += 16) {
            const __mmask16 mask = TailMask(num_dimensions - i);
            const __m512 values_vec = _mm512_maskz_loadu_ps(mask, values + i);
            lower_vec = _mm512_mask_min_ps(lower_vec, mask, lower_vec, values_vec);
            upper_vec = _mm512_mask_max_ps(upper_vec, mask, upper_vec, values_vec);
        }
        lower = _mm512_reduce_min_ps(lower_vec);
        upper = _mm512_reduce_max_ps(upper_vec);
    }

    // Returns the squared norm of the dequantized values
    static float QuantizeToU8(
        const query_t* PDX_RESTRICT values,
        size_t num_dimensions,
        float lower,
        float step,
        uint8_t* PDX_RESTRICT codes
    ) {
        const __m512 lower_vec = _mm512_set1_ps(lower);
        const __m512 inverse_step_vec = _mm512_set1_ps(1.0f / step);
        const __m512 step_vec = _mm512_set1_ps(step);
        __m512 squared_norm_vec = _mm512_setzero_ps();
        for (size_t i = 0; i < num_dimensions; i += 16) {
            const __mmask16 mask = TailMask(num_dimensions - i);
            _mm_mask_storeu_epi8(
                codes + i,
                mask,
                Quantize16<255>(
                    values + i, mask, lower_vec, inverse_step_vec, step_vec, squared_norm_vec
                )
            );
        }
        return _mm512_reduce_add_ps(squared_norm_vec);
    }

    // Dimension 2i in the low nibble of byte i. Returns the squared norm of the dequantized values
    static float QuantizeToU4(
        const query_t* PDX_RESTRICT values,
        size_t num_dimensions,
        float lower,
        float step,
        uint8_t* PDX_RESTRICT codes
    ) {
        const __m512 lower_vec = _mm512_set1_ps(lower);
        const __m512 inverse_step_vec = _mm512_set1_ps(1.0f / step);
        const __m512 step_vec = _mm512_set1_ps(step);
        // maddubs with these weights packs each pair of codes as even | odd << 4
        const __m128i code_pair_weights = _mm_set1_epi16(0x1001);
        __m512 squared_norm_vec = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            const __m128i codes_vec = Quantize16<15>(
                values + i, 0xFFFF, lower_vec, inverse_step_vec, step_vec, squared_norm_vec
            );
            const __m128i code_pairs = _mm_maddubs_epi16(codes_vec, code_pair_weights);
            _mm_storel_epi64(
                reinterpret_cast<__m128i*>(codes + i / 2), _mm_packus_epi16(code_pairs, code_pairs)
            );
        }
        return _mm512_reduce_add_ps(squared_norm_vec) +
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
        const __m512 lower_vec = _mm512_set1_ps(lower);
        const __m512 step_vec = _mm512_set1_ps(step);
        const __m128i nibble_mask = _mm_set1_epi8(0x0F);
        size_t i = 0;
        for (; i + 32 <= num_dimensions; i += 32) {
            const __m128i code_pairs =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i / 2));
            const __m128i low_nibbles = _mm_and_si128(code_pairs, nibble_mask);
            const __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(code_pairs, 4), nibble_mask);
            const __m512 codes_low_vec =
                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(low_nibbles, high_nibbles)
                ));
            const __m512 codes_high_vec =
                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(low_nibbles, high_nibbles)
                ));
            _mm512_storeu_ps(out + i, _mm512_fmadd_ps(codes_low_vec, step_vec, lower_vec));
            _mm512_storeu_ps(out + i + 16, _mm512_fmadd_ps(codes_high_vec, step_vec, lower_vec));
        }
        scalar_computer::DequantizeFromU4(codes + i / 2, num_dimensions - i, lower, step, out + i);
    }

    static void QuantizeToInt8(
        const query_t* PDX_RESTRICT values,
        size_t num_dimensions,
        float inverse_scale,
        int8_t* PDX_RESTRICT out
    ) {
        const __m512 inverse_scale_vec = _mm512_set1_ps(inverse_scale);
        for (size_t i = 0; i < num_dimensions; i += 16) {
            const __mmask16 mask = TailMask(num_dimensions - i);
            const __m512i rounded = _mm512_cvtps_epi32(
                _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, values + i), inverse_scale_vec)
            );
            _mm_mask_storeu_epi8(out + i, mask, _mm512_cvtsepi32_epi8(rounded));
        }
    }

    // TODO(@lkuffo, low): stored vertically, in blocks of 16 centroids whose 64-byte groups hold 4
    // dimensions of each, the LVQ8 codes would take one _mm512_dpbusd_epi32 per group (against the
    // 4 query bytes broadcast by _mm512_set1_epi32) for the dots of all 16 centroids, with no
    // horizontal sums. Measured 1.87x over this kernel on Zen 5 (geomean of 56 cells). It needs a
    // blocked Centroids layout and a new CentroidQuantization nibble. Reads padded_size (a multiple
    // of LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES) bytes of query and codes. query_codes_sum: the sum of
    // the query's codes (the NEON kernel needs it)
    static int32_t HorizontalDotInt8WithU8(
        const int8_t* PDX_RESTRICT query,
        const uint8_t* PDX_RESTRICT codes,
        size_t padded_size,
        int32_t /*query_codes_sum*/
    ) {
        __m512i dot_vec_0 = _mm512_setzero_si512();
        __m512i dot_vec_1 = _mm512_setzero_si512();
        __m512i dot_vec_2 = _mm512_setzero_si512();
        __m512i dot_vec_3 = _mm512_setzero_si512();
        size_t i = 0;
        for (; i + 256 <= padded_size; i += 256) {
            dot_vec_0 = _mm512_dpbusd_epi32(
                dot_vec_0, _mm512_loadu_si512(codes + i), _mm512_loadu_si512(query + i)
            );
            dot_vec_1 = _mm512_dpbusd_epi32(
                dot_vec_1, _mm512_loadu_si512(codes + i + 64), _mm512_loadu_si512(query + i + 64)
            );
            dot_vec_2 = _mm512_dpbusd_epi32(
                dot_vec_2, _mm512_loadu_si512(codes + i + 128), _mm512_loadu_si512(query + i + 128)
            );
            dot_vec_3 = _mm512_dpbusd_epi32(
                dot_vec_3, _mm512_loadu_si512(codes + i + 192), _mm512_loadu_si512(query + i + 192)
            );
        }
        for (; i < padded_size; i += 64) {
            dot_vec_0 = _mm512_dpbusd_epi32(
                dot_vec_0, _mm512_loadu_si512(codes + i), _mm512_loadu_si512(query + i)
            );
        }
        return _mm512_reduce_add_epi32(_mm512_add_epi32(
            _mm512_add_epi32(dot_vec_0, dot_vec_1), _mm512_add_epi32(dot_vec_2, dot_vec_3)
        ));
    }

    // Byte i of codes holds dimension 2i (low nibble) and 2i+1 (high), so even_query[i] and
    // odd_query[i] are those dimensions of the query. Reads padded_size bytes of each
    static int32_t HorizontalDotInt8WithU4(
        const int8_t* PDX_RESTRICT even_query,
        const int8_t* PDX_RESTRICT odd_query,
        const uint8_t* PDX_RESTRICT codes,
        size_t padded_size
    ) {
        __m512i dot_vec_0 = _mm512_setzero_si512();
        __m512i dot_vec_1 = _mm512_setzero_si512();
        const __m512i nibble_mask = _mm512_set1_epi8(0x0F);
        for (size_t i = 0; i < padded_size; i += 64) {
            const __m512i code_pairs = _mm512_loadu_si512(codes + i);
            dot_vec_0 = _mm512_dpbusd_epi32(
                dot_vec_0,
                _mm512_and_si512(code_pairs, nibble_mask),
                _mm512_loadu_si512(even_query + i)
            );
            dot_vec_1 = _mm512_dpbusd_epi32(
                dot_vec_1,
                _mm512_and_si512(_mm512_srli_epi16(code_pairs, 4), nibble_mask),
                _mm512_loadu_si512(odd_query + i)
            );
        }
        return _mm512_reduce_add_epi32(_mm512_add_epi32(dot_vec_0, dot_vec_1));
    }

  private:
    static __mmask16 TailMask(size_t remaining) {
        return remaining >= 16 ? static_cast<__mmask16>(0xFFFF)
                               : static_cast<__mmask16>((1u << remaining) - 1);
    }

    // The codes of the 16 values in mask (round to nearest even, clamped to [0, MAX_CODE]), 0 in
    // the other lanes; adds the squares of their dequantized values to squared_norm_vec
    template <int MAX_CODE>
    static __m128i Quantize16(
        const query_t* PDX_RESTRICT values,
        const __mmask16 mask,
        const __m512 lower_vec,
        const __m512 inverse_step_vec,
        const __m512 step_vec,
        __m512& squared_norm_vec
    ) {
        const __m512 scaled = _mm512_mul_ps(
            _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, values), lower_vec), inverse_step_vec
        );
        const __m512i codes_vec = _mm512_maskz_mov_epi32(
            mask,
            _mm512_min_epi32(
                _mm512_max_epi32(_mm512_cvtps_epi32(scaled), _mm512_setzero_si512()),
                _mm512_set1_epi32(MAX_CODE)
            )
        );
        const __m512 dequantized =
            _mm512_fmadd_ps(_mm512_cvtepi32_ps(codes_vec), step_vec, lower_vec);
        squared_norm_vec = _mm512_mask3_fmadd_ps(dequantized, dequantized, squared_norm_vec, mask);
        return _mm512_cvtepi32_epi8(codes_vec);
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
        __m512i res;
        __m512i vec2_u8;
        __m512i vec1_u8;
        __m512i diff_u8;
        __m256i y_res;
        __m256i y_vec2_u8;
        __m256i y_vec1_u8;
        __m256i y_diff_u8;
        // dpbusd reads its second operand as signed, so a difference d >= 128 is squared as
        // d * (d - 256). dpbusd(diff, diff & 0x80) yields -128 * sum(d | d >= 128); subtracting it
        // twice adds the missing 256 * d back (exact in two's complement).
        const __m512i zero = _mm512_setzero_si512();
        const __m512i high_bit = _mm512_set1_epi8(static_cast<char>(0x80));
        const __m256i y_zero = _mm256_setzero_si256();
        const __m256i y_high_bit = _mm256_set1_epi8(static_cast<char>(0x80));
        const uint32_t* query_grouped = reinterpret_cast<const uint32_t*>(query);
        size_t dim_idx = start_dimension;
        for (; dim_idx + 4 <= end_dimension; dim_idx += 4) {
            uint32_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
            if constexpr (!SKIP_PRUNED) {
                // To load the query efficiently I will load it as uint32_t (4 bytes packed in 1
                // word)
                uint32_t query_value = query_grouped[dimension_idx / 4];
                // And then broadcast it to the register
                vec1_u8 = _mm512_set1_epi32(query_value);
                for (; i + 16 <= n_vectors; i += 16) {
                    // Read 64 bytes of data (64 values) with 4 dimensions of 16 vectors
                    res = _mm512_loadu_si512(&distances_p[i]);
                    vec2_u8 = _mm512_loadu_si512(&data[offset_to_dimension_start + i * 4]
                    ); // This 4 is because everytime I read 4 dimensions
                    diff_u8 = _mm512_or_si512(
                        _mm512_subs_epu8(vec1_u8, vec2_u8), _mm512_subs_epu8(vec2_u8, vec1_u8)
                    );
                    // _mm512_storeu_si512(
                    //     &distances_p[i], _mm512_dpbusds_epi32(res, diff_u8, diff_u8)
                    // );
                    const __m512i corr =
                        _mm512_dpbusd_epi32(zero, diff_u8, _mm512_and_si512(diff_u8, high_bit));
                    res = _mm512_dpbusd_epi32(res, diff_u8, diff_u8);
                    _mm512_storeu_si512(
                        &distances_p[i], _mm512_sub_epi32(res, _mm512_slli_epi32(corr, 1))
                    );
                }
                y_vec1_u8 = _mm256_set1_epi32(query_value);
                for (; i + 8 <= n_vectors; i += 8) {
                    // Read 32 bytes of data (32 values) with 4 dimensions of 8 vectors
                    y_res = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&distances_p[i]));
                    y_vec2_u8 = _mm256_loadu_epi8(&data[offset_to_dimension_start + i * 4]
                    ); // This 4 is because everytime I read 4 dimensions
                    y_diff_u8 = _mm256_or_si256(
                        _mm256_subs_epu8(y_vec1_u8, y_vec2_u8),
                        _mm256_subs_epu8(y_vec2_u8, y_vec1_u8)
                    );
                    // _mm256_storeu_si256(
                    //     reinterpret_cast<__m256i*>(&distances_p[i]),
                    //     _mm256_dpbusds_epi32(y_res, y_diff_u8, y_diff_u8)
                    // );
                    const __m256i y_corr = _mm256_dpbusd_epi32(
                        y_zero, y_diff_u8, _mm256_and_si256(y_diff_u8, y_high_bit)
                    );
                    y_res = _mm256_dpbusd_epi32(y_res, y_diff_u8, y_diff_u8);
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(&distances_p[i]),
                        _mm256_sub_epi32(y_res, _mm256_slli_epi32(y_corr, 1))
                    );
                }
            }
            // Scalar tail (vectors).
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED) {
                    vector_idx = pruning_positions[vector_idx];
                }
                int to_multiply_a =
                    query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int to_multiply_b = query[dimension_idx + 1] -
                                    data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int to_multiply_c = query[dimension_idx + 2] -
                                    data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int to_multiply_d = query[dimension_idx + 3] -
                                    data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] +=
                    (to_multiply_a * to_multiply_a) + (to_multiply_b * to_multiply_b) +
                    (to_multiply_c * to_multiply_c) + (to_multiply_d * to_multiply_d);
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
        // __m512i d2_i32_vec = _mm512_setzero_si512();
        __m512i d2_low_i32 = _mm512_setzero_si512();
        __m512i d2_high_i32 = _mm512_setzero_si512();
        const __m512i zero = _mm512_setzero_si512();
        __m512i a_u8_vec, b_u8_vec;

    simsimd_l2sq_u8_ice_cycle:
        if (num_dimensions < 64) {
            const __mmask64 mask =
                static_cast<__mmask64>(_bzhi_u64(0xFFFFFFFFFFFFFFFF, num_dimensions));
            a_u8_vec = _mm512_maskz_loadu_epi8(mask, vector1);
            b_u8_vec = _mm512_maskz_loadu_epi8(mask, vector2);
            num_dimensions = 0;
        } else {
            a_u8_vec = _mm512_loadu_si512(vector1);
            b_u8_vec = _mm512_loadu_si512(vector2);
            vector1 += 64, vector2 += 64, num_dimensions -= 64;
        }

        // Substracting unsigned vectors in AVX-512 is done by saturating subtraction:
        __m512i d_u8_vec = _mm512_or_si512(
            _mm512_subs_epu8(a_u8_vec, b_u8_vec), _mm512_subs_epu8(b_u8_vec, a_u8_vec)
        );

        // dpbusd would read the second operand as signed (wrong for differences >= 128): widen to
        // 16 bits and square with the i16 dot product on two independent accumulators
        // d2_i32_vec = _mm512_dpbusds_epi32(d2_i32_vec, d_u8_vec, d_u8_vec);
        const __m512i d_low_i16 = _mm512_unpacklo_epi8(d_u8_vec, zero);
        const __m512i d_high_i16 = _mm512_unpackhi_epi8(d_u8_vec, zero);
        d2_low_i32 = _mm512_dpwssd_epi32(d2_low_i32, d_low_i16, d_low_i16);
        d2_high_i32 = _mm512_dpwssd_epi32(d2_high_i32, d_high_i16, d_high_i16);
        if (num_dimensions)
            goto simsimd_l2sq_u8_ice_cycle;
        // return _mm512_reduce_add_epi32(d2_i32_vec);
        return static_cast<distance_t>(
            _mm512_reduce_add_epi32(_mm512_add_epi32(d2_low_i32, d2_high_i32))
        );
    };
};

} // namespace PDX

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

struct SAQCompactionTable {
    uint32_t packed_lanes[256];
};

// For each mask of 8 surviving 32-bit lanes, the indices that move them to the front, 3 bits each
constexpr SAQCompactionTable MakeSAQCompactionTable() {
    SAQCompactionTable table{};
    for (uint32_t mask = 0; mask < 256; ++mask) {
        uint32_t packed = 0;
        uint32_t out_lane = 0;
        for (uint32_t lane = 0; lane < 8; ++lane) {
            if ((mask >> lane) & 1u) {
                packed |= lane << (3 * out_lane++);
            }
        }
        table.packed_lanes[mask] = packed;
    }
    return table;
}

inline constexpr SAQCompactionTable SAQ_COMPACTION_TABLE = MakeSAQCompactionTable();

template <>
class SIMDComputer<DistanceMetric::L2SQ, Quantization::SAQ> {
  public:
    static size_t ScanHead(
        const uint8_t* PDX_RESTRICT head,
        size_t num_vectors,
        size_t stride,
        const int8_t* PDX_RESTRICT query_codes,
        float dot_scale,
        float dot_offset,
        float query_squared_norm,
        const uint16_t* PDX_RESTRICT head_rescales,
        const uint16_t* PDX_RESTRICT squared_norms,
        const uint8_t* PDX_RESTRICT selection,
        float threshold,
        uint32_t* PDX_RESTRICT positions,
        float* PDX_RESTRICT estimates,
        float* PDX_RESTRICT rescales
    ) {
        int32_t group_codes[NUM_HEAD_GROUPS];
        std::memcpy(group_codes, query_codes, sizeof(group_codes));
        const __m256 dot_scale_vec = _mm256_set1_ps(dot_scale);
        const __m256 dot_offset_vec = _mm256_set1_ps(dot_offset);
        const __m256 query_squared_norm_vec = _mm256_set1_ps(query_squared_norm);
        const __m256 threshold_vec = _mm256_set1_ps(threshold);
        const __m256i lane_offsets = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        size_t num_kept = 0;
        for (size_t vector_idx = 0; vector_idx < num_vectors; vector_idx += 8) {
            const size_t num_valid = std::min<size_t>(8, num_vectors - vector_idx);
            const __m256i dot =
                HeadDots8(head + vector_idx * U8_INTERLEAVE_SIZE, stride, num_valid, group_codes);
            const __m256 rescale = LoadHalves8(head_rescales + vector_idx, num_valid);
            const __m256 estimate = _mm256_fmadd_ps(
                rescale,
                _mm256_fmsub_ps(_mm256_cvtepi32_ps(dot), dot_scale_vec, dot_offset_vec),
                _mm256_add_ps(
                    LoadHalves8(squared_norms + vector_idx, num_valid), query_squared_norm_vec
                )
            );
            auto keep_mask =
                static_cast<uint32_t>(
                    _mm256_movemask_ps(_mm256_cmp_ps(estimate, threshold_vec, _CMP_LT_OQ))
                ) &
                ((1u << num_valid) - 1);
            if (selection != nullptr) {
                uint8_t selected_bytes[8] = {};
                std::memcpy(selected_bytes, selection + vector_idx, num_valid);
                const __m256i selected = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(selected_bytes))
                );
                keep_mask &= static_cast<uint32_t>(_mm256_movemask_ps(
                    _mm256_castsi256_ps(_mm256_cmpgt_epi32(selected, _mm256_setzero_si256()))
                ));
            }
            num_kept += Compact8(
                keep_mask,
                _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(vector_idx)), lane_offsets),
                estimate,
                rescale,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept
            );
        }
        return num_kept;
    }

    // ScanHead over the vectors at positions only, 8 positions per batch with one gather per
    // group; keeps those below threshold, compacted in place
    static size_t ScanHeadPruning(
        const uint8_t* PDX_RESTRICT head,
        size_t stride,
        const int8_t* PDX_RESTRICT query_codes,
        float dot_scale,
        float dot_offset,
        float query_squared_norm,
        const uint16_t* PDX_RESTRICT head_rescales,
        const uint16_t* PDX_RESTRICT squared_norms,
        float threshold,
        uint32_t* PDX_RESTRICT positions,
        float* PDX_RESTRICT estimates,
        float* PDX_RESTRICT rescales,
        size_t num_positions
    ) {
        int32_t group_codes[NUM_HEAD_GROUPS];
        std::memcpy(group_codes, query_codes, sizeof(group_codes));
        const __m256 dot_scale_vec = _mm256_set1_ps(dot_scale);
        const __m256 dot_offset_vec = _mm256_set1_ps(dot_offset);
        const __m256 query_squared_norm_vec = _mm256_set1_ps(query_squared_norm);
        const __m256 threshold_vec = _mm256_set1_ps(threshold);
        size_t num_kept = 0;
        for (size_t position_idx = 0; position_idx < num_positions; position_idx += 8) {
            const size_t num_valid = std::min<size_t>(8, num_positions - position_idx);
            // Lanes past the last position hold position 0, which every gather can read
            const __m256i position_vec = _mm256_maskload_epi32(
                reinterpret_cast<const int*>(positions + position_idx), ValidLanes8(num_valid)
            );
            const __m256i level_offsets = _mm256_slli_epi32(position_vec, 2);
            __m256i dots[4] = {
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256()
            };
            for (size_t group_idx = 0; group_idx < NUM_HEAD_GROUPS; ++group_idx) {
                const __m256i group_vectors = _mm256_i32gather_epi32(
                    reinterpret_cast<const int*>(head + group_idx * U8_INTERLEAVE_SIZE * stride),
                    level_offsets,
                    1
                );
                dots[group_idx % 4] =
                    AddHeadGroupDots(dots[group_idx % 4], group_vectors, group_codes[group_idx]);
            }
            const __m256i dot = FinishHeadDots(_mm256_add_epi32(
                _mm256_add_epi32(dots[0], dots[1]), _mm256_add_epi32(dots[2], dots[3])
            ));
            uint16_t lane_rescales[8] = {};
            uint16_t lane_squared_norms[8] = {};
            for (size_t lane = 0; lane < num_valid; ++lane) {
                lane_rescales[lane] = head_rescales[positions[position_idx + lane]];
                lane_squared_norms[lane] = squared_norms[positions[position_idx + lane]];
            }
            const __m256 rescale = LoadHalves8(lane_rescales);
            const __m256 estimate = _mm256_fmadd_ps(
                rescale,
                _mm256_fmsub_ps(_mm256_cvtepi32_ps(dot), dot_scale_vec, dot_offset_vec),
                _mm256_add_ps(LoadHalves8(lane_squared_norms), query_squared_norm_vec)
            );
            num_kept += Compact8(
                static_cast<uint32_t>(
                    _mm256_movemask_ps(_mm256_cmp_ps(estimate, threshold_vec, _CMP_LT_OQ))
                ) & ((1u << num_valid) - 1),
                position_vec,
                estimate,
                rescale,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept
            );
        }
        return num_kept;
    }

    template <uint32_t WIDTH>
    static size_t ScanTail(
        const uint8_t* PDX_RESTRICT records,
        const int8_t* PDX_RESTRICT query_codes,
        float dot_scale,
        float dot_offset,
        const uint16_t* PDX_RESTRICT segment_rescales,
        float threshold,
        uint32_t* PDX_RESTRICT positions,
        float* PDX_RESTRICT estimates,
        float* PDX_RESTRICT rescales,
        size_t num_survivors
    ) {
        const __m256i query[2] = {LoadBytes32(query_codes), LoadBytes32(query_codes + 32)};
        const __m256 dot_scale_vec = _mm256_set1_ps(dot_scale);
        const __m256 dot_offset_vec = _mm256_set1_ps(dot_offset);
        const __m256 threshold_vec = _mm256_set1_ps(threshold);
        size_t num_kept = 0;
        for (size_t first_survivor = 0; first_survivor < num_survivors; first_survivor += 16) {
            const size_t batch_size = std::min<size_t>(16, num_survivors - first_survivor);
            int32_t lane_dots[16] = {};
            uint16_t lane_rescales[16] = {};
            for (size_t lane = 0; lane < batch_size; ++lane) {
                const uint32_t position = positions[first_survivor + lane];
                lane_dots[lane] =
                    TailDot<WIDTH>(records + static_cast<size_t>(position) * 8 * WIDTH, query);
                if (segment_rescales != nullptr) {
                    lane_rescales[lane] = segment_rescales[position];
                }
            }
            for (size_t half = 0; half * 8 < batch_size; ++half) {
                const size_t first = first_survivor + half * 8;
                const size_t num_valid = std::min<size_t>(8, batch_size - half * 8);
                const __m256 rescale = segment_rescales != nullptr
                                           ? LoadHalves8(lane_rescales + half * 8)
                                           : _mm256_loadu_ps(rescales + first);
                const __m256 estimate = _mm256_fmadd_ps(
                    rescale,
                    _mm256_fmsub_ps(
                        _mm256_cvtepi32_ps(LoadBytes32(lane_dots + half * 8)),
                        dot_scale_vec,
                        dot_offset_vec
                    ),
                    _mm256_loadu_ps(estimates + first)
                );
                num_kept += Compact8(
                    static_cast<uint32_t>(
                        _mm256_movemask_ps(_mm256_cmp_ps(estimate, threshold_vec, _CMP_LT_OQ))
                    ) & ((1u << num_valid) - 1),
                    LoadBytes32(positions + first),
                    estimate,
                    rescale,
                    positions + num_kept,
                    estimates + num_kept,
                    rescales + num_kept
                );
            }
        }
        return num_kept;
    }

    static void ComputeHeadEstimates(
        const uint8_t* PDX_RESTRICT head,
        size_t num_vectors,
        size_t stride,
        const int8_t* PDX_RESTRICT query_codes,
        float dot_scale,
        float dot_offset,
        float query_squared_norm,
        const uint16_t* PDX_RESTRICT head_rescales,
        const uint16_t* PDX_RESTRICT squared_norms,
        float* PDX_RESTRICT estimates
    ) {
        int32_t group_codes[NUM_HEAD_GROUPS];
        std::memcpy(group_codes, query_codes, sizeof(group_codes));
        const __m256 dot_scale_vec = _mm256_set1_ps(dot_scale);
        const __m256 dot_offset_vec = _mm256_set1_ps(dot_offset);
        const __m256 query_squared_norm_vec = _mm256_set1_ps(query_squared_norm);
        for (size_t vector_idx = 0; vector_idx < num_vectors; vector_idx += 8) {
            const size_t num_valid = std::min<size_t>(8, num_vectors - vector_idx);
            const __m256i dot =
                HeadDots8(head + vector_idx * U8_INTERLEAVE_SIZE, stride, num_valid, group_codes);
            const __m256 estimate = _mm256_fmadd_ps(
                LoadHalves8(head_rescales + vector_idx, num_valid),
                _mm256_fmsub_ps(_mm256_cvtepi32_ps(dot), dot_scale_vec, dot_offset_vec),
                _mm256_add_ps(
                    LoadHalves8(squared_norms + vector_idx, num_valid), query_squared_norm_vec
                )
            );
            if (num_valid == 8) {
                _mm256_storeu_ps(estimates + vector_idx, estimate);
            } else {
                _mm256_maskstore_ps(estimates + vector_idx, ValidLanes8(num_valid), estimate);
            }
        }
    }

    template <uint32_t WIDTH>
    static void AccumulateTailEstimates(
        const uint8_t* PDX_RESTRICT records,
        size_t num_vectors,
        const int8_t* PDX_RESTRICT query_codes,
        float dot_scale,
        float dot_offset,
        const uint16_t* PDX_RESTRICT segment_rescales,
        float* PDX_RESTRICT estimates
    ) {
        const __m256i query[2] = {LoadBytes32(query_codes), LoadBytes32(query_codes + 32)};
        const __m256 dot_scale_vec = _mm256_set1_ps(dot_scale);
        const __m256 dot_offset_vec = _mm256_set1_ps(dot_offset);
        for (size_t vector_idx = 0; vector_idx < num_vectors; vector_idx += 8) {
            const size_t num_valid = std::min<size_t>(8, num_vectors - vector_idx);
            const __m256i valid_lanes = ValidLanes8(num_valid);
            int32_t lane_dots[8] = {};
            for (size_t lane = 0; lane < num_valid; ++lane) {
                lane_dots[lane] = TailDot<WIDTH>(records + (vector_idx + lane) * 8 * WIDTH, query);
            }
            const __m256 estimate = _mm256_fmadd_ps(
                LoadHalves8(segment_rescales + vector_idx, num_valid),
                _mm256_fmsub_ps(
                    _mm256_cvtepi32_ps(LoadBytes32(lane_dots)), dot_scale_vec, dot_offset_vec
                ),
                num_valid == 8 ? _mm256_loadu_ps(estimates + vector_idx)
                               : _mm256_maskload_ps(estimates + vector_idx, valid_lanes)
            );
            if (num_valid == 8) {
                _mm256_storeu_ps(estimates + vector_idx, estimate);
            } else {
                _mm256_maskstore_ps(estimates + vector_idx, valid_lanes, estimate);
            }
        }
    }

    // The codec's kernels (scalar_computer documents them), 16 values per step
    using scalar_computer = ScalarComputer<DistanceMetric::L2SQ, Quantization::SAQ>;

    static float ComputeResidual(
        const float* PDX_RESTRICT embedding,
        const float* PDX_RESTRICT mean,
        size_t num_dimensions,
        float* PDX_RESTRICT residual
    ) {
        Float32x16 accumulator = Zero16();
        size_t j = 0;
        for (; j + 16 <= num_dimensions; j += 16) {
            const Float32x16 value = Sub16(Load16(embedding + j), Load16(mean + j));
            Store16(residual + j, value);
            accumulator = Fma16(value, value, accumulator);
        }
        float squared_norm = ReduceAdd16(accumulator);
        if (j < num_dimensions) {
            squared_norm += scalar_computer::ComputeResidual(
                embedding + j, mean + j, num_dimensions - j, residual + j
            );
        }
        return squared_norm;
    }

    static float EncodeSegment(
        const float* PDX_RESTRICT values,
        size_t num_values,
        uint32_t width,
        float scale,
        uint8_t* PDX_RESTRICT codes
    ) {
        if (num_values % 16 != 0) {
            return scalar_computer::EncodeSegment(values, num_values, width, scale, codes);
        }
        const int32_t half = int32_t{1} << (width - 1);
        const int32_t max_level = half - 1;
        Float32x16 squares = Zero16();
        for (size_t j = 0; j < num_values; j += 16) {
            const Float32x16 value = Load16(values + j);
            squares = Fma16(value, value, squares);
        }
        const float squared_norm = ReduceAdd16(squares);
        if (squared_norm <= 0.0f) {
            std::fill_n(codes, num_values, static_cast<uint8_t>(half));
            return 0.0f;
        }
        const Float32x16 epsilon = Set16(SAQ_LEVEL_ROUNDING_EPSILON);
        const Float32x16 point_five = Set16(0.5f);
        float k = scale / std::sqrt(squared_norm);
        if (max_level > 0) {
            float best_objective = -1.0f;
            float best_k = k;
            for (uint32_t point = 0; point < SAQ_SCALE_GRID_POINTS; ++point) {
                const float candidate = k * scalar_computer::GridPointFactor(point);
                const Float32x16 candidate_vec = Set16(candidate);
                Float32x16 numerator = Zero16();
                Float32x16 denominator = Zero16();
                for (size_t j = 0; j < num_values; j += 16) {
                    const Float32x16 magnitude = Abs16(Load16(values + j));
                    const Float32x16 value = Add16(
                        ToFloat32x16(Levels16(Fma16(magnitude, candidate_vec, epsilon), max_level)),
                        point_five
                    );
                    numerator = Fma16(value, magnitude, numerator);
                    denominator = Fma16(value, value, denominator);
                }
                const float objective =
                    ReduceAdd16(numerator) / std::sqrt(ReduceAdd16(denominator));
                if (objective > best_objective) {
                    best_objective = objective;
                    best_k = candidate;
                }
            }
            k = best_k;
        }
        const Float32x16 k_vec = Set16(k);
        Float32x16 inner_product = Zero16();
        for (size_t j = 0; j < num_values; j += 16) {
            const Float32x16 value = Load16(values + j);
            const Float32x16 magnitude = Abs16(value);
            const Int32x16 level = Levels16(Fma16(magnitude, k_vec, epsilon), max_level);
            inner_product = Fma16(Add16(ToFloat32x16(level), point_five), magnitude, inner_product);
            StoreCodes16(codes + j, level, value, half);
        }
        return squared_norm / ReduceAdd16(inner_product);
    }

    static void PackRecord(
        const uint8_t* PDX_RESTRICT levels,
        uint32_t width,
        uint8_t* PDX_RESTRICT record
    ) {
        static constexpr void (*PACK_RECORDS[SAQ_MAX_CODE_WIDTH + 1])(const uint8_t*, uint8_t*) =
            {nullptr,
             PackRecordOfWidth<1>,
             PackRecordOfWidth<2>,
             PackRecordOfWidth<3>,
             PackRecordOfWidth<4>,
             PackRecordOfWidth<5>,
             PackRecordOfWidth<6>,
             PackRecordOfWidth<7>,
             PackRecordOfWidth<8>};
        PACK_RECORDS[width](levels, record);
    }

    static void DecodeRecord(
        const uint8_t* PDX_RESTRICT record,
        uint32_t width,
        float rescale,
        float center,
        const float* PDX_RESTRICT mean,
        size_t num_values,
        float* PDX_RESTRICT out
    ) {
        if (num_values != SAQ_CHUNK_SIZE) {
            scalar_computer::DecodeRecord(record, width, rescale, center, mean, num_values, out);
            return;
        }
        static constexpr void (*DECODE_RECORDS
                                   [SAQ_MAX_CODE_WIDTH +
                                    1])(const uint8_t*, float, float, const float*, float*) =
            {nullptr,
             DecodeRecordOfWidth<1>,
             DecodeRecordOfWidth<2>,
             DecodeRecordOfWidth<3>,
             DecodeRecordOfWidth<4>,
             DecodeRecordOfWidth<5>,
             DecodeRecordOfWidth<6>,
             DecodeRecordOfWidth<7>,
             DecodeRecordOfWidth<8>};
        DECODE_RECORDS[width](record, rescale, center, mean, out);
    }

  private:
    // 16 lanes in 2 registers, the codec's unit of work
    struct Float32x16 {
        __m256 low, high;
    };
    struct Int32x16 {
        __m256i low, high;
    };

    static Float32x16 Zero16() { return Set16(0.0f); }

    static Float32x16 Set16(float x) {
        const __m256 v = _mm256_set1_ps(x);
        return {v, v};
    }

    static Float32x16 Load16(const float* PDX_RESTRICT p) {
        return {_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8)};
    }

    static void Store16(float* PDX_RESTRICT p, const Float32x16& a) {
        _mm256_storeu_ps(p, a.low);
        _mm256_storeu_ps(p + 8, a.high);
    }

    static Float32x16 Sub16(const Float32x16& a, const Float32x16& b) {
        return {_mm256_sub_ps(a.low, b.low), _mm256_sub_ps(a.high, b.high)};
    }

    static Float32x16 Add16(const Float32x16& a, const Float32x16& b) {
        return {_mm256_add_ps(a.low, b.low), _mm256_add_ps(a.high, b.high)};
    }

    // a * b + c
    static Float32x16 Fma16(const Float32x16& a, const Float32x16& b, const Float32x16& c) {
        return {_mm256_fmadd_ps(a.low, b.low, c.low), _mm256_fmadd_ps(a.high, b.high, c.high)};
    }

    static Float32x16 Abs16(const Float32x16& a) {
        const __m256 magnitude_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        return {_mm256_and_ps(a.low, magnitude_mask), _mm256_and_ps(a.high, magnitude_mask)};
    }

    static float ReduceAdd16(const Float32x16& a) {
        const __m256 sum = _mm256_add_ps(a.low, a.high);
        __m128 halves = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        halves = _mm_hadd_ps(halves, halves);
        return _mm_cvtss_f32(_mm_hadd_ps(halves, halves));
    }

    // min(trunc(x), max_level)
    static Int32x16 Levels16(const Float32x16& x, int32_t max_level) {
        const __m256i max_vec = _mm256_set1_epi32(max_level);
        return {
            _mm256_min_epi32(_mm256_cvttps_epi32(x.low), max_vec),
            _mm256_min_epi32(_mm256_cvttps_epi32(x.high), max_vec)
        };
    }

    static Float32x16 ToFloat32x16(const Int32x16& a) {
        return {_mm256_cvtepi32_ps(a.low), _mm256_cvtepi32_ps(a.high)};
    }

    // The bytes half + level where value >= 0, half - 1 - level (half + ~level) elsewhere
    static void StoreCodes16(
        uint8_t* PDX_RESTRICT codes,
        const Int32x16& level,
        const Float32x16& value,
        int32_t half
    ) {
        const __m256 zero = _mm256_setzero_ps();
        const __m256i half_vec = _mm256_set1_epi32(half);
        const __m256i low_codes = _mm256_add_epi32(
            half_vec,
            _mm256_xor_si256(
                level.low, _mm256_castps_si256(_mm256_cmp_ps(value.low, zero, _CMP_LT_OQ))
            )
        );
        const __m256i high_codes = _mm256_add_epi32(
            half_vec,
            _mm256_xor_si256(
                level.high, _mm256_castps_si256(_mm256_cmp_ps(value.high, zero, _CMP_LT_OQ))
            )
        );
        // packus interleaves the 128-bit lanes; the permute puts the 16 words back in order
        const __m256i words =
            _mm256_permute4x64_epi64(_mm256_packus_epi32(low_codes, high_codes), 0xD8);
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(codes),
            _mm_packus_epi16(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1))
        );
    }

    // ScalarComputer::PackRecord for one width
    template <uint32_t WIDTH>
    static void PackRecordOfWidth(
        const uint8_t* PDX_RESTRICT levels,
        uint8_t* PDX_RESTRICT record
    ) {
        const __m256i low_levels = LoadBytes32(levels);
        const __m256i high_levels = LoadBytes32(levels + 32);
        if constexpr (WIDTH == 8) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(record), low_levels);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(record + 32), high_levels);
            return;
        }
        if constexpr ((WIDTH & 4u) != 0) {
            const __m256i bytes = _mm256_or_si256(
                _mm256_and_si256(low_levels, _mm256_set1_epi8(0x0F)),
                _mm256_and_si256(
                    _mm256_slli_epi16(high_levels, 4), _mm256_set1_epi8(static_cast<char>(0xF0))
                )
            );
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(record), bytes);
            record += 32;
        }
        if constexpr ((WIDTH & 2u) != 0) {
            constexpr int SHIFT = (WIDTH & 4u) != 0 ? 4 : 0;
            const __m256i two_bit_mask = _mm256_set1_epi8(3);
            const __m256i low_parts =
                _mm256_and_si256(_mm256_srli_epi16(low_levels, SHIFT), two_bit_mask);
            const __m256i high_parts =
                _mm256_and_si256(_mm256_srli_epi16(high_levels, SHIFT), two_bit_mask);
            const __m128i bytes = _mm_or_si128(
                _mm_or_si128(
                    _mm256_castsi256_si128(low_parts),
                    _mm_slli_epi16(_mm256_extracti128_si256(low_parts, 1), 2)
                ),
                _mm_or_si128(
                    _mm_slli_epi16(_mm256_castsi256_si128(high_parts), 4),
                    _mm_slli_epi16(_mm256_extracti128_si256(high_parts, 1), 6)
                )
            );
            _mm_storeu_si128(reinterpret_cast<__m128i*>(record), bytes);
            record += 16;
        }
        if constexpr ((WIDTH & 1u) != 0) {
            // Each level's top bit to its byte's bit 7, then one movemask per 32 levels
            constexpr int UP = 8 - static_cast<int>(WIDTH);
            const auto low_mask =
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(low_levels, UP)));
            const auto high_mask =
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(high_levels, UP)));
            const uint64_t bits =
                static_cast<uint64_t>(low_mask) | (static_cast<uint64_t>(high_mask) << 32);
            std::memcpy(record, &bits, sizeof(bits));
        }
    }

    // ScalarComputer::DecodeRecord for one width and a whole chunk, levels widened in registers
    template <uint32_t WIDTH>
    static void DecodeRecordOfWidth(
        const uint8_t* PDX_RESTRICT record,
        float rescale,
        float center,
        const float* PDX_RESTRICT mean,
        float* PDX_RESTRICT out
    ) {
        __m256i levels[2];
        UnpackLevels<WIDTH>(record, levels);
        const __m256 rescale_vec = _mm256_set1_ps(rescale);
        const __m256 center_vec = _mm256_set1_ps(center);
        for (size_t half_idx = 0; half_idx < 2; ++half_idx) {
            const __m128i low = _mm256_castsi256_si128(levels[half_idx]);
            const __m128i high = _mm256_extracti128_si256(levels[half_idx], 1);
            const __m128i pieces[4] = {low, _mm_srli_si128(low, 8), high, _mm_srli_si128(high, 8)};
            for (size_t piece = 0; piece < 4; ++piece) {
                const size_t offset = 32 * half_idx + 8 * piece;
                const __m256 floats = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pieces[piece]));
                _mm256_storeu_ps(
                    out + offset,
                    _mm256_fmadd_ps(
                        _mm256_sub_ps(floats, center_vec),
                        rescale_vec,
                        _mm256_loadu_ps(mean + offset)
                    )
                );
            }
        }
    }

    static __m256i LoadBytes32(const void* address) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(address));
    }

    static __m128i LoadBytes16(const void* address) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(address));
    }

    static int32_t ReduceAdd(const __m256i vec) {
        __m128i halves =
            _mm_add_epi32(_mm256_castsi256_si128(vec), _mm256_extracti128_si256(vec, 1));
        halves = _mm_hadd_epi32(halves, halves);
        return _mm_cvtsi128_si32(_mm_hadd_epi32(halves, halves));
    }

    static __m256 LoadHalves8(const uint16_t* PDX_RESTRICT halves) {
#if defined(__F16C__)
        return _mm256_cvtph_ps(LoadBytes16(halves));
#else
        float values[8];
        for (size_t i = 0; i < 8; ++i) {
            values[i] = ReadHalf(reinterpret_cast<const uint8_t*>(halves + i));
        }
        return _mm256_loadu_ps(values);
#endif
    }

    // Each 32-bit lane plus the dot of its 4 head levels (8 vectors x 4 levels of one group) with
    // the group's 4 query codes. Without VNNI the lanes come out as vectors 0 1 4 5 | 2 3 6 7,
    // which FinishHeadDots reorders
    static __m256i AddHeadGroupDots(__m256i acc, __m256i levels, int32_t codes) {
#if defined(__AVXVNNI__)
        return _mm256_dpbusd_avx_epi32(acc, levels, _mm256_set1_epi32(codes));
#else
        const __m256i query = _mm256_cvtepi8_epi16(_mm_set1_epi32(codes));
        const __m256i low_vectors =
            _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(levels)), query);
        const __m256i high_vectors =
            _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(levels, 1)), query);
        return _mm256_add_epi32(acc, _mm256_hadd_epi32(low_vectors, high_vectors));
#endif
    }

    static __m256i FinishHeadDots(__m256i dots) {
#if defined(__AVXVNNI__)
        return dots;
#else
        return _mm256_permute4x64_epi64(dots, 0xD8);
#endif
    }

    // The dot of 32 levels with 32 query codes, added into 8 partial sums. Levels below 128 (widths
    // up to 7) keep maddubs' pair sums within s16
    template <bool SMALL_LEVELS>
    static __m256i AddLevelDots(__m256i acc, __m256i levels, __m256i query) {
#if defined(__AVXVNNI__)
        return _mm256_dpbusd_avx_epi32(acc, levels, query);
#else
        if constexpr (SMALL_LEVELS) {
            return _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(levels, query), _mm256_set1_epi16(1))
            );
        } else {
            const __m256i low = _mm256_madd_epi16(
                _mm256_cvtepu8_epi16(_mm256_castsi256_si128(levels)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(query))
            );
            const __m256i high = _mm256_madd_epi16(
                _mm256_cvtepu8_epi16(_mm256_extracti128_si256(levels, 1)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(query, 1))
            );
            return _mm256_add_epi32(acc, _mm256_add_epi32(low, high));
        }
#endif
    }

    static constexpr size_t NUM_HEAD_GROUPS = SAQ_CHUNK_SIZE / U8_INTERLEAVE_SIZE;

    // All ones in the first num_valid of the 8 32-bit lanes
    static __m256i ValidLanes8(size_t num_valid) {
        return _mm256_cmpgt_epi32(
            _mm256_set1_epi32(static_cast<int>(num_valid)),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
        );
    }

    // The first num_valid (up to 8) halves, the rest zero, reading none past them
    static __m256 LoadHalves8(const uint16_t* PDX_RESTRICT halves, size_t num_valid) {
        if (num_valid == 8) {
            return LoadHalves8(halves);
        }
        uint16_t valid_halves[8] = {};
        std::memcpy(valid_halves, halves, num_valid * sizeof(uint16_t));
        return LoadHalves8(valid_halves);
    }

    // The head dots of the num_valid (up to 8) vectors whose levels start at levels (groups stride
    // vectors apart), reading no levels past them
    static __m256i HeadDots8(
        const uint8_t* PDX_RESTRICT levels,
        size_t stride,
        size_t num_valid,
        const int32_t group_codes[NUM_HEAD_GROUPS]
    ) {
        const __m256i valid_lanes = ValidLanes8(num_valid);
        __m256i dots[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};
        for (size_t group_idx = 0; group_idx < NUM_HEAD_GROUPS; ++group_idx) {
            const uint8_t* group_levels = levels + group_idx * U8_INTERLEAVE_SIZE * stride;
            const __m256i group_vectors =
                num_valid == 8 ? LoadBytes32(group_levels)
                               : _mm256_maskload_epi32(
                                     reinterpret_cast<const int*>(group_levels), valid_lanes
                                 );
            dots[group_idx % 2] =
                AddHeadGroupDots(dots[group_idx % 2], group_vectors, group_codes[group_idx]);
        }
        return FinishHeadDots(_mm256_add_epi32(dots[0], dots[1]));
    }

    // The dot of a WIDTH-bit tail record with its query chunk
    template <uint32_t WIDTH>
    static int32_t TailDot(const uint8_t* PDX_RESTRICT record, const __m256i query[2]) {
        __m256i levels[2];
        UnpackLevels<WIDTH>(record, levels);
        const __m256i dot = AddLevelDots<(WIDTH < 8)>(_mm256_setzero_si256(), levels[0], query[0]);
        return ReduceAdd(AddLevelDots<(WIDTH < 8)>(dot, levels[1], query[1]));
    }

    // The 64 levels of a WIDTH-bit tail record (ScalarComputer::PackRecord's format), in order
    template <uint32_t WIDTH>
    static void UnpackLevels(const uint8_t* PDX_RESTRICT record, __m256i levels[2]) {
        if constexpr (WIDTH == 8) {
            levels[0] = LoadBytes32(record);
            levels[1] = LoadBytes32(record + 32);
        } else {
            levels[0] = _mm256_setzero_si256();
            levels[1] = _mm256_setzero_si256();
            if constexpr ((WIDTH & 4u) != 0) {
                const __m256i bytes = LoadBytes32(record);
                const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
                levels[0] = _mm256_and_si256(bytes, nibble_mask);
                levels[1] = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nibble_mask);
                record += 32;
            }
            if constexpr ((WIDTH & 2u) != 0) {
                constexpr int SHIFT = (WIDTH & 4u) != 0 ? 4 : 0;
                // Both 128-bit lanes get the 16 bytes; each takes the bit pairs of its 16
                // dimensions
                const __m256i bytes = _mm256_broadcastsi128_si256(LoadBytes16(record));
                const __m256i two_bit_mask = _mm256_set1_epi8(3);
                __m256i low_parts = _mm256_and_si256(
                    _mm256_srlv_epi32(bytes, _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2)),
                    two_bit_mask
                );
                __m256i high_parts = _mm256_and_si256(
                    _mm256_srlv_epi32(bytes, _mm256_setr_epi32(4, 4, 4, 4, 6, 6, 6, 6)),
                    two_bit_mask
                );
                if constexpr (SHIFT != 0) {
                    low_parts = _mm256_slli_epi16(low_parts, SHIFT);
                    high_parts = _mm256_slli_epi16(high_parts, SHIFT);
                }
                levels[0] = _mm256_or_si256(levels[0], low_parts);
                levels[1] = _mm256_or_si256(levels[1], high_parts);
                record += 16;
            }
            if constexpr ((WIDTH & 1u) != 0) {
                uint64_t bits;
                std::memcpy(&bits, record, sizeof(bits));
                const __m256i mask_bytes = _mm256_set1_epi64x(static_cast<long long>(bits));
                const __m256i lane_bits = _mm256_setr_epi8(
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    -128,
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    -128,
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    -128,
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    -128
                );
                const __m256i top_bit = _mm256_set1_epi8(static_cast<char>(1u << (WIDTH - 1)));
                // Lane j of dimensions 32h to 32h + 31 takes mask byte 4h + j / 8 and tests bit j %
                // 8
                const auto expand = [&](__m256i byte_indices) {
                    const __m256i selected =
                        _mm256_and_si256(_mm256_shuffle_epi8(mask_bytes, byte_indices), lane_bits);
                    return _mm256_and_si256(_mm256_cmpeq_epi8(selected, lane_bits), top_bit);
                };
                levels[0] = _mm256_or_si256(
                    levels[0],
                    expand(_mm256_setr_epi8(
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3,
                        3
                    ))
                );
                levels[1] = _mm256_or_si256(
                    levels[1],
                    expand(_mm256_setr_epi8(
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        5,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        6,
                        7,
                        7,
                        7,
                        7,
                        7,
                        7,
                        7,
                        7
                    ))
                );
            }
        }
    }

    // Stores the lanes of keep_mask's surviving vectors at the front of the outputs (all 8 lanes
    // are written), returns how many survived
    static size_t Compact8(
        uint32_t keep_mask,
        __m256i positions_in,
        __m256 estimates_in,
        __m256 rescales_in,
        uint32_t* PDX_RESTRICT positions,
        float* PDX_RESTRICT estimates,
        float* PDX_RESTRICT rescales
    ) {
        const __m256i lanes = _mm256_and_si256(
            _mm256_srlv_epi32(
                _mm256_set1_epi32(static_cast<int>(SAQ_COMPACTION_TABLE.packed_lanes[keep_mask])),
                _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21)
            ),
            _mm256_set1_epi32(7)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(positions), _mm256_permutevar8x32_epi32(positions_in, lanes)
        );
        _mm256_storeu_ps(estimates, _mm256_permutevar8x32_ps(estimates_in, lanes));
        _mm256_storeu_ps(rescales, _mm256_permutevar8x32_ps(rescales_in, lanes));
        return static_cast<size_t>(__builtin_popcount(keep_mask));
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

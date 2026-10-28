#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"
#include <cassert>
#include <cstdint>
#include <immintrin.h>

namespace PDX {

template <DistanceMetric alpha, Quantization Q>
class SIMDComputer {};

struct ClusterSketchReversedBitsTable {
    uint8_t bytes[256];
};

// Bit k of a comparison mask is dimension k, which a sketch code byte holds at bit 7 - k
constexpr ClusterSketchReversedBitsTable MakeClusterSketchReversedBitsTable() {
    ClusterSketchReversedBitsTable table{};
    for (uint32_t value = 0; value < 256; ++value) {
        uint32_t reversed = 0;
        for (uint32_t bit = 0; bit < 8; ++bit) {
            reversed |= ((value >> bit) & 1u) << (7 - bit);
        }
        table.bytes[value] = static_cast<uint8_t>(reversed);
    }
    return table;
}

inline constexpr ClusterSketchReversedBitsTable CLUSTER_SKETCH_REVERSED_BITS_TABLE =
    MakeClusterSketchReversedBitsTable();

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

    // scalar_computer documents it. RaBitQ-Library's accumulate_hacc: 64 code bytes are one group of
    // 4 codebooks of the lookup table; the u16 sums of a lane gain at most 2 * 255 per code byte, so
    // they flush into u32 every 128 code bytes (1024 dimensions)
    static bool SketchBatchMayHoldNeighbor(
        const uint8_t* PDX_RESTRICT batch,
        size_t code_size_in_bytes,
        const uint8_t* PDX_RESTRICT lookup_table,
        float lookup_table_step,
        float lookup_table_base_sum,
        float centering_correction,
        float squared_distance_to_mean,
        float error_scale,
        float threshold
    ) {
        constexpr size_t CODEBOOK_GROUP_SIZE_IN_BYTES = 128;
        constexpr size_t COLUMNS_PER_FLUSH = 128;
        const __m512i nibble_mask = _mm512_set1_epi8(0x0F);
        // Lanes 0-15 and 16-31
        __m512i sums[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
        for (size_t first = 0; first < code_size_in_bytes; first += COLUMNS_PER_FLUSH) {
            const size_t last = std::min(code_size_in_bytes, first + COLUMNS_PER_FLUSH);
            // [low-byte table, high-byte table][lanes 0-15: words, words >> 8; lanes 16-31: same]
            __m512i partial_sums[2][4];
            for (auto& byte_table : partial_sums) {
                for (auto& partial : byte_table) {
                    partial = _mm512_setzero_si512();
                }
            }
            for (size_t column = first; column < last; column += 2) {
                // The 128-bit lanes are codebooks 2 * column to 2 * column + 3
                const __m512i codes = _mm512_loadu_si512(batch + column * CLUSTER_SKETCH_BATCH_SIZE);
                const __m512i low_nibbles = _mm512_and_si512(codes, nibble_mask);
                const __m512i high_nibbles =
                    _mm512_and_si512(_mm512_srli_epi16(codes, 4), nibble_mask);
                const uint8_t* tables =
                    lookup_table + (column / 2) * CODEBOOK_GROUP_SIZE_IN_BYTES;
                for (size_t byte_table = 0; byte_table < 2; ++byte_table) {
                    const __m512i table = _mm512_loadu_si512(tables + byte_table * 64);
                    const __m512i low_lanes = _mm512_shuffle_epi8(table, low_nibbles);
                    const __m512i high_lanes = _mm512_shuffle_epi8(table, high_nibbles);
                    __m512i* partial = partial_sums[byte_table];
                    partial[0] = _mm512_add_epi16(partial[0], low_lanes);
                    partial[1] = _mm512_add_epi16(partial[1], _mm512_srli_epi16(low_lanes, 8));
                    partial[2] = _mm512_add_epi16(partial[2], high_lanes);
                    partial[3] = _mm512_add_epi16(partial[3], _mm512_srli_epi16(high_lanes, 8));
                }
            }
            // Per byte table: lanes 0-15 and 16-31 as u32, in lane order (the even positions are
            // lanes 0-7, the odd ones lanes 8-15)
            __m512i lane_sums[2][2];
            for (size_t byte_table = 0; byte_table < 2; ++byte_table) {
                const __m512i* partial = partial_sums[byte_table];
                for (size_t lane_half = 0; lane_half < 2; ++lane_half) {
                    const __m256i words = _mm256_add_epi16(
                        _mm512_castsi512_si256(partial[2 * lane_half]),
                        _mm512_extracti64x4_epi64(partial[2 * lane_half], 1)
                    );
                    const __m256i odd_positions = _mm256_add_epi16(
                        _mm512_castsi512_si256(partial[2 * lane_half + 1]),
                        _mm512_extracti64x4_epi64(partial[2 * lane_half + 1], 1)
                    );
                    const __m256i even_positions =
                        _mm256_sub_epi16(words, _mm256_slli_epi16(odd_positions, 8));
                    lane_sums[byte_table][lane_half] = _mm512_add_epi32(
                        _mm512_cvtepu16_epi32(
                            _mm256_permute2f128_si256(even_positions, odd_positions, 0x21)
                        ),
                        _mm512_cvtepu16_epi32(_mm256_blend_epi32(even_positions, odd_positions, 0xF0))
                    );
                }
            }
            for (size_t lane_half = 0; lane_half < 2; ++lane_half) {
                sums[lane_half] = _mm512_add_epi32(
                    sums[lane_half],
                    _mm512_add_epi32(
                        lane_sums[0][lane_half], _mm512_slli_epi32(lane_sums[1][lane_half], 8)
                    )
                );
            }
        }
        const auto* f_add =
            reinterpret_cast<const float*>(batch + code_size_in_bytes * CLUSTER_SKETCH_BATCH_SIZE);
        const float* f_rescale = f_add + CLUSTER_SKETCH_BATCH_SIZE;
        const float* f_error = f_rescale + CLUSTER_SKETCH_BATCH_SIZE;
        const __m512 step_vec = _mm512_set1_ps(lookup_table_step);
        const __m512 offset_vec = _mm512_set1_ps(lookup_table_base_sum + centering_correction);
        const __m512 distance_vec = _mm512_set1_ps(squared_distance_to_mean);
        const __m512 error_scale_vec = _mm512_set1_ps(error_scale);
        const __m512 threshold_vec = _mm512_set1_ps(threshold);
        for (size_t lane_half = 0; lane_half < 2; ++lane_half) {
            const size_t lane = lane_half * 16;
            const __m512 dot =
                _mm512_fmadd_ps(_mm512_cvtepi32_ps(sums[lane_half]), step_vec, offset_vec);
            const __m512 estimate = _mm512_fmadd_ps(
                _mm512_loadu_ps(f_rescale + lane),
                dot,
                _mm512_add_ps(_mm512_loadu_ps(f_add + lane), distance_vec)
            );
            const __m512 lower_bound =
                _mm512_fnmadd_ps(_mm512_loadu_ps(f_error + lane), error_scale_vec, estimate);
            // A NaN bound compares false (ordered), so it never prunes a cluster
            if (_mm512_cmp_ps_mask(lower_bound, threshold_vec, _CMP_GT_OQ) != 0xFFFF) {
                return true;
            }
        }
        return false;
    }

    // scalar_computer documents it. Per 8 dimensions, a code byte from the comparison mask with its
    // bits reversed; 32 dimensions per step, then 8 (a masked half)
    static void ComputeSketchCodeAndSums(
        const float* PDX_RESTRICT embedding,
        const float* PDX_RESTRICT mean,
        size_t num_dimensions,
        uint8_t* PDX_RESTRICT code,
        float& squared_norm,
        float& absolute_sum,
        float& mean_sum_by_sign
    ) {
        const __m512 zero = _mm512_setzero_ps();
        __m512 squared_norm_vec[2] = {zero, zero};
        __m512 absolute_sum_vec[2] = {zero, zero};
        __m512 mean_sum_by_sign_vec[2] = {zero, zero};
        // Adds a residual to the sums; returns its positive lanes
        const auto accumulate = [&](const __m512 embedding_vec, const __m512 mean_vec, size_t half) {
            const __m512 residual = _mm512_sub_ps(embedding_vec, mean_vec);
            const __mmask16 positive = _mm512_cmp_ps_mask(residual, zero, _CMP_GT_OQ);
            squared_norm_vec[half] = _mm512_fmadd_ps(residual, residual, squared_norm_vec[half]);
            absolute_sum_vec[half] = _mm512_add_ps(absolute_sum_vec[half], _mm512_abs_ps(residual));
            mean_sum_by_sign_vec[half] = _mm512_mask_add_ps(
                mean_sum_by_sign_vec[half], positive, mean_sum_by_sign_vec[half], mean_vec
            );
            mean_sum_by_sign_vec[half] = _mm512_mask_sub_ps(
                mean_sum_by_sign_vec[half],
                static_cast<__mmask16>(~positive),
                mean_sum_by_sign_vec[half],
                mean_vec
            );
            return positive;
        };
        size_t i = 0;
        for (; i + 32 <= num_dimensions; i += 32) {
            uint32_t masks = 0;
            for (size_t half = 0; half < 2; ++half) {
                const size_t offset = i + 16 * half;
                masks |= static_cast<uint32_t>(accumulate(
                             _mm512_loadu_ps(embedding + offset), _mm512_loadu_ps(mean + offset), half
                         ))
                         << (16 * half);
            }
            uint32_t four_bytes = 0;
            for (size_t byte = 0; byte < 4; ++byte) {
                four_bytes |=
                    static_cast<uint32_t>(
                        CLUSTER_SKETCH_REVERSED_BITS_TABLE.bytes[(masks >> (8 * byte)) & 0xFF]
                    )
                    << (8 * byte);
            }
            std::memcpy(code + i / 8, &four_bytes, sizeof(four_bytes));
        }
        constexpr auto EIGHT_DIMENSIONS = static_cast<__mmask16>(0x00FF);
        for (; i + 8 <= num_dimensions; i += 8) {
            const __mmask16 positive = accumulate(
                _mm512_maskz_loadu_ps(EIGHT_DIMENSIONS, embedding + i),
                _mm512_maskz_loadu_ps(EIGHT_DIMENSIONS, mean + i),
                0
            );
            code[i / 8] = CLUSTER_SKETCH_REVERSED_BITS_TABLE.bytes[positive & 0xFF];
        }
        squared_norm += _mm512_reduce_add_ps(_mm512_add_ps(squared_norm_vec[0], squared_norm_vec[1]));
        absolute_sum += _mm512_reduce_add_ps(_mm512_add_ps(absolute_sum_vec[0], absolute_sum_vec[1]));
        mean_sum_by_sign +=
            _mm512_reduce_add_ps(_mm512_add_ps(mean_sum_by_sign_vec[0], mean_sum_by_sign_vec[1]));
        scalar_computer::ComputeSketchCodeAndSums(
            embedding + i,
            mean + i,
            num_dimensions - i,
            code + i / 8,
            squared_norm,
            absolute_sum,
            mean_sum_by_sign
        );
    }

    // scalar_computer documents it. Rows i and i + 16 share a register, so one transpose pair
    // turns the 32 rows into columns
    static void PackSketchColumns(
        const uint8_t* PDX_RESTRICT rows,
        size_t row_stride,
        size_t first_column,
        size_t num_columns,
        uint8_t* PDX_RESTRICT batch
    ) {
        constexpr size_t HALF_COLUMN = CLUSTER_SKETCH_BATCH_SIZE / 2;
        __m256i lanes[HALF_COLUMN];
        for (size_t lane = 0; lane < HALF_COLUMN; ++lane) {
            lanes[lane] = _mm256_loadu2_m128i(
                reinterpret_cast<const __m128i*>(
                    rows + (lane + HALF_COLUMN) * row_stride + first_column
                ),
                reinterpret_cast<const __m128i*>(rows + lane * row_stride + first_column)
            );
        }
        TransposeBytes16x16Pair(lanes);
        for (size_t column = 0; column < num_columns; ++column) {
            PackSketchColumn(lanes[column], batch + (first_column + column) * CLUSTER_SKETCH_BATCH_SIZE);
        }
    }

    // scalar_computer documents it
    static void UnpackSketchColumns(
        const uint8_t* PDX_RESTRICT batch,
        size_t first_column,
        size_t num_columns,
        size_t num_rows,
        uint8_t* PDX_RESTRICT rows,
        size_t row_stride
    ) {
        constexpr size_t HALF_COLUMN = CLUSTER_SKETCH_BATCH_SIZE / 2;
        __m256i lanes[HALF_COLUMN];
        for (size_t column = 0; column < CLUSTER_SKETCH_TRANSPOSE_COLUMNS; ++column) {
            lanes[column] =
                column < num_columns
                    ? UnpackSketchColumn(batch + (first_column + column) * CLUSTER_SKETCH_BATCH_SIZE)
                    : _mm256_setzero_si256();
        }
        TransposeBytes16x16Pair(lanes);
        alignas(32) uint8_t row_bytes[CLUSTER_SKETCH_BATCH_SIZE];
        const auto row = [&](size_t lane) { return rows + lane * row_stride + first_column; };
        for (size_t lane = 0; lane < HALF_COLUMN; ++lane) {
            if (num_columns == CLUSTER_SKETCH_TRANSPOSE_COLUMNS) {
                if (lane < num_rows) {
                    _mm_storeu_si128(
                        reinterpret_cast<__m128i*>(row(lane)), _mm256_castsi256_si128(lanes[lane])
                    );
                }
                if (lane + HALF_COLUMN < num_rows) {
                    _mm_storeu_si128(
                        reinterpret_cast<__m128i*>(row(lane + HALF_COLUMN)),
                        _mm256_extracti128_si256(lanes[lane], 1)
                    );
                }
            } else {
                _mm256_store_si256(reinterpret_cast<__m256i*>(row_bytes), lanes[lane]);
                if (lane < num_rows) {
                    std::memcpy(row(lane), row_bytes, num_columns);
                }
                if (lane + HALF_COLUMN < num_rows) {
                    std::memcpy(row(lane + HALF_COLUMN), row_bytes + HALF_COLUMN, num_columns);
                }
            }
        }
    }

    // scalar_computer documents it. Per column (its halves in the two 128-bit lanes), the lane's
    // nibbles broadcast (vpshufb) and merged under the lane's mask; 16 columns per step
    static void StoreSketchLane(
        const uint8_t* PDX_RESTRICT code,
        size_t num_columns,
        uint8_t* PDX_RESTRICT columns,
        uint32_t lane
    ) {
        constexpr size_t HALF_COLUMN = CLUSTER_SKETCH_BATCH_SIZE / 2;
        const bool high_lanes = lane >= HALF_COLUMN;
        const __m128i half_mask = _mm_and_si128(
            _mm_cmpeq_epi8(
                _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                _mm_set1_epi8(static_cast<char>(CLUSTER_SKETCH_POSITION_OF_LANE[lane % HALF_COLUMN]))
            ),
            _mm_set1_epi8(static_cast<char>(high_lanes ? 0xF0 : 0x0F))
        );
        const __m256i lane_mask = _mm256_set_m128i(half_mask, half_mask);
        const __m128i low_nibble = _mm_set1_epi8(0x0F);
        const __m256i one = _mm256_set1_epi8(1);
        size_t column = 0;
        for (; column + 16 <= num_columns; column += 16) {
            const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + column));
            __m128i upper_nibbles = _mm_and_si128(_mm_srli_epi16(bytes, 4), low_nibble);
            __m128i lower_nibbles = _mm_and_si128(bytes, low_nibble);
            if (high_lanes) {
                upper_nibbles = _mm_slli_epi16(upper_nibbles, 4);
                lower_nibbles = _mm_slli_epi16(lower_nibbles, 4);
            }
            const __m256i nibbles = _mm256_set_m128i(lower_nibbles, upper_nibbles);
            __m256i index = _mm256_setzero_si256();
            for (size_t c = 0; c < 16; ++c) {
                auto* packed =
                    reinterpret_cast<__m256i*>(columns + (column + c) * CLUSTER_SKETCH_BATCH_SIZE);
                const __m256i values = _mm256_shuffle_epi8(nibbles, index);
                _mm256_storeu_si256(
                    packed,
                    _mm256_or_si256(
                        _mm256_andnot_si256(lane_mask, _mm256_loadu_si256(packed)),
                        _mm256_and_si256(lane_mask, values)
                    )
                );
                index = _mm256_add_epi8(index, one);
            }
        }
        scalar_computer::StoreSketchLane(
            code + column,
            num_columns - column,
            columns + column * CLUSTER_SKETCH_BATCH_SIZE,
            lane
        );
    }

    // scalar_computer documents it. Per column, the lane's byte of both halves broadcast (vpshufb)
    // and slid into a 16-byte accumulator per half (vpalignr), byte c holding column c after 16
    static void ReadSketchLane(
        const uint8_t* PDX_RESTRICT columns,
        size_t num_columns,
        uint32_t lane,
        uint8_t* PDX_RESTRICT code
    ) {
        constexpr size_t HALF_COLUMN = CLUSTER_SKETCH_BATCH_SIZE / 2;
        const bool high_lanes = lane >= HALF_COLUMN;
        const __m256i position =
            _mm256_set1_epi8(static_cast<char>(CLUSTER_SKETCH_POSITION_OF_LANE[lane % HALF_COLUMN]));
        const __m128i low_nibble = _mm_set1_epi8(0x0F);
        size_t column = 0;
        for (; column + 16 <= num_columns; column += 16) {
            __m256i halves = _mm256_setzero_si256();
            for (size_t c = 0; c < 16; ++c) {
                const __m256i lane_bytes = _mm256_shuffle_epi8(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        columns + (column + c) * CLUSTER_SKETCH_BATCH_SIZE
                    )),
                    position
                );
                halves = _mm256_alignr_epi8(lane_bytes, halves, 1);
            }
            __m128i upper_nibbles = _mm256_castsi256_si128(halves);
            __m128i lower_nibbles = _mm256_extracti128_si256(halves, 1);
            if (high_lanes) {
                upper_nibbles = _mm_srli_epi16(upper_nibbles, 4);
                lower_nibbles = _mm_srli_epi16(lower_nibbles, 4);
            }
            upper_nibbles = _mm_and_si128(upper_nibbles, low_nibble);
            lower_nibbles = _mm_and_si128(lower_nibbles, low_nibble);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(code + column),
                _mm_or_si128(_mm_slli_epi16(upper_nibbles, 4), lower_nibbles)
            );
        }
        scalar_computer::ReadSketchLane(
            columns + column * CLUSTER_SKETCH_BATCH_SIZE, num_columns - column, lane, code + column
        );
    }

    // scalar_computer documents it. Per 16 columns, ReadSketchLane's loop gathers the source lane's
    // nibbles into one register (both halves), then StoreSketchLane's loop writes them: every source
    // column is read before a target column is written
    static void CopySketchLane(
        const uint8_t* source_columns,
        uint32_t source_lane,
        uint8_t* target_columns,
        uint32_t target_lane,
        size_t num_columns
    ) {
        constexpr size_t HALF_COLUMN = CLUSTER_SKETCH_BATCH_SIZE / 2;
        const __m256i source_position = _mm256_set1_epi8(
            static_cast<char>(CLUSTER_SKETCH_POSITION_OF_LANE[source_lane % HALF_COLUMN])
        );
        const __m128i half_mask = _mm_and_si128(
            _mm_cmpeq_epi8(
                _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                _mm_set1_epi8(
                    static_cast<char>(CLUSTER_SKETCH_POSITION_OF_LANE[target_lane % HALF_COLUMN])
                )
            ),
            _mm_set1_epi8(static_cast<char>(target_lane >= HALF_COLUMN ? 0xF0 : 0x0F))
        );
        const __m256i target_mask = _mm256_set_m128i(half_mask, half_mask);
        const __m256i low_nibble = _mm256_set1_epi8(0x0F);
        const __m256i one = _mm256_set1_epi8(1);
        size_t column = 0;
        for (; column + 16 <= num_columns; column += 16) {
            __m256i nibbles = _mm256_setzero_si256();
            for (size_t c = 0; c < 16; ++c) {
                const __m256i lane_bytes = _mm256_shuffle_epi8(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        source_columns + (column + c) * CLUSTER_SKETCH_BATCH_SIZE
                    )),
                    source_position
                );
                nibbles = _mm256_alignr_epi8(lane_bytes, nibbles, 1);
            }
            if (source_lane >= HALF_COLUMN) {
                nibbles = _mm256_srli_epi16(nibbles, 4);
            }
            nibbles = _mm256_and_si256(nibbles, low_nibble);
            if (target_lane >= HALF_COLUMN) {
                nibbles = _mm256_slli_epi16(nibbles, 4);
            }
            __m256i index = _mm256_setzero_si256();
            for (size_t c = 0; c < 16; ++c) {
                auto* packed = reinterpret_cast<__m256i*>(
                    target_columns + (column + c) * CLUSTER_SKETCH_BATCH_SIZE
                );
                _mm256_storeu_si256(
                    packed,
                    _mm256_or_si256(
                        _mm256_andnot_si256(target_mask, _mm256_loadu_si256(packed)),
                        _mm256_and_si256(target_mask, _mm256_shuffle_epi8(nibbles, index))
                    )
                );
                index = _mm256_add_epi8(index, one);
            }
        }
        scalar_computer::CopySketchLane(
            source_columns + column * CLUSTER_SKETCH_BATCH_SIZE,
            source_lane,
            target_columns + column * CLUSTER_SKETCH_BATCH_SIZE,
            target_lane,
            num_columns - column
        );
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

    // One sketch column from its code byte of lanes 0-15 (low 128 bits) and 16-31 (high 128 bits)
    static void PackSketchColumn(const __m256i lanes, uint8_t* PDX_RESTRICT packed) {
        const __m128i low_lanes = _mm256_castsi256_si128(lanes);
        const __m128i high_lanes = _mm256_extracti128_si256(lanes, 1);
        const __m128i low_nibble = _mm_set1_epi8(0x0F);
        const __m128i high_nibble = _mm_set1_epi8(static_cast<char>(0xF0));
        const __m128i upper_nibbles = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(low_lanes, 4), low_nibble),
            _mm_and_si128(high_lanes, high_nibble)
        );
        const __m128i lower_nibbles = _mm_or_si128(
            _mm_and_si128(low_lanes, low_nibble),
            _mm_and_si128(_mm_slli_epi16(high_lanes, 4), high_nibble)
        );
        const __m128i lane_at_position =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(CLUSTER_SKETCH_LANE_AT_POSITION));
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(packed),
            _mm256_shuffle_epi8(
                _mm256_set_m128i(lower_nibbles, upper_nibbles),
                _mm256_set_m128i(lane_at_position, lane_at_position)
            )
        );
    }

    static __m256i UnpackSketchColumn(const uint8_t* PDX_RESTRICT packed) {
        const __m128i position_of_lane =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(CLUSTER_SKETCH_POSITION_OF_LANE));
        const __m256i nibbles = _mm256_shuffle_epi8(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed)),
            _mm256_set_m128i(position_of_lane, position_of_lane)
        );
        const __m128i upper_nibbles = _mm256_castsi256_si128(nibbles);
        const __m128i lower_nibbles = _mm256_extracti128_si256(nibbles, 1);
        const __m128i low_nibble = _mm_set1_epi8(0x0F);
        const __m128i high_nibble = _mm_set1_epi8(static_cast<char>(0xF0));
        const __m128i low_lanes = _mm_or_si128(
            _mm_and_si128(_mm_slli_epi16(upper_nibbles, 4), high_nibble),
            _mm_and_si128(lower_nibbles, low_nibble)
        );
        const __m128i high_lanes = _mm_or_si128(
            _mm_and_si128(upper_nibbles, high_nibble),
            _mm_and_si128(_mm_srli_epi16(lower_nibbles, 4), low_nibble)
        );
        return _mm256_set_m128i(high_lanes, low_lanes);
    }

    // A 16 x 16 byte transpose in each 128-bit half, in place. Each round interleaves rows i and
    // i + 8, rotating the 8 bits of (row, byte) by one; four rounds swap them
    static void TransposeBytes16x16Pair(__m256i* rows) {
        __m256i interleaved[16];
        for (size_t round = 0; round < 4; ++round) {
            for (size_t i = 0; i < 8; ++i) {
                interleaved[2 * i] = _mm256_unpacklo_epi8(rows[i], rows[i + 8]);
                interleaved[2 * i + 1] = _mm256_unpackhi_epi8(rows[i], rows[i + 8]);
            }
            for (size_t i = 0; i < 16; ++i) {
                rows[i] = interleaved[i];
            }
        }
    }
};

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
        const uint16_t* PDX_RESTRICT residual_norms,
        const uint8_t* PDX_RESTRICT selection,
        float threshold,
        uint32_t* PDX_RESTRICT positions,
        float* PDX_RESTRICT estimates,
        float* PDX_RESTRICT rescales
    ) {
        __m512i query_groups[NUM_HEAD_GROUPS];
        LoadHeadQuery(query_codes, query_groups);
        const __m512 dot_scale_vec = _mm512_set1_ps(dot_scale);
        const __m512 dot_offset_vec = _mm512_set1_ps(dot_offset);
        const __m512 query_squared_norm_vec = _mm512_set1_ps(query_squared_norm);
        const __m512 threshold_vec = _mm512_set1_ps(threshold);
        const __m512i lane_offsets =
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        size_t num_kept = 0;
        for (size_t vector_idx = 0; vector_idx < num_vectors; vector_idx += 16) {
            const size_t remaining = num_vectors - vector_idx;
            const __mmask16 valid = remaining >= 16 ? static_cast<__mmask16>(0xFFFF)
                                                    : static_cast<__mmask16>((1u << remaining) - 1);
            const __mmask64 valid_levels =
                remaining >= 16 ? ~__mmask64{0} : (__mmask64{1} << (4 * remaining)) - 1;
            const __m512i dot = HeadDots16(
                head + vector_idx * U8_INTERLEAVE_SIZE, stride, valid_levels, query_groups
            );
            const __m512 rescale =
                _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(valid, head_rescales + vector_idx));
            const __m512 residual_norm =
                _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(valid, residual_norms + vector_idx));
            const __m512 estimate = _mm512_fmadd_ps(
                rescale,
                _mm512_fmsub_ps(_mm512_cvtepi32_ps(dot), dot_scale_vec, dot_offset_vec),
                _mm512_fmadd_ps(residual_norm, residual_norm, query_squared_norm_vec)
            );
            __mmask16 keep = valid;
            if (selection != nullptr) {
                const __m128i selected = _mm_maskz_loadu_epi8(valid, selection + vector_idx);
                keep = _mm_test_epi8_mask(selected, selected);
            }
            keep = _mm512_mask_cmp_ps_mask(keep, estimate, threshold_vec, _CMP_LT_OQ);
            num_kept += Compact16(
                keep,
                _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(vector_idx)), lane_offsets),
                estimate,
                rescale,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept
            );
        }
        return num_kept;
    }

    // ScanHead over the vectors at positions only, 16 positions per batch with one gather per
    // group; keeps those below threshold, compacted in place
    static size_t ScanHeadPruning(
        const uint8_t* PDX_RESTRICT head,
        size_t stride,
        const int8_t* PDX_RESTRICT query_codes,
        float dot_scale,
        float dot_offset,
        float query_squared_norm,
        const uint16_t* PDX_RESTRICT head_rescales,
        const uint16_t* PDX_RESTRICT residual_norms,
        float threshold,
        uint32_t* PDX_RESTRICT positions,
        float* PDX_RESTRICT estimates,
        float* PDX_RESTRICT rescales,
        size_t num_positions
    ) {
        __m512i query_groups[NUM_HEAD_GROUPS];
        LoadHeadQuery(query_codes, query_groups);
        const __m512 dot_scale_vec = _mm512_set1_ps(dot_scale);
        const __m512 dot_offset_vec = _mm512_set1_ps(dot_offset);
        const __m512 query_squared_norm_vec = _mm512_set1_ps(query_squared_norm);
        const __m512 threshold_vec = _mm512_set1_ps(threshold);
        size_t num_kept = 0;
        for (size_t position_idx = 0; position_idx < num_positions; position_idx += 16) {
            const size_t remaining = num_positions - position_idx;
            const __mmask16 valid = remaining >= 16 ? static_cast<__mmask16>(0xFFFF)
                                                    : static_cast<__mmask16>((1u << remaining) - 1);
            const __m512i position_vec = _mm512_maskz_loadu_epi32(valid, positions + position_idx);
            const __m512i level_offsets = _mm512_slli_epi32(position_vec, 2);
            __m512i dots[4] = {
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512()
            };
            for (size_t group_idx = 0; group_idx < NUM_HEAD_GROUPS; ++group_idx) {
                dots[group_idx % 4] = _mm512_dpbusd_epi32(
                    dots[group_idx % 4],
                    _mm512_mask_i32gather_epi32(
                        _mm512_setzero_si512(),
                        valid,
                        level_offsets,
                        head + group_idx * U8_INTERLEAVE_SIZE * stride,
                        1
                    ),
                    query_groups[group_idx]
                );
            }
            const __m512i dot = _mm512_add_epi32(
                _mm512_add_epi32(dots[0], dots[1]), _mm512_add_epi32(dots[2], dots[3])
            );
            // Each gather reads 4 bytes from a half; the low 2 are the position's
            const __m512i half_offsets = _mm512_slli_epi32(position_vec, 1);
            const __m512 rescale =
                _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_mask_i32gather_epi32(
                    _mm512_setzero_si512(), valid, half_offsets, head_rescales, 1
                )));
            const __m512 residual_norm =
                _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_mask_i32gather_epi32(
                    _mm512_setzero_si512(), valid, half_offsets, residual_norms, 1
                )));
            const __m512 estimate = _mm512_fmadd_ps(
                rescale,
                _mm512_fmsub_ps(_mm512_cvtepi32_ps(dot), dot_scale_vec, dot_offset_vec),
                _mm512_fmadd_ps(residual_norm, residual_norm, query_squared_norm_vec)
            );
            num_kept += Compact16(
                _mm512_mask_cmp_ps_mask(valid, estimate, threshold_vec, _CMP_LT_OQ),
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
        const LevelUnpacker<WIDTH> unpack;
        const __m512i query = _mm512_loadu_si512(query_codes);
        const __m512 dot_scale_vec = _mm512_set1_ps(dot_scale);
        const __m512 dot_offset_vec = _mm512_set1_ps(dot_offset);
        const __m512 threshold_vec = _mm512_set1_ps(threshold);
        size_t num_kept = 0;
        for (size_t first_survivor = 0; first_survivor < num_survivors; first_survivor += 16) {
            const size_t batch_size = std::min<size_t>(16, num_survivors - first_survivor);
            const __mmask16 valid = batch_size == 16
                                        ? static_cast<__mmask16>(0xFFFF)
                                        : static_cast<__mmask16>((1u << batch_size) - 1);
            const __m512i batch_positions =
                _mm512_maskz_loadu_epi32(valid, positions + first_survivor);
            const __m512i dot = TailDots16(unpack, query, [&](size_t lane) {
                const uint32_t position =
                    positions[first_survivor + (lane < batch_size ? lane : 0)];
                return records + static_cast<size_t>(position) * 8 * WIDTH;
            });
            __m512 rescale;
            if (segment_rescales != nullptr) {
                const __m512i gathered = _mm512_mask_i32gather_epi32(
                    _mm512_setzero_si512(), valid, batch_positions, segment_rescales, 2
                );
                rescale = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(gathered));
            } else {
                rescale = _mm512_maskz_loadu_ps(valid, rescales + first_survivor);
            }
            const __m512 estimate = _mm512_fmadd_ps(
                rescale,
                _mm512_fmsub_ps(_mm512_cvtepi32_ps(dot), dot_scale_vec, dot_offset_vec),
                _mm512_maskz_loadu_ps(valid, estimates + first_survivor)
            );
            num_kept += Compact16(
                _mm512_mask_cmp_ps_mask(valid, estimate, threshold_vec, _CMP_LT_OQ),
                batch_positions,
                estimate,
                rescale,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept
            );
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
        const uint16_t* PDX_RESTRICT residual_norms,
        float* PDX_RESTRICT estimates
    ) {
        __m512i query_groups[NUM_HEAD_GROUPS];
        LoadHeadQuery(query_codes, query_groups);
        const __m512 dot_scale_vec = _mm512_set1_ps(dot_scale);
        const __m512 dot_offset_vec = _mm512_set1_ps(dot_offset);
        const __m512 query_squared_norm_vec = _mm512_set1_ps(query_squared_norm);
        for (size_t vector_idx = 0; vector_idx < num_vectors; vector_idx += 16) {
            const size_t remaining = num_vectors - vector_idx;
            const __mmask16 valid = remaining >= 16 ? static_cast<__mmask16>(0xFFFF)
                                                    : static_cast<__mmask16>((1u << remaining) - 1);
            const __mmask64 valid_levels =
                remaining >= 16 ? ~__mmask64{0} : (__mmask64{1} << (4 * remaining)) - 1;
            const __m512i dot = HeadDots16(
                head + vector_idx * U8_INTERLEAVE_SIZE, stride, valid_levels, query_groups
            );
            const __m512 rescale =
                _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(valid, head_rescales + vector_idx));
            const __m512 residual_norm =
                _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(valid, residual_norms + vector_idx));
            _mm512_mask_storeu_ps(
                estimates + vector_idx,
                valid,
                _mm512_fmadd_ps(
                    rescale,
                    _mm512_fmsub_ps(_mm512_cvtepi32_ps(dot), dot_scale_vec, dot_offset_vec),
                    _mm512_fmadd_ps(residual_norm, residual_norm, query_squared_norm_vec)
                )
            );
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
        const LevelUnpacker<WIDTH> unpack;
        const __m512i query = _mm512_loadu_si512(query_codes);
        const __m512 dot_scale_vec = _mm512_set1_ps(dot_scale);
        const __m512 dot_offset_vec = _mm512_set1_ps(dot_offset);
        for (size_t first_vector = 0; first_vector < num_vectors; first_vector += 16) {
            const size_t batch_size = std::min<size_t>(16, num_vectors - first_vector);
            const __mmask16 valid = batch_size == 16
                                        ? static_cast<__mmask16>(0xFFFF)
                                        : static_cast<__mmask16>((1u << batch_size) - 1);
            const __m512i dot = TailDots16(unpack, query, [&](size_t lane) {
                return records + (first_vector + (lane < batch_size ? lane : 0)) * 8 * WIDTH;
            });
            const __m512 rescale =
                _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(valid, segment_rescales + first_vector));
            _mm512_mask_storeu_ps(
                estimates + first_vector,
                valid,
                _mm512_fmadd_ps(
                    rescale,
                    _mm512_fmsub_ps(_mm512_cvtepi32_ps(dot), dot_scale_vec, dot_offset_vec),
                    _mm512_maskz_loadu_ps(valid, estimates + first_vector)
                )
            );
        }
    }

    using scalar_computer = ScalarComputer<DistanceMetric::L2SQ, Quantization::SAQ>;

    static float ComputeResidual(
        const float* PDX_RESTRICT embedding,
        const float* PDX_RESTRICT mean,
        size_t num_dimensions,
        float* PDX_RESTRICT residual
    ) {
        __m512 accumulator = _mm512_setzero_ps();
        size_t j = 0;
        for (; j + 16 <= num_dimensions; j += 16) {
            const __m512 value =
                _mm512_sub_ps(_mm512_loadu_ps(embedding + j), _mm512_loadu_ps(mean + j));
            _mm512_storeu_ps(residual + j, value);
            accumulator = _mm512_fmadd_ps(value, value, accumulator);
        }
        float squared_norm = _mm512_reduce_add_ps(accumulator);
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
        __m512 squares = _mm512_setzero_ps();
        for (size_t j = 0; j < num_values; j += 16) {
            const __m512 value = _mm512_loadu_ps(values + j);
            squares = _mm512_fmadd_ps(value, value, squares);
        }
        const float squared_norm = _mm512_reduce_add_ps(squares);
        if (squared_norm <= 0.0f) {
            std::fill_n(codes, num_values, static_cast<uint8_t>(half));
            return 0.0f;
        }
        const __m512 epsilon = _mm512_set1_ps(SAQ_LEVEL_ROUNDING_EPSILON);
        const __m512 point_five = _mm512_set1_ps(0.5f);
        const __m512i max_level_vec = _mm512_set1_epi32(max_level);
        const auto levels_of = [&](__m512 magnitude, __m512 k_vec) {
            return _mm512_min_epi32(
                _mm512_cvttps_epi32(_mm512_fmadd_ps(magnitude, k_vec, epsilon)), max_level_vec
            );
        };
        float k = scale / std::sqrt(squared_norm);
        if (max_level > 0) {
            float best_objective = -1.0f;
            float best_k = k;
            for (uint32_t point = 0; point < SAQ_SCALE_GRID_POINTS; ++point) {
                const float candidate = k * scalar_computer::GridPointFactor(point);
                const __m512 candidate_vec = _mm512_set1_ps(candidate);
                __m512 numerator = _mm512_setzero_ps();
                __m512 denominator = _mm512_setzero_ps();
                for (size_t j = 0; j < num_values; j += 16) {
                    const __m512 magnitude = _mm512_abs_ps(_mm512_loadu_ps(values + j));
                    const __m512 value = _mm512_add_ps(
                        _mm512_cvtepi32_ps(levels_of(magnitude, candidate_vec)), point_five
                    );
                    numerator = _mm512_fmadd_ps(value, magnitude, numerator);
                    denominator = _mm512_fmadd_ps(value, value, denominator);
                }
                const float objective =
                    _mm512_reduce_add_ps(numerator) / std::sqrt(_mm512_reduce_add_ps(denominator));
                if (objective > best_objective) {
                    best_objective = objective;
                    best_k = candidate;
                }
            }
            k = best_k;
        }
        const __m512 k_vec = _mm512_set1_ps(k);
        const __m512i half_vec = _mm512_set1_epi32(half);
        const __m512i below_half_vec = _mm512_set1_epi32(half - 1);
        __m512 inner_product = _mm512_setzero_ps();
        for (size_t j = 0; j < num_values; j += 16) {
            const __m512 value = _mm512_loadu_ps(values + j);
            const __m512 magnitude = _mm512_abs_ps(value);
            const __m512i level = levels_of(magnitude, k_vec);
            inner_product = _mm512_fmadd_ps(
                _mm512_add_ps(_mm512_cvtepi32_ps(level), point_five), magnitude, inner_product
            );
            // half + level where value >= 0, half - 1 - level elsewhere
            const __mmask16 negative = _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_LT_OQ);
            const __m512i level_codes = _mm512_mask_sub_epi32(
                _mm512_add_epi32(half_vec, level), negative, below_half_vec, level
            );
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(codes + j), _mm512_cvtepi32_epi8(level_codes)
            );
        }
        return squared_norm / _mm512_reduce_add_ps(inner_product);
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
    // ScalarComputer::PackRecord for one width
    template <uint32_t WIDTH>
    static void PackRecordOfWidth(
        const uint8_t* PDX_RESTRICT levels,
        uint8_t* PDX_RESTRICT record
    ) {
        const __m512i all_levels = _mm512_loadu_si512(levels);
        if constexpr (WIDTH == 8) {
            _mm512_storeu_si512(record, all_levels);
            return;
        }
        if constexpr ((WIDTH & 4u) != 0) {
            const __m256i bytes = _mm256_or_si256(
                _mm256_and_si256(_mm512_castsi512_si256(all_levels), _mm256_set1_epi8(0x0F)),
                _mm256_and_si256(
                    _mm256_slli_epi16(_mm512_extracti64x4_epi64(all_levels, 1), 4),
                    _mm256_set1_epi8(static_cast<char>(0xF0))
                )
            );
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(record), bytes);
            record += 32;
        }
        if constexpr ((WIDTH & 2u) != 0) {
            constexpr int SHIFT = (WIDTH & 4u) != 0 ? 4 : 0;
            const __m512i parts =
                _mm512_and_si512(_mm512_srli_epi16(all_levels, SHIFT), _mm512_set1_epi8(3));
            const __m128i bytes = _mm_or_si128(
                _mm_or_si128(
                    _mm512_castsi512_si128(parts),
                    _mm_slli_epi16(_mm512_extracti32x4_epi32(parts, 1), 2)
                ),
                _mm_or_si128(
                    _mm_slli_epi16(_mm512_extracti32x4_epi32(parts, 2), 4),
                    _mm_slli_epi16(_mm512_extracti32x4_epi32(parts, 3), 6)
                )
            );
            _mm_storeu_si128(reinterpret_cast<__m128i*>(record), bytes);
            record += 16;
        }
        if constexpr ((WIDTH & 1u) != 0) {
            const uint64_t bits = _mm512_test_epi8_mask(
                all_levels, _mm512_set1_epi8(static_cast<char>(1u << (WIDTH - 1)))
            );
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
        const __m512i levels = LevelUnpacker<WIDTH>{}(record);
        const __m512 rescale_vec = _mm512_set1_ps(rescale);
        const __m512 center_vec = _mm512_set1_ps(center);
        const __m128i quarters[4] = {
            _mm512_castsi512_si128(levels),
            _mm512_extracti32x4_epi32(levels, 1),
            _mm512_extracti32x4_epi32(levels, 2),
            _mm512_extracti32x4_epi32(levels, 3)
        };
        for (size_t quarter = 0; quarter < 4; ++quarter) {
            const __m512 floats = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(quarters[quarter]));
            _mm512_storeu_ps(
                out + 16 * quarter,
                _mm512_fmadd_ps(
                    _mm512_sub_ps(floats, center_vec),
                    rescale_vec,
                    _mm512_loadu_ps(mean + 16 * quarter)
                )
            );
        }
    }

    // The 64 levels of a WIDTH-bit tail record (ScalarComputer::PackRecord's format), in order
    template <uint32_t WIDTH>
    class LevelUnpacker {
      public:
        __m512i operator()(const uint8_t* PDX_RESTRICT record) const {
            if constexpr (WIDTH == 8) {
                return _mm512_loadu_si512(record);
            }
            __m512i levels = _mm512_setzero_si512();
            if constexpr ((WIDTH & 4u) != 0) {
                const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(record));
                const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
                levels = _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_and_si256(bytes, nibble_mask)),
                    _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nibble_mask),
                    1
                );
                record += 32;
            }
            if constexpr ((WIDTH & 2u) != 0) {
                constexpr int SHIFT = (WIDTH & 4u) != 0 ? 4 : 0;
                // Every 128-bit lane gets the 16 bytes; lane i takes their bits 2i and 2i + 1
                const __m512i bytes =
                    _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(record))
                    );
                const __m512i parts = _mm512_and_si512(
                    _mm512_srlv_epi32(
                        bytes, _mm512_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6)
                    ),
                    _mm512_set1_epi8(3)
                );
                levels =
                    _mm512_or_si512(levels, SHIFT != 0 ? _mm512_slli_epi16(parts, SHIFT) : parts);
                record += 16;
            }
            if constexpr ((WIDTH & 1u) != 0) {
                uint64_t bits;
                std::memcpy(&bits, record, sizeof(bits));
                levels = _mm512_or_si512(
                    levels,
                    _mm512_maskz_mov_epi8(
                        _cvtu64_mask64(bits), _mm512_set1_epi8(static_cast<char>(1u << (WIDTH - 1)))
                    )
                );
            }
            return levels;
        }
    };

    // Lane i of the result is the sum of the 16 lanes of dots[i]
    static __m512i Reduce16(const __m512i* dots) {
        __m512i pairs[8];
        __m512i quads[4];
        for (size_t i = 0; i < 8; ++i) {
            pairs[i] = _mm512_add_epi32(
                _mm512_unpacklo_epi32(dots[2 * i], dots[2 * i + 1]),
                _mm512_unpackhi_epi32(dots[2 * i], dots[2 * i + 1])
            );
        }
        for (size_t i = 0; i < 4; ++i) {
            quads[i] = _mm512_add_epi32(
                _mm512_unpacklo_epi64(pairs[2 * i], pairs[2 * i + 1]),
                _mm512_unpackhi_epi64(pairs[2 * i], pairs[2 * i + 1])
            );
        }
        const __m512i low = _mm512_add_epi32(
            _mm512_shuffle_i32x4(quads[0], quads[1], _MM_SHUFFLE(2, 0, 2, 0)),
            _mm512_shuffle_i32x4(quads[0], quads[1], _MM_SHUFFLE(3, 1, 3, 1))
        );
        const __m512i high = _mm512_add_epi32(
            _mm512_shuffle_i32x4(quads[2], quads[3], _MM_SHUFFLE(2, 0, 2, 0)),
            _mm512_shuffle_i32x4(quads[2], quads[3], _MM_SHUFFLE(3, 1, 3, 1))
        );
        return _mm512_add_epi32(
            _mm512_shuffle_i32x4(low, high, _MM_SHUFFLE(2, 0, 2, 0)),
            _mm512_shuffle_i32x4(low, high, _MM_SHUFFLE(3, 1, 3, 1))
        );
    }

    // Stores keep's surviving lanes at the front of the outputs (all 16 lanes are written),
    // returns how many survived
    static size_t Compact16(
        __mmask16 keep,
        __m512i positions_in,
        __m512 estimates_in,
        __m512 rescales_in,
        uint32_t* PDX_RESTRICT positions,
        float* PDX_RESTRICT estimates,
        float* PDX_RESTRICT rescales
    ) {
        _mm512_storeu_si512(positions, _mm512_maskz_compress_epi32(keep, positions_in));
        _mm512_storeu_ps(estimates, _mm512_maskz_compress_ps(keep, estimates_in));
        _mm512_storeu_ps(rescales, _mm512_maskz_compress_ps(keep, rescales_in));
        return static_cast<size_t>(_mm_popcnt_u32(keep));
    }

    static constexpr size_t NUM_HEAD_GROUPS = SAQ_CHUNK_SIZE / U8_INTERLEAVE_SIZE;

    // The head query's groups of 4 codes, each in every lane
    static void LoadHeadQuery(
        const int8_t* PDX_RESTRICT query_codes,
        __m512i query_groups[NUM_HEAD_GROUPS]
    ) {
        for (size_t group_idx = 0; group_idx < NUM_HEAD_GROUPS; ++group_idx) {
            int32_t group_codes;
            std::memcpy(
                &group_codes, query_codes + group_idx * U8_INTERLEAVE_SIZE, sizeof(int32_t)
            );
            query_groups[group_idx] = _mm512_set1_epi32(group_codes);
        }
    }

    // The head dots of the 16 vectors whose levels start at levels (groups stride vectors apart),
    // reading only the bytes of valid_levels
    static __m512i HeadDots16(
        const uint8_t* PDX_RESTRICT levels,
        size_t stride,
        __mmask64 valid_levels,
        const __m512i query_groups[NUM_HEAD_GROUPS]
    ) {
        __m512i dots[4] = {
            _mm512_setzero_si512(),
            _mm512_setzero_si512(),
            _mm512_setzero_si512(),
            _mm512_setzero_si512()
        };
        for (size_t group_idx = 0; group_idx < NUM_HEAD_GROUPS; ++group_idx) {
            dots[group_idx % 4] = _mm512_dpbusd_epi32(
                dots[group_idx % 4],
                _mm512_maskz_loadu_epi8(
                    valid_levels, levels + group_idx * U8_INTERLEAVE_SIZE * stride
                ),
                query_groups[group_idx]
            );
        }
        return _mm512_add_epi32(
            _mm512_add_epi32(dots[0], dots[1]), _mm512_add_epi32(dots[2], dots[3])
        );
    }

    // The dots of 16 tail records with the query chunk, lane l's record at record_of_lane(l)
    template <uint32_t WIDTH, class RecordOfLane>
    static __m512i TailDots16(
        const LevelUnpacker<WIDTH>& unpack,
        __m512i query,
        RecordOfLane record_of_lane
    ) {
        __m512i dots[16];
        for (size_t lane = 0; lane < 16; ++lane) {
            dots[lane] =
                _mm512_dpbusd_epi32(_mm512_setzero_si512(), unpack(record_of_lane(lane)), query);
        }
        return Reduce16(dots);
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

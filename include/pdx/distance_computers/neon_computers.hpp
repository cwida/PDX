#pragma once

#include "arm_neon.h"
#include "pdx/common.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>

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
        // PDX_PROFILE_SCOPE("Search/SIMD");
#if defined(__APPLE__)
        distance_t distance = 0.0;
        PDX_VECTORIZE_LOOP
        for (size_t i = 0; i < num_dimensions; ++i) {
            distance_t diff = vector1[i] - vector2[i];
            distance += diff * diff;
        }
        return distance;
#else
        float32x4_t sum_vec = vdupq_n_f32(0);
        size_t i = 0;
        for (; i + 4 <= num_dimensions; i += 4) {
            float32x4_t a_vec = vld1q_f32(vector1 + i);
            float32x4_t b_vec = vld1q_f32(vector2 + i);
            float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
            sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
        }
        distance_t distance = vaddvq_f32(sum_vec);
        for (; i < num_dimensions; ++i) {
            distance_t diff = vector1[i] - vector2[i];
            distance += diff * diff;
        }
        return distance;
#endif
    };

    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        size_t j = 0;
        for (; j + 4 <= d; j += 4) {
            float32x4_t vec = vld1q_f32(data + j);
            const uint32x4_t mask = vld1q_u32(masks + j);
            vec = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vec), mask));
            vst1q_f32(out + j, vec);
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
        float32x4_t dot_vec_0 = vdupq_n_f32(0);
        float32x4_t dot_vec_1 = vdupq_n_f32(0);
        float32x4_t dot_vec_2 = vdupq_n_f32(0);
        float32x4_t dot_vec_3 = vdupq_n_f32(0);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            const uint8x16_t codes_u8 = vld1q_u8(codes + i);
            const uint16x8_t codes_low_u16 = vmovl_u8(vget_low_u8(codes_u8));
            const uint16x8_t codes_high_u16 = vmovl_high_u8(codes_u8);
            dot_vec_0 = vfmaq_f32(
                dot_vec_0,
                vld1q_f32(query + i),
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes_low_u16)))
            );
            dot_vec_1 = vfmaq_f32(
                dot_vec_1, vld1q_f32(query + i + 4), vcvtq_f32_u32(vmovl_high_u16(codes_low_u16))
            );
            dot_vec_2 = vfmaq_f32(
                dot_vec_2,
                vld1q_f32(query + i + 8),
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes_high_u16)))
            );
            dot_vec_3 = vfmaq_f32(
                dot_vec_3, vld1q_f32(query + i + 12), vcvtq_f32_u32(vmovl_high_u16(codes_high_u16))
            );
        }
        float dot =
            vaddvq_f32(vaddq_f32(vaddq_f32(dot_vec_0, dot_vec_1), vaddq_f32(dot_vec_2, dot_vec_3)));
        for (; i < num_dimensions; ++i) {
            dot += query[i] * static_cast<float>(codes[i]);
        }
        return dot;
    }

    static void AccumulateDotWithU8x16(
        const query_t* PDX_RESTRICT query,
        const uint8x16_t codes_u8,
        float32x4_t& dot_vec_0,
        float32x4_t& dot_vec_1,
        float32x4_t& dot_vec_2,
        float32x4_t& dot_vec_3
    ) {
        const uint16x8_t codes_low_u16 = vmovl_u8(vget_low_u8(codes_u8));
        const uint16x8_t codes_high_u16 = vmovl_high_u8(codes_u8);
        dot_vec_0 = vfmaq_f32(
            dot_vec_0, vld1q_f32(query), vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes_low_u16)))
        );
        dot_vec_1 = vfmaq_f32(
            dot_vec_1, vld1q_f32(query + 4), vcvtq_f32_u32(vmovl_high_u16(codes_low_u16))
        );
        dot_vec_2 = vfmaq_f32(
            dot_vec_2, vld1q_f32(query + 8), vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes_high_u16)))
        );
        dot_vec_3 = vfmaq_f32(
            dot_vec_3, vld1q_f32(query + 12), vcvtq_f32_u32(vmovl_high_u16(codes_high_u16))
        );
    }

    static float HorizontalDotWithU4(
        const query_t* PDX_RESTRICT query,
        const uint8_t* PDX_RESTRICT codes,
        size_t num_dimensions
    ) {
        float32x4_t dot_vec_0 = vdupq_n_f32(0);
        float32x4_t dot_vec_1 = vdupq_n_f32(0);
        float32x4_t dot_vec_2 = vdupq_n_f32(0);
        float32x4_t dot_vec_3 = vdupq_n_f32(0);
        const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
        size_t i = 0;
        for (; i + 32 <= num_dimensions; i += 32) {
            const uint8x16_t code_pairs = vld1q_u8(codes + i / 2);
            const uint8x16_t low_nibbles = vandq_u8(code_pairs, nibble_mask);
            const uint8x16_t high_nibbles = vshrq_n_u8(code_pairs, 4);
            AccumulateDotWithU8x16(
                query + i,
                vzip1q_u8(low_nibbles, high_nibbles),
                dot_vec_0,
                dot_vec_1,
                dot_vec_2,
                dot_vec_3
            );
            AccumulateDotWithU8x16(
                query + i + 16,
                vzip2q_u8(low_nibbles, high_nibbles),
                dot_vec_0,
                dot_vec_1,
                dot_vec_2,
                dot_vec_3
            );
        }
        float dot =
            vaddvq_f32(vaddq_f32(vaddq_f32(dot_vec_0, dot_vec_1), vaddq_f32(dot_vec_2, dot_vec_3)));
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
        float32x4_t lower_vec = vdupq_n_f32(std::numeric_limits<float>::infinity());
        float32x4_t upper_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        size_t i = 0;
        for (; i + 4 <= num_dimensions; i += 4) {
            const float32x4_t values_vec = vld1q_f32(values + i);
            lower_vec = vminq_f32(lower_vec, values_vec);
            upper_vec = vmaxq_f32(upper_vec, values_vec);
        }
        lower = vminvq_f32(lower_vec);
        upper = vmaxvq_f32(upper_vec);
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
        const float32x4_t lower_vec = vdupq_n_f32(lower);
        const float32x4_t inverse_step_vec = vdupq_n_f32(1.0f / step);
        const float32x4_t step_vec = vdupq_n_f32(step);
        float32x4_t squared_norm_vec = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            vst1q_u8(
                codes + i,
                Quantize16<255>(values + i, lower_vec, inverse_step_vec, step_vec, squared_norm_vec)
            );
        }
        return vaddvq_f32(squared_norm_vec) +
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
        const float32x4_t lower_vec = vdupq_n_f32(lower);
        const float32x4_t inverse_step_vec = vdupq_n_f32(1.0f / step);
        const float32x4_t step_vec = vdupq_n_f32(step);
        float32x4_t squared_norm_vec = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            const uint8x16_t codes_vec =
                Quantize16<15>(values + i, lower_vec, inverse_step_vec, step_vec, squared_norm_vec);
            const uint8x16_t code_pairs = vorrq_u8(
                vuzp1q_u8(codes_vec, codes_vec), vshlq_n_u8(vuzp2q_u8(codes_vec, codes_vec), 4)
            );
            vst1_u8(codes + i / 2, vget_low_u8(code_pairs));
        }
        return vaddvq_f32(squared_norm_vec) +
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
        const float32x4_t lower_vec = vdupq_n_f32(lower);
        const float32x4_t step_vec = vdupq_n_f32(step);
        const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
        const auto store_16_dequantized = [&](const uint8x16_t codes_u8, float* destination) {
            const uint16x8_t codes_low_u16 = vmovl_u8(vget_low_u8(codes_u8));
            const uint16x8_t codes_high_u16 = vmovl_high_u8(codes_u8);
            vst1q_f32(
                destination,
                vfmaq_f32(
                    lower_vec, vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes_low_u16))), step_vec
                )
            );
            vst1q_f32(
                destination + 4,
                vfmaq_f32(lower_vec, vcvtq_f32_u32(vmovl_high_u16(codes_low_u16)), step_vec)
            );
            vst1q_f32(
                destination + 8,
                vfmaq_f32(
                    lower_vec, vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes_high_u16))), step_vec
                )
            );
            vst1q_f32(
                destination + 12,
                vfmaq_f32(lower_vec, vcvtq_f32_u32(vmovl_high_u16(codes_high_u16)), step_vec)
            );
        };
        size_t i = 0;
        for (; i + 32 <= num_dimensions; i += 32) {
            const uint8x16_t code_pairs = vld1q_u8(codes + i / 2);
            const uint8x16_t low_nibbles = vandq_u8(code_pairs, nibble_mask);
            const uint8x16_t high_nibbles = vshrq_n_u8(code_pairs, 4);
            store_16_dequantized(vzip1q_u8(low_nibbles, high_nibbles), out + i);
            store_16_dequantized(vzip2q_u8(low_nibbles, high_nibbles), out + i + 16);
        }
        scalar_computer::DequantizeFromU4(codes + i / 2, num_dimensions - i, lower, step, out + i);
    }

    static void QuantizeToInt8(
        const query_t* PDX_RESTRICT values,
        size_t num_dimensions,
        float inverse_scale,
        int8_t* PDX_RESTRICT out
    ) {
        const float32x4_t inverse_scale_vec = vdupq_n_f32(inverse_scale);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            int32x4_t rounded[4];
            for (size_t quarter = 0; quarter < 4; ++quarter) {
                rounded[quarter] =
                    vcvtnq_s32_f32(vmulq_f32(vld1q_f32(values + i + 4 * quarter), inverse_scale_vec)
                    );
            }
            const int16x8_t rounded_low =
                vcombine_s16(vqmovn_s32(rounded[0]), vqmovn_s32(rounded[1]));
            const int16x8_t rounded_high =
                vcombine_s16(vqmovn_s32(rounded[2]), vqmovn_s32(rounded[3]));
            vst1q_s8(out + i, vcombine_s8(vqmovn_s16(rounded_low), vqmovn_s16(rounded_high)));
        }
        scalar_computer::QuantizeToInt8(values + i, num_dimensions - i, inverse_scale, out + i);
    }

    // TODO(@lkuffo, low): stored vertically, in blocks of 16 centroids whose 64-byte groups hold 4
    // dimensions of each, the LVQ8 codes would take 4 vdotq_laneq_s32 per group (each against the 4
    // query bytes of one lane) for the dots of all 16 centroids, with no horizontal sums. Its i8mm
    // form (vusdotq_laneq_s32) measured 1.67x over this kernel on the M4 (geomean of 56 cells). It
    // needs a blocked Centroids layout and a new CentroidQuantization nibble. Reads padded_size (a
    // multiple of LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES) bytes of query and codes. query_codes_sum:
    // the sum of the query's codes
    static int32_t HorizontalDotInt8WithU8(
        const int8_t* PDX_RESTRICT query,
        const uint8_t* PDX_RESTRICT codes,
        size_t padded_size,
        int32_t query_codes_sum
    ) {
        int32x4_t dot_vec_0 = vdupq_n_s32(0);
        int32x4_t dot_vec_1 = vdupq_n_s32(0);
        int32x4_t dot_vec_2 = vdupq_n_s32(0);
        int32x4_t dot_vec_3 = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
        // SDOT multiplies s8 by s8: the codes XOR 0x80 are c - 128, so the dot lacks 128 times the
        // sum of the query's codes
        const uint8x16_t code_bias = vdupq_n_u8(0x80);
        const auto biased_codes = [code_bias](const uint8_t* address) {
            return vreinterpretq_s8_u8(veorq_u8(vld1q_u8(address), code_bias));
        };
        for (size_t i = 0; i < padded_size; i += 64) {
            dot_vec_0 = vdotq_s32(dot_vec_0, biased_codes(codes + i), vld1q_s8(query + i));
            dot_vec_1 =
                vdotq_s32(dot_vec_1, biased_codes(codes + i + 16), vld1q_s8(query + i + 16));
            dot_vec_2 =
                vdotq_s32(dot_vec_2, biased_codes(codes + i + 32), vld1q_s8(query + i + 32));
            dot_vec_3 =
                vdotq_s32(dot_vec_3, biased_codes(codes + i + 48), vld1q_s8(query + i + 48));
        }
        const int32_t code_bias_correction = 128 * query_codes_sum;
#else
        for (size_t i = 0; i < padded_size; i += 16) {
            AccumulateWideningDotInt8WithU8x16(
                vld1q_s8(query + i), vld1q_u8(codes + i), dot_vec_0, dot_vec_1, dot_vec_2, dot_vec_3
            );
        }
        static_cast<void>(query_codes_sum);
        const int32_t code_bias_correction = 0;
#endif
        return vaddvq_s32(
                   vaddq_s32(vaddq_s32(dot_vec_0, dot_vec_1), vaddq_s32(dot_vec_2, dot_vec_3))
               ) +
               code_bias_correction;
    }

    // Byte i of codes holds dimension 2i (low nibble) and 2i+1 (high), so even_query[i] and
    // odd_query[i] are those dimensions of the query. Reads padded_size bytes of each
    static int32_t HorizontalDotInt8WithU4(
        const int8_t* PDX_RESTRICT even_query,
        const int8_t* PDX_RESTRICT odd_query,
        const uint8_t* PDX_RESTRICT codes,
        size_t padded_size
    ) {
        int32x4_t dot_vec_0 = vdupq_n_s32(0);
        int32x4_t dot_vec_1 = vdupq_n_s32(0);
        int32x4_t dot_vec_2 = vdupq_n_s32(0);
        int32x4_t dot_vec_3 = vdupq_n_s32(0);
        const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
        for (size_t i = 0; i < padded_size; i += 32) {
            const uint8x16_t code_pairs_0 = vld1q_u8(codes + i);
            const uint8x16_t code_pairs_1 = vld1q_u8(codes + i + 16);
            const uint8x16_t low_nibbles_0 = vandq_u8(code_pairs_0, nibble_mask);
            const uint8x16_t high_nibbles_0 = vshrq_n_u8(code_pairs_0, 4);
            const uint8x16_t low_nibbles_1 = vandq_u8(code_pairs_1, nibble_mask);
            const uint8x16_t high_nibbles_1 = vshrq_n_u8(code_pairs_1, 4);
#if defined(__ARM_FEATURE_DOTPROD)
            // The nibbles fit in s8, so the signed dot product needs no correction
            dot_vec_0 =
                vdotq_s32(dot_vec_0, vreinterpretq_s8_u8(low_nibbles_0), vld1q_s8(even_query + i));
            dot_vec_1 =
                vdotq_s32(dot_vec_1, vreinterpretq_s8_u8(high_nibbles_0), vld1q_s8(odd_query + i));
            dot_vec_2 = vdotq_s32(
                dot_vec_2, vreinterpretq_s8_u8(low_nibbles_1), vld1q_s8(even_query + i + 16)
            );
            dot_vec_3 = vdotq_s32(
                dot_vec_3, vreinterpretq_s8_u8(high_nibbles_1), vld1q_s8(odd_query + i + 16)
            );
#else
            AccumulateWideningDotInt8WithU8x16(
                vld1q_s8(even_query + i), low_nibbles_0, dot_vec_0, dot_vec_1, dot_vec_2, dot_vec_3
            );
            AccumulateWideningDotInt8WithU8x16(
                vld1q_s8(odd_query + i), high_nibbles_0, dot_vec_0, dot_vec_1, dot_vec_2, dot_vec_3
            );
            AccumulateWideningDotInt8WithU8x16(
                vld1q_s8(even_query + i + 16),
                low_nibbles_1,
                dot_vec_0,
                dot_vec_1,
                dot_vec_2,
                dot_vec_3
            );
            AccumulateWideningDotInt8WithU8x16(
                vld1q_s8(odd_query + i + 16),
                high_nibbles_1,
                dot_vec_0,
                dot_vec_1,
                dot_vec_2,
                dot_vec_3
            );
#endif
        }
        return vaddvq_s32(
            vaddq_s32(vaddq_s32(dot_vec_0, dot_vec_1), vaddq_s32(dot_vec_2, dot_vec_3))
        );
    }

    // scalar_computer documents it. The u16 sums of a lane gain at most 2 * 255 per code byte, so
    // they flush into u32 every 128 code bytes (1024 dimensions, RaBitQ-Library's kSafeChunkDim)
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
        constexpr size_t HALF_COLUMN = CLUSTER_SKETCH_BATCH_SIZE / 2;
        constexpr size_t CODEBOOK_GROUP_SIZE_IN_BYTES = 128;
        constexpr size_t COLUMNS_PER_FLUSH = 128;
        const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
        // [lanes 0-15, lanes 16-31][positions 0-7, positions 8-15][first 4, last 4 positions]
        uint32x4_t sums[2][2][2];
        for (auto& lane_half : sums) {
            for (auto& position_half : lane_half) {
                position_half[0] = vdupq_n_u32(0);
                position_half[1] = vdupq_n_u32(0);
            }
        }
        for (size_t first = 0; first < code_size_in_bytes; first += COLUMNS_PER_FLUSH) {
            const size_t last = std::min(code_size_in_bytes, first + COLUMNS_PER_FLUSH);
            // [lanes 0-15, lanes 16-31][low-byte table, high-byte table][positions 0-7, 8-15]
            uint16x8_t partial_sums[2][2][2];
            for (auto& lane_half : partial_sums) {
                for (auto& table : lane_half) {
                    table[0] = vdupq_n_u16(0);
                    table[1] = vdupq_n_u16(0);
                }
            }
            for (size_t column = first; column < last; ++column) {
                const uint8_t* codes = batch + column * CLUSTER_SKETCH_BATCH_SIZE;
                const uint8_t* tables =
                    lookup_table + (column / 2) * CODEBOOK_GROUP_SIZE_IN_BYTES + (column % 2) * 32;
                for (size_t half = 0; half < 2; ++half) {
                    const uint8x16_t code = vld1q_u8(codes + half * HALF_COLUMN);
                    const uint8x16_t nibbles[2] = {vandq_u8(code, nibble_mask), vshrq_n_u8(code, 4)};
                    const uint8x16_t table[2] = {
                        vld1q_u8(tables + half * 16), vld1q_u8(tables + 64 + half * 16)
                    };
                    for (size_t lane_half = 0; lane_half < 2; ++lane_half) {
                        for (size_t byte_table = 0; byte_table < 2; ++byte_table) {
                            const uint8x16_t looked_up =
                                vqtbl1q_u8(table[byte_table], nibbles[lane_half]);
                            uint16x8_t* partial = partial_sums[lane_half][byte_table];
                            partial[0] = vaddw_u8(partial[0], vget_low_u8(looked_up));
                            partial[1] = vaddw_high_u8(partial[1], looked_up);
                        }
                    }
                }
            }
            for (size_t lane_half = 0; lane_half < 2; ++lane_half) {
                for (size_t position_half = 0; position_half < 2; ++position_half) {
                    const uint16x8_t low_bytes = partial_sums[lane_half][0][position_half];
                    const uint16x8_t high_bytes = partial_sums[lane_half][1][position_half];
                    uint32x4_t* sum = sums[lane_half][position_half];
                    sum[0] = vaddq_u32(
                        sum[0],
                        vaddq_u32(
                            vmovl_u16(vget_low_u16(low_bytes)),
                            vshll_n_u16(vget_low_u16(high_bytes), 8)
                        )
                    );
                    sum[1] = vaddq_u32(
                        sum[1],
                        vaddq_u32(vmovl_high_u16(low_bytes), vshll_high_n_u16(high_bytes, 8))
                    );
                }
            }
        }
        const auto* f_add =
            reinterpret_cast<const float*>(batch + code_size_in_bytes * CLUSTER_SKETCH_BATCH_SIZE);
        const float* f_rescale = f_add + CLUSTER_SKETCH_BATCH_SIZE;
        const float* f_error = f_rescale + CLUSTER_SKETCH_BATCH_SIZE;
        const float32x4_t step_vec = vdupq_n_f32(lookup_table_step);
        const float32x4_t offset_vec = vdupq_n_f32(lookup_table_base_sum + centering_correction);
        const float32x4_t distance_vec = vdupq_n_f32(squared_distance_to_mean);
        const float32x4_t error_scale_vec = vdupq_n_f32(error_scale);
        const float32x4_t threshold_vec = vdupq_n_f32(threshold);
        for (size_t lane_half = 0; lane_half < 2; ++lane_half) {
            // Position p holds lane p / 2 + 8 * (p % 2): the even positions are lanes 0-7
            const uint32x4_t (&position_sums)[2][2] = sums[lane_half];
            const uint32x4_t lane_sums[4] = {
                vuzp1q_u32(position_sums[0][0], position_sums[0][1]),
                vuzp1q_u32(position_sums[1][0], position_sums[1][1]),
                vuzp2q_u32(position_sums[0][0], position_sums[0][1]),
                vuzp2q_u32(position_sums[1][0], position_sums[1][1])
            };
            for (size_t quarter = 0; quarter < 4; ++quarter) {
                const size_t lane = lane_half * HALF_COLUMN + quarter * 4;
                const float32x4_t dot =
                    vfmaq_f32(offset_vec, vcvtq_f32_u32(lane_sums[quarter]), step_vec);
                const float32x4_t estimate = vfmaq_f32(
                    vaddq_f32(vld1q_f32(f_add + lane), distance_vec), vld1q_f32(f_rescale + lane), dot
                );
                const float32x4_t lower_bound =
                    vfmsq_f32(estimate, vld1q_f32(f_error + lane), error_scale_vec);
                // A NaN bound compares false, so it never prunes a cluster
                if (vminvq_u32(vcgtq_f32(lower_bound, threshold_vec)) == 0) {
                    return true;
                }
            }
        }
        return false;
    }

  private:
    // Without DOTPROD: widens 16 bytes of query and codes to s16, then multiply-accumulates into
    // s32
    static void AccumulateWideningDotInt8WithU8x16(
        const int8x16_t query_s8,
        const uint8x16_t codes_u8,
        int32x4_t& dot_vec_0,
        int32x4_t& dot_vec_1,
        int32x4_t& dot_vec_2,
        int32x4_t& dot_vec_3
    ) {
        const int16x8_t codes_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(codes_u8)));
        const int16x8_t codes_high = vreinterpretq_s16_u16(vmovl_high_u8(codes_u8));
        const int16x8_t query_low = vmovl_s8(vget_low_s8(query_s8));
        const int16x8_t query_high = vmovl_high_s8(query_s8);
        dot_vec_0 = vmlal_s16(dot_vec_0, vget_low_s16(codes_low), vget_low_s16(query_low));
        dot_vec_1 = vmlal_high_s16(dot_vec_1, codes_low, query_low);
        dot_vec_2 = vmlal_s16(dot_vec_2, vget_low_s16(codes_high), vget_low_s16(query_high));
        dot_vec_3 = vmlal_high_s16(dot_vec_3, codes_high, query_high);
    }

    // The codes of 16 values (round to nearest even, clamped to [0, MAX_CODE]); adds the squares of
    // their dequantized values to squared_norm_vec
    template <int MAX_CODE>
    static uint8x16_t Quantize16(
        const query_t* PDX_RESTRICT values,
        const float32x4_t lower_vec,
        const float32x4_t inverse_step_vec,
        const float32x4_t step_vec,
        float32x4_t& squared_norm_vec
    ) {
        int32x4_t codes_vec[4];
        for (size_t quarter = 0; quarter < 4; ++quarter) {
            const int32x4_t rounded = vcvtnq_s32_f32(
                vmulq_f32(vsubq_f32(vld1q_f32(values + 4 * quarter), lower_vec), inverse_step_vec)
            );
            codes_vec[quarter] =
                vminq_s32(vmaxq_s32(rounded, vdupq_n_s32(0)), vdupq_n_s32(MAX_CODE));
            const float32x4_t dequantized =
                vfmaq_f32(lower_vec, vcvtq_f32_s32(codes_vec[quarter]), step_vec);
            squared_norm_vec = vfmaq_f32(squared_norm_vec, dequantized, dequantized);
        }
        const uint16x8_t codes_low_u16 =
            vcombine_u16(vqmovun_s32(codes_vec[0]), vqmovun_s32(codes_vec[1]));
        const uint16x8_t codes_high_u16 =
            vcombine_u16(vqmovun_s32(codes_vec[2]), vqmovun_s32(codes_vec[3]));
        return vcombine_u8(vqmovn_u16(codes_low_u16), vqmovn_u16(codes_high_u16));
    }
};

// Equivalent of vdotq_u32(acc, a, a) for squared accumulation.
static inline uint32x4_t squared_dot_accumulate(uint32x4_t acc, uint8x16_t a) {
#ifdef __ARM_FEATURE_DOTPROD
    return vdotq_u32(acc, a, a);
#else
    uint16x8_t sq_lo = vmull_u8(vget_low_u8(a), vget_low_u8(a));
    uint16x8_t sq_hi = vmull_u8(vget_high_u8(a), vget_high_u8(a));
    uint32x4_t partial_lo = vpaddlq_u16(sq_lo);
    uint32x4_t partial_hi = vpaddlq_u16(sq_hi);
    return vaddq_u32(acc, vpaddq_u32(partial_lo, partial_hi));
#endif
}

struct SAQCompactionTable {
    uint8_t shuffles[16][16];
};

// For each mask of 4 surviving 32-bit lanes, the byte shuffle that moves them to the front
constexpr SAQCompactionTable MakeSAQCompactionTable() {
    SAQCompactionTable table{};
    for (uint32_t mask = 0; mask < 16; ++mask) {
        uint32_t out_lane = 0;
        for (uint32_t lane = 0; lane < 4; ++lane) {
            if ((mask >> lane) & 1u) {
                for (uint32_t byte = 0; byte < 4; ++byte) {
                    table.shuffles[mask][out_lane * 4 + byte] =
                        static_cast<uint8_t>(lane * 4 + byte);
                }
                ++out_lane;
            }
        }
        for (uint32_t byte = out_lane * 4; byte < 16; ++byte) {
            table.shuffles[mask][byte] = 0xFF;
        }
    }
    return table;
}

inline constexpr SAQCompactionTable SAQ_COMPACTION_TABLE = MakeSAQCompactionTable();

// The 1-bit part of a tail record is a 64-bit mask: lane j of the 16 levels i takes mask byte
// 2i + j / 8 and tests bit j % 8 of it
inline constexpr uint8_t SAQ_BIT_PART_BYTE_INDICES[4][16] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5},
    {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7}
};
inline constexpr uint8_t SAQ_BIT_PART_LANE_BITS[16] =
    {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

template <>
class SIMDComputer<DistanceMetric::L2SQ, Quantization::SAQ> {
    using scalar_computer = ScalarComputer<DistanceMetric::L2SQ, Quantization::SAQ>;

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
        int8x16_t query_groups[NUM_HEAD_GROUPS];
        const int32x4_t dot_bias =
            vdupq_n_s32(DOT_LEVELS_BIAS * LoadHeadQuery(query_codes, query_groups));
        const float32x4_t dot_scale_vec = vdupq_n_f32(dot_scale);
        const float32x4_t dot_offset_vec = vdupq_n_f32(dot_offset);
        const float32x4_t query_squared_norm_vec = vdupq_n_f32(query_squared_norm);
        const float32x4_t threshold_vec = vdupq_n_f32(threshold);
        const uint32x4_t lane_offsets = {0, 1, 2, 3};
        size_t num_kept = 0;
        size_t vector_idx = 0;
        for (; vector_idx + 4 <= num_vectors; vector_idx += 4) {
            const int32x4_t dot =
                HeadDots4(head + vector_idx * U8_INTERLEAVE_SIZE, stride, query_groups, dot_bias);
            const float32x4_t rescale =
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(head_rescales + vector_idx)));
            const float32x4_t residual_norm =
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(residual_norms + vector_idx)));
            const float32x4_t estimate = vfmaq_f32(
                vfmaq_f32(query_squared_norm_vec, residual_norm, residual_norm),
                rescale,
                vsubq_f32(vmulq_f32(vcvtq_f32_s32(dot), dot_scale_vec), dot_offset_vec)
            );
            uint32x4_t keep = vcltq_f32(estimate, threshold_vec);
            if (selection != nullptr) {
                uint32_t selected;
                std::memcpy(&selected, selection + vector_idx, sizeof(uint32_t));
                const uint32x4_t selected_lanes =
                    vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(selected)))));
                keep = vandq_u32(keep, vtstq_u32(selected_lanes, selected_lanes));
            }
            num_kept += Compact4(
                keep,
                vaddq_u32(vdupq_n_u32(static_cast<uint32_t>(vector_idx)), lane_offsets),
                estimate,
                rescale,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept
            );
        }
        if (vector_idx < num_vectors) {
            const size_t num_remainder_kept = scalar_computer::ScanHead(
                head + vector_idx * U8_INTERLEAVE_SIZE,
                num_vectors - vector_idx,
                stride,
                query_codes,
                dot_scale,
                dot_offset,
                query_squared_norm,
                head_rescales + vector_idx,
                residual_norms + vector_idx,
                selection != nullptr ? selection + vector_idx : nullptr,
                threshold,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept
            );
            for (size_t kept_idx = num_kept; kept_idx < num_kept + num_remainder_kept; ++kept_idx) {
                positions[kept_idx] += static_cast<uint32_t>(vector_idx);
            }
            num_kept += num_remainder_kept;
        }
        return num_kept;
    }

    // ScanHead over the vectors at positions only, 4 positions per batch with one lane load per
    // vector and group; keeps those below threshold, compacted in place
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
        int8x16_t query_groups[NUM_HEAD_GROUPS];
        const int32x4_t dot_bias =
            vdupq_n_s32(DOT_LEVELS_BIAS * LoadHeadQuery(query_codes, query_groups));
        const float32x4_t dot_scale_vec = vdupq_n_f32(dot_scale);
        const float32x4_t dot_offset_vec = vdupq_n_f32(dot_offset);
        const float32x4_t query_squared_norm_vec = vdupq_n_f32(query_squared_norm);
        const float32x4_t threshold_vec = vdupq_n_f32(threshold);
        size_t num_kept = 0;
        size_t position_idx = 0;
        for (; position_idx + 4 <= num_positions; position_idx += 4) {
            const uint32x4_t position_vec = vld1q_u32(positions + position_idx);
            const size_t level_offsets[4] = {
                static_cast<size_t>(positions[position_idx]) * U8_INTERLEAVE_SIZE,
                static_cast<size_t>(positions[position_idx + 1]) * U8_INTERLEAVE_SIZE,
                static_cast<size_t>(positions[position_idx + 2]) * U8_INTERLEAVE_SIZE,
                static_cast<size_t>(positions[position_idx + 3]) * U8_INTERLEAVE_SIZE
            };
            int32x4_t dots[4] = {dot_bias, vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
            for (size_t group_idx = 0; group_idx < NUM_HEAD_GROUPS; ++group_idx) {
                const uint8_t* group_levels = head + group_idx * U8_INTERLEAVE_SIZE * stride;
                uint32x4_t levels = vdupq_n_u32(0);
                levels = vld1q_lane_u32(
                    reinterpret_cast<const uint32_t*>(group_levels + level_offsets[0]), levels, 0
                );
                levels = vld1q_lane_u32(
                    reinterpret_cast<const uint32_t*>(group_levels + level_offsets[1]), levels, 1
                );
                levels = vld1q_lane_u32(
                    reinterpret_cast<const uint32_t*>(group_levels + level_offsets[2]), levels, 2
                );
                levels = vld1q_lane_u32(
                    reinterpret_cast<const uint32_t*>(group_levels + level_offsets[3]), levels, 3
                );
                dots[group_idx % 4] = DotLevels(
                    dots[group_idx % 4], vreinterpretq_u8_u32(levels), query_groups[group_idx]
                );
            }
            const int32x4_t dot =
                vaddq_s32(vaddq_s32(dots[0], dots[1]), vaddq_s32(dots[2], dots[3]));
            const uint16_t lane_rescales[4] = {
                head_rescales[positions[position_idx]],
                head_rescales[positions[position_idx + 1]],
                head_rescales[positions[position_idx + 2]],
                head_rescales[positions[position_idx + 3]]
            };
            const uint16_t lane_residual_norms[4] = {
                residual_norms[positions[position_idx]],
                residual_norms[positions[position_idx + 1]],
                residual_norms[positions[position_idx + 2]],
                residual_norms[positions[position_idx + 3]]
            };
            const float32x4_t rescale = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(lane_rescales)));
            const float32x4_t residual_norm =
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(lane_residual_norms)));
            const float32x4_t estimate = vfmaq_f32(
                vfmaq_f32(query_squared_norm_vec, residual_norm, residual_norm),
                rescale,
                vsubq_f32(vmulq_f32(vcvtq_f32_s32(dot), dot_scale_vec), dot_offset_vec)
            );
            num_kept += Compact4(
                vcltq_f32(estimate, threshold_vec),
                position_vec,
                estimate,
                rescale,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept
            );
        }
        if (position_idx < num_positions) {
            // The last positions move up behind the kept ones, then the scalar kernel
            const size_t num_remaining = num_positions - position_idx;
            std::memmove(
                positions + num_kept, positions + position_idx, num_remaining * sizeof(uint32_t)
            );
            num_kept += scalar_computer::ScanHeadPruning(
                head,
                stride,
                query_codes,
                dot_scale,
                dot_offset,
                query_squared_norm,
                head_rescales,
                residual_norms,
                threshold,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept,
                num_remaining
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
        int8x16_t query[4];
        const int32_t codes_sum = LoadTailQuery(query_codes, query);
        const int32_t dot_bias = WIDTH == 8 ? DOT_LEVELS_BIAS * codes_sum : 0;
        const float32x4_t dot_scale_vec = vdupq_n_f32(dot_scale);
        const float32x4_t dot_offset_vec = vdupq_n_f32(dot_offset);
        const float32x4_t threshold_vec = vdupq_n_f32(threshold);
        size_t num_kept = 0;
        size_t survivor_idx = 0;
        for (; survivor_idx + 4 <= num_survivors; survivor_idx += 4) {
            int32_t lane_dots[4];
            uint16_t lane_rescales[4];
            for (size_t lane = 0; lane < 4; ++lane) {
                const uint32_t position = positions[survivor_idx + lane];
                lane_dots[lane] = TailDot<WIDTH>(
                    records + static_cast<size_t>(position) * 8 * WIDTH, query, dot_bias
                );
                if (segment_rescales != nullptr) {
                    lane_rescales[lane] = segment_rescales[position];
                }
            }
            const float32x4_t rescale =
                segment_rescales != nullptr
                    ? vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(lane_rescales)))
                    : vld1q_f32(rescales + survivor_idx);
            const float32x4_t estimate = vfmaq_f32(
                vld1q_f32(estimates + survivor_idx),
                rescale,
                vsubq_f32(
                    vmulq_f32(vcvtq_f32_s32(vld1q_s32(lane_dots)), dot_scale_vec), dot_offset_vec
                )
            );
            num_kept += Compact4(
                vcltq_f32(estimate, threshold_vec),
                vld1q_u32(positions + survivor_idx),
                estimate,
                rescale,
                positions + num_kept,
                estimates + num_kept,
                rescales + num_kept
            );
        }
        if (survivor_idx < num_survivors) {
            const size_t num_remainder_kept = scalar_computer::ScanTail<WIDTH>(
                records,
                query_codes,
                dot_scale,
                dot_offset,
                segment_rescales,
                threshold,
                positions + survivor_idx,
                estimates + survivor_idx,
                rescales + survivor_idx,
                num_survivors - survivor_idx
            );
            std::memmove(
                positions + num_kept,
                positions + survivor_idx,
                num_remainder_kept * sizeof(uint32_t)
            );
            std::memmove(
                estimates + num_kept, estimates + survivor_idx, num_remainder_kept * sizeof(float)
            );
            std::memmove(
                rescales + num_kept, rescales + survivor_idx, num_remainder_kept * sizeof(float)
            );
            num_kept += num_remainder_kept;
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
        int8x16_t query_groups[NUM_HEAD_GROUPS];
        const int32x4_t dot_bias =
            vdupq_n_s32(DOT_LEVELS_BIAS * LoadHeadQuery(query_codes, query_groups));
        const float32x4_t dot_scale_vec = vdupq_n_f32(dot_scale);
        const float32x4_t dot_offset_vec = vdupq_n_f32(dot_offset);
        const float32x4_t query_squared_norm_vec = vdupq_n_f32(query_squared_norm);
        size_t vector_idx = 0;
        for (; vector_idx + 4 <= num_vectors; vector_idx += 4) {
            const int32x4_t dot =
                HeadDots4(head + vector_idx * U8_INTERLEAVE_SIZE, stride, query_groups, dot_bias);
            const float32x4_t rescale =
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(head_rescales + vector_idx)));
            const float32x4_t residual_norm =
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(residual_norms + vector_idx)));
            vst1q_f32(
                estimates + vector_idx,
                vfmaq_f32(
                    vfmaq_f32(query_squared_norm_vec, residual_norm, residual_norm),
                    rescale,
                    vsubq_f32(vmulq_f32(vcvtq_f32_s32(dot), dot_scale_vec), dot_offset_vec)
                )
            );
        }
        if (vector_idx < num_vectors) {
            scalar_computer::ComputeHeadEstimates(
                head + vector_idx * U8_INTERLEAVE_SIZE,
                num_vectors - vector_idx,
                stride,
                query_codes,
                dot_scale,
                dot_offset,
                query_squared_norm,
                head_rescales + vector_idx,
                residual_norms + vector_idx,
                estimates + vector_idx
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
        int8x16_t query[4];
        const int32_t codes_sum = LoadTailQuery(query_codes, query);
        const int32_t dot_bias = WIDTH == 8 ? DOT_LEVELS_BIAS * codes_sum : 0;
        const float32x4_t dot_scale_vec = vdupq_n_f32(dot_scale);
        const float32x4_t dot_offset_vec = vdupq_n_f32(dot_offset);
        size_t vector_idx = 0;
        for (; vector_idx + 4 <= num_vectors; vector_idx += 4) {
            int32_t lane_dots[4];
            for (size_t lane = 0; lane < 4; ++lane) {
                lane_dots[lane] =
                    TailDot<WIDTH>(records + (vector_idx + lane) * 8 * WIDTH, query, dot_bias);
            }
            const float32x4_t rescale =
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(segment_rescales + vector_idx)));
            vst1q_f32(
                estimates + vector_idx,
                vfmaq_f32(
                    vld1q_f32(estimates + vector_idx),
                    rescale,
                    vsubq_f32(
                        vmulq_f32(vcvtq_f32_s32(vld1q_s32(lane_dots)), dot_scale_vec),
                        dot_offset_vec
                    )
                )
            );
        }
        if (vector_idx < num_vectors) {
            scalar_computer::AccumulateTailEstimates<WIDTH>(
                records + vector_idx * 8 * WIDTH,
                num_vectors - vector_idx,
                query_codes,
                dot_scale,
                dot_offset,
                segment_rescales + vector_idx,
                estimates + vector_idx
            );
        }
    }

    // The codec's kernels (scalar_computer documents them), 16 values per step
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
    // 16 lanes in 4 registers, the codec's unit of work
    struct Float32x16 {
        float32x4_t v[4];
    };
    struct Int32x16 {
        int32x4_t v[4];
    };

    static Float32x16 Zero16() { return Set16(0.0f); }

    static Float32x16 Set16(float x) {
        const float32x4_t v = vdupq_n_f32(x);
        return {{v, v, v, v}};
    }

    static Float32x16 Load16(const float* PDX_RESTRICT p) {
        return {{vld1q_f32(p), vld1q_f32(p + 4), vld1q_f32(p + 8), vld1q_f32(p + 12)}};
    }

    static void Store16(float* PDX_RESTRICT p, const Float32x16& a) {
        for (size_t i = 0; i < 4; ++i) {
            vst1q_f32(p + 4 * i, a.v[i]);
        }
    }

    static Float32x16 Sub16(const Float32x16& a, const Float32x16& b) {
        Float32x16 out;
        for (size_t i = 0; i < 4; ++i) {
            out.v[i] = vsubq_f32(a.v[i], b.v[i]);
        }
        return out;
    }

    static Float32x16 Add16(const Float32x16& a, const Float32x16& b) {
        Float32x16 out;
        for (size_t i = 0; i < 4; ++i) {
            out.v[i] = vaddq_f32(a.v[i], b.v[i]);
        }
        return out;
    }

    // a * b + c
    static Float32x16 Fma16(const Float32x16& a, const Float32x16& b, const Float32x16& c) {
        Float32x16 out;
        for (size_t i = 0; i < 4; ++i) {
            out.v[i] = vfmaq_f32(c.v[i], a.v[i], b.v[i]);
        }
        return out;
    }

    static Float32x16 Abs16(const Float32x16& a) {
        Float32x16 out;
        for (size_t i = 0; i < 4; ++i) {
            out.v[i] = vabsq_f32(a.v[i]);
        }
        return out;
    }

    static float ReduceAdd16(const Float32x16& a) {
        return vaddvq_f32(vaddq_f32(vaddq_f32(a.v[0], a.v[1]), vaddq_f32(a.v[2], a.v[3])));
    }

    // min(trunc(x), max_level)
    static Int32x16 Levels16(const Float32x16& x, int32_t max_level) {
        const int32x4_t max_vec = vdupq_n_s32(max_level);
        Int32x16 out;
        for (size_t i = 0; i < 4; ++i) {
            out.v[i] = vminq_s32(vcvtq_s32_f32(x.v[i]), max_vec);
        }
        return out;
    }

    static Float32x16 ToFloat32x16(const Int32x16& a) {
        Float32x16 out;
        for (size_t i = 0; i < 4; ++i) {
            out.v[i] = vcvtq_f32_s32(a.v[i]);
        }
        return out;
    }

    // The bytes half + level where value >= 0, half - 1 - level (half + ~level) elsewhere
    static void StoreCodes16(
        uint8_t* PDX_RESTRICT codes,
        const Int32x16& level,
        const Float32x16& value,
        int32_t half
    ) {
        const int32x4_t half_vec = vdupq_n_s32(half);
        uint16x4_t words[4];
        for (size_t i = 0; i < 4; ++i) {
            const int32x4_t negative =
                vreinterpretq_s32_u32(vcltq_f32(value.v[i], vdupq_n_f32(0.0f)));
            words[i] = vqmovun_s32(vaddq_s32(half_vec, veorq_s32(level.v[i], negative)));
        }
        vst1q_u8(
            codes,
            vcombine_u8(
                vqmovn_u16(vcombine_u16(words[0], words[1])),
                vqmovn_u16(vcombine_u16(words[2], words[3]))
            )
        );
    }

    // ScalarComputer::PackRecord for one width
    template <uint32_t WIDTH>
    static void PackRecordOfWidth(
        const uint8_t* PDX_RESTRICT levels,
        uint8_t* PDX_RESTRICT record
    ) {
        const uint8x16_t parts_in[4] = {
            vld1q_u8(levels), vld1q_u8(levels + 16), vld1q_u8(levels + 32), vld1q_u8(levels + 48)
        };
        if constexpr (WIDTH == 8) {
            for (size_t i = 0; i < 4; ++i) {
                vst1q_u8(record + 16 * i, parts_in[i]);
            }
            return;
        }
        if constexpr ((WIDTH & 4u) != 0) {
            const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
            vst1q_u8(
                record, vorrq_u8(vandq_u8(parts_in[0], nibble_mask), vshlq_n_u8(parts_in[2], 4))
            );
            vst1q_u8(
                record + 16,
                vorrq_u8(vandq_u8(parts_in[1], nibble_mask), vshlq_n_u8(parts_in[3], 4))
            );
            record += 32;
        }
        if constexpr ((WIDTH & 2u) != 0) {
            constexpr int SHIFT = (WIDTH & 4u) != 0 ? 4 : 0;
            const uint8x16_t two_bit_mask = vdupq_n_u8(3);
            uint8x16_t parts[4];
            for (size_t i = 0; i < 4; ++i) {
                if constexpr (SHIFT != 0) {
                    parts[i] = vandq_u8(vshrq_n_u8(parts_in[i], SHIFT), two_bit_mask);
                } else {
                    parts[i] = vandq_u8(parts_in[i], two_bit_mask);
                }
            }
            vst1q_u8(
                record,
                vorrq_u8(
                    vorrq_u8(parts[0], vshlq_n_u8(parts[1], 2)),
                    vorrq_u8(vshlq_n_u8(parts[2], 4), vshlq_n_u8(parts[3], 6))
                )
            );
            record += 16;
        }
        if constexpr ((WIDTH & 1u) != 0) {
            // The top bit of each level, weighted by its lane's bit and summed per 8 levels
            const uint8x16_t top_bit = vdupq_n_u8(static_cast<uint8_t>(1u << (WIDTH - 1)));
            const uint8x16_t lane_bits = vld1q_u8(SAQ_BIT_PART_LANE_BITS);
            uint8_t mask_bytes[8];
            for (size_t i = 0; i < 4; ++i) {
                const uint64x2_t sums = vpaddlq_u32(
                    vpaddlq_u16(vpaddlq_u8(vandq_u8(vtstq_u8(parts_in[i], top_bit), lane_bits)))
                );
                mask_bytes[2 * i] = static_cast<uint8_t>(vgetq_lane_u64(sums, 0));
                mask_bytes[2 * i + 1] = static_cast<uint8_t>(vgetq_lane_u64(sums, 1));
            }
            std::memcpy(record, mask_bytes, sizeof(mask_bytes));
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
        uint8x16_t levels[4];
        UnpackLevels<WIDTH>(record, levels);
        const float32x4_t rescale_vec = vdupq_n_f32(rescale);
        const float32x4_t center_vec = vdupq_n_f32(center);
        for (size_t i = 0; i < 4; ++i) {
            const uint16x8_t words[2] = {
                vmovl_u8(vget_low_u8(levels[i])), vmovl_high_u8(levels[i])
            };
            for (size_t k = 0; k < 2; ++k) {
                const float32x4_t floats[2] = {
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(words[k]))),
                    vcvtq_f32_u32(vmovl_high_u16(words[k]))
                };
                for (size_t q = 0; q < 2; ++q) {
                    const size_t offset = 16 * i + 8 * k + 4 * q;
                    vst1q_f32(
                        out + offset,
                        vfmaq_f32(
                            vld1q_f32(mean + offset), vsubq_f32(floats[q], center_vec), rescale_vec
                        )
                    );
                }
            }
        }
    }

    // Stores the lanes of keep's surviving vectors at the front of the outputs (all 4 lanes are
    // written), returns how many survived
    static size_t Compact4(
        uint32x4_t keep,
        uint32x4_t positions_in,
        float32x4_t estimates_in,
        float32x4_t rescales_in,
        uint32_t* positions,
        float* estimates,
        float* rescales
    ) {
        const uint32x4_t lane_bits = {1, 2, 4, 8};
        const uint32_t mask = vaddvq_u32(vandq_u32(keep, lane_bits));
        const uint8x16_t shuffle = vld1q_u8(SAQ_COMPACTION_TABLE.shuffles[mask]);
        vst1q_u32(
            positions, vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(positions_in), shuffle))
        );
        vst1q_f32(
            estimates, vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(estimates_in), shuffle))
        );
        vst1q_f32(
            rescales, vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(rescales_in), shuffle))
        );
        return static_cast<size_t>(PDX_POPCOUNT(mask));
    }

    static int32x4_t WideningLaneDots(int32x4_t acc, uint8x16_t levels, int8x16_t query) {
        const int16x8_t levels_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(levels)));
        const int16x8_t levels_high = vreinterpretq_s16_u16(vmovl_high_u8(levels));
        const int16x8_t query_low = vmovl_s8(vget_low_s8(query));
        const int16x8_t query_high = vmovl_high_s8(query);
        const int32x4_t products_0 = vmull_s16(vget_low_s16(levels_low), vget_low_s16(query_low));
        const int32x4_t products_1 = vmull_high_s16(levels_low, query_low);
        const int32x4_t products_2 = vmull_s16(vget_low_s16(levels_high), vget_low_s16(query_high));
        const int32x4_t products_3 = vmull_high_s16(levels_high, query_high);
        return vaddq_s32(
            acc, vpaddq_s32(vpaddq_s32(products_0, products_1), vpaddq_s32(products_2, products_3))
        );
    }

    // Each lane plus the dot of its 4 levels with its 4 query codes; levels below 128 are valid s8
    static int32x4_t DotSmallLevels(int32x4_t acc, uint8x16_t levels, int8x16_t query) {
#if defined(__ARM_FEATURE_DOTPROD)
        return vdotq_s32(acc, vreinterpretq_s8_u8(levels), query);
#else
        return WideningLaneDots(acc, levels, query);
#endif
    }

#if defined(__ARM_FEATURE_DOTPROD)
    // SDOT multiplies s8 by s8: levels XOR 0x80 are level - 128, so the dots lack 128 times the
    // sum of the query codes, which the caller adds
    static constexpr int32_t DOT_LEVELS_BIAS = 128;
#else
    static constexpr int32_t DOT_LEVELS_BIAS = 0;
#endif

    // DotSmallLevels for levels up to 255, short by DOT_LEVELS_BIAS times the query codes' sum
    static int32x4_t DotLevels(int32x4_t acc, uint8x16_t levels, int8x16_t query) {
#if defined(__ARM_FEATURE_DOTPROD)
        return vdotq_s32(acc, vreinterpretq_s8_u8(veorq_u8(levels, vdupq_n_u8(0x80))), query);
#else
        return WideningLaneDots(acc, levels, query);
#endif
    }

    static constexpr size_t NUM_HEAD_GROUPS = SAQ_CHUNK_SIZE / U8_INTERLEAVE_SIZE;

    // The head query's groups of 4 codes, each in every lane; returns the codes' sum
    static int32_t LoadHeadQuery(
        const int8_t* PDX_RESTRICT query_codes,
        int8x16_t query_groups[NUM_HEAD_GROUPS]
    ) {
        int32_t codes_sum = 0;
        for (size_t group_idx = 0; group_idx < NUM_HEAD_GROUPS; ++group_idx) {
            int32_t group_codes;
            std::memcpy(
                &group_codes, query_codes + group_idx * U8_INTERLEAVE_SIZE, sizeof(int32_t)
            );
            query_groups[group_idx] = vreinterpretq_s8_s32(vdupq_n_s32(group_codes));
            for (size_t k = 0; k < U8_INTERLEAVE_SIZE; ++k) {
                codes_sum += query_codes[group_idx * U8_INTERLEAVE_SIZE + k];
            }
        }
        return codes_sum;
    }

    // The head dots of the 4 vectors whose levels start at levels (groups stride vectors apart)
    static int32x4_t HeadDots4(
        const uint8_t* PDX_RESTRICT levels,
        size_t stride,
        const int8x16_t query_groups[NUM_HEAD_GROUPS],
        int32x4_t dot_bias
    ) {
        int32x4_t dots[4] = {dot_bias, vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
        for (size_t group_idx = 0; group_idx < NUM_HEAD_GROUPS; ++group_idx) {
            dots[group_idx % 4] = DotLevels(
                dots[group_idx % 4],
                vld1q_u8(levels + group_idx * U8_INTERLEAVE_SIZE * stride),
                query_groups[group_idx]
            );
        }
        return vaddq_s32(vaddq_s32(dots[0], dots[1]), vaddq_s32(dots[2], dots[3]));
    }

    // A tail chunk's query codes; returns their sum
    static int32_t LoadTailQuery(const int8_t* PDX_RESTRICT query_codes, int8x16_t query[4]) {
        int32_t codes_sum = 0;
        for (size_t i = 0; i < 4; ++i) {
            query[i] = vld1q_s8(query_codes + i * 16);
            codes_sum += vaddlvq_s8(query[i]);
        }
        return codes_sum;
    }

    // The dot of a WIDTH-bit tail record with its query chunk; dot_bias completes DotLevels' at 8
    // bits
    template <uint32_t WIDTH>
    static int32_t TailDot(
        const uint8_t* PDX_RESTRICT record,
        const int8x16_t query[4],
        int32_t dot_bias
    ) {
        uint8x16_t levels[4];
        UnpackLevels<WIDTH>(record, levels);
        int32x4_t dot = vdupq_n_s32(0);
        for (size_t i = 0; i < 4; ++i) {
            dot = WIDTH == 8 ? DotLevels(dot, levels[i], query[i])
                             : DotSmallLevels(dot, levels[i], query[i]);
        }
        return vaddvq_s32(dot) + dot_bias;
    }

    // The 64 levels of a WIDTH-bit tail record (ScalarComputer::PackRecord's format), in order
    template <uint32_t WIDTH>
    static void UnpackLevels(const uint8_t* PDX_RESTRICT record, uint8x16_t levels[4]) {
        if constexpr (WIDTH == 8) {
            for (size_t i = 0; i < 4; ++i) {
                levels[i] = vld1q_u8(record + i * 16);
            }
        } else {
            for (size_t i = 0; i < 4; ++i) {
                levels[i] = vdupq_n_u8(0);
            }
            if constexpr ((WIDTH & 4u) != 0) {
                const uint8x16_t low_bytes = vld1q_u8(record);
                const uint8x16_t high_bytes = vld1q_u8(record + 16);
                const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
                levels[0] = vandq_u8(low_bytes, nibble_mask);
                levels[1] = vandq_u8(high_bytes, nibble_mask);
                levels[2] = vshrq_n_u8(low_bytes, 4);
                levels[3] = vshrq_n_u8(high_bytes, 4);
                record += 32;
            }
            if constexpr ((WIDTH & 2u) != 0) {
                constexpr int SHIFT = (WIDTH & 4u) != 0 ? 4 : 0;
                const uint8x16_t bytes = vld1q_u8(record);
                const uint8x16_t two_bit_mask = vdupq_n_u8(3);
                const uint8x16_t parts[4] = {
                    vandq_u8(bytes, two_bit_mask),
                    vandq_u8(vshrq_n_u8(bytes, 2), two_bit_mask),
                    vandq_u8(vshrq_n_u8(bytes, 4), two_bit_mask),
                    vshrq_n_u8(bytes, 6)
                };
                for (size_t i = 0; i < 4; ++i) {
                    levels[i] =
                        vorrq_u8(levels[i], SHIFT != 0 ? vshlq_n_u8(parts[i], SHIFT) : parts[i]);
                }
                record += 16;
            }
            if constexpr ((WIDTH & 1u) != 0) {
                const uint8x8_t mask_bytes = vld1_u8(record);
                const uint8x16_t both_halves = vcombine_u8(mask_bytes, mask_bytes);
                const uint8x16_t lane_bits = vld1q_u8(SAQ_BIT_PART_LANE_BITS);
                const uint8x16_t top_bit = vdupq_n_u8(static_cast<uint8_t>(1u << (WIDTH - 1)));
                for (size_t i = 0; i < 4; ++i) {
                    const uint8x16_t spread =
                        vqtbl1q_u8(both_halves, vld1q_u8(SAQ_BIT_PART_BYTE_INDICES[i]));
                    levels[i] = vorrq_u8(levels[i], vandq_u8(vtstq_u8(spread, lane_bits), top_bit));
                }
            }
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
        size_t dim_idx = start_dimension;
        for (; dim_idx + 4 <= end_dimension; dim_idx += 4) {
            uint32_t dimension_idx = dim_idx;
            uint32_t query_value;
            std::memcpy(&query_value, &query[dimension_idx], sizeof(query_value));
            uint8x8_t vals = vreinterpret_u8_u32(vdup_n_u32(query_value));
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
            if constexpr (!SKIP_PRUNED) {
                const uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
                const uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
                for (; i + 4 <= n_vectors; i += 4) {
                    // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
                    uint32x4_t res = vld1q_u32(&distances_p[i]);
                    uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 4]
                    ); // This 4 is because everytime I read 4 dimensions
                    uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
                    vst1q_u32(&distances_p[i], squared_dot_accumulate(res, diff_u8));
                }
            }
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
        uint32x4_t sum_vec = vdupq_n_u32(0);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            uint8x16_t a_vec = vld1q_u8(vector1 + i);
            uint8x16_t b_vec = vld1q_u8(vector2 + i);
            uint8x16_t d_vec = vabdq_u8(a_vec, b_vec);
            sum_vec = squared_dot_accumulate(sum_vec, d_vec);
        }
        distance_t distance = vaddvq_u32(sum_vec);
        for (; i < num_dimensions; ++i) {
            int n = static_cast<int>(vector1[i]) - vector2[i];
            distance += n * n;
        }
        return distance;
    };
};

} // namespace PDX

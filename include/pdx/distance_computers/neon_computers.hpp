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
    // dimensions of each, the LVQ8 codes would take 4 vusdotq_laneq_s32 per group (each against the
    // 4 query bytes of one lane) for the dots of all 16 centroids, with no horizontal sums.
    // Measured 1.67x over this kernel on the M4 (geomean of 56 cells). It needs a blocked Centroids
    // layout and a new CentroidQuantization nibble. Reads padded_size (a multiple of
    // LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES) bytes of query and codes
    static int32_t HorizontalDotInt8WithU8(
        const int8_t* PDX_RESTRICT query,
        const uint8_t* PDX_RESTRICT codes,
        size_t padded_size
    ) {
        int32x4_t dot_vec_0 = vdupq_n_s32(0);
        int32x4_t dot_vec_1 = vdupq_n_s32(0);
        int32x4_t dot_vec_2 = vdupq_n_s32(0);
        int32x4_t dot_vec_3 = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
        for (size_t i = 0; i < padded_size; i += 64) {
            dot_vec_0 = vusdotq_s32(dot_vec_0, vld1q_u8(codes + i), vld1q_s8(query + i));
            dot_vec_1 = vusdotq_s32(dot_vec_1, vld1q_u8(codes + i + 16), vld1q_s8(query + i + 16));
            dot_vec_2 = vusdotq_s32(dot_vec_2, vld1q_u8(codes + i + 32), vld1q_s8(query + i + 32));
            dot_vec_3 = vusdotq_s32(dot_vec_3, vld1q_u8(codes + i + 48), vld1q_s8(query + i + 48));
        }
#else
        for (size_t i = 0; i < padded_size; i += 16) {
            AccumulateWideningDotInt8WithU8x16(
                vld1q_s8(query + i), vld1q_u8(codes + i), dot_vec_0, dot_vec_1, dot_vec_2, dot_vec_3
            );
        }
#endif
        return vaddvq_s32(
            vaddq_s32(vaddq_s32(dot_vec_0, dot_vec_1), vaddq_s32(dot_vec_2, dot_vec_3))
        );
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
#if defined(__ARM_FEATURE_MATMUL_INT8)
            dot_vec_0 = vusdotq_s32(dot_vec_0, low_nibbles_0, vld1q_s8(even_query + i));
            dot_vec_1 = vusdotq_s32(dot_vec_1, high_nibbles_0, vld1q_s8(odd_query + i));
            dot_vec_2 = vusdotq_s32(dot_vec_2, low_nibbles_1, vld1q_s8(even_query + i + 16));
            dot_vec_3 = vusdotq_s32(dot_vec_3, high_nibbles_1, vld1q_s8(odd_query + i + 16));
#elif defined(__ARM_FEATURE_DOTPROD)
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

  private:
    // Without i8mm: widens 16 bytes of query and codes to s16, then multiply-accumulates into s32
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

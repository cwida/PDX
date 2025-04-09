#ifndef PDX_QUANTIZERS_H
#define PDX_QUANTIZERS_H

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif

#include <cstdint>
#include <cstdio>
#include <cmath>
#include "pdx/common.hpp"

namespace PDX {

class Quantizer {

public:
    alignas(64) float transformed_raw_query[4096]{};
    alignas(64) float normalized_query[4096]{};

public:
    void NormalizeQuery(const float * src) {
        float sum = 0.0f;
        for (size_t i = 0; i < num_dimensions; ++i) {
            sum += src[i] * src[i];
        }
        float norm = std::sqrt(sum);
        for (size_t i = 0; i < num_dimensions; ++i) {
            normalized_query[i] = src[i] / norm;
        }
    }
    size_t num_dimensions = 0;

    void SetD(size_t d){
        num_dimensions = d;
    }

    virtual void ScaleQuery(const float * src, int32_t * dst) {};
    virtual void PrepareQuery(const int32_t * scaled_query, uint8_t *query, const int32_t *for_bases){

    };
};

template <Quantization q=U8>
class LEPQuantizer : public Quantizer {

public:
    using QUANTIZED_QUERY_TYPE = QuantizedVectorType_t<q>;
    alignas(64) inline static int32_t scaled_query[4096];
    alignas(64) inline static int32_t dim_clip_value[4096];
    alignas(64) inline static QUANTIZED_QUERY_TYPE quantized_query[4096];
    LEPQuantizer(){
        lep_exponent = 1000;
        if constexpr (q == Quantization::U8){
            lep_factor = 1.0;
            MAX_VALUE = 255;
        }
    }

    int lep_exponent;
    float lep_factor;
    uint8_t MAX_VALUE;

    void SetExponent(int exponent){
        lep_exponent = exponent;
    }

    void ScaleQuery(const float * src) {
        for (size_t i = 0; i < num_dimensions; ++i) {
            scaled_query[i] = static_cast<int>(std::round(src[i] * lep_exponent * lep_factor));
        }
    }

    // TODO: There are several ways to optimize this
    void PrepareQuery(const int32_t *for_bases){
        for (size_t i = 0; i < num_dimensions; i += 16) {
            // Load 8 int32 values in two NEON registers
            int32x4_t sub_a = vld1q_s32(scaled_query + i);
            int32x4_t sub_b = vld1q_s32(scaled_query + i + 4);
            int32x4_t sub_c = vld1q_s32(scaled_query + i + 8);
            int32x4_t sub_d = vld1q_s32(scaled_query + i + 12);
            int32x4_t for_a = vld1q_s32(for_bases + i);
            int32x4_t for_b = vld1q_s32(for_bases + i + 4);
            int32x4_t for_c = vld1q_s32(for_bases + i + 8);
            int32x4_t for_d = vld1q_s32(for_bases + i + 12);

            int32x4_t input_low_1 = vsubq_s32(sub_a, for_a);
            int32x4_t input_high_1 = vsubq_s32(sub_b, for_b);
            int32x4_t input_low_2 = vsubq_s32(sub_c, for_c);
            int32x4_t input_high_2 = vsubq_s32(sub_d, for_d);

            vst1q_s32(dim_clip_value + i, input_low_1);
            vst1q_s32(dim_clip_value + i + 4, input_high_1);
            vst1q_s32(dim_clip_value + i + 8, input_low_2);
            vst1q_s32(dim_clip_value + i + 12, input_high_2);

            // Narrow from int32 to int16 (saturating)
            int16x4_t narrowed_low_1 = vqmovn_s32(input_low_1);
            int16x4_t narrowed_high_1 = vqmovn_s32( input_high_1);
            int16x4_t narrowed_low_2 = vqmovn_s32(input_low_2);
            int16x4_t narrowed_high_2 = vqmovn_s32(input_high_2);

            // Combine into a single int16x8_t vector
            int16x8_t combined_1 = vcombine_s16(narrowed_low_1, narrowed_high_1);
            int16x8_t combined_2 = vcombine_s16(narrowed_low_2, narrowed_high_2);

            // Narrow from int16 to uint8 (saturating)
            uint8x8_t result_1 = vqmovun_s16(combined_1);
            uint8x8_t result_2 = vqmovun_s16(combined_2);

            // Mask out values that were clamped to 15 (set them to 0)
            // TODO: 15 is going to be clamped to 0 to detect it as an exception
            uint8x8_t mask_1 = vceq_u8(result_1, vdup_n_u8(MAX_VALUE)); // Create mask where result == 255
            uint8x8_t mask_2 = vceq_u8(result_2, vdup_n_u8(MAX_VALUE)); // Create mask where result == 255
            result_1 = vbic_u8(result_1, mask_1); // Zero out those values
            result_2 = vbic_u8(result_2, mask_2);

            //uint8x16_t combined_result = vcombine_u8(result_1, result_2);
            //vst1q_u8(query + i, combined_result);

            // Store result
            vst1_u8(quantized_query + i, result_1);
            vst1_u8(quantized_query + i + 8, result_2);
        }
    };

};

}; // namespace PDX

#endif //PDX_QUANTIZERS_H

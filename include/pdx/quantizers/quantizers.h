#ifndef PDX_QUANTIZERS_H
#define PDX_QUANTIZERS_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include "pdx/common.hpp"
#include "pdx/quantizers/bitpacker/unpacker_neon.h"

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

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
//            lep_factor = 1.0;
//            MAX_VALUE = 255;
//            SKIP_DECOMP_FACTOR = 1.0;
            // TODO: U7 needs to clamp to 127
            lep_factor = 0.5;
            MAX_VALUE = 127;
            SKIP_DECOMP_FACTOR = 1.0;
        } else if constexpr (q == Quantization::U4){
            lep_factor = 0.0625;
            MAX_VALUE = 15;
            SKIP_DECOMP_FACTOR = 0.5;
        } else if constexpr (q == Quantization::U6){
            lep_factor = 0.25;
            MAX_VALUE = 63;
            SKIP_DECOMP_FACTOR = 0.75;
        }
    }

    int lep_exponent;
    float lep_factor;
    uint8_t MAX_VALUE;
    float SKIP_DECOMP_FACTOR;

    void SetExponent(int exponent){
        lep_exponent = exponent;
    }

    void ScaleQuery(const float * src) {
        for (size_t i = 0; i < num_dimensions; ++i) {
            scaled_query[i] = static_cast<int>(std::round(src[i] * lep_exponent * lep_factor));
        }
    }

    void Decompress(uint8_t * data, uint8_t * out, size_t length){
        uint8_t * out_ptr = out;
        uint8_t * data_ptr = data;
        // TODO: Handle remain properly, for now I will let it overflow slightly at the end by aligning n_vectors to 32
        for (uint32_t pos = 0; pos < AlignValue<uint32_t, 1024>(length); pos+=1024){
            auto data_pos = (size_t)(pos * SKIP_DECOMP_FACTOR); // To skip correctly
            if constexpr (q == U4) {
                Unpacker::unpack_4bw_8ow_128crw_8uf(data_ptr + data_pos, out_ptr+pos);
            } else if constexpr (q == U6) {
                Unpacker::unpack_6bw_8ow_128crw_8uf(data_ptr + data_pos, out_ptr+pos);
            }

        }
    }

    // TODO: Separate the neon/avx512 code to somewhere else
    void PrepareQuery(const int32_t *for_bases){
        // TODO: Not multiple of 16/64 in dimensions
#ifdef __ARM_NEON
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
#elif defined(__AVX512F__)
        __m512i zero = _mm512_setzero_si512();
        __m512i cur_max  = _mm512_set1_epi8(MAX_VALUE);
        for (size_t i = 0; i < num_dimensions; i += 64) {
            // Load 64 int32 values in four AVX512 registers
            __m512i sub_a = _mm512_load_si512(scaled_query + i);
            __m512i sub_b = _mm512_load_si512(scaled_query + i + 16);
            __m512i sub_c = _mm512_load_si512(scaled_query + i + 32);
            __m512i sub_d = _mm512_load_si512(scaled_query + i + 48);
            __m512i for_a = _mm512_load_si512(for_bases + i);
            __m512i for_b = _mm512_load_si512(for_bases + i + 16);
            __m512i for_c = _mm512_load_si512(for_bases + i + 32);
            __m512i for_d = _mm512_load_si512(for_bases + i + 48);

            // Subtract in 4 registers
            __m512i input_low_1 = _mm512_sub_epi32(sub_a, for_a);
            __m512i input_high_1 = _mm512_sub_epi32(sub_b, for_b);
            __m512i input_low_2 = _mm512_sub_epi32(sub_c, for_c);
            __m512i input_high_2 = _mm512_sub_epi32(sub_d, for_d);

            // Store the result of the subtraction
            _mm512_store_epi32(dim_clip_value + i, input_low_1);
            _mm512_store_epi32(dim_clip_value + i + 16, input_high_1);
            _mm512_store_epi32(dim_clip_value + i + 32, input_low_2);
            _mm512_store_epi32(dim_clip_value + i + 48, input_high_2);

            // Narrow from int32 to int8 (saturating, max value is set on overflow)
            __m128i result_1 = _mm512_cvtsepi32_epi8(input_low_1);
            __m128i result_2 = _mm512_cvtsepi32_epi8(input_high_1);
            __m128i result_3 = _mm512_cvtsepi32_epi8(input_low_2);
            __m128i result_4 = _mm512_cvtsepi32_epi8(input_high_2);

            uint8_t first = _mm_extract_epi8(result_1, 0);
            //std::cout << "first: " << +first << "\n";

            __m512i result = _mm512_undefined_epi32();  // start with an undefined 512-bit register

            // TODO: Probably this would be the bottleneck of my function
            result = _mm512_inserti64x2(result, result_1, 0); // insert into slot 0
            result = _mm512_inserti64x2(result, result_2, 1); // insert into slot 1
            result = _mm512_inserti64x2(result, result_3, 2); // insert into slot 2
            result = _mm512_inserti64x2(result, result_4, 3); // insert into slot 3

            // Mask out values that were clamped (set them to 0)
            // First we check which values are NOT equal to MAX_VALUE, these are 1
            //__mmask64 mask = _mm512_cmpneq_epu8_mask(result, _mm512_set1_epi8(MAX_VALUE)); // Create mask where result == 255
            // Then we mask out the values which were zero
            //result = _mm512_maskz_mov_epi8(mask, result);

            //__m512i zero = _mm512_setzero_si512();
            //__m512i max  = _mm512_set1_epi8(MAX_VALUE);
            result = _mm512_max_epi8(result, zero);
            result = _mm512_min_epi32(result, cur_max);

            // Store quantized query (64 values)
            _mm512_storeu_epi8(quantized_query + i, result);
        }
#else
        // Seems that in AVX512, this code is equally as performant as the explicit SIMD one
        for (size_t i = 0; i < num_dimensions; ++i){
                int rounded = (int)(scaled_query[i] - for_bases[i]);
                dim_clip_value[i] = rounded;
                if (rounded > MAX_VALUE || rounded < 0) {
                        quantized_query[i] = 0;
                }else {
                        quantized_query[i] = static_cast<uint8_t>(rounded);
                }
        }
#endif
    };

};

}; // namespace PDX

#endif //PDX_QUANTIZERS_H

#pragma once
#ifndef PDX_NEON_COMPUTERS_HPP
#define PDX_NEON_COMPUTERS_HPP

#include <cstdint>
#include <cstdio>
#include "arm_neon.h"
#include <iostream>
#include "pdx/common.hpp"

namespace PDX {

template<DistanceFunction alpha, Quantization q>
class SIMDComputer {};

template <>
class SIMDComputer<L2, Quantization::U8>{
public:
    using DISTANCE_TYPE = DistanceType_t<U8>;
    using QUERY_TYPE = QuantizedVectorType_t<U8>;
    using DATA_TYPE = DataType_t<U8>;

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
            const int32_t * dim_clip_value = nullptr,
            const float * scaling_factors = nullptr
    ){
        // Compute L2
        // TODO: Handle tail in dimension length, for now im not going to worry on that
        // as all the datasets are divisible by 4
        // Todo: template this 4 parameter
        // TODO: Get this distance functions out of here to another folder / file structure

        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                dimension_idx = indices_dimensions[dim_idx];
            }
            uint8x8_t vals = vld1_u8(&query[dimension_idx]);
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
            if constexpr (!SKIP_PRUNED){
                uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
                uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
                for (; i + 4 <= n_vectors; i+=4) {
                    // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
                    uint32x4_t res = vld1q_u32(&distances_p[i]);
                    uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 4]); // This 4 is because everytime I read 4 dimensions
                    uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
                    vst1q_u32(&distances_p[i], vdotq_u32(res, diff_u8, diff_u8));
                }
            }
            // n_vectors % 4 (rest)
//#ifdef BENCHMARK_TIME
//            end_to_end_clock.Tic();
//#endif
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED){
                    vector_idx = pruning_positions[vector_idx];
                }
                /*
                uint32x2_t res = vdup_n_s32(0);
                // Not needed
                //result = vld1_lane_s32(&distances_p[vector_idx], result, 0);
                uint8x8_t vec2_u8 = vld1_u8(&data[offset_to_dimension_start + (vector_idx * 4)]);
                uint8x8_t diff_u8 = vabd_u8(vec1_u8, vec2_u8);
                res = vdot_u32(res, diff_u8, diff_u8);
                distances_p[vector_idx] += vget_lane_u32(res, 0);
                */
                // I am sure I will have 4 dims
                // L2
                int to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (to_multiply_a * to_multiply_a) +
                                           (to_multiply_b * to_multiply_b) +
                                           (to_multiply_c * to_multiply_c) +
                                           (to_multiply_d * to_multiply_d);

            }
        }
        size_t group = start_dimension;
        size_t loop_c = 0;
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; ++dim_idx) {
            // TODO: Do something more clean
            if (loop_c == 4) {
                group += 4;
                loop_c = 0;
            }
            // Todo: detect clip up, but this is super rare
            if (dim_clip_value[dim_idx] < 0) {
                for (size_t j=0; j < n_vectors; ++j) {
                    size_t vector_idx = j;
                    size_t offset_to_dimension_start = group * total_vectors;
                    if constexpr (SKIP_PRUNED){
                        vector_idx = pruning_positions[vector_idx];
                    }
                    // L2
                    // TODO: Altho this clipping fixing is meaningless to the total runtime
                    // We can pushup this term to the end of the calculation, and only add it on vectors
                    // when we are merging to the heap.
                    // If I remove it completely, and not push it up before adding it to the queue,
                    // it will only work if the same query dimension always clips in all the buckets
                    // otherwise I could get some errors
                    // ACTUALLY THIS ALWAYS WORK BECAUSE dim_clip_value[dim_idx] IS ALWAYS NEGATIVE
                    // IF THE CLIPPING IS POSITIVE, I COULD BE PRUNING VECTORS INCORRECTLY
                    // (in other context, here it works because it is an instant correction)
                    // In fact TODO: Check that dim_clip_value[dim_idx] is always negative
                    distances_p[vector_idx] -= 2 * data[offset_to_dimension_start + (vector_idx * 4) + (dim_idx % 4)] * dim_clip_value[dim_idx];
                    distances_p[vector_idx] += dim_clip_value[dim_idx] * dim_clip_value[dim_idx];
                }
            }
            loop_c += 1;
        }
    }

    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p,
            const float * scaling_factors = nullptr
    ){
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (size_t vector_idx = 0; vector_idx < 64; ++vector_idx) {
                /*
                uint32x2_t res = vdup_n_s32(0);
                // Not needed
                //result = vld1_lane_s32(&distances_p[vector_idx], result, 0);
                uint8x8_t vec2_u8 = vld1_u8(&data[offset_to_dimension_start + (vector_idx * 4)]);
                uint8x8_t diff_u8 = vabd_u8(vec1_u8, vec2_u8);
                res = vdot_u32(res, diff_u8, diff_u8);
                distances_p[vector_idx] += vget_lane_u32(res, 0);
                */
                // I am sure I will have 4 dims
                // L2
                int to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (to_multiply_a * to_multiply_a) +
                                           (to_multiply_b * to_multiply_b) +
                                           (to_multiply_c * to_multiply_c) +
                                           (to_multiply_d * to_multiply_d);

            }
        }
//        uint32x4_t res[16];
//        // Load initial values
//        for (size_t i = 0; i < 16; ++i) {
//            res[i] = vdupq_n_u32(0);
//        }
//        // Compute L2
//        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
//            uint32_t dimension_idx = dim_idx;
//            uint8x8_t vals = vld1_u8(&query[dimension_idx]);
//            uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
//            uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
//            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
//            for (int i = 0; i < 16; ++i) { // total: 64 vectors * 4 dimensions each (at 1 byte per value = 2048-bits)
//                // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
//                uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 16]);
//                uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
//                res[i] = vdotq_u32(res[i], diff_u8, diff_u8);
//            }
//        }
//        // Store results back
//        for (int i = 0; i < 16; ++i) {
//            vst1q_u32(&distances_p[i * 4], res[i]);
//        }
    }

    static DISTANCE_TYPE Horizontal(
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_dimensions,
            const float * scaling_factors = nullptr
    ){
        uint32x4_t sum_vec = vdupq_n_u32(0);
        size_t i = 0;
        for (; i + 16 <= num_dimensions; i += 16) {
            uint8x16_t a_vec = vld1q_u8(vector1 + i);
            uint8x16_t b_vec = vld1q_u8(vector2 + i);
            uint8x16_t d_vec = vabdq_u8(a_vec, b_vec);
            sum_vec = vdotq_u32(sum_vec, d_vec, d_vec);
        }
        DISTANCE_TYPE distance = vaddvq_u32(sum_vec);
        for (; i < num_dimensions; ++i) {
            int n = (int)vector1[i] - vector2[i];
            distance += n * n;
        }
        return distance;
    };
};


template <>
class SIMDComputer<L2, Quantization::U4>{
public:
    using DISTANCE_TYPE = DistanceType_t<U4>;
    using QUERY_TYPE = QuantizedVectorType_t<U4>;
    using DATA_TYPE = DataType_t<U4>;

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
            const int32_t * dim_clip_value = nullptr,
            const float * scaling_factors = nullptr
    ){
        // I dont need this because I am doing dot product 4 into 1
        // Maybe it is worth trying doing the lookup and then dot product with a vector of ones.
        static const uint8_t u4_squares_lookup[16] = {
                0, 1, 4, 9, 16, 25, 36, 49, 64,
                81, 100, 121, 144, 169, 196, 225
        };
        uint8x16_t u4_lookup_table = vld1q_u8(u4_squares_lookup);
        uint8x16_t ones_tmp = vdupq_n_u8(1);
        uint8x16_t nibble_mask = vdupq_n_u8(0x0F);

        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            uint8x8_t vals = vld1_u8(&query[dimension_idx]);
            size_t offset_to_dimension_start = (dimension_idx * total_vectors) / 2; // Every dimensions is 2
            size_t i = 0;
            // TODO: RE ADD
            // DIRECT SIMD
//            if constexpr (!SKIP_PRUNED){
//                uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
//                uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
//                for (; i <= n_vectors - 8; i+=8) {
//                    // Read 16 bytes of data (16 values) with 4 dimensions of 8 vectors (32 nibbles)
//                    uint32x4_t res_1 = vld1q_u32(&distances_p[i]);
//                    uint32x4_t res_2 = vld1q_u32(&distances_p[i+4]);
//
//                    // loading 128-bits --> 32 nibbles
//                    uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 2]); // *2 because each dimension is two
//
//                    // Extract nibbles (i don't need the lookup table as uint4 are already the correct rep. of uint8)
//                    uint8x16_t a_even = vandq_u8(vec2_u8, nibble_mask);
//                    uint8x16_t a_odd = vandq_u8(vshrq_n_u8(vec2_u8, 4), nibble_mask);
//
//                    // Right now they are:
//                    // a_even = [x0, x2, x4, x6, ...]
//                    // a_odd  = [x1, x3, x5, x7, ...]
//                    //        [v1, v1, v2, v2  ...]
//                    // I need to unpack lo and unpack hi to put 4 of each vector consecutively
////                    uint8x16_t r_first = vcombine_u8(vget_high_u8(a_even), vget_high_u8(a_odd));
////                    uint8x16_t r_second = vcombine_u8(vget_low_u8(a_even), vget_low_u8(a_odd));
//                    uint8x16_t r_first = vzip1q_u8(a_even, a_odd);
//                    uint8x16_t r_second = vzip2q_u8(a_even, a_odd);
//                    // Now they are:
//                    // r_first  = [x0, x1, x2, x3, ...]
//                    // r_second = [x4, x5, x6, x7, ...]
//                    // TODO: Probably a smarter layout would help me to avoid this vcombine_u8
//
//                    // Abs diff
//                    uint8x16_t d_first = vabdq_u8(r_first, vec1_u8);
//                    uint8x16_t d_second = vabdq_u8(r_second, vec1_u8);
//
//                    // Option 1:
//                    vst1q_u32(&distances_p[i], vdotq_u32(res_1, d_first, d_first));
//                    vst1q_u32(&distances_p[i+4], vdotq_u32(res_2, d_second, d_second));
//
//                    // Option 2:
////                    uint8x16_t sq_lo = vqtbl1q_u8(u4_lookup_table, r_lo);
////                    uint8x16_t sq_hi = vqtbl1q_u8(u4_lookup_table, r_hi);
////                    vst1q_u32(&distances_p[i], vdotq_u32(res_1, d_lo, ones_tmp));
////                    vst1q_u32(&distances_p[i+4], vdotq_u32(res_2, d_hi, ones_tmp));
//                }
//            }
            // n_vectors % 4 (rest)
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED){
                    vector_idx = pruning_positions[vector_idx];
                }
                uint8_t n_1 = data[offset_to_dimension_start + (vector_idx * 2)];
                uint8_t n_2 = data[offset_to_dimension_start + (vector_idx * 2) + 1];

                // TODO: I think this is wrong :)
                int to_multiply_a = query[dimension_idx] - static_cast<uint8_t>(n_1 & 0x0F);
                int to_multiply_b = query[dimension_idx + 1] - static_cast<uint8_t>((n_1 >> 4) & 0x0F);
                int to_multiply_c = query[dimension_idx + 2] - static_cast<uint8_t>(n_2 & 0x0F);
                int to_multiply_d = query[dimension_idx + 3] - static_cast<uint8_t>((n_2 >> 4) & 0x0F);
                distances_p[vector_idx] += (to_multiply_a * to_multiply_a) +
                                           (to_multiply_b * to_multiply_b) +
                                           (to_multiply_c * to_multiply_c) +
                                           (to_multiply_d * to_multiply_d);
            }
        }
        size_t group = start_dimension;
        size_t loop_c = 0;
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; ++dim_idx) {
            if (loop_c == 4) {
                group += 4;
                loop_c = 0;
            }
            if (dim_clip_value[dim_idx] < 0) {
                for (size_t j=0; j < n_vectors; ++j) {
                    size_t vector_idx = j;
                    size_t offset_to_dimension_start = (group * total_vectors) / 2;
                    if constexpr (SKIP_PRUNED){
                        vector_idx = pruning_positions[vector_idx];
                    }
                    uint8_t val;
                    if (dim_idx % 2 != 0) {
                        val = (data[offset_to_dimension_start + (vector_idx * 2) + (dim_idx % 2)] >> 4) & 0x0F;
                    } else {
                        val = data[offset_to_dimension_start + (vector_idx * 2) + (dim_idx % 2)] & 0x0F;
                    }
                    distances_p[vector_idx] -= 2 * val * dim_clip_value[dim_idx];
                    distances_p[vector_idx] += dim_clip_value[dim_idx] * dim_clip_value[dim_idx];
                }
            }
            loop_c += 1;
        }
    }

    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p,
            const float * scaling_factors = nullptr
    ){
        // TODO
    }

    static DISTANCE_TYPE Horizontal(
        const DATA_TYPE *__restrict vector1,
        const DATA_TYPE *__restrict vector2,
        size_t num_dimensions,
        const float * scaling_factors = nullptr
    ){
        return 0;
    };
};


template <>
class SIMDComputer<L2, Quantization::F32>{
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
            const uint32_t * pruning_positions,
            const uint32_t * indices_dimensions,
            const int32_t * dim_clip_value,
            const float * scaling_factors
    ){
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                auto true_vector_idx = vector_idx;
                if constexpr(SKIP_PRUNED){
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
            DISTANCE_TYPE * distances_p,
            const float * scaling_factors
    ){
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
            size_t num_dimensions,
            const float * scaling_factors
    ){
        float32x4_t sum_vec = vdupq_n_f32(0);
        size_t i = 0;
        for (; i + 4 <= num_dimensions; i += 4) {
            float32x4_t a_vec = vld1q_f32(vector1 + i);
            float32x4_t b_vec = vld1q_f32(vector2 + i);
            float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
            sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
        }
        DISTANCE_TYPE distance = vaddvq_f32(sum_vec);
        for (; i < num_dimensions; ++i) {
            float diff = vector1[i] - vector2[i];
            distance += diff * diff;
        }
        return distance;
    };
};

template <>
class SIMDComputer<L2, Quantization::ASYMMETRIC_U8>{
public:
    using DISTANCE_TYPE = DistanceType_t<ASYMMETRIC_U8>;
    using QUERY_TYPE = QuantizedVectorType_t<ASYMMETRIC_U8>;
    using DATA_TYPE = DataType_t<ASYMMETRIC_U8>;

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
            const uint32_t * pruning_positions,
            const uint32_t * indices_dimensions,
            const int32_t * dim_clip_value,
            const float * scaling_factors
    ){
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                dimension_idx = indices_dimensions[dim_idx];
            }
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
#pragma clang loop vectorize(enable)
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED) {
                    vector_idx = pruning_positions[vector_idx];
                }

                // float to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                // float to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
                // float to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
                // float to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
                // distances_p[vector_idx] += (to_multiply_a * to_multiply_a) +
                //                            (to_multiply_b * to_multiply_b) +
                //                            (to_multiply_c * to_multiply_c) +
                //                            (to_multiply_d * to_multiply_d);

                // L2
                // Auto-vectorizes correctly in NEON
                float to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                float to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
                float to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
                float to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (to_multiply_a * to_multiply_a * scaling_factors[dimension_idx]) +
                                           (to_multiply_b * to_multiply_b * scaling_factors[dimension_idx + 1]) +
                                           (to_multiply_c * to_multiply_c * scaling_factors[dimension_idx + 2]) +
                                           (to_multiply_d * to_multiply_d * scaling_factors[dimension_idx + 3]);


                // IP Idea
                // float to_multiply_a = query[dimension_idx] * data[offset_to_dimension_start + (vector_idx * 4)];
                // float to_multiply_b = query[dimension_idx + 1] * data[offset_to_dimension_start + (vector_idx * 4) + 1];
                // float to_multiply_c = query[dimension_idx + 2] * data[offset_to_dimension_start + (vector_idx * 4) + 2];
                // float to_multiply_d = query[dimension_idx + 3] * data[offset_to_dimension_start + (vector_idx * 4) + 3];
                // distances_p[vector_idx] += (to_multiply_a) +
                //                            (to_multiply_b) +
                //                            (to_multiply_c) +
                //                            (to_multiply_d);

            }
        }
    }

    // Defer to the scalar kernel
    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p,
            const float * scaling_factors
    ){
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            size_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {

                // float to_multiply = query[dimension_idx] - (float)data[offset_to_dimension_start + vector_idx];
                // distances_p[vector_idx] += to_multiply * to_multiply;

                float to_multiply = query[dimension_idx] - (float)data[offset_to_dimension_start + vector_idx];
                distances_p[vector_idx] += to_multiply * to_multiply * scaling_factors[dimension_idx];

                // IP idea
                //distances_p[vector_idx] += query[dimension_idx] * data[offset_to_dimension_start + vector_idx];
            }
        }
    }

    static DISTANCE_TYPE Horizontal(
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_dimensions,
            const float * scaling_factors
    ){
        size_t i = 0;
        float32x4_t sum_vec = vdupq_n_f32(0);
        // for (; i + 4 < num_dimensions; i+=4) {
        //     float32x4_t vec_a = vld1q_f32(vector1 + i);
        //     float32x4_t vec_c = vld1q_f32(scaling_factors + i);
        //     // From uint8 to float32
        //     uint8x8_t u8_vec_b = vld1_u8(vector2 + i);
        //     uint16x8_t u16_vec_b = vmovl_u8(u8_vec_b);
        //     uint32x4_t u32_vec_b = vmovl_u16(vget_low_u16(u16_vec_b));
        //     float32x4_t vec_b = vcvtq_f32_u32(u32_vec_b);
        //
        //     float32x4_t diff_vec = vsubq_f32(vec_a, vec_b);
        //     // Using FMADD
        //     float32x4_t tmp = vfmaq_f32(vdupq_n_f32(0), diff_vec, diff_vec);
        //     sum_vec = vfmaq_f32(sum_vec, tmp, vec_c);
        //
        //     // Using MUL
        //     // float32x4_t sq = vmulq_f32(diff_vec, diff_vec);
        //     // float32x4_t term = vmulq_f32(sq, vec_c);
        //     // sum_vec = vaddq_f32(sum_vec, term);
        //
        // }
        DISTANCE_TYPE distance = vaddvq_f32(sum_vec);
#pragma clang loop vectorize(enable)
        for (; i < num_dimensions; ++i) {
            // float diff = vector1[i] - (float)vector2[i];
            // distance += diff * diff;

            float diff = vector1[i] - (float)vector2[i];
            distance += diff * diff * scaling_factors[i];

            // IP idea
            // distance += vector1[i] * vector2[i];
        }
        return distance;
    };
};


template <>
class SIMDComputer<L2, Quantization::ASYMMETRIC_LEP_U8>{
public:
    using DISTANCE_TYPE = DistanceType_t<ASYMMETRIC_LEP_U8>;
    using QUERY_TYPE = QuantizedVectorType_t<ASYMMETRIC_LEP_U8>;
    using DATA_TYPE = DataType_t<ASYMMETRIC_LEP_U8>;

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
            const uint32_t * pruning_positions,
            const uint32_t * indices_dimensions,
            const int32_t * dim_clip_value,
            const float * scaling_factors,
            // NEW
            const QUERY_TYPE *__restrict exceptions_query,
            const DATA_TYPE *__restrict exceptions_data,
            const uint16_t *__restrict exceptions_positions,
            size_t n_exceptions,
            const float * scaling_factors_exceptions
    ){
        const uint8_t EXC_ESCAPE_CODE_SCALAR = 15;
        const uint8x8_t EXC_ESCAPE_CODE = vdup_n_u8(15);
        const uint8x8_t MASK_TO_COUNT_EXCEPTIONS = vdup_n_u8(1);
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=2) {
            uint32_t dimension_idx = dim_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                dimension_idx = indices_dimensions[dim_idx];
            }
            size_t offset_to_dimension_start = dimension_idx * total_vectors / 2;
            size_t i = 0;
            if constexpr (!SKIP_PRUNED) {
                const uint8x8_t mask_lo_nibble = vdup_n_u8(0x0F);
                float32x4_t vec_a_orig_0 = vdupq_n_f32(query[dimension_idx]);
                float32x4_t vec_a_orig_1 = vdupq_n_f32(query[dimension_idx + 1]);

                float32x4_t vec_c_orig_0 = vdupq_n_f32(scaling_factors[dimension_idx]);
                float32x4_t vec_c_orig_1 = vdupq_n_f32(scaling_factors[dimension_idx + 1]);

                // Loading data that corresponds to exceptions:
                // Query
                float32x4_t exc_query_0 = vdupq_n_f32(exceptions_query[dimension_idx]);
                float32x4_t exc_query_1 = vdupq_n_f32(exceptions_query[dimension_idx + 1]);
                // Scaling Factors
                float32x4_t exc_scaling_0 = vdupq_n_f32(scaling_factors_exceptions[dimension_idx]);
                float32x4_t exc_scaling_1 = vdupq_n_f32(scaling_factors_exceptions[dimension_idx + 1]);
                // Data itself
                size_t exc_start_0 = dimension_idx * n_exceptions;
                size_t exc_start_1 = (dimension_idx + 1) * n_exceptions;
                uint16_t exc_offset_0 = 0;
                uint16_t exc_offset_1 = 0;

                float bad_term_0 = query[dimension_idx] * query[dimension_idx] * scaling_factors[dimension_idx];
                float bad_term_1 = query[dimension_idx + 1] * query[dimension_idx + 1] * scaling_factors[dimension_idx + 1];

                for (; i+8 <= total_vectors; i+=8) {
                    float32x4_t res_low = vld1q_f32(&distances_p[i]);
                    float32x4_t res_high = vld1q_f32(&distances_p[i + 4]);

                    // From uint8 to float32
                    uint8x8_t u8_vec_b = vld1_u8(data + offset_to_dimension_start + i);

                    uint8x8_t u8_vec_b_0 = vshr_n_u8(u8_vec_b, 4); // High
                    uint8x8_t u8_vec_b_1 = vand_u8(u8_vec_b, mask_lo_nibble); // Low

                    //
                    // Detect and Patch exceptions
                    // Detect ESCAPE_CODE mask
                    // uint8x8_t exc_mask_0 = vceq_u8(u8_vec_b_0, EXC_ESCAPE_CODE);
                    // uint8x8_t exc_mask_1 = vceq_u8(u8_vec_b_1, EXC_ESCAPE_CODE);
                    // // Detect where I must read exceptions from
                    // uint8x8_t next_exceptions_0 = vld1_u8(exceptions_data + exc_start_0 + exc_offset_0);
                    // uint8x8_t next_exceptions_1 = vld1_u8(exceptions_data + exc_start_1 + exc_offset_1);
                    // // Increase offset counters of exception array
                    //exc_offset_0 += vaddv_u8(vand_u8(exc_mask_0, MASK_TO_COUNT_EXCEPTIONS));
                    //exc_offset_1 += vaddv_u8(vand_u8(exc_mask_1, MASK_TO_COUNT_EXCEPTIONS));
                    // // Mask original vectors
                    // u8_vec_b_0 = vbsl_u8(exc_mask_0, next_exceptions_0, u8_vec_b_0);
                    // u8_vec_b_1 = vbsl_u8(exc_mask_1, next_exceptions_1, u8_vec_b_1);
                    // // Interleave with exceptions vectors
                    // uint32x4_t u32_mask_0 = vreinterpretq_u32_u8(vcombine_u8(exc_mask_0, exc_mask_0));
                    // uint32x4_t u32_mask_1 = vreinterpretq_u32_u8(vcombine_u8(exc_mask_1, exc_mask_1));
                    // float32x4_t vec_a_0 = vbslq_f32(u32_mask_0, exc_query_0, vec_a_orig_0);
                    // float32x4_t vec_c_0 = vbslq_f32(u32_mask_0, exc_scaling_0, vec_c_orig_0);
                    // float32x4_t vec_a_1 = vbslq_f32(u32_mask_1, exc_query_1, vec_a_orig_1);
                    // float32x4_t vec_c_1 = vbslq_f32(u32_mask_1, exc_scaling_1, vec_c_orig_1);
                    ////////////////////////////////////

                    uint16x8_t u16_vec_b_0 = vmovl_u8(u8_vec_b_0);
                    float32x4_t vec_b_0_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_vec_b_0)));
                    // In NEON programming, low is left
                    float32x4_t vec_b_0_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_vec_b_0)));

                    uint16x8_t u16_vec_b_1 = vmovl_u8(u8_vec_b_1);
                    float32x4_t vec_b_1_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_vec_b_1)));
                    float32x4_t vec_b_1_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_vec_b_1)));

                    float32x4_t diff_vec_0_low = vsubq_f32(vec_a_orig_0, vec_b_0_low);
                    float32x4_t diff_vec_1_low = vsubq_f32(vec_a_orig_1, vec_b_1_low);
                    float32x4_t diff_vec_0_high = vsubq_f32(vec_a_orig_0, vec_b_0_high);
                    float32x4_t diff_vec_1_high = vsubq_f32(vec_a_orig_1, vec_b_1_high);

                    // Using MUL
                    float32x4_t tmp_0_low = vmulq_f32(diff_vec_0_low, diff_vec_0_low);
                    float32x4_t tmp_1_low = vmulq_f32(diff_vec_1_low, diff_vec_1_low);
                    float32x4_t tmp_0_high = vmulq_f32(diff_vec_0_high, diff_vec_0_high);
                    float32x4_t tmp_1_high = vmulq_f32(diff_vec_1_high, diff_vec_1_high);
                    res_low = vfmaq_f32(res_low, tmp_0_low, vec_c_orig_0);
                    res_low = vfmaq_f32(res_low, tmp_1_low, vec_c_orig_1);
                    res_high = vfmaq_f32(res_high, tmp_0_high, vec_c_orig_0);
                    res_high = vfmaq_f32(res_high, tmp_1_high, vec_c_orig_1);

                    vst1q_f32(&distances_p[i], res_low);
                    vst1q_f32(&distances_p[i+4], res_high);

                }
            }

            size_t exc_start_0 = dimension_idx * n_exceptions;
            size_t exc_start_1 = (dimension_idx + 1) * n_exceptions;
            size_t exc_offset_0 = 0;
            size_t exc_offset_1 = 0;

            #pragma clang loop vectorize(enable)
             for (; i < n_vectors; ++i) {
                 size_t vector_idx = i;
                 if constexpr (SKIP_PRUNED){
                     vector_idx = pruning_positions[vector_idx];
                 }
                 // L2
                 // TODO: Inline patching of exceptions
                 /*
                  *
                  * if data[offset_to_dimension_start + (vector_idx * 4)] == ESCAPE_CODE:
                  *  use exception data value
                  *  use exception query value
                  *  use exception scaling factor
                  *  advance exception_array by 1
                  * The other option would be to do another loop that goes through data
                  * masking it, and applying the correction. Probably faster.
                  */
                 uint8_t n_1 = data[offset_to_dimension_start + vector_idx];
                 uint8_t nibble_0 = (n_1 & 0xF0) >> 4;
                 uint8_t nibble_1 = n_1 & 0x0F;

                 float query_dim_0 = query[dimension_idx];
                 float query_dim_1 = query[dimension_idx + 1];

                 float scale_0 = scaling_factors[dimension_idx];
                 float scale_1 = scaling_factors[dimension_idx + 1];

                 // if (nibble_0 == EXC_ESCAPE_CODE_SCALAR) {
                 //     nibble_0 = exceptions_data[exc_start_0 + exc_offset_0];
                 //     query_dim_0 = exceptions_query[dimension_idx];
                 //     scale_0 = scaling_factors_exceptions[dimension_idx];
                 //     exc_offset_0 += 1;
                 // }
                 // if (nibble_1 == EXC_ESCAPE_CODE_SCALAR) {
                 //     nibble_1 = exceptions_data[exc_start_1 + exc_offset_1];
                 //     query_dim_1 = exceptions_query[dimension_idx + 1];
                 //     scale_1 = scaling_factors_exceptions[dimension_idx + 1];
                 //     exc_offset_1 += 1;
                 // }

                 float to_multiply_a = query_dim_0 - (float)nibble_0; // High
                 float to_multiply_b = query_dim_1 - (float)nibble_1; // Low

                 distances_p[vector_idx] += (to_multiply_a * to_multiply_a * scale_0) +
                                            (to_multiply_b * to_multiply_b * scale_1);
             }


            // float bad_term_0 = query[dimension_idx] * query[dimension_idx] * scaling_factors[dimension_idx];
            // float bad_term_1 = query[dimension_idx + 1] * query[dimension_idx + 1] * scaling_factors[dimension_idx + 1];
            // size_t exc_idx = 0;
            // size_t exc_offset_to_dimension_start_0 = dimension_idx * n_exceptions;
            // size_t exc_offset_to_dimension_start_1 = (dimension_idx + 1) * n_exceptions;
            // for (; exc_idx < n_exceptions; ++exc_idx) {
            //     uint16_t vector_idx_0 = exceptions_positions[exc_offset_to_dimension_start_0 + exc_idx];
            //     uint16_t vector_idx_1 = exceptions_positions[exc_offset_to_dimension_start_1 + exc_idx];
            //     // Calculate the real L2
            //     float good_term_0 = exceptions_query[dimension_idx] - exceptions_data[exc_offset_to_dimension_start_0 + exc_idx];
            //     good_term_0 = good_term_0 * good_term_0 * scaling_factors_exceptions[dimension_idx];
            //
            //     float good_term_1 = exceptions_query[dimension_idx + 1] - exceptions_data[exc_offset_to_dimension_start_1 + exc_idx];
            //     good_term_1 = good_term_1 * good_term_1 * scaling_factors_exceptions[dimension_idx + 1];
            //
            //     distances_p[vector_idx_0] += good_term_0 - bad_term_0;
            //     distances_p[vector_idx_1] += good_term_1 - bad_term_1;
            // }
        }
    }

    // Defer to the scalar kernel
    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p,
            const float * scaling_factors
    ){
        // TODO or not needed
    }

    static DISTANCE_TYPE Horizontal(
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_nibbles,
            const float * scaling_factors
    ){
        const uint8_t EXC_ESCAPE_CODE_SCALAR = 15;
        size_t num_words = num_nibbles / 2;
        size_t cur_dim = 0;
        size_t i = 0;
        DISTANCE_TYPE distance = 0.0;
        for (; i < num_words; i++) {
            uint8_t nibble_high = (vector2[i] & 0xF0) >> 4;
            uint8_t nibble_low = (vector2[i] & 0x0F);

            float diff_high = vector1[cur_dim] - (float)(nibble_high);
            float diff_low = vector1[cur_dim+1] - (float)(nibble_low);

            distance += diff_high * diff_high * scaling_factors[cur_dim];
            distance += diff_low * diff_low * scaling_factors[cur_dim+1];

            cur_dim += 2;
        }
        return distance;

        // #pragma clang loop vectorize(enable)
        // for (; i < num_words; i++) {
        //     uint8_t nibble_high = (vector2[i] & 0xF0) >> 4;
        //     uint8_t nibble_low = (vector2[i] & 0x0F);
        //
        //     if (nibble_high != EXC_ESCAPE_CODE_SCALAR) {
        //         float diff_high = vector1[cur_dim] - (float)(nibble_high);
        //         distance += diff_high * diff_high * scaling_factors[cur_dim];
        //     }
        //     if (nibble_low != EXC_ESCAPE_CODE_SCALAR) {
        //         float diff_low = vector1[cur_dim+1] - (float)(nibble_low);
        //         distance += diff_low * diff_low * scaling_factors[cur_dim+1];
        //     }
        //
        //     cur_dim += 2;
        // }
        return distance;

    };

    // Defer to the scalar kernel
    template<bool USE_DIMENSIONS_REORDER, bool SKIP_PRUNED>
    static void PatchVertical(
            const QUERY_TYPE *__restrict quant_query,
            const QUERY_TYPE *__restrict exceptions_query,
            const DATA_TYPE *__restrict exceptions_data,
            const uint16_t *__restrict exceptions_positions,
            size_t n_exceptions,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p,
            const uint32_t * pruning_positions,
            const uint32_t * indices_dimensions,
            const int32_t * dim_clip_value,
            const float * scaling_factors,
            const float * scaling_factors_exceptions
    ){
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=1) {
            uint32_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * n_exceptions;
            size_t i = 0;
            // Correct current L2
            // This bad term can be computer on the fly, but my guess is it will not take much time
            float bad_term = quant_query[dimension_idx] * quant_query[dimension_idx] * scaling_factors[dimension_idx];
            for (; i < n_exceptions; ++i) {
                uint16_t vector_idx = exceptions_positions[offset_to_dimension_start + i];
                // Calculate the real L2
                float good_term = exceptions_query[dimension_idx] - exceptions_data[offset_to_dimension_start + i];
                good_term = good_term * good_term * scaling_factors_exceptions[dimension_idx];

                // Since there was never a bad term, this is wrong.
                //distances_p[vector_idx] += good_term - bad_term;
                distances_p[vector_idx] += good_term - bad_term;
            }
        }
    }

};


template <>
class SIMDComputer<NEGATIVE_L2, Quantization::F32>{
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
            const uint32_t * pruning_positions,
            const uint32_t * indices_dimensions,
            const int32_t * dim_clip_value
    ){
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                auto true_vector_idx = vector_idx;
                if constexpr(SKIP_PRUNED){
                    true_vector_idx = pruning_positions[vector_idx];
                }
                distances_p[true_vector_idx] -= 2 * query[true_dimension_idx] * data[offset_to_dimension_start + true_vector_idx];
            }
        }
    }

    // Defer to the scalar kernel
    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p
    ){
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            size_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
                distances_p[vector_idx] -= 2 * query[dimension_idx] * data[offset_to_dimension_start + vector_idx];
            }
        }
    }

    static DISTANCE_TYPE Horizontal(
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_dimensions
    ){
        return 0.0;
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
        float32x4_t sum_vec = vdupq_n_f32(0);
        size_t i = 0;
        for (; i + 4 <= num_dimensions; i += 4) {
            float32x4_t a_vec = vld1q_f32(vector1 + i);
            float32x4_t b_vec = vld1q_f32(vector2 + i);
            sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
        }
        DISTANCE_TYPE distance = vaddvq_f32(sum_vec);
        for (; i < num_dimensions; ++i) {
            distance += vector1[i] * vector2[i];
        }
        return distance;
    };

};


} // namespace PDX


#endif //PDX_NEON_COMPUTERS_HPP

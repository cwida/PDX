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
            const int32_t * dim_clip_value = nullptr
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
            // TODO: RE ADD
//            if constexpr (!SKIP_PRUNED){
//                uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
//                uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
//                for (; i <= n_vectors - 4; i+=4) {
//                    // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
//                    uint32x4_t res = vld1q_u32(&distances_p[i]);
//                    uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 4]);
//                    uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
//                    vst1q_u32(&distances_p[i], vdotq_u32(res, diff_u8, diff_u8));
//                }
//            }
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
            DISTANCE_TYPE * distances_p
    ){
        uint32x4_t res[16];
        // Load initial values
        for (size_t i = 0; i < 16; ++i) {
            res[i] = vdupq_n_u32(0);
        }
        // Compute L2
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            uint8x8_t vals = vld1_u8(&query[dimension_idx]);
            uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
            uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (int i = 0; i < 16; ++i) { // total: 64 vectors * 4 dimensions each (at 1 byte per value = 2048-bits)
                // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
                uint8x16_t vec2_u8 = vld1q_u8(&data[offset_to_dimension_start + i * 16]);
                uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
                // TODO: RE ADD
                //res[i] = vdotq_u32(res[i], diff_u8, diff_u8);
            }
        }
        // Store results back
        for (int i = 0; i < 16; ++i) {
            vst1q_u32(&distances_p[i * 4], res[i]);
        }
    }

    static DISTANCE_TYPE Horizontal(const DATA_TYPE *__restrict vector1, const DATA_TYPE *__restrict vector2, size_t num_dimensions){
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
                float to_multiply = query[true_dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
                distances_p[true_vector_idx] += to_multiply * to_multiply;
//                if constexpr (L_ALPHA == IP) { // TODO: This is a special IP for BSA
//                    distances_p[true_vector_idx] -=
//                            2 * query[true_dimension_idx] * data[offset_to_dimension_start + true_vector_idx];
//                }
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
                float to_multiply = query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
                distances_p[vector_idx] += to_multiply * to_multiply;
//                if constexpr (L_ALPHA == IP){
//                    distances_p[vector_idx] -= 2 * query[dimension_idx] * data[offset_to_dimension_start + vector_idx];
//                }
            }
        }
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

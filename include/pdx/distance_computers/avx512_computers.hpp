#ifndef PDX_AVX512_COMPUTERS_HPP
#define PDX_AVX512_COMPUTERS_HPP

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <immintrin.h>
#include "pdx/common.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"

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
        __m512i res;
        __m512i vec2_u8;
        __m512i vec1_u8;
        __m512i diff_u8;
        __m256i y_res;
        __m256i y_vec2_u8;
        __m256i y_vec1_u8;
        __m256i y_diff_u8;
        uint32_t * query_grouped = (uint32_t *)query;
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                dimension_idx = indices_dimensions[dim_idx];
            }
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
            if constexpr (!SKIP_PRUNED){
                // To load the query efficiently I will load it as uint32_t (4 bytes packed in 1 word)
                uint32_t query_value = query_grouped[dimension_idx / 4];
                // And then broadcast it to the register
                vec1_u8 = _mm512_set1_epi32(query_value);
                for (; i + 16 <= n_vectors; i+=16) {
                    // Read 64 bytes of data (64 values) with 4 dimensions of 16 vectors
                    res = _mm512_load_si512(&distances_p[i]);
                    vec2_u8 = _mm512_loadu_si512(&data[offset_to_dimension_start + i * 4]); // This 4 is because everytime I read 4 dimensions
                    diff_u8 = _mm512_or_si512(_mm512_subs_epu8(vec1_u8, vec2_u8), _mm512_subs_epu8(vec2_u8, vec1_u8));
                    _mm512_store_epi32(&distances_p[i], _mm512_dpbusds_epi32(res, diff_u8, diff_u8));
                }
                y_vec1_u8 = _mm256_set1_epi32(query_value);
                for (; i + 8 <= n_vectors; i+=8) {
                    // Read 32 bytes of data (32 values) with 4 dimensions of 8 vectors
                    y_res = _mm256_load_epi32(&distances_p[i]);
                    y_vec2_u8 = _mm256_loadu_epi8(&data[offset_to_dimension_start + i * 4]); // This 4 is because everytime I read 4 dimensions
                    y_diff_u8 = _mm256_or_si256(_mm256_subs_epu8(y_vec1_u8, y_vec2_u8), _mm256_subs_epu8(y_vec2_u8, y_vec1_u8));
                    _mm256_store_epi32(&distances_p[i], _mm256_dpbusds_epi32(y_res, y_diff_u8, y_diff_u8));
                }
            }
            // rest
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED){
                    vector_idx = pruning_positions[vector_idx];
                }
                int to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (to_multiply_a * to_multiply_a) +
                                           (to_multiply_b * to_multiply_b) +
                                           (to_multiply_c * to_multiply_c) +
                                           (to_multiply_d * to_multiply_d);
            }
            // TODO: I can prune here (?)
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
        __m512i res[4];
        __m512i vec2_u8;
        __m512i vec1_u8;
        __m512i diff_u8;
        uint32_t * query_grouped = (uint32_t *)query;
        // Load 64 initial values
        for (size_t i = 0; i < 4; ++i) {
            res[i] = _mm512_load_si512(&distances_p[i * 16]);
        }
        // Compute L2
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            // To load the query efficiently I will load it as uint32_t (4 bytes packed in 1 word)
            uint32_t query_value = query_grouped[dimension_idx / 4];
            // And then broadcast it to the register
            vec1_u8 = _mm512_set1_epi32(query_value);
            size_t offset_to_dimension_start = dimension_idx * PDX_VECTOR_SIZE;
            for (int i = 0; i < 4; ++i) { // total: 64 vectors (4 iterations of 16 vectors) * 4 dimensions each (at 1 byte per value = 2048-bits)
                // Read 64 bytes of data (64 values) with 4 dimensions of 16 vectors
                vec2_u8 = _mm512_loadu_si512(&data[offset_to_dimension_start + i * 64]);
                diff_u8 = _mm512_or_si512(_mm512_subs_epu8(vec1_u8, vec2_u8), _mm512_subs_epu8(vec2_u8, vec1_u8));
                // I can use this asymmetric dot product as my values are actually 7-bit
                // Hence, the [sign] properties of the second operand is ignored
                // As results will never be negative, it can be stored on res[i] without issues
                // and it saturates to MAX_INT
                res[i] = _mm512_dpbusds_epi32(res[i], diff_u8, diff_u8);
            }
        }
        // Store results back
        for (int i = 0; i < 4; ++i) {
            _mm512_store_epi32(&distances_p[i * 16], res[i]);
        }
    }

    static DISTANCE_TYPE Horizontal(
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_dimensions,
            const float * scaling_factors = nullptr
    ){
        __m512i d2_i32_vec = _mm512_setzero_si512();
        __m512i a_u8_vec, b_u8_vec, d_u8_vec;

simsimd_l2sq_u8_ice_cycle:
        if (num_dimensions < 64) {
            __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, num_dimensions);
            a_u8_vec = _mm512_maskz_loadu_epi8(mask, vector1);
            b_u8_vec = _mm512_maskz_loadu_epi8(mask, vector2);
            num_dimensions = 0;
        }
        else {
            a_u8_vec = _mm512_loadu_si512(vector1);
            b_u8_vec = _mm512_loadu_si512(vector2);
            vector1 += 64, vector2 += 64, num_dimensions -= 64;
        }

        // Substracting unsigned vectors in AVX-512 is done by saturating subtraction:
        d_u8_vec = _mm512_or_si512(_mm512_subs_epu8(a_u8_vec, b_u8_vec), _mm512_subs_epu8(b_u8_vec, a_u8_vec));

        // Multiply and accumulate at `int8` level which are actually uint7, accumulate at `int32` level:
        d2_i32_vec = _mm512_dpbusds_epi32(d2_i32_vec, d_u8_vec, d_u8_vec);
        if (num_dimensions) goto simsimd_l2sq_u8_ice_cycle;
        return _mm512_reduce_add_epi32(d2_i32_vec);
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
        // TODO
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
    using scalar_computer = ScalarComputer<L2, Quantization::F32>;

    alignas(64) static DISTANCE_TYPE pruning_distances_tmp[4096];

    static void GatherDistances(
        size_t n_vectors,
        DISTANCE_TYPE * distances_p,
        const uint32_t * pruning_positions
    ){
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            auto true_vector_idx = pruning_positions[vector_idx];
            pruning_distances_tmp[vector_idx] = distances_p[true_vector_idx];
        }
    }

    template <bool USE_DIMENSIONS_REORDER>
    static void GatherBasedKernel(
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
        GatherDistances(n_vectors, distances_p, pruning_positions);
        __m512 data_vec, d_vec, cur_dist_vec;
        __m256 data_vec_m256, d_vec_m256, cur_dist_vec_m256;
        // Then we move data to be sequential
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            if constexpr (USE_DIMENSIONS_REORDER) {
                true_dimension_idx = indices_dimensions[dimension_idx];
            }
            __m512 query_vec;
            query_vec = _mm512_set1_ps(query[true_dimension_idx]);
//            if constexpr (L_ALPHA == IP){
//                query_vec = _mm512_set1_ps(-2 * query[true_dimension_idx]);
//            }
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            const float * tmp_data = data + offset_to_dimension_start;
            // Now we do the sequential distance calculation loop which would use SIMD
            // Up to 16
            size_t i = 0;
            for (; i + 16 < n_vectors; i+=16) {
                cur_dist_vec = _mm512_load_ps(&pruning_distances_tmp[i]);
                data_vec = _mm512_i32gather_ps(
                        _mm512_load_epi32(&pruning_positions[i]),
                        tmp_data, sizeof(DISTANCE_TYPE)
                );
                d_vec = _mm512_sub_ps(data_vec, query_vec);
                cur_dist_vec = _mm512_fmadd_ps(d_vec, d_vec, cur_dist_vec);
//                if constexpr (L_ALPHA == IP) { // TODO: This is a special IP for BSA
//                    cur_dist_vec = _mm512_fmadd_ps(data_vec, query_vec, cur_dist_vec);
//                }
                _mm512_store_ps(&pruning_distances_tmp[i], cur_dist_vec);
            }
            __m256 query_vec_m256;
            query_vec_m256 = _mm256_set1_ps(query[true_dimension_idx]);
//            if constexpr (L_ALPHA == IP){
//                query_vec_m256 = _mm256_set1_ps(-2 * query[true_dimension_idx]);
//            }
            // Up to 8
            for (; i + 8 < n_vectors; i+=8) {
                cur_dist_vec_m256 = _mm256_load_ps(&pruning_distances_tmp[i]);
                data_vec_m256 = _mm256_i32gather_ps(
                        tmp_data, _mm256_load_epi32(&pruning_positions[i]),
                        sizeof(DISTANCE_TYPE)
                );
                d_vec_m256 = _mm256_sub_ps(data_vec_m256, query_vec_m256);
                cur_dist_vec_m256 = _mm256_fmadd_ps(d_vec_m256, d_vec_m256, cur_dist_vec_m256);
//                if constexpr (L_ALPHA == IP) { // TODO: This is a special IP for BSA
//                    cur_dist_vec_m256 = _mm256_fmadd_ps(data_vec_m256, query_vec_m256, cur_dist_vec_m256);
//                }
                _mm256_store_ps(&pruning_distances_tmp[i], cur_dist_vec_m256);
            }
            // Tail
            for (; i < n_vectors; i++){
                float to_multiply = query[true_dimension_idx] - tmp_data[pruning_positions[i]];
                pruning_distances_tmp[i] += to_multiply * to_multiply;
//                if constexpr (L_ALPHA == IP) { // TODO: This is a special IP for BSA
//                    pruning_distances_tmp[i] -= 2 * query[true_dimension_idx] * tmp_data[pruning_positions[i]];
//                }
            }
        }
        // We now move distances back
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            auto true_vector_idx = pruning_positions[vector_idx];
            distances_p[true_vector_idx] = pruning_distances_tmp[vector_idx];
        }
    }

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
            const int32_t * dim_clip_value = nullptr,
            const float * scaling_factors = nullptr
    ){
        // SIMD is less efficient when looping on the array of not-yet pruned vectors
        // A way to improve the performance by ~20% is using a GATHER intrinsic. However this only works on Intel microarchs.
        // In AMD (Zen 4, Zen 3) using a GATHER is shooting ourselves in the foot (~80 uops)
        // __AVX512FP16__ macro let us detect Intel architectures (from Sapphire Rapids onwards)
#if false && defined(__AVX512FP16__)
        if (n_vectors >= 8) {
            GatherBasedKernel<USE_DIMENSIONS_REORDER>(
                    query, data, n_vectors, total_vectors, start_dimension, end_dimension,
                    distances_p, pruning_positions, indices_dimensions
            );
            return;
        }
#endif
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
            const float * scaling_factors = nullptr
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
            const float * scaling_factors = nullptr
    ){
        __m512 d2_vec = _mm512_setzero();
        __m512 a_vec, b_vec;
        simsimd_l2sq_f32_skylake_cycle:
        if (num_dimensions < 16) {
            __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, num_dimensions);
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
        if (num_dimensions)
            goto simsimd_l2sq_f32_skylake_cycle;

        // _simsimd_reduce_f32x16_skylake
        __m512 x = _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
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
        const __m512i dumb_mask = _mm512_setr_epi32(
            0xFFFFFFFF, 0,         0,         0,
            0,          0xFFFFFFFF,0,         0,
            0,          0,         0xFFFFFFFF,0,
            0,          0,         0,         0xFFFFFFFF
        );
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=4) {
            uint32_t dimension_idx = dim_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                dimension_idx = indices_dimensions[dim_idx];
            }
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;

            // In this asymmetric kernel we cannot advance 64 at a time
            // if constexpr (!SKIP_PRUNED){
            //     __m512i vec1 = _mm512_broadcast_f32x4(_mm_loadu_ps(query + dimension_idx));
            //     __m512i vec_scales = _mm512_broadcast_f32x4(_mm_loadu_ps(scaling_factors + dimension_idx));
            //     for (; i + 4 <= n_vectors; i+=4) {
            //         // Unfortunately, I am only going 4 vectors at a time (4x4)
            //         //__m512 res = _mm512_load_ps(&distances_p[i]); // 16 values, but only going to use 4
            //         __m512 res = _mm512_broadcast_f32x4(_mm_load_ps(&distances_p[i])); // 16 values, but only going to use 4
            //         res = _mm512_and_ps(res, _mm512_castsi512_ps(dumb_mask));
            //
            //         __m512 vec2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_load_si128((__m128i*)&data[offset_to_dimension_start + i * 4]))); // 16 values at a time from 4 vectors
            //
            //         // Problem: 4 values of vec2 need to sum only to 1 value of res
            //         __m512 diff = _mm512_sub_ps(vec1, vec2);
            //         __m512 tmp_ = _mm512_mul_ps(diff, diff);
            //         res = _mm512_fmadd_ps(tmp_, vec_scales, res);
            //
            //         // 2) pairwise sum:  [a0+a1, a1+a0, a2+a3, a3+a2, ...] per 128-bit lane
            //         __m512 t1 = _mm512_permute_ps(res, _MM_PERM_BADC);
            //         __m512 s1 = _mm512_add_ps(res, t1);
            //
            //         // 3) cross-pair sum: [ (a0+a1)+(a2+a3) replicated ] per lane
            //         __m512 t2 = _mm512_permute_ps(s1, _MM_PERM_CDAB);
            //         __m512 s2 = _mm512_add_ps(s1, t2);
            //
            //         // 4) s2 now holds the 4-sum in *every* element of each 128-bit lane:
            //         //    idxs 0,1,2,3 all = res1; 4–7 = res2; 8–11 = res3; 12–15 = res4
            //         __m128 l0 = _mm512_castps512_ps128(s2);           // lane 0 → [res1,res1,res1,res1]
            //         __m128 l1 = _mm512_extractf32x4_ps(s2, 1);        // lane 1 → [res2,…]
            //         __m128 l2 = _mm512_extractf32x4_ps(s2, 2);        // lane 2 → [res3,…]
            //         __m128 l3 = _mm512_extractf32x4_ps(s2, 3);        // lane 3 → [res4,…]
            //         // 5) convert low element of each to scalar and store
            //         distances_p[i] = _mm_cvtss_f32(l0);
            //         distances_p[i + 1] = _mm_cvtss_f32(l1);
            //         distances_p[i + 2] = _mm_cvtss_f32(l2);
            //         distances_p[i + 3] = _mm_cvtss_f32(l3);
            //
            //         // Cannot use dot-product
            //         //_mm512_store_ps(&distances_p[i], res);
            //     }
            //     // __m512i vec_256 = _mm256_broadcast_f32x4(_mm_loadu_ps(query + dimension_idx));
            //     // __m512i vec_scales_256 = _mm256_broadcast_f32x4(_mm_loadu_ps(scaling_factors + dimension_idx));
            //     // for (; i + 8 <= n_vectors; i+=8) {
            //     //     // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
            //     //     __m256 res = _mm256_load_ps(&distances_p[i]); // 16 values at a time
            //     //     __m256 vec2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_load_si128((__m128i*)&data[offset_to_dimension_start + i * 4])));
            //     //
            //     //     __m256 diff = _mm256_sub_ps(vec1, vec2);
            //     //     __m256 tmp_ = _mm256_mul_ps(diff, diff);
            //     //     res = _mm256_fmadd_ps(tmp_, vec_scales, res);
            //     //     // Cannot use dot-product
            //     // }
            // }
            #pragma clang loop vectorize(enable)
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED) {
                    vector_idx = pruning_positions[vector_idx];
                }

                // L2
                float to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                float to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
                float to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
                float to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (to_multiply_a * to_multiply_a * scaling_factors[dimension_idx]) +
                                           (to_multiply_b * to_multiply_b * scaling_factors[dimension_idx + 1]) +
                                           (to_multiply_c * to_multiply_c * scaling_factors[dimension_idx + 2]) +
                                           (to_multiply_d * to_multiply_d * scaling_factors[dimension_idx + 3]);
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
                float to_multiply = query[dimension_idx] - (float)data[offset_to_dimension_start + vector_idx];
                distances_p[vector_idx] += to_multiply * to_multiply * scaling_factors[dimension_idx];
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
        __m512 sum_vec = _mm512_setzero_ps();
        for (; i + 16 < num_dimensions; i+=16) {
            __m512 vec1 = _mm512_load_ps(&vector1[i]);
            __m512 vec2 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_load_si128((__m128i*)&vector2[i])));
            __m512 vec_scale = _mm512_load_ps(&scaling_factors[i]);
            __m512 diff = _mm512_sub_ps(vec1, vec2);
            __m512 tmp = _mm512_mul_ps(diff, diff);
            sum_vec = _mm512_fmadd_ps(tmp, vec_scale, sum_vec);
        }
        DISTANCE_TYPE distance = _mm512_reduce_add_ps(sum_vec);
        //DISTANCE_TYPE distance = 0;
        #pragma clang loop vectorize(enable)
        for (; i < num_dimensions; ++i) {
            float diff = vector1[i] - (float)vector2[i];
            distance += diff * diff * scaling_factors[i];
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
        const __m128i EXC_ESCAPE_CODE = _mm_set1_epi8(15);
        const __m512 EXC_ESCAPE_CODE_512 = _mm512_set1_ps(15);
        const __m128i MASK_TO_COUNT_EXCEPTIONS = _mm_set1_epi8(1);
        __m128i low_mask = _mm_set1_epi8(0x0f);
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=2) {
            uint32_t dimension_idx = dim_idx;
            if constexpr (USE_DIMENSIONS_REORDER){
                dimension_idx = indices_dimensions[dim_idx];
            }
            size_t offset_to_dimension_start = dimension_idx * total_vectors / 2;
            float query_dim_0 = query[dimension_idx];
            float query_dim_1 = query[dimension_idx + 1];

            float scale_0 = scaling_factors[dimension_idx];
            float scale_1 = scaling_factors[dimension_idx + 1];
            size_t i = 0;
            size_t exc_start_0 = dimension_idx * n_exceptions;
            size_t exc_start_1 = (dimension_idx + 1) * n_exceptions;
            uint16_t exc_offset_0 = 0;
            uint16_t exc_offset_1 = 0;
            // In this asymmetric kernel we cannot advance 64 at a time
            if constexpr (!SKIP_PRUNED){
                __m512 vec_a_orig_0 = _mm512_set1_ps(query_dim_0);
                __m512 vec_a_orig_1 = _mm512_set1_ps(query_dim_1);
                __m512 vec_c_orig_0 = _mm512_set1_ps(scale_0);
                __m512 vec_c_orig_1 = _mm512_set1_ps(scale_1);

                // Loading data that corresponds to exceptions:
                // Query
                __m512 exc_query_0 = _mm512_set1_ps(exceptions_query[dimension_idx]);
                __m512 exc_query_1 = _mm512_set1_ps(exceptions_query[dimension_idx + 1]);
                // Scaling Factors
                __m512 exc_scaling_0 = _mm512_set1_ps(scaling_factors_exceptions[dimension_idx]);
                __m512 exc_scaling_1 = _mm512_set1_ps(scaling_factors_exceptions[dimension_idx + 1]);
                // Data itself
                /////////////////////////////////////////////////
                __m128i next_exceptions_0 = _mm_loadu_si128((__m128i*)(exceptions_data + exc_start_0 + exc_offset_0));
                __m128i next_exceptions_1 = _mm_loadu_si128((__m128i*)(exceptions_data + exc_start_1 + exc_offset_1));
                for (; i + 16 <= n_vectors; i+=16) {
                    __m512 res = _mm512_load_ps(&distances_p[i]); // touching 16 vectors

                    // Load 16 uint8 values
                    __m128i raw_data = _mm_loadu_si128((__m128i*)&data[offset_to_dimension_start + i]);

                    // From uint4 to uint8
                    __m128i raw_data_0 = _mm_and_si128(_mm_srli_epi16(raw_data, 4), low_mask);
                    __m128i raw_data_1 = _mm_and_si128(raw_data, low_mask);

                    // Detect and Patch exceptions
                    //Detect ESCAPE_CODE mask
                    __mmask16 exc_mask_0 = _mm_cmpeq_epi8_mask(raw_data_0, EXC_ESCAPE_CODE);
                    __mmask16 exc_mask_1 = _mm_cmpeq_epi8_mask(raw_data_1, EXC_ESCAPE_CODE);
                    // Detect where I must read exceptions from
                    // __m128i next_exceptions_0 = _mm_loadu_si128((__m128i*)(exceptions_data + exc_start_0 + exc_offset_0));
                    // __m128i next_exceptions_1 = _mm_loadu_si128((__m128i*)(exceptions_data + exc_start_1 + exc_offset_1));
                    // // Increase offset counters of exception array
                    // exc_offset_0 += _mm_popcnt_u32((uint32_t)exc_mask_0);
                    // exc_offset_1 += _mm_popcnt_u32((uint32_t)exc_mask_1);
                    // Mask original vectors
                    // raw_data_0 = _mm_mask_expand_epi8(raw_data_0, exc_mask_0, next_exceptions_0);
                    // raw_data_1 = _mm_mask_expand_epi8(raw_data_1, exc_mask_1, next_exceptions_1);
                    // Interleave with exceptions vectors
                    __m512 vec_a_0 = _mm512_mask_blend_ps(exc_mask_0,  vec_a_orig_0, EXC_ESCAPE_CODE_512);
                    //__m512 vec_c_0 = _mm512_mask_blend_ps(exc_mask_0, vec_c_orig_0, exc_scaling_0);
                    __m512 vec_a_1 = _mm512_mask_blend_ps(exc_mask_1, vec_a_orig_1, EXC_ESCAPE_CODE_512);
                    // __m512 vec_c_1 = _mm512_mask_blend_ps(exc_mask_1, vec_c_orig_1, exc_scaling_1);
                    ////////////////////////////////////

                    // DELETE LATER
                    // raw_data_0 = _mm_mask_mov_epi8(raw_data_0, exc_mask_0, _mm_setzero_si128());
                    // raw_data_1 = _mm_mask_mov_epi8(raw_data_1, exc_mask_1, _mm_setzero_si128());
                    // __m512 vec_a_0 = _mm512_mask_mov_ps(vec_a_orig_0, exc_mask_0,   _mm512_setzero_ps());
                    // __m512 vec_c_0 = _mm512_mask_mov_ps(vec_c_orig_0, exc_mask_0, _mm512_setzero_ps());
                    // __m512 vec_a_1 = _mm512_mask_mov_ps(vec_a_orig_1, exc_mask_1, _mm512_setzero_ps());
                    // __m512 vec_c_1 = _mm512_mask_mov_ps(vec_c_orig_1, exc_mask_1, _mm512_setzero_ps());
                    /// until here ////

                    // From uint8 to float
                    __m512 vec_b_0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(raw_data_0)); // 16 values at a time from 2 vectors
                    __m512 vec_b_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(raw_data_1));

                    // DELETE LATER
                    // Sub, multiply and fmadd on dimension 0
                    // __m512 diff_0 = _mm512_sub_ps(vec_a_orig_0, vec_b_0);
                    // __m512 tmp_0 = _mm512_mul_ps(diff_0, diff_0);
                    // res = _mm512_fmadd_ps(tmp_0, vec_c_orig_0, res);
                    // // Sub, multiply and fmadd on dimension 1
                    // __m512 diff_1 = _mm512_sub_ps(vec_a_orig_1, vec_b_1);
                    // __m512 tmp_1 = _mm512_mul_ps(diff_1, diff_1);
                    // res = _mm512_fmadd_ps(tmp_1, vec_c_orig_1, res);
                    //////////

                    // Sub, multiply and fmadd on dimension 0
                    __m512 diff_0 = _mm512_sub_ps(vec_a_0, vec_b_0);
                    __m512 tmp_0 = _mm512_mul_ps(diff_0, diff_0);
                    res = _mm512_fmadd_ps(tmp_0, vec_c_orig_0, res);
                    // Sub, multiply and fmadd on dimension 1
                    __m512 diff_1 = _mm512_sub_ps(vec_a_1, vec_b_1);
                    __m512 tmp_1 = _mm512_mul_ps(diff_1, diff_1);
                    res = _mm512_fmadd_ps(tmp_1, vec_c_orig_1, res);
                    // Store distances
                    _mm512_store_ps(&distances_p[i], res);
                }
                // std::cout << exc_offset_0 << ", " <<  n_exceptions << std::endl;
                // std::cout << exc_offset_1 << ", " <<  n_exceptions << std::endl;
                // assert(exc_offset_0 == n_exceptions);
                // assert(exc_offset_1 == n_exceptions);
            }
            // for (; i < n_vectors; ++i) {
            //     size_t vector_idx = i;
            //     if constexpr (SKIP_PRUNED){
            //         vector_idx = pruning_positions[vector_idx];
            //     }
            //     // L2
            //     // TODO: Inline patching of exceptions
            //     /*
            //      *
            //      * if data[offset_to_dimension_start + (vector_idx * 4)] == ESCAPE_CODE:
            //      *  use exception data value
            //      *  use exception query value
            //      *  use exception scaling factor
            //      *  advance exception_array by 1
            //      * The other option would be to do another loop that goes through data
            //      * masking it, and applying the correction. Probably faster.
            //      */
            //
            //     uint8_t n_1 = data[offset_to_dimension_start + vector_idx];
            //     uint8_t nibble_0 = (n_1 & 0xF0) >> 4;
            //     uint8_t nibble_1 = n_1 & 0x0F;
            //
            //     // When we SKIP PRUNED, we do not patch inplace (for now)
            //     if constexpr (SKIP_PRUNED) {
            //         if (nibble_0 != EXC_ESCAPE_CODE_SCALAR) {
            //             float diff_high = query_dim_0 - (float)(nibble_0);
            //             distances_p[vector_idx] += diff_high * diff_high * scale_0;
            //         }
            //         if (nibble_1 != EXC_ESCAPE_CODE_SCALAR) {
            //             float diff_low = query_dim_1 - (float)(nibble_1);
            //             distances_p[vector_idx] += diff_low * diff_low * scale_1;
            //         }
            //     }
            //     else {
            //         if (nibble_0 != EXC_ESCAPE_CODE_SCALAR) {
            //             float diff_high = query_dim_0 - (float)(nibble_0);
            //             distances_p[vector_idx] += diff_high * diff_high * scale_0;
            //         }
            //         else {
            //             float diff_high = exceptions_query[dimension_idx] - exceptions_data[exc_start_0 + exc_offset_0];
            //             distances_p[vector_idx] += (diff_high * diff_high * scaling_factors_exceptions[dimension_idx]);
            //             exc_offset_0 += 1;
            //         }
            //         if (nibble_1 != EXC_ESCAPE_CODE_SCALAR) {
            //             float diff_low = query_dim_1 - (float)(nibble_1);
            //             distances_p[vector_idx] += diff_low * diff_low * scale_1;
            //         }
            //         else {
            //             float diff_low = exceptions_query[dimension_idx + 1] - exceptions_data[exc_start_1 + exc_offset_1];
            //             distances_p[vector_idx] += (diff_low * diff_low * scaling_factors_exceptions[dimension_idx + 1]);
            //             exc_offset_1 += 1;
            //         }
            //     }
            //
            //     // float to_multiply_a = query_dim_0 - (float)nibble_0; // High
            //     // float to_multiply_b = query_dim_1 - (float)nibble_1; // Low
            //
            //     // distances_p[vector_idx] += (to_multiply_a * to_multiply_a * scale_0) +
            //     //                            (to_multiply_b * to_multiply_b * scale_1);
            // }
            // if constexpr (!SKIP_PRUNED) {
            //     if (n_exceptions != exc_offset_0) {
            //         std::cout << n_exceptions << " exceptions present [0].\n";
            //         std::cout << exc_offset_0 << " exceptions were skipped [0].\n";
            //     }
            //     if (n_exceptions != exc_offset_1) {
            //         std::cout << n_exceptions << " exceptions present [1].\n";
            //         std::cout << exc_offset_0 << " exceptions were skipped [1].\n";
            //     }
            //     assert(n_exceptions == exc_offset_0);
            //     assert(n_exceptions == exc_offset_1);
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
        // TODO
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
        #pragma clang loop vectorize(enable)
        for (; i < num_words; i++) {
            uint8_t nibble_high = (vector2[i] & 0xF0) >> 4;
            uint8_t nibble_low = (vector2[i] & 0x0F);

            if (nibble_high != EXC_ESCAPE_CODE_SCALAR) {
                float diff_high = vector1[cur_dim] - (float)(nibble_high);
                distance += diff_high * diff_high * scaling_factors[cur_dim];
            }
            if (nibble_low != EXC_ESCAPE_CODE_SCALAR) {
                float diff_low = vector1[cur_dim+1] - (float)(nibble_low);
                distance += diff_low * diff_low * scaling_factors[cur_dim+1];
            }

            cur_dim += 2;
        }
        // for (; i < num_words; i++) {
        //     uint8_t nibble_high = (vector2[i] & 0xF0) >> 4;
        //     uint8_t nibble_low = (vector2[i] & 0x0F);
        //
        //     float diff_high = vector1[cur_dim] - (float)(nibble_high);
        //     float diff_low = vector1[cur_dim+1] - (float)(nibble_low);
        //
        //     distance += diff_high * diff_high * scaling_factors[cur_dim];
        //     distance += diff_low * diff_low * scaling_factors[cur_dim+1];
        //
        //     cur_dim += 2;
        // }
        return distance;
        // for (; i < num_dimensions; ++i) {
        //     float diff = vector1[i] - (float)vector2[i];
        //     distance += diff * diff * scaling_factors[i];
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
        //std::cout << n_exceptions << std::endl;
        alignas(64) static float distance_correction[256];
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx+=1) {
            uint32_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * n_exceptions;
            size_t i = 0;
            // Correct current L2
            // This bad term can be computer on the fly, but my guess is it will not take much time
            float bad_term = quant_query[dimension_idx] * quant_query[dimension_idx] * scaling_factors[dimension_idx];
            // __m512 vec_bad_term = _mm512_set1_ps(bad_term);
            __m512 vec_exc_query = _mm512_set1_ps(exceptions_query[dimension_idx]);
            __m512 vec_exc_scaling_factor = _mm512_set1_ps(scaling_factors_exceptions[dimension_idx]);

            // for (; i + 16 < n_exceptions; i+=16) {
            //     //__m512 vec_ids = _mm512_load_ps(exceptions_positions + offset_to_dimension_start + i);
            //
            //     __m128i raw_exc_data = _mm_loadu_si128((__m128i*)&exceptions_data[offset_to_dimension_start + i]);
            //     __m512 vec_exc_data = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(raw_exc_data));
            //
            //     __m512 vec_good_terms = _mm512_sub_ps(vec_exc_query, vec_exc_data);
            //     __m512 tmp = _mm512_mul_ps(vec_good_terms, vec_good_terms);
            //     __m512 dis_correction = _mm512_mul_ps(tmp, vec_exc_scaling_factor);
            //     //dis_correction = _mm512_sub_ps(dis_correction, vec_bad_term);
            //     _mm512_store_ps(&distance_correction[i], dis_correction);
            //
            //     // Scalar kernel
            //     // uint16_t vector_idx = exceptions_positions[offset_to_dimension_start + i];
            //     // Calculate the real L2
            //     //float good_term = exceptions_query[dimension_idx] - exceptions_data[offset_to_dimension_start + i];
            //     //good_term = good_term * good_term * scaling_factors_exceptions[dimension_idx];
            //     //distances_p[vector_idx] += good_term - bad_term;
            // }
            #pragma clang loop vectorize(enable)
            for (; i < n_exceptions; i++) {
                // Scalar kernel
                float good_term = exceptions_query[dimension_idx] - exceptions_data[offset_to_dimension_start + i];
                good_term = good_term * good_term * scaling_factors_exceptions[dimension_idx];
                distance_correction[i] = good_term; //- bad_term;
            }
            size_t j = 0;
            for (; j < n_exceptions; j++) {
                // I cannot use scatter as this is an accumulation
                uint16_t vector_idx = exceptions_positions[offset_to_dimension_start + j];
                distances_p[vector_idx] += distance_correction[j];
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
            const int32_t * dim_clip_value,
            const float * scaling_factors = nullptr
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
            DISTANCE_TYPE * distances_p,
            const float * scaling_factors = nullptr
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
            size_t num_dimensions,
            const float * scaling_factors = nullptr
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
            const int32_t * dim_clip_value = nullptr,
            const float * scaling_factors = nullptr
    ){
        // TODO
    }

    // Defer to the scalar kernel
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
            const QUERY_TYPE *__restrict vector1,
            const DATA_TYPE *__restrict vector2,
            size_t num_dimensions,
            const float * scaling_factors = nullptr
    ){
        __m512 d2_vec = _mm512_setzero();
        __m512 a_vec, b_vec;

simsimd_ip_f32_skylake_cycle:
        if (num_dimensions < 16) {
            __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, num_dimensions);
            a_vec = _mm512_maskz_loadu_ps(mask, vector1);
            b_vec = _mm512_maskz_loadu_ps(mask, vector2);
            num_dimensions = 0;
        } else {
            a_vec = _mm512_loadu_ps(vector1);
            b_vec = _mm512_loadu_ps(vector2);
            vector1 += 16, vector2 += 16, num_dimensions -= 16;
        }
        d2_vec = _mm512_fmadd_ps(a_vec, b_vec, d2_vec);
        if (num_dimensions)
            goto simsimd_ip_f32_skylake_cycle;

        // _simsimd_reduce_f32x16_skylake
        __m512 x = _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
    };

};

}

// TODO: Defer to the scalar kernel on Vertical but on VerticalPruning use the SIMD if Sapphire Rapids

#endif //PDX_AVX512_COMPUTERS_HPP

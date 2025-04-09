#ifndef PDX_AVX512_COMPUTERS_HPP
#define PDX_AVX512_COMPUTERS_HPP

#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include "pdx/common.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"

namespace PDX {

template<DistanceFunction alpha, Quantization q>
class SIMDComputer {};

template <>
class SIMDComputer<L2, Quantization::U8>{

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
            const int32_t * dim_clip_value = nullptr
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
            const int32_t * dim_clip_value = nullptr
    ){
        // SIMD is less efficient when looping on the array of not-yet pruned vectors
        // A way to improve the performance by ~20% is using a GATHER intrinsic. However this only works on Intel microarchs.
        // In AMD (Zen 4, Zen 3) using a GATHER is shooting ourselves in the foot (~80 uops)
        // __AVX512FP16__ macro let us detect Intel architectures (from Sapphire Rapids onwards)
#if defined(__AVX512FP16__)
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
            DISTANCE_TYPE * distances_p
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
            size_t num_dimensions
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

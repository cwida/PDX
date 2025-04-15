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
                for (; i <= n_vectors - 16; i+=16) {
                    // Read 64 bytes of data (64 values) with 4 dimensions of 16 vectors
                    res = _mm512_load_si512(&distances_p[i]);
                    vec2_u8 = _mm512_loadu_si512(&data[offset_to_dimension_start + i * 4]); // This 4 is because everytime I read 4 dimensions
                    diff_u8 = _mm512_or_si512(_mm512_subs_epu8(vec1_u8, vec2_u8), _mm512_subs_epu8(vec2_u8, vec1_u8));
                    _mm512_store_epi32(&distances_p[i], _mm512_dpbusds_epi32(res, diff_u8, diff_u8));
                }
                y_vec1_u8 = _mm256_set1_epi32(query_value);
                for (; i <= n_vectors - 8; i+=8) {
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
            DISTANCE_TYPE * distances_p
    ){
        __m512i res[4];
        __m512i vec2_u8;
        __m512i vec1_u8;
        __m512i diff_u8;
        uint32_t * query_grouped = (uint32_t *)query;
        // Load 64 initial values
        for (size_t i = 0; i < 4; ++i) {
            res[i] = _mm512_setzero_si512();
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
                _mm512_dpbusds_epi32(res[i], diff_u8, diff_u8);
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
            size_t num_dimensions){
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
            const int32_t * dim_clip_value = nullptr
    ){
        // TODO
    }

    static void Vertical(
            const QUERY_TYPE *__restrict query,
            const DATA_TYPE *__restrict data,
            size_t start_dimension,
            size_t end_dimension,
            DISTANCE_TYPE * distances_p
    ){
        // TODO
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

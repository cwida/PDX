#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#endif


template<typename T>
struct KNNCandidate {
    uint32_t index;
    float distance;
};

template<>
struct KNNCandidate<float> {
    uint32_t index;
    float distance;
};

template<>
struct KNNCandidate<uint8_t> {
    uint32_t index;
    uint32_t distance;
};

template<typename T>
struct DistanceType {
    using type = float; // default for f32
};
template<>
struct DistanceType<uint8_t> {
    using type = uint32_t;
};

template<typename T>
using DistanceType_t = typename DistanceType<T>::type;

template<typename T>
struct VectorComparator {
    bool operator() (const KNNCandidate<T>& a, const KNNCandidate<T>& b) {
        return a.distance < b.distance;
    }
};

template<typename T>
struct VectorComparatorInverse {
    bool operator() (const KNNCandidate<T>& a, const KNNCandidate<T>& b) {
        return a.distance > b.distance;
    }
};

enum VectorSearchKernel {
    F32_SIMD_IP,
    F32_SIMD_L2,
    F32_PDX_IP,
    F32_PDX_L2,

    U8_SIMD_L2,
    U8_SIMD_IP,
    U8_PDX_L2,
    U8_PDX_IP
};

static constexpr size_t F32_PDX_VECTOR_SIZE = 64;
static constexpr size_t U8_PDX_VECTOR_SIZE = 64;

alignas(64) static float distances_f32[64];
alignas(64) static uint32_t distances_u8[64];

static constexpr size_t U8_N_REGISTERS_NEON = 16;
static constexpr size_t U8_N_REGISTERS_AVX = 4;

////////////////
// SIMD
////////////////


inline float f32_simd_ip(
    const float *first_vector,
    const float *second_vector,
    const size_t d
) {
#if defined(__APPLE__)
    float distance = 0.0;
#pragma clang loop vectorize(enable)
    for (size_t i = 0; i < d; ++i) {
        distance += first_vector[i] * second_vector[i];
    }
    return distance;
#elif defined(__ARM_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0);
    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float32x4_t a_vec = vld1q_f32(first_vector + i);
        float32x4_t b_vec = vld1q_f32(second_vector + i);
        sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
    }
    float distance = vaddvq_f32(sum_vec);
    for (; i < d; ++i) {
        distance += first_vector[i] * second_vector[i];
    }
    return distance;
#elif defined(__AVX512F__)
    __m512 d2_vec = _mm512_setzero();
    __m512 a_vec, b_vec;
    size_t num_dimensions = d;

    simsimd_ip_f32_skylake_cycle:
        if (num_dimensions < 16) {
            __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, num_dimensions);
            a_vec = _mm512_maskz_loadu_ps(mask, first_vector);
            b_vec = _mm512_maskz_loadu_ps(mask, second_vector);
            num_dimensions = 0;
        } else {
            a_vec = _mm512_loadu_ps(first_vector);
            b_vec = _mm512_loadu_ps(second_vector);
            first_vector += 16, second_vector += 16, num_dimensions -= 16;
        }
    d2_vec = _mm512_fmadd_ps(a_vec, b_vec, d2_vec);
    if (num_dimensions)
        goto simsimd_ip_f32_skylake_cycle;

    // _simsimd_reduce_f32x16_skylake
    __m512 x = _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
    __m128 r = _mm512_castps512_ps128(_mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1))));
    r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(_mm_hadd_ps(r, r));
#endif
}


inline float f32_simd_l2(
    const float *first_vector,
    const float *second_vector,
    const size_t d
) {
#if defined(__APPLE__)
    float distance = 0.0;
#pragma clang loop vectorize(enable)
    for (size_t i = 0; i < d; ++i) {
        float diff = first_vector[i] - second_vector[i];
        distance += diff * diff;
    }
    return distance;
#elif defined(__ARM_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0);
    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float32x4_t a_vec = vld1q_f32(first_vector + i);
        float32x4_t b_vec = vld1q_f32(second_vector + i);
        float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
        sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
    }
    float distance = vaddvq_f32(sum_vec);
#pragma clang loop vectorize(enable)
    for (; i < d; ++i) {
        float diff = first_vector[i] - second_vector[i];
        distance += diff * diff;
    }
    return distance;
#elif defined(__AVX512F__)
    __m512 d2_vec = _mm512_setzero();
    __m512 a_vec, b_vec;
    size_t num_dimensions = d;

    simsimd_l2sq_f32_skylake_cycle:
        if (d < 16) {
            __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, num_dimensions);
            a_vec = _mm512_maskz_loadu_ps(mask, first_vector);
            b_vec = _mm512_maskz_loadu_ps(mask, second_vector);
            num_dimensions = 0;
        } else {
            a_vec = _mm512_loadu_ps(first_vector);
            b_vec = _mm512_loadu_ps(second_vector);
            first_vector += 16, second_vector += 16, num_dimensions -= 16;
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
#endif
}


inline uint32_t u8_simd_l2(
    const uint8_t *first_vector,
    const uint8_t *second_vector,
    const size_t d
) {
#if defined(__ARM_NEON)
    uint32x4_t sum_vec = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        uint8x16_t a_vec = vld1q_u8(first_vector + i);
        uint8x16_t b_vec = vld1q_u8(second_vector + i);
        uint8x16_t d_vec = vabdq_u8(a_vec, b_vec);
        sum_vec = vdotq_u32(sum_vec, d_vec, d_vec);
    }
    uint32_t distance = vaddvq_u32(sum_vec);
    for (; i < d; ++i) {
        const int n = first_vector[i] - second_vector[i];
        distance += n * n;
    }
    return distance;
#elif defined(__AVX512F__)
    __m512i d2_i32_vec = _mm512_setzero_si512();
    __m512i a_u8_vec, b_u8_vec;
    size_t num_dimensions = d;

    simsimd_l2sq_u8_ice_cycle:
            if (num_dimensions < 64) {
                const __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, num_dimensions);
                a_u8_vec = _mm512_maskz_loadu_epi8(mask, first_vector);
                b_u8_vec = _mm512_maskz_loadu_epi8(mask, second_vector);
                num_dimensions = 0;
            }
            else {
                a_u8_vec = _mm512_loadu_si512(first_vector);
                b_u8_vec = _mm512_loadu_si512(second_vector);
                first_vector += 64, second_vector += 64, num_dimensions -= 64;
            }

    // Substracting unsigned vectors in AVX-512 is done by saturating subtraction:
    __m512i d_u8_vec = _mm512_or_si512(_mm512_subs_epu8(a_u8_vec, b_u8_vec), _mm512_subs_epu8(b_u8_vec, a_u8_vec));

    // Multiply and accumulate at `int8` level which are actually uint7, accumulate at `int32` level:
    d2_i32_vec = _mm512_dpbusds_epi32(d2_i32_vec, d_u8_vec, d_u8_vec);
    if (num_dimensions) goto simsimd_l2sq_u8_ice_cycle;
    return _mm512_reduce_add_epi32(d2_i32_vec);
#endif
};

inline uint32_t u8_simd_ip(
    const uint8_t *first_vector,
    const uint8_t *second_vector,
    const size_t d
) {
#if defined(__ARM_NEON)
    uint32x4_t sum_vec = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        uint8x16_t a_vec = vld1q_u8(first_vector + i);
        uint8x16_t b_vec = vld1q_u8(second_vector + i);
        sum_vec = vdotq_u32(sum_vec, a_vec, b_vec);
    }
    uint32_t distance = vaddvq_u32(sum_vec);
    for (; i < d; ++i) {
        distance += first_vector[i] * second_vector[i];
    }
    return distance;
#elif defined(__AVX512F__)
    __m512i d2_i32_vec = _mm512_setzero_si512();
    __m512i a_u8_vec, b_u8_vec;
    size_t num_dimensions = d;

    simsimd_l2sq_u8_ice_cycle:
            if (num_dimensions < 64) {
                const __mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, num_dimensions);
                a_u8_vec = _mm512_maskz_loadu_epi8(mask, first_vector);
                b_u8_vec = _mm512_maskz_loadu_epi8(mask, second_vector);
                num_dimensions = 0;
            }
            else {
                a_u8_vec = _mm512_loadu_si512(first_vector);
                b_u8_vec = _mm512_loadu_si512(second_vector);
                first_vector += 64, second_vector += 64, num_dimensions -= 64;
            }

    // Multiply and accumulate at `int8` level which are actually uint7, accumulate at `int32` level:
    d2_i32_vec = _mm512_dpbusds_epi32(d2_i32_vec, a_u8_vec, b_u8_vec);
    if (num_dimensions) goto simsimd_l2sq_u8_ice_cycle;
    return _mm512_reduce_add_epi32(d2_i32_vec);
#endif
};

////////////////
// PDX
////////////////


inline void f32_pdx_ip(
    const float *first_vector,
    const float *second_vector,
    const size_t d
) {
    memset((void*) distances_f32, 0.0, F32_PDX_VECTOR_SIZE * sizeof(float));
    for (size_t dim_idx = 0; dim_idx < d; dim_idx++) {
        const size_t dimension_idx = dim_idx;
        const size_t offset_to_dimension_start = dimension_idx * F32_PDX_VECTOR_SIZE;
        for (size_t vector_idx = 0; vector_idx < F32_PDX_VECTOR_SIZE; ++vector_idx) {
            distances_f32[vector_idx] += second_vector[dimension_idx] * first_vector[offset_to_dimension_start + vector_idx];
        }
    }
}

inline void f32_pdx_l1(
    const float *first_vector,
    const float *second_vector,
    const size_t d
) {
    memset((void*) distances_f32, 0.0, F32_PDX_VECTOR_SIZE * sizeof(float));
    for (size_t dim_idx = 0; dim_idx < d; dim_idx++) {
        const size_t dimension_idx = dim_idx;
        const size_t offset_to_dimension_start = dimension_idx * F32_PDX_VECTOR_SIZE;
        for (size_t vector_idx = 0; vector_idx < F32_PDX_VECTOR_SIZE; ++vector_idx) {
            float to_abs = second_vector[dimension_idx] - first_vector[offset_to_dimension_start + vector_idx];
            distances_f32[vector_idx] += std::fabs(to_abs);
        }
    }
}

inline void f32_pdx_l2(
    const float *first_vector,
    const float *second_vector,
    const size_t d
) {
    memset((void*) distances_f32, 0.0, F32_PDX_VECTOR_SIZE * sizeof(float));
    for (size_t dim_idx = 0; dim_idx < d; dim_idx++) {
        const size_t dimension_idx = dim_idx;
        const size_t offset_to_dimension_start = dimension_idx * F32_PDX_VECTOR_SIZE;
        for (size_t vector_idx = 0; vector_idx < F32_PDX_VECTOR_SIZE; ++vector_idx) {
            float to_multiply = second_vector[dimension_idx] - first_vector[offset_to_dimension_start + vector_idx];
            distances_f32[vector_idx] += to_multiply * to_multiply;
        }
    }
}

inline void u8_pdx_l2(
    const uint8_t *first_vector,
    const uint8_t *second_vector,
    const size_t d
) {
    memset((void*) distances_u8, 0, U8_PDX_VECTOR_SIZE * sizeof(uint32_t));
#if defined(__ARM_NEON)
    uint32x4_t res[U8_N_REGISTERS_NEON];
    const uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    // Load initial values
    for (size_t i = 0; i < U8_N_REGISTERS_NEON; ++i) {
        res[i] = vdupq_n_u32(0);
    }
    // Compute L2
    for (size_t dim_idx = 0; dim_idx < d; dim_idx+=4) {
        const uint32_t dimension_idx = dim_idx;
        const uint8x8_t vals = vld1_u8(&second_vector[dimension_idx]);
        const uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
        const size_t offset_to_dimension_start = dimension_idx * U8_PDX_VECTOR_SIZE;
        for (int i = 0; i < U8_N_REGISTERS_NEON; ++i) { // total: 64 vectors * 4 dimensions each (at 1 byte per value = 2048-bits)
            // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
            const uint8x16_t vec2_u8 = vld1q_u8(&first_vector[offset_to_dimension_start + i * 16]);
            const uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
            res[i] = vdotq_u32(res[i], diff_u8, diff_u8);
        }
    }
    // Store results back
    for (int i = 0; i < U8_N_REGISTERS_NEON; ++i) {
        vst1q_u32(&distances_u8[i * 4], res[i]);
    }
#elif defined(__AVX512F__)
    __m512i res[U8_N_REGISTERS_AVX];
    const uint32_t * query_grouped = (uint32_t *)second_vector;
    // Load 64 initial values
    for (size_t i = 0; i < U8_N_REGISTERS_AVX; ++i) {
        res[i] = _mm512_load_si512(&distances_u8[i * 16]);
    }
    // Compute L2
    for (size_t dim_idx = 0; dim_idx < d; dim_idx+=4) {
        const uint32_t dimension_idx = dim_idx;
        // To load the query efficiently I will load it as uint32_t (4 bytes packed in 1 word)
        const uint32_t query_value = query_grouped[dimension_idx / 4];
        // And then broadcast it to the register
        const __m512i vec1_u8 = _mm512_set1_epi32(query_value);
        const size_t offset_to_dimension_start = dimension_idx * U8_PDX_VECTOR_SIZE;
        for (int i = 0; i < U8_N_REGISTERS_AVX; ++i) { // total: 64 vectors (4 iterations of 16 vectors) * 4 dimensions each (at 1 byte per value = 2048-bits)
            // Read 64 bytes of data (64 values) with 4 dimensions of 16 vectors
            const __m512i vec2_u8 = _mm512_loadu_si512(&first_vector[offset_to_dimension_start + i * 64]);
            const __m512i diff_u8 = _mm512_or_si512(_mm512_subs_epu8(vec1_u8, vec2_u8), _mm512_subs_epu8(vec2_u8, vec1_u8));
            // I can use this asymmetric dot product as my values are actually 7-bit
            // Hence, the [sign] properties of the second operand is ignored
            // As results will never be negative, it can be stored on res[i] without issues
            res[i] = _mm512_dpbusds_epi32(res[i], diff_u8, diff_u8);
        }
    }
    // Store results back
    for (int i = 0; i < U8_N_REGISTERS_AVX; ++i) {
        _mm512_store_epi32(&distances_u8[i * 16], res[i]);
    }
#endif
};

inline void u8_pdx_ip(
    const uint8_t *first_vector,
    const uint8_t *second_vector,
    const size_t d
) {
    memset((void*) distances_u8, 0, U8_PDX_VECTOR_SIZE * sizeof(uint32_t));
#if defined(__ARM_NEON)
    uint32x4_t res[U8_N_REGISTERS_NEON];
    const uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    // Load initial values
    for (size_t i = 0; i < U8_N_REGISTERS_NEON; ++i) {
        res[i] = vdupq_n_u32(0);
    }
    // Compute L2
    for (size_t dim_idx = 0; dim_idx < d; dim_idx+=4) {
        const uint32_t dimension_idx = dim_idx;
        const uint8x8_t vals = vld1_u8(&second_vector[dimension_idx]);
        const uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
        const size_t offset_to_dimension_start = dimension_idx * U8_PDX_VECTOR_SIZE;
        for (int i = 0; i < 16; ++i) { // total: 64 vectors * 4 dimensions each (at 1 byte per value = 2048-bits)
            // Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
            const uint8x16_t vec2_u8 = vld1q_u8(&first_vector[offset_to_dimension_start + i * 16]);
            res[i] = vdotq_u32(res[i], vec2_u8, vec1_u8);
        }
    }
    // Store results back
    for (int i = 0; i < U8_N_REGISTERS_NEON; ++i) {
        vst1q_u32(&distances_u8[i * 4], res[i]);
    }
#elif defined(__AVX512F__)
    __m512i res[U8_N_REGISTERS_AVX];
    const uint32_t * query_grouped = (uint32_t *)second_vector;
    // Load 64 initial values
    for (size_t i = 0; i < U8_N_REGISTERS_AVX; ++i) {
        res[i] = _mm512_load_si512(&distances_u8[i * 16]);
    }
    // Compute L2
    for (size_t dim_idx = 0; dim_idx < d; dim_idx+=4) {
        const uint32_t dimension_idx = dim_idx;
        // To load the query efficiently I will load it as uint32_t (4 bytes packed in 1 word)
        const uint32_t query_value = query_grouped[dimension_idx / 4];
        // And then broadcast it to the register
        const __m512i vec1_u8 = _mm512_set1_epi32(query_value);
        const size_t offset_to_dimension_start = dimension_idx * U8_PDX_VECTOR_SIZE;
        for (int i = 0; i < U8_N_REGISTERS_AVX; ++i) { // total: 64 vectors (4 iterations of 16 vectors) * 4 dimensions each (at 1 byte per value = 2048-bits)
            // Read 64 bytes of data (64 values) with 4 dimensions of 16 vectors
            const __m512i vec2_u8 = _mm512_loadu_si512(&first_vector[offset_to_dimension_start + i * 64]);
            // I can use this asymmetric dot product as my values are actually 7-bit
            // Hence, the [sign] properties of the second operand is ignored
            // As results will never be negative, it can be stored on res[i] without issues
            res[i] = _mm512_dpbusds_epi32(res[i], vec1_u8, vec2_u8);
        }
    }
    // Store results back
    for (int i = 0; i < U8_N_REGISTERS_AVX; ++i) {
        _mm512_store_epi32(&distances_u8[i * 16], res[i]);
    }
#endif
};

template <VectorSearchKernel kernel=F32_SIMD_IP, bool FILTERED=false, typename T=float>
std::vector<KNNCandidate<T>> standalone_simd(
    const T *first_vector,
    const T *second_vector,
    const size_t d,
    const size_t num_queries,
    const size_t num_vectors,
    const size_t knn,
    const size_t * positions = nullptr
) {
    std::vector<KNNCandidate<T>> result(knn * num_queries);
    std::vector<KNNCandidate<T>> all_distances(num_vectors);
    const T* query = second_vector;
    for (size_t i = 0; i < num_queries; ++i) {
        const T* data = first_vector;
        // Fill all_distances by direct indexing
        for (size_t j = 0; j < num_vectors; ++j) {
            if constexpr (FILTERED) {
                data = data + (positions[j] * d);
            }
            DistanceType_t<T> current_distance;
            if constexpr (kernel == F32_SIMD_IP){
                current_distance = f32_simd_ip(data, query, d);
            } else if constexpr (kernel == F32_SIMD_L2) {
                current_distance = f32_simd_l2(data, query, d);
            } else if constexpr (kernel == U8_SIMD_L2) {
                current_distance = u8_simd_l2(data, query, d);
            } else if constexpr (kernel == U8_SIMD_IP) {
                current_distance = u8_simd_ip(data, query, d);
            }
            all_distances[j].index = static_cast<uint32_t>(j);
            all_distances[j].distance = current_distance;
            if constexpr (!FILTERED) {
                data += d;
            }
        }

        // Partial sort to get top-k
        if constexpr (
            kernel == F32_SIMD_IP
            || kernel == U8_SIMD_IP
        ) {
            std::partial_sort(
                all_distances.begin(),
                all_distances.begin() + knn,
                all_distances.end(),
                VectorComparatorInverse<T>()
            );
        } else {
            std::partial_sort(
                all_distances.begin(),
                all_distances.begin() + knn,
                all_distances.end(),
                VectorComparator<T>()
            );
        }
        // Copy top-k results to result vector
        for (size_t k = 0; k < knn; ++k) {
            result[i * knn + k] = all_distances[k];
        }
        query += d;
    }
    return result;
}

template <VectorSearchKernel kernel=F32_PDX_IP, int PDX_BLOCK_SIZE=64, typename T=float>
std::vector<KNNCandidate<T>> standalone_pdx(
    const T *first_vector,
    const T *second_vector,
    const size_t d,
    const size_t num_queries,
    const size_t num_vectors,
    const size_t knn
) {
    std::vector<KNNCandidate<T>> result(knn * num_queries);
    std::vector<KNNCandidate<T>> all_distances(num_vectors);
    const T* query = second_vector;
    for (size_t i = 0; i < num_queries; ++i) {
        const T* data = first_vector;
        // Fill all_distances by direct indexing
        size_t global_offset = 0;
        for (size_t j = 0; j < num_vectors; j+=PDX_BLOCK_SIZE) {
            if constexpr (kernel == F32_PDX_IP){
                f32_pdx_ip(data, query, d);
            } else if constexpr (kernel == F32_PDX_L2) {
                f32_pdx_l2(data, query, d);
            } else if constexpr (kernel == U8_PDX_L2) {
                u8_pdx_l2(data, query, d);
            } else if constexpr (kernel == U8_PDX_IP) {
                u8_pdx_ip(data, query, d);
            }
            // TODO: Ugly (could be a bottleneck on PDX kernels)
            for (uint32_t z = 0; z < PDX_BLOCK_SIZE; ++z) {
                all_distances[global_offset].index = global_offset;
                if constexpr (std::is_same_v<T, float>){
                    all_distances[global_offset].distance = distances_f32[z];
                } else if constexpr (std::is_same_v<T, uint8_t>) {
                    all_distances[global_offset].distance = distances_u8[z];
                }
                global_offset++;
            }
            data += d * PDX_BLOCK_SIZE;
        }

        // Partial sort to get top-k
        if constexpr (
            kernel == F32_PDX_IP
            || kernel == U8_PDX_IP
        ) {
            std::partial_sort(
                all_distances.begin(),
                all_distances.begin() + knn,
                all_distances.end(),
                VectorComparatorInverse<T>()
            );
        } else {
            std::partial_sort(
                all_distances.begin(),
                all_distances.begin() + knn,
                all_distances.end(),
                VectorComparator<T>()
            );
        }

        // Copy top-k results to result vector
        for (size_t k = 0; k < knn; ++k) {
            result[i * knn + k] = all_distances[k];
        }
        query += d;
    }
    return result;
}

std::vector<KNNCandidate<float>> standalone_f32(
    const VectorSearchKernel kernel,
    const float *first_vector,
    const float *second_vector,
    const size_t d,
    const size_t num_queries,
    const size_t num_vectors,
    const size_t knn
) {
    switch (kernel) {
        case F32_PDX_IP:
            return standalone_pdx<F32_PDX_IP>(first_vector, second_vector, d, num_queries, num_vectors, knn);
        case F32_PDX_L2:
            return standalone_pdx<F32_PDX_L2>(first_vector, second_vector, d, num_queries, num_vectors, knn);

        case F32_SIMD_IP:
            return standalone_simd<F32_SIMD_IP>(first_vector, second_vector, d, num_queries, num_vectors, knn);
        case F32_SIMD_L2:
            return standalone_simd<F32_SIMD_L2>(first_vector, second_vector, d, num_queries, num_vectors, knn);

        default:
            return standalone_pdx<F32_PDX_IP>(first_vector, second_vector, d, num_queries, num_vectors, knn);
    }
}

std::vector<KNNCandidate<uint8_t>> filtered_standalone_u8(
    const VectorSearchKernel kernel,
    const uint8_t *first_vector,
    const uint8_t *second_vector,
    const size_t d,
    const size_t num_queries,
    const size_t num_vectors,
    const size_t knn,
    const size_t *positions
) {
    switch (kernel) {
        case U8_SIMD_L2:
            return standalone_simd<U8_SIMD_L2, true>(first_vector, second_vector, d, num_queries, num_vectors, knn, positions);
        default:
            return standalone_simd<U8_SIMD_L2, true>(first_vector, second_vector, d, num_queries, num_vectors, knn, positions);
    }
}

std::vector<KNNCandidate<uint8_t>> standalone_u8(
    const VectorSearchKernel kernel,
    const uint8_t *first_vector,
    const uint8_t *second_vector,
    const size_t d,
    const size_t num_queries,
    const size_t num_vectors,
    const size_t knn
) {
    switch (kernel) {
        case U8_PDX_L2:
            return standalone_pdx<U8_PDX_L2, U8_PDX_VECTOR_SIZE>(first_vector, second_vector, d, num_queries, num_vectors, knn);
        case U8_PDX_IP:
            return standalone_pdx<U8_PDX_IP, U8_PDX_VECTOR_SIZE>(first_vector, second_vector, d, num_queries, num_vectors, knn);

        case U8_SIMD_L2:
            return standalone_simd<U8_SIMD_L2>(first_vector, second_vector, d, num_queries, num_vectors, knn);
        case U8_SIMD_IP:
            return standalone_simd<U8_SIMD_IP>(first_vector, second_vector, d, num_queries, num_vectors, knn);

        default:
            return standalone_pdx<U8_PDX_L2>(first_vector, second_vector, d, num_queries, num_vectors, knn);
    }
}

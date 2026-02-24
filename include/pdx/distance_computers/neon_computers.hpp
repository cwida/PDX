#pragma once

#include <cstdint>
#include "arm_neon.h"
#include "pdx/common.hpp"

namespace PDX {

template <DistanceMetric alpha, Quantization Q>
class SIMDComputer {};

template <>
class SIMDComputer<DistanceMetric::L2SQ, Quantization::F32> {
public:
	using distance_t = pdx_distance_t<F32>;
	using query_t = pdx_quantized_embedding_t<F32>;
	using data_t = pdx_data_t<F32>;

	template <bool SKIP_PRUNED>
	static void Vertical(const query_t *PDX_RESTRICT query, const data_t *PDX_RESTRICT data, size_t n_vectors,
	                     size_t total_vectors, size_t start_dimension, size_t end_dimension, distance_t *distances_p,
	                     const uint32_t *pruning_positions = nullptr) {
		size_t dimensions_jump_factor = total_vectors;
		for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension; ++dimension_idx) {
			size_t offset_to_dimension_start = dimension_idx * dimensions_jump_factor;
			for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
				auto true_vector_idx = vector_idx;
				if constexpr (SKIP_PRUNED) {
					true_vector_idx = pruning_positions[vector_idx];
				}
				distance_t to_multiply = query[dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
				distances_p[true_vector_idx] += to_multiply * to_multiply;
			}
		}
	}

	static distance_t Horizontal(const query_t *PDX_RESTRICT vector1, const data_t *PDX_RESTRICT vector2,
	                             size_t num_dimensions) {
#if defined(__APPLE__)
		distance_t distance = 0.0;
#pragma clang loop vectorize(enable)
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
	static void Vertical(const query_t *PDX_RESTRICT query, const data_t *PDX_RESTRICT data, size_t n_vectors,
	                     size_t total_vectors, size_t start_dimension, size_t end_dimension, distance_t *distances_p,
	                     const uint32_t *pruning_positions = nullptr) {
		size_t dim_idx = start_dimension;
		for (; dim_idx + 4 <= end_dimension; dim_idx += 4) {
			uint32_t dimension_idx = dim_idx;
			uint8x8_t vals = vld1_u8(&query[dimension_idx]);
			size_t offset_to_dimension_start = dimension_idx * total_vectors;
			size_t i = 0;
			if constexpr (!SKIP_PRUNED) {
				const uint8x16_t idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
				const uint8x16_t vec1_u8 = vqtbl1q_u8(vcombine_u8(vals, vals), idx);
				for (; i + 4 <= n_vectors; i += 4) {
					// Read 16 bytes of data (16 values) with 4 dimensions of 4 vectors
					uint32x4_t res = vld1q_u32(&distances_p[i]);
					uint8x16_t vec2_u8 = vld1q_u8(
					    &data[offset_to_dimension_start + i * 4]); // This 4 is because everytime I read 4 dimensions
					uint8x16_t diff_u8 = vabdq_u8(vec1_u8, vec2_u8);
					vst1q_u32(&distances_p[i], squared_dot_accumulate(res, diff_u8));
				}
			}
			for (; i < n_vectors; ++i) {
				size_t vector_idx = i;
				if constexpr (SKIP_PRUNED) {
					vector_idx = pruning_positions[vector_idx];
				}
				int to_multiply_a = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
				int to_multiply_b = query[dimension_idx + 1] - data[offset_to_dimension_start + (vector_idx * 4) + 1];
				int to_multiply_c = query[dimension_idx + 2] - data[offset_to_dimension_start + (vector_idx * 4) + 2];
				int to_multiply_d = query[dimension_idx + 3] - data[offset_to_dimension_start + (vector_idx * 4) + 3];
				distances_p[vector_idx] += (to_multiply_a * to_multiply_a) + (to_multiply_b * to_multiply_b) +
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

	static distance_t Horizontal(const query_t *PDX_RESTRICT vector1, const data_t *PDX_RESTRICT vector2,
	                             size_t num_dimensions) {
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

#pragma once

#include "pdx/common.hpp"
#include <cstdint>

namespace PDX {

template <DistanceMetric alpha, Quantization Q>
class ScalarComputer {};

template <>
class ScalarComputer<DistanceMetric::L2SQ, Quantization::F32> {
  public:
    using distance_t = pdx_distance_t<F32>;
    using query_t = pdx_quantized_embedding_t<F32>;
    using data_t = pdx_data_t<F32>;

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
        distance_t distance = 0.0;
#pragma clang loop vectorize(enable)
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance_t to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
            distance += to_multiply * to_multiply;
        }
        return distance;
    };

    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        auto data_bits = reinterpret_cast<const uint32_t*>(data);
        auto out_bits = reinterpret_cast<uint32_t*>(out);
#pragma clang loop vectorize(enable)
        for (size_t j = 0; j < d; ++j) {
            out_bits[j] = data_bits[j] ^ masks[j];
        }
    }
};

template <>
class ScalarComputer<DistanceMetric::L2SQ, Quantization::U8> {
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
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            for (size_t i = 0; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED) {
                    vector_idx = pruning_positions[vector_idx];
                }
                int da = query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int db = query[dimension_idx + 1] -
                         data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int dc = query[dimension_idx + 2] -
                         data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int dd = query[dimension_idx + 3] -
                         data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] += (da * da) + (db * db) + (dc * dc) + (dd * dd);
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
        distance_t distance = 0;
        for (size_t i = 0; i < num_dimensions; ++i) {
            int diff = static_cast<int>(vector1[i]) - static_cast<int>(vector2[i]);
            distance += diff * diff;
        }
        return distance;
    };
};

} // namespace PDX

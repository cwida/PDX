#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/quantizers/scalar.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace PDX {

struct LVQParams {
    float quantization_base;
    float quantization_step;
    float dequantized_squared_norm;
};

template <uint8_t BIT_WIDTH = 8>
class LVQQuantizer : public Quantizer {
    static_assert(BIT_WIDTH == 8, "Only 8-bit LVQ is implemented");

  public:
    static constexpr uint8_t MAX_VALUE = (1 << BIT_WIDTH) - 1;

    explicit LVQQuantizer(size_t num_dimensions) : Quantizer(num_dimensions) {}

    LVQParams QuantizeEmbedding(
        const float* PDX_RESTRICT embedding,
        uint8_t* PDX_RESTRICT output_codes
    ) const {
        float lower = embedding[0];
        float upper = embedding[0];
        for (size_t i = 1; i < num_dimensions; ++i) {
            lower = std::min(lower, embedding[i]);
            upper = std::max(upper, embedding[i]);
        }
        const float range = upper - lower;
        const float step = (range > 0) ? range / static_cast<float>(MAX_VALUE) : 1.0f;
        const float inverse_step = 1.0f / step;
        float dequantized_squared_norm = 0.0f;
        for (size_t i = 0; i < num_dimensions; ++i) {
            const float code = std::min(
                static_cast<float>(MAX_VALUE),
                std::max(0.0f, std::nearbyint((embedding[i] - lower) * inverse_step))
            );
            output_codes[i] = static_cast<uint8_t>(code);
            const float dequantized = lower + step * code;
            dequantized_squared_norm += dequantized * dequantized;
        }
        return {lower, step, dequantized_squared_norm};
    }

    void DequantizeEmbedding(
        const uint8_t* PDX_RESTRICT codes,
        const LVQParams& params,
        float* PDX_RESTRICT output_embedding
    ) const {
        PDX_VECTORIZE_LOOP
        for (size_t i = 0; i < num_dimensions; ++i) {
            output_embedding[i] =
                params.quantization_base + params.quantization_step * static_cast<float>(codes[i]);
        }
    }

    float ComputeL2SQ(
        const float* PDX_RESTRICT query,
        const float query_sum,
        const float query_squared_norm,
        const uint8_t* PDX_RESTRICT codes,
        const LVQParams& params
    ) const {
        const float dot = DistanceComputer<DistanceMetric::L2SQ, F32>::HorizontalDotWithU8(
            query, codes, num_dimensions
        );
        return query_squared_norm + params.dequantized_squared_norm -
               2.0f * (params.quantization_base * query_sum + params.quantization_step * dot);
    }
};

} // namespace PDX

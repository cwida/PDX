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
    static_assert(BIT_WIDTH == 8 || BIT_WIDTH == 4, "LVQ is implemented for 8 and 4 bits");

  public:
    static constexpr uint8_t MAX_VALUE = (1 << BIT_WIDTH) - 1;

    static constexpr size_t GetCodesSizeInBytes(size_t num_dimensions) {
        return BIT_WIDTH == 8 ? num_dimensions : (num_dimensions + 1) / 2;
    }

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
            if constexpr (BIT_WIDTH == 8) {
                output_codes[i] = static_cast<uint8_t>(code);
            } else if (i % 2 == 0) {
                output_codes[i / 2] = static_cast<uint8_t>(code);
            } else {
                output_codes[i / 2] |= static_cast<uint8_t>(static_cast<uint8_t>(code) << 4);
            }
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
        if constexpr (BIT_WIDTH == 8) {
            PDX_VECTORIZE_LOOP
            for (size_t i = 0; i < num_dimensions; ++i) {
                output_embedding[i] = params.quantization_base +
                                      params.quantization_step * static_cast<float>(codes[i]);
            }
        } else {
            size_t i = 0;
            for (; i + 2 <= num_dimensions; i += 2) {
                const uint8_t code_pair = codes[i / 2];
                output_embedding[i] = params.quantization_base +
                                      params.quantization_step * static_cast<float>(code_pair & 0x0F);
                output_embedding[i + 1] = params.quantization_base +
                                          params.quantization_step * static_cast<float>(code_pair >> 4);
            }
            if (i < num_dimensions) {
                output_embedding[i] =
                    params.quantization_base +
                    params.quantization_step * static_cast<float>(codes[i / 2] & 0x0F);
            }
        }
    }

    float ComputeL2SQ(
        const float* PDX_RESTRICT query,
        const float query_sum,
        const float query_squared_norm,
        const uint8_t* PDX_RESTRICT codes,
        const LVQParams& params
    ) const {
        float dot;
        if constexpr (BIT_WIDTH == 8) {
            dot = DistanceComputer<DistanceMetric::L2SQ, F32>::HorizontalDotWithU8(
                query, codes, num_dimensions
            );
        } else {
            dot = DistanceComputer<DistanceMetric::L2SQ, F32>::HorizontalDotWithU4(
                query, codes, num_dimensions
            );
        }
        return query_squared_norm + params.dequantized_squared_norm -
               2.0f * (params.quantization_base * query_sum + params.quantization_step * dot);
    }
};

} // namespace PDX

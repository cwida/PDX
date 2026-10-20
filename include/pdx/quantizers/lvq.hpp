#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/quantizers/scalar.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>

namespace PDX {

struct LVQParams {
    float quantization_base;
    float quantization_step;
    float dequantized_squared_norm;
};

// A search query quantized to int8 for the LVQ centroid kernels: value ~ scale * code, with
// scale = max |value| / 127. It holds two layouts, each zero-padded to whole
// LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES chunks: the dimensions in order (for LVQ8 codes), and the even
// and the odd dimensions apart (for LVQ4 codes, whose byte i holds dimensions 2i and 2i + 1).
class LVQInt8Query {
  public:
    static constexpr float MAX_ABSOLUTE_CODE = 127.0f;

    // transformed_query: normalized if the index needs it, then rotated
    LVQInt8Query(const float* PDX_RESTRICT transformed_query, uint32_t num_dimensions)
        : num_dimensions(num_dimensions),
          padded_codes_size(AlignValue<size_t, LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES>(num_dimensions)),
          padded_code_pairs_size(
              AlignValue<size_t, LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES>((num_dimensions + 1) / 2)
          ),
          codes(new int8_t[padded_codes_size + 2 * padded_code_pairs_size]()) {
        using computer = DistanceComputer<DistanceMetric::L2SQ, F32>;
        float lower, upper;
        computer::MinMax(transformed_query, num_dimensions, lower, upper);
        const float max_absolute_value = std::max(-lower, upper);
        scale = (max_absolute_value > 0) ? max_absolute_value / MAX_ABSOLUTE_CODE : 1.0f;
        computer::QuantizeToInt8(transformed_query, num_dimensions, 1.0f / scale, codes.get());
        int8_t* even_dimension_codes = codes.get() + padded_codes_size;
        int8_t* odd_dimension_codes = even_dimension_codes + padded_code_pairs_size;
        PDX_VECTORIZE_LOOP
        for (size_t pair_idx = 0; pair_idx < num_dimensions / 2; ++pair_idx) {
            even_dimension_codes[pair_idx] = codes[2 * pair_idx];
            odd_dimension_codes[pair_idx] = codes[2 * pair_idx + 1];
        }
        if (num_dimensions % 2 == 1) {
            even_dimension_codes[num_dimensions / 2] = codes[num_dimensions - 1];
        }
        PDX_VECTORIZE_LOOP
        for (size_t i = 0; i < num_dimensions; ++i) {
            sum += transformed_query[i];
            squared_norm += transformed_query[i] * transformed_query[i];
        }
    }

    uint32_t GetNumDimensions() const { return num_dimensions; }
    // The LVQ8 layout
    const int8_t* GetCodesPtr() const { return codes.get(); }
    size_t GetPaddedCodesSize() const { return padded_codes_size; }
    // The LVQ4 layout
    const int8_t* GetEvenDimensionCodesPtr() const { return codes.get() + padded_codes_size; }
    const int8_t* GetOddDimensionCodesPtr() const {
        return codes.get() + padded_codes_size + padded_code_pairs_size;
    }
    size_t GetPaddedCodePairsSize() const { return padded_code_pairs_size; }
    float GetScale() const { return scale; }
    float GetSum() const { return sum; }
    float GetSquaredNorm() const { return squared_norm; }

  private:
    uint32_t num_dimensions;
    size_t padded_codes_size;
    size_t padded_code_pairs_size;
    std::unique_ptr<int8_t[]> codes;
    float scale = 1.0f;
    float sum = 0.0f;
    float squared_norm = 0.0f;
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
        float lower, upper;
        DistanceComputer<DistanceMetric::L2SQ, F32>::MinMax(
            embedding, num_dimensions, lower, upper
        );
        const float range = upper - lower;
        const float step = (range > 0) ? range / static_cast<float>(MAX_VALUE) : 1.0f;
        float dequantized_squared_norm;
        if constexpr (BIT_WIDTH == 8) {
            dequantized_squared_norm = DistanceComputer<DistanceMetric::L2SQ, F32>::QuantizeToU8(
                embedding, num_dimensions, lower, step, output_codes
            );
        } else {
            dequantized_squared_norm = DistanceComputer<DistanceMetric::L2SQ, F32>::QuantizeToU4(
                embedding, num_dimensions, lower, step, output_codes
            );
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
            DistanceComputer<DistanceMetric::L2SQ, F32>::DequantizeFromU4(
                codes,
                num_dimensions,
                params.quantization_base,
                params.quantization_step,
                output_embedding
            );
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

    // ComputeL2SQ with the query's dot to the codes taken in int8. The codes must be readable for
    // the query's padded size (see LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES)
    float ComputeL2SQWithInt8Query(
        const LVQInt8Query& query,
        const uint8_t* PDX_RESTRICT codes,
        const LVQParams& params
    ) const {
        int32_t dot;
        if constexpr (BIT_WIDTH == 8) {
            dot = DistanceComputer<DistanceMetric::L2SQ, F32>::HorizontalDotInt8WithU8(
                query.GetCodesPtr(), codes, query.GetPaddedCodesSize()
            );
        } else {
            dot = DistanceComputer<DistanceMetric::L2SQ, F32>::HorizontalDotInt8WithU4(
                query.GetEvenDimensionCodesPtr(),
                query.GetOddDimensionCodesPtr(),
                codes,
                query.GetPaddedCodePairsSize()
            );
        }
        return query.GetSquaredNorm() + params.dequantized_squared_norm -
               2.0f * (params.quantization_base * query.GetSum() +
                       params.quantization_step * query.GetScale() * static_cast<float>(dot));
    }
};

} // namespace PDX

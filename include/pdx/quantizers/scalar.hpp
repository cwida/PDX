#pragma once

#include "pdx/common.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace PDX {

struct ScalarQuantizationParams {
    float quantization_base;
    float quantization_scale;
};

class Quantizer {

  public:
    explicit Quantizer(size_t num_dimensions) : num_dimensions(num_dimensions) {}
    virtual ~Quantizer() = default;

  public:
    void NormalizeQuery(const float* src, float* out) const {
        float sum = 0.0f;
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < num_dimensions; ++i) {
            sum += src[i] * src[i];
        }

        if (sum == 0.0f) {
            return;
        }

        // float inverse_norm = 1.0f / std::sqrt(sum);
        float norm = std::sqrt(sum);
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < num_dimensions; ++i) {
            out[i] = src[i] / norm; // * inverse_norm;
        }
    }

    const size_t num_dimensions;

};

template <Quantization Q = U8>
class ScalarQuantizer : public Quantizer {
  public:
    using quantized_embedding_t = pdx_quantized_embedding_t<Q>;

    explicit ScalarQuantizer(size_t num_dimensions) : Quantizer(num_dimensions) {}

#ifdef __AVX512F__
    // We rely on _mm512_dpbusds_epi32 that has asymmetric operands
    static constexpr uint8_t MAX_VALUE = 127;
#else
    static constexpr uint8_t MAX_VALUE = 255;
#endif

    static ScalarQuantizationParams ComputeQuantizationParams(
        const float* embeddings,
        const size_t total_elements
    ) {
        float global_min = std::numeric_limits<float>::max();
        float global_max = std::numeric_limits<float>::lowest();
#pragma omp parallel for reduction(min : global_min) reduction(max : global_max) num_threads(PDX::g_n_threads)
        for (size_t i = 0; i < total_elements; ++i) {
            global_min = std::min(global_min, embeddings[i]);
            global_max = std::max(global_max, embeddings[i]);
        }
        const float range = global_max - global_min;
        return {global_min, (range > 0) ? static_cast<float>(MAX_VALUE) / range : 1.0f};
    }

    void QuantizeEmbedding(
        const float* embedding,
        const float quantization_base,
        const float quantization_scale,
        quantized_embedding_t* output_quantized_embedding
    ) {
        for (size_t i = 0; i < num_dimensions; ++i) {
            const int rounded =
                static_cast<int>(std::round((embedding[i] - quantization_base) * quantization_scale)
                );
            if (PDX_UNLIKELY(rounded > MAX_VALUE)) {
                output_quantized_embedding[i] = MAX_VALUE;
            } else if (PDX_UNLIKELY(rounded < 0)) {
                output_quantized_embedding[i] = 0;
            } else {
                output_quantized_embedding[i] = static_cast<uint8_t>(rounded);
            }
        }
    }
};

} // namespace PDX

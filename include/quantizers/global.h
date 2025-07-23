#ifndef PDX_QUANTIZERS_H
#define PDX_QUANTIZERS_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include "common.hpp"

namespace PDX {

class Quantizer {

public:
    virtual ~Quantizer() = default;

public:
    void NormalizeQuery(const float * src, float * out) {
        float sum = 0.0f;
        for (size_t i = 0; i < num_dimensions; ++i) {
            sum += src[i] * src[i];
        }
        float norm = std::sqrt(sum);
        for (size_t i = 0; i < num_dimensions; ++i) {
            out[i] = src[i] / norm;
        }
    }
    size_t num_dimensions = 0;

    void SetD(size_t d){
        num_dimensions = d;
    }

    virtual void ScaleQuery(const float * src, int32_t * dst) {};
};

template <Quantization q=U8>
class Global8Quantizer : public Quantizer {
public:
    using QUANTIZED_QUERY_TYPE = QuantizedVectorType_t<q>;
    alignas(64) inline static int32_t dim_clip_value[4096];
    alignas(64) inline static QUANTIZED_QUERY_TYPE quantized_query[4096];

    Global8Quantizer(){
        if constexpr (q == Quantization::U8){
            MAX_VALUE = 255;
        }
    }

    uint8_t MAX_VALUE;

    void PrepareQuery(
        const float *query,
        const float for_base,
        const float scale_factor
    ){
        for (size_t i = 0; i < num_dimensions; ++i){
            // Scale factor is global in symmetric kernel
            int rounded = std::round((query[i] - for_base) * scale_factor);
            dim_clip_value[i] = rounded;
            if (rounded > MAX_VALUE || rounded < 0) {
                    quantized_query[i] = 0;
            }else {
                    quantized_query[i] = static_cast<uint8_t>(rounded);
            }
        }
    };

};

}; // namespace PDX

#endif //PDX_QUANTIZERS_H

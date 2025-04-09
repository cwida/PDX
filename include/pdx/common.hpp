#ifndef PDX_COMMON_HPP
#define PDX_COMMON_HPP

#include <cstdint>
#include <cstdio>


namespace PDX {

    template<class T, T val=8>
    static constexpr uint32_t AlignValue(T n) {
        return ((n + (val - 1)) / val) * val;
    }

    enum DimensionsOrder {
        SEQUENTIAL,
        DISTANCE_TO_MEANS,
        DECREASING,
        DISTANCE_TO_MEANS_IMPROVED,
        DECREASING_IMPROVED,
        DIMENSION_ZONES
    };

    enum DistanceFunction {
        L2,
        IP,
        L1,
        NEGATIVE_L2 // Only the negative term of L2 (-2*q[i]*d[i])
    };

    enum Quantization {
        F32,
        F16,
        BF,
        U8,
        U6,
        U4
    };

    static constexpr size_t PDX_VECTOR_SIZE = 64;

    // TODO: Do the same for indexes?
    template<Quantization q>
    struct DistanceType {
        using type = uint32_t; // default for U8, U6, U4
    };
    template<>
    struct DistanceType<F32> {
        using type = float;
    };
    template<Quantization q>
    using DistanceType_t = typename DistanceType<q>::type;

    // TODO: Do the same for indexes?
    template<Quantization q>
    struct DataType {
        using type = uint8_t; // default for U8, U6, U4
    };
    template<>
    struct DataType<F32> {
        using type = float;
    };
    template<Quantization q>
    using DataType_t = typename DataType<q>::type;


    template<Quantization q>
    struct QuantizedVectorType {
        using type = uint8_t; // default for U8, U6, U4
    };
    template<>
    struct QuantizedVectorType<F32> {
        using type = float;
    };
    template<Quantization q>
    using QuantizedVectorType_t = typename QuantizedVectorType<q>::type;


    template<PDX::Quantization q>
    struct KNNCandidate {
        uint32_t index;
        PDX::DistanceType_t<q> distance;
    };

    template<PDX::Quantization q>
    struct VectorComparator {
        bool operator() (const KNNCandidate<q>& a, const KNNCandidate<q>& b) {
            return a.distance < b.distance;
        }
    };

    template <Quantization q>
    struct Vectorgroup { // default for U8, U6, U4
        uint32_t num_embeddings{};
        uint32_t *indices = nullptr;
        uint8_t *data = nullptr;
        int32_t *for_bases{};
    };

    template<>
    struct Vectorgroup<F32> {
        uint32_t num_embeddings{};
        uint32_t *indices = nullptr;
        float *data = nullptr;
    };

    //template<>
    //struct VectorgroupLEPX {
    //    uint32_t num_embeddings{};
    //    uint32_t *indices = nullptr;
    //    alignas(8) uint8_t *data = nullptr;
    //    uint8_t *exceptions{};
    //    int32_t *for_bases{};
    //    uint32_t exceptions_n{};
    //    uint16_t * exceptions_per_dimension{};
    //};

};

#endif //PDX_COMMON_HPP

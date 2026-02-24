#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <queue>
#include <random>

#ifndef PDX_RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define PDX_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define PDX_RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
#define PDX_RESTRICT __restrict__
#else
#define PDX_RESTRICT
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PDX_LIKELY(x) __builtin_expect(!!(x), 1)
#define PDX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define PDX_LIKELY(x) (x)
#define PDX_UNLIKELY(x) (x)
#endif

namespace PDX {

static constexpr float PROPORTION_HORIZONTAL_DIM = 0.75f;
static constexpr size_t D_THRESHOLD_FOR_DCT_ROTATION = 512;
static constexpr size_t PDX_MAX_DIMS = 16384;
static constexpr size_t H_DIM_SIZE = 64;
static constexpr size_t U8_INTERLEAVE_SIZE = 4;
static constexpr uint32_t DIMENSIONS_FETCHING_SIZES[20] = {16,  16,  32,  32,   32,   32,   64,
                                                           64,  64,  64,  128,  128,  128,  128,
                                                           256, 256, 512, 1024, 2048, 16384};

static constexpr bool AllFetchingSizesMultipleOfU8InterleaveSize() {
    for (auto s : DIMENSIONS_FETCHING_SIZES) {
        if (s % U8_INTERLEAVE_SIZE != 0) {
            return false;
        }
    }
    return true;
}
static_assert(
    AllFetchingSizesMultipleOfU8InterleaveSize(),
    "All DIMENSIONS_FETCHING_SIZES must be multiples of U8_INTERLEAVE_SIZE"
);

// Epsilon0 parameter of ADSampling (Reference: https://dl.acm.org/doi/abs/10.1145/3589282)
static constexpr float ADSAMPLING_PRUNING_AGGRESIVENESS = 1.5f;

template <class T, T val = 8>
static constexpr uint32_t AlignValue(T n) {
    return ((n + (val - 1)) / val) * val;
}

enum class DistanceMetric { L2SQ, COSINE, IP };

enum Quantization { F32, U8, F16, BF };

enum class PDXIndexType : uint8_t { PDX_F32 = 0, PDX_U8 = 1, PDX_TREE_F32 = 2, PDX_TREE_U8 = 3 };

// TODO: Do the same for indexes?
template <Quantization Q>
struct DistanceType {
    using type = uint32_t;
};
template <>
struct DistanceType<F32> {
    using type = float;
};
template <Quantization Q>
using pdx_distance_t = typename DistanceType<Q>::type;

// TODO: Do the same for indexes?
template <Quantization Q>
struct DataType {
    using type = uint8_t; // U8
};
template <>
struct DataType<F32> {
    using type = float;
};
template <Quantization Q>
using pdx_data_t = typename DataType<Q>::type;

template <Quantization Q>
struct QuantizedVectorType {
    using type = uint8_t; // U8
};
template <>
struct QuantizedVectorType<F32> {
    using type = float;
};
template <Quantization Q>
using pdx_quantized_embedding_t = typename QuantizedVectorType<Q>::type;

using eigen_matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct KNNCandidate {
    uint32_t index;
    float distance;
};

struct VectorComparator {
    bool operator()(const KNNCandidate& a, const KNNCandidate& b) {
        return a.distance < b.distance;
    }
};

using Heap = std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>;

struct PDXDimensionSplit {
    const uint32_t horizontal_dimensions;
    const uint32_t vertical_dimensions;
};

[[nodiscard]] static inline constexpr PDXDimensionSplit GetPDXDimensionSplit(
    const uint32_t num_dimensions
) {
    auto local_proportion_horizontal_dim = PROPORTION_HORIZONTAL_DIM;
    if (num_dimensions <= 128) {
        local_proportion_horizontal_dim = 0.25;
    }
    auto horizontal_d =
        static_cast<uint32_t>(static_cast<float>(num_dimensions) * local_proportion_horizontal_dim);
    auto vertical_d = static_cast<uint32_t>(num_dimensions - horizontal_d);
    if (horizontal_d % H_DIM_SIZE > 0) {
        horizontal_d = ((horizontal_d + H_DIM_SIZE / 2) / H_DIM_SIZE) * H_DIM_SIZE;
        vertical_d = num_dimensions - horizontal_d;
    }
    if (!vertical_d) {
        horizontal_d = H_DIM_SIZE;
        vertical_d = num_dimensions - horizontal_d;
    }
    if (num_dimensions <= H_DIM_SIZE) {
        horizontal_d = 0;
        vertical_d = num_dimensions;
    }

    assert(horizontal_d + vertical_d == num_dimensions);

    return {horizontal_d, vertical_d};
};

static_assert(GetPDXDimensionSplit(4).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(4).vertical_dimensions == 4);

static_assert(GetPDXDimensionSplit(33).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(33).vertical_dimensions == 33);

static_assert(GetPDXDimensionSplit(64).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(64).vertical_dimensions == 64);

static_assert(GetPDXDimensionSplit(65).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(65).vertical_dimensions == 65);

static_assert(GetPDXDimensionSplit(100).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(100).vertical_dimensions == 100);

static_assert(GetPDXDimensionSplit(127).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(127).vertical_dimensions == 127);

static_assert(GetPDXDimensionSplit(128).horizontal_dimensions == 64);
static_assert(GetPDXDimensionSplit(128).vertical_dimensions == 64);

static_assert(GetPDXDimensionSplit(256).horizontal_dimensions == 192);
static_assert(GetPDXDimensionSplit(256).vertical_dimensions == 64);

static_assert(GetPDXDimensionSplit(1024).horizontal_dimensions == 768);
static_assert(GetPDXDimensionSplit(1024).vertical_dimensions == 256);

static_assert(GetPDXDimensionSplit(1028).horizontal_dimensions == 768);
static_assert(GetPDXDimensionSplit(1028).vertical_dimensions == 260);

[[nodiscard]] inline constexpr uint32_t ComputeNumberOfClusters(const uint32_t num_embeddings) {
    if (num_embeddings < 500000) {
        return std::ceil(2 * std::sqrt(num_embeddings));
    } else if (num_embeddings < 2500000) {
        return std::ceil(4 * std::sqrt(num_embeddings));
    } else {
        return std::ceil(8 * std::sqrt(num_embeddings));
    }
}

[[nodiscard]] inline constexpr bool DistanceMetricRequiresNormalization(
    const PDX::DistanceMetric distance_metric
) {
    return distance_metric == PDX::DistanceMetric::COSINE ||
           distance_metric == PDX::DistanceMetric::IP;
}

} // namespace PDX

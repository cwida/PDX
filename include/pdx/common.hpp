#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numkong/cast.h>
#include <queue>
#include <random>

#include "superkmeans/allocator.h"
#include "superkmeans/executor.h"

#define PDX_ENSURE_POSITIVE(x)                                                                     \
    if ((x) <= 0) {                                                                                \
        throw std::invalid_argument("Value must be positive: " #x);                                \
    }

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

#ifndef PDX_ALWAYS_INLINE
#if __has_cpp_attribute(gnu::always_inline)
#define PDX_ALWAYS_INLINE [[gnu::always_inline]]
#elif defined(__GNUC__) || defined(__clang__)
#define PDX_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define PDX_ALWAYS_INLINE __forceinline
#else
#define PDX_ALWAYS_INLINE
#endif
#endif

#ifndef PDX_NO_INLINE
#if defined(__GNUC__) || defined(__clang__)
#define PDX_NO_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define PDX_NO_INLINE __declspec(noinline)
#else
#define PDX_NO_INLINE
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PDX_LIKELY(x) __builtin_expect(!!(x), 1)
#define PDX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define PDX_LIKELY(x) (x)
#define PDX_UNLIKELY(x) (x)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PDX_PREFETCH(addr, rw, locality) __builtin_prefetch((addr), (rw), (locality))
#elif defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>
#define PDX_PREFETCH(addr, rw, locality) __prefetch(addr)
#elif defined(_MSC_VER)
#include <xmmintrin.h>
#define PDX_PREFETCH(addr, rw, locality)                                                           \
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define PDX_PREFETCH(addr, rw, locality) ((void) 0)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PDX_POPCOUNT(x) __builtin_popcount(x)
#elif defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>
#define PDX_POPCOUNT(x) _CountOneBits(static_cast<unsigned long>(x))
#elif defined(_MSC_VER)
#include <intrin.h>
#define PDX_POPCOUNT(x) __popcnt(static_cast<unsigned int>(x))
#endif

#if defined(__clang__)
#define PDX_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define PDX_VECTORIZE_LOOP _Pragma("GCC ivdep")
#else
#define PDX_VECTORIZE_LOOP
#endif

namespace PDX {

// Runs PDX's parallel loops; the index holds one (see IPDXIndex) and passes it down.
using ParallelExecutor = skmeans::ParallelExecutor;
// Where PDX's large buffers come from; the index binds one (see IPDXIndex) and passes it down.
using MemoryAllocator = skmeans::MemoryAllocator;

static constexpr float PROPORTION_HORIZONTAL_DIM = 0.75f;
static constexpr size_t D_THRESHOLD_FOR_DCT_ROTATION = 512;

inline bool IsPowerOf2(const uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}
static constexpr size_t PDX_MAX_DIMS = 16384;
static constexpr size_t H_DIM_SIZE = 64;
static constexpr size_t U8_INTERLEAVE_SIZE = 4;
// The int8-query kernels of LVQ centroids read the codes in whole chunks of this many bytes: the
// query is zero-padded to a multiple of it, and the encoded centroids keep this many bytes after
// the last one.
static constexpr size_t LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES = 64;
static constexpr uint32_t DIMENSIONS_FETCHING_SIZES[20] = {16,  16,  32,  32,   32,   32,   64,
                                                           64,  64,  64,  128,  128,  128,  128,
                                                           256, 256, 512, 1024, 2048, 16384};

static constexpr float CENTROID_PERTURBATION_EPS = 1.0f / 1024.0f;
static constexpr size_t FLAT_SEARCHER_BULK_SEARCH_SIZE_THRESHOLD = 512;

static constexpr uint32_t SAQ_MIN_BITS_PER_DIMENSION = 3;
static constexpr uint32_t SAQ_MAX_BITS_PER_DIMENSION = 4;
static constexpr uint32_t SAQ_CHUNK_SIZE = H_DIM_SIZE;
static constexpr uint32_t SAQ_MIN_DIMENSIONS = SAQ_CHUNK_SIZE;
// Below SAQ_DP_PLAN_MIN_DIMENSIONS the plan is one segment at the declared bits, its fp16 factors
// on top of the budget (the DP's segments lose recall there). From SAQ_PDX_U8_HEAD_MIN_DIMENSIONS
// the head chunk is its own 8-bit segment in the PDX U8 layout (recall-neutral only from there)
static constexpr uint32_t SAQ_DP_PLAN_MIN_DIMENSIONS = 256;
static constexpr uint32_t SAQ_PDX_U8_HEAD_MIN_DIMENSIONS = 640;
static constexpr uint32_t SAQ_MAX_CODE_WIDTH = 8;
static constexpr size_t SAQ_FIT_SAMPLE_SIZE = 122880;
// The SAQ scans read and write up to one batch (16 lanes) past the last vector
static constexpr size_t SAQ_SCAN_PADDING = 16;
// The SAQ pruning bound's calibration: rows of the index as queries, each against its nearest
// neighbours by estimate in its nearest clusters
static constexpr size_t SAQ_CALIBRATION_QUERIES = 1000;
static constexpr uint32_t SAQ_CALIBRATION_CLUSTERS = 16;
static constexpr size_t SAQ_CALIBRATION_NEIGHBORS = 20;
static constexpr double SAQ_RECALIBRATION_GROWTH_FACTOR = 1.25;
// Filtered SAQ scans by the fraction of a cluster's rows that pass: below the first, the start runs
// the pruning kernels (the rows that don't pass skip their tail chunks); below the second, the head
// reads only the passing rows instead of scanning every row
// Measured in AVX512, (Zen 5, Granite Rapids), AVX2, and ARM NEON (M4 Pro)
static constexpr float SAQ_PRUNING_START_MAX_PASSING_FRACTION = 0.70f;
static constexpr float SAQ_POSITIONS_HEAD_MAX_PASSING_FRACTION = 0.10f;
// The SAQ encoder's scale per segment: the mean best scale of this many random Gaussian vectors of
// the segment's length, computed at fit (RaBitQ-Library's get_const_scaling_factors); each row then
// takes the best of SAQ_SCALE_GRID_POINTS scales spread over +-SAQ_SCALE_GRID_SPAN around it
static constexpr size_t SAQ_CONSTANT_SCALE_VECTORS = 100;
static constexpr uint32_t SAQ_SCALE_GRID_POINTS = 4;
static constexpr float SAQ_SCALE_GRID_SPAN = 0.25f;
static constexpr float SAQ_LEVEL_ROUNDING_EPSILON = 1e-5f;

static constexpr float CLUSTER_SKETCH_EPSILON = 1.9f;
static constexpr size_t CLUSTER_SKETCH_BATCH_SIZE = 32;
static constexpr uint32_t CLUSTER_SKETCH_DIMENSION_ALIGNMENT = 16;
static constexpr size_t CLUSTER_SKETCH_MEAN_SAMPLE_SIZE = 12288;
static constexpr size_t CLUSTER_SKETCH_TRANSPOSE_COLUMNS = 16;
// Position j of a half column holds lane CLUSTER_SKETCH_LANE_AT_POSITION[j] (and that lane + 16)
alignas(16) static constexpr uint8_t
    CLUSTER_SKETCH_LANE_AT_POSITION[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
alignas(16) static constexpr uint8_t
    CLUSTER_SKETCH_POSITION_OF_LANE[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};

// Maintenance (SPFresh-like)
// When a cluster splits, points may be stolen from / reassigned to at most
// SPLIT_MAX_NEIGHBOR_CLUSTERS
static constexpr size_t SPLIT_MAX_NEIGHBOR_CLUSTERS = 32;
// The 2-means cluster split runs this many iterations
static constexpr uint32_t SPLIT_KMEANS_ITERS = 4;
static constexpr float TWO_MEANS_TOL = 1e-4f;
static constexpr uint32_t SPLIT_CASCADE_MAX_DEPTH = 32;
static constexpr uint32_t MAX_CLUSTER_SIZE_LOWER_BOUND = 32;
static constexpr size_t CLUSTER_STORAGE_ORDER_MAX_CHAIN_CLUSTERS = 4096;
static constexpr uint32_t DELETED_MARKER = std::numeric_limits<uint32_t>::max();

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

template <class T>
std::unique_ptr<T[]> CopyArray(const std::unique_ptr<T[]>& source, size_t size) {
    std::unique_ptr<T[]> copy(new T[size]);
    std::copy(source.get(), source.get() + size, copy.get());
    return copy;
}

// TODO(@lkuffo, low): Convert the factors of many embeddings at once (nk_cast)
inline void WriteHalf(float value, uint8_t* out) {
    nk_f16_t half;
    nk_f32_to_f16(&value, &half);
    std::memcpy(out, &half, sizeof(half));
}

inline float ReadHalf(const uint8_t* in) {
    nk_f16_t half;
    std::memcpy(&half, in, sizeof(half));
    nk_f32_t value;
    nk_f16_to_f32(&half, &value);
    return value;
}

enum class DistanceMetric : uint8_t { L2SQ, COSINE, IP };

enum Quantization : uint8_t { F32, U8, F16, BF, SAQ };

enum class PDXIndexType : uint8_t {
    PDX_F32 = 0,
    PDX_U8 = 1,
    PDX_TREE_F32 = 2,
    PDX_TREE_U8 = 3,
    PDX_FLAT = 4,
    PDX_SAQ = 5
};

enum class CentroidQuantization : uint8_t { F32 = 0, LVQ8 = 1, LVQ4 = 2 };

enum class ClusterStorageOrder : uint8_t { ID = 0, NEAREST_CENTROID_CHAIN = 1 };

enum class ClusterSketchCheck : uint8_t { NEVER = 0, WHEN_CLUSTER_NOT_CACHED = 1, ALWAYS = 2 };

// A saved index's type byte: its PDXIndexType in the low nibble, its CentroidQuantization in the
// high
[[nodiscard]] constexpr uint8_t MakeIndexTypeByte(
    PDXIndexType type,
    CentroidQuantization centroid_quantization
) {
    return static_cast<uint8_t>(
        static_cast<uint8_t>(type) | (static_cast<uint8_t>(centroid_quantization) << 4)
    );
}

[[nodiscard]] constexpr PDXIndexType GetIndexTypeFromByte(uint8_t type_byte) {
    return static_cast<PDXIndexType>(type_byte & 0x0F);
}

[[nodiscard]] constexpr CentroidQuantization GetCentroidQuantizationFromByte(uint8_t type_byte) {
    return static_cast<CentroidQuantization>(type_byte >> 4);
}

static constexpr uint8_t PDX_SERIALIZATION_VERSION = 6;

template <Quantization Q>
struct DistanceType {
    using type = uint32_t;
};
template <>
struct DistanceType<F32> {
    using type = float;
};
template <>
struct DistanceType<SAQ> {
    using type = float;
};
template <Quantization Q>
using pdx_distance_t = typename DistanceType<Q>::type;

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

inline void GenerateRotationMatrix(
    uint32_t num_dimensions,
    uint32_t seed,
    float* out,
    bool fix_signs = false
) {
    std::mt19937 gen(seed);
    Eigen::Map<eigen_matrix_t> rotation(out, num_dimensions, num_dimensions);
    std::normal_distribution<float> normal_dist;
    for (size_t i = 0; i < num_dimensions; ++i) {
        for (size_t j = 0; j < num_dimensions; ++j) {
            rotation(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = normal_dist(gen);
        }
    }
    const Eigen::HouseholderQR<Eigen::MatrixXf> qr(rotation);
    rotation = qr.householderQ() * Eigen::MatrixXf::Identity(num_dimensions, num_dimensions);
    if (fix_signs) {
        for (Eigen::Index j = 0; j < num_dimensions; ++j) {
            if (qr.matrixQR()(j, j) < 0) {
                rotation.col(j) *= -1;
            }
        }
    }
}

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

struct CentroidDistanceComparator {
    const float* distances_to_centroids;
    bool operator()(uint32_t a, uint32_t b) const {
        return distances_to_centroids[a] < distances_to_centroids[b];
    }
};

struct PDXDimensionSplit {
    const uint32_t horizontal_dimensions;
    const uint32_t vertical_dimensions;
};

[[nodiscard]] static constexpr PDXDimensionSplit GetPDXDimensionSplit(const uint32_t num_dimensions
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

[[nodiscard]] constexpr uint32_t ComputeNumberOfClusters(const uint32_t num_embeddings) {
    if (num_embeddings < 500000) {
        return std::ceil(2 * std::sqrt(num_embeddings));
    } else if (num_embeddings < 2500000) {
        return std::ceil(4 * std::sqrt(num_embeddings));
    } else {
        return std::ceil(8 * std::sqrt(num_embeddings));
    }
}

[[nodiscard]] constexpr bool DistanceMetricRequiresNormalization(
    const PDX::DistanceMetric distance_metric
) {
    return distance_metric == PDX::DistanceMetric::COSINE ||
           distance_metric == PDX::DistanceMetric::IP;
}

} // namespace PDX

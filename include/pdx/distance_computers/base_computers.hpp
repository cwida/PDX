#pragma once

#include "pdx/common.hpp"

#ifdef __ARM_NEON
#include "pdx/distance_computers/neon_computers.hpp"
#endif

#if defined(__AVX2__) && !defined(__AVX512F__)
#include "pdx/distance_computers/avx2_computers.hpp"
#endif

#ifdef __AVX512F__
#include "pdx/distance_computers/avx512_computers.hpp"
#endif

// Fallback to scalar computer.
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
#include "pdx/distance_computers/scalar_computers.hpp"
#endif

// TODO: Support SVE

namespace PDX {

template <DistanceMetric alpha, Quantization Q>
class DistanceComputer {};

template <>
class DistanceComputer<DistanceMetric::L2SQ, Quantization::F32> {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarComputer<DistanceMetric::L2SQ, F32>;
#else
    using computer = SIMDComputer<DistanceMetric::L2SQ, F32>;
#endif

  public:
    constexpr static auto VerticalPruning = computer::Vertical<true>;
    constexpr static auto Vertical = computer::Vertical<false>;

    constexpr static auto Horizontal = computer::Horizontal;
    constexpr static auto FlipSign = computer::FlipSign;
    constexpr static auto HorizontalDotWithU8 = computer::HorizontalDotWithU8;
    constexpr static auto HorizontalDotWithU4 = computer::HorizontalDotWithU4;
    constexpr static auto MinMax = computer::MinMax;
    constexpr static auto QuantizeToU8 = computer::QuantizeToU8;
    constexpr static auto QuantizeToU4 = computer::QuantizeToU4;
    constexpr static auto DequantizeFromU4 = computer::DequantizeFromU4;
    constexpr static auto QuantizeToInt8 = computer::QuantizeToInt8;
    constexpr static auto HorizontalDotInt8WithU8 = computer::HorizontalDotInt8WithU8;
    constexpr static auto HorizontalDotInt8WithU4 = computer::HorizontalDotInt8WithU4;
};

template <>
class DistanceComputer<DistanceMetric::L2SQ, Quantization::U8> {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarComputer<DistanceMetric::L2SQ, U8>;
#else
    using computer = SIMDComputer<DistanceMetric::L2SQ, U8>;
#endif

  public:
    constexpr static auto VerticalPruning = computer::Vertical<true>;
    constexpr static auto Vertical = computer::Vertical<false>;

    constexpr static auto Horizontal = computer::Horizontal;
};

template <>
class DistanceComputer<DistanceMetric::L2SQ, Quantization::SAQ> {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarComputer<DistanceMetric::L2SQ, SAQ>;
#else
    using computer = SIMDComputer<DistanceMetric::L2SQ, SAQ>;
#endif

  public:
    using ScanTailFunction = size_t (*)(
        const uint8_t*,
        const int8_t*,
        float,
        float,
        const uint16_t*,
        float,
        uint32_t*,
        float*,
        float*,
        size_t
    );

    constexpr static auto ScanHead = computer::ScanHead;
    constexpr static auto PackRecord = ScalarComputer<DistanceMetric::L2SQ, SAQ>::PackRecord;
    constexpr static auto UnpackRecord = ScalarComputer<DistanceMetric::L2SQ, SAQ>::UnpackRecord;

    static ScanTailFunction GetScanTail(uint32_t width) {
        static constexpr ScanTailFunction SCAN_TAILS[SAQ_MAX_CODE_WIDTH + 1] = {
            nullptr,
            computer::ScanTail<1>,
            computer::ScanTail<2>,
            computer::ScanTail<3>,
            computer::ScanTail<4>,
            computer::ScanTail<5>,
            computer::ScanTail<6>,
            computer::ScanTail<7>,
            computer::ScanTail<8>
        };
        return SCAN_TAILS[width];
    }

    using AccumulateTailEstimatesFunction =
        void (*)(const uint8_t*, size_t, const int8_t*, float, float, const uint16_t*, float*);

    constexpr static auto ComputeHeadEstimates = computer::ComputeHeadEstimates;

    static AccumulateTailEstimatesFunction GetAccumulateTailEstimates(uint32_t width) {
        static constexpr AccumulateTailEstimatesFunction ACCUMULATE_TAIL_ESTIMATES[SAQ_MAX_CODE_WIDTH + 1] = {
            nullptr,
            computer::AccumulateTailEstimates<1>,
            computer::AccumulateTailEstimates<2>,
            computer::AccumulateTailEstimates<3>,
            computer::AccumulateTailEstimates<4>,
            computer::AccumulateTailEstimates<5>,
            computer::AccumulateTailEstimates<6>,
            computer::AccumulateTailEstimates<7>,
            computer::AccumulateTailEstimates<8>
        };
        return ACCUMULATE_TAIL_ESTIMATES[width];
    }
};

} // namespace PDX

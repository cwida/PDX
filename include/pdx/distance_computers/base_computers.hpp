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

} // namespace PDX

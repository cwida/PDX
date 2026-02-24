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

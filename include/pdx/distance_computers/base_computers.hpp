#pragma once
#ifndef PDX_BASE_COMPUTERS_HPP
#define PDX_BASE_COMPUTERS_HPP

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include "pdx/common.hpp"

#ifdef __ARM_NEON
#include "neon_computers.hpp"
#endif

#if defined(__AVX2__) && !defined(__AVX512F__)
#include "avx2_computers.hpp"
#endif

#ifdef __AVX512F__
#include "avx512_computers.hpp"
#endif

// TODO: Support SVE

namespace PDX {

template <DistanceFunction alpha, Quantization q>
class DistanceComputer {};

template<>
class DistanceComputer<L2, Quantization::U8> {
    using computer = SIMDComputer<L2, U8>;
public:
    constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
    constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
    constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
    constexpr static auto Vertical = computer::VerticalPruning<false, false>;

    constexpr static auto VerticalBlock = SIMDComputer<L2, U8>::Vertical;
    constexpr static auto Horizontal = SIMDComputer<L2, U8>::Horizontal;
};

template<>
class DistanceComputer<L2, Quantization::F32> {
    using computer = SIMDComputer<L2, F32>;
public:
    constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
    constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
    constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
    constexpr static auto Vertical = computer::VerticalPruning<false, false>;

    constexpr static auto VerticalBlock = SIMDComputer<L2, F32>::Vertical;
    constexpr static auto Horizontal = SIMDComputer<L2, F32>::Horizontal;
};

//template <Quantization q>
//class DistanceComputer<IP, q> {
//
//};
//
//template <Quantization q>
//class DistanceComputer<L1, q> {
//
//};
}; // namespace PDX


#endif //PDX_BASE_COMPUTERS_HPP
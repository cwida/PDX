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
class DistanceComputer<L2, Quantization::F32> {
    using computer = SIMDComputer<L2, F32>;
public:
    constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
    constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
    constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
    constexpr static auto Vertical = computer::VerticalPruning<false, false>;

    constexpr static auto VerticalBlock = computer::Vertical;
    constexpr static auto Horizontal = computer::Horizontal;
};

template<>
class DistanceComputer<NEGATIVE_L2, Quantization::F32> {
    using computer = SIMDComputer<NEGATIVE_L2, F32>;
public:
    constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
    constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
    constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
    constexpr static auto Vertical = computer::VerticalPruning<false, false>;

    constexpr static auto VerticalBlock = computer::Vertical;
    constexpr static auto Horizontal = computer::Horizontal;
};

template<>
class DistanceComputer<L2, Quantization::U8> {
    using computer = SIMDComputer<L2, U8>;
public:
    constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
    constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
    constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
    constexpr static auto Vertical = computer::VerticalPruning<false, false>;

    constexpr static auto VerticalBlock = computer::Vertical;
    constexpr static auto Horizontal = computer::Horizontal;
};

template<>
class DistanceComputer<L2, Quantization::ASYMMETRIC_U8> {
    using computer = SIMDComputer<L2, ASYMMETRIC_U8>;
public:
    constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
    constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
    constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
    constexpr static auto Vertical = computer::VerticalPruning<false, false>;

    constexpr static auto VerticalBlock = computer::Vertical;
    constexpr static auto Horizontal = computer::Horizontal;
};

template<>
class DistanceComputer<L2, Quantization::U6> {
    using computer = SIMDComputer<L2, U8>; // TODO: Use a proper computer for U6 that fuse decomp and distance calc
public:
    constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
    constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
    constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
    constexpr static auto Vertical = computer::VerticalPruning<false, false>;

    constexpr static auto VerticalBlock = computer::Vertical;
    constexpr static auto Horizontal = computer::Horizontal;
};

template<>
class DistanceComputer<L2, Quantization::U4> {
    using computer = SIMDComputer<L2, U4>;
public:
    constexpr static auto VerticalReorderedPruning = computer::VerticalPruning<true, true>;
    constexpr static auto VerticalPruning = computer::VerticalPruning<false, true>;
    constexpr static auto VerticalReordered = computer::VerticalPruning<true, false>;
    constexpr static auto Vertical = computer::VerticalPruning<false, false>;

    constexpr static auto VerticalBlock = computer::Vertical;
    constexpr static auto Horizontal = computer::Horizontal;
};


template<>
class DistanceComputer<IP, Quantization::F32> {
    using computer = SIMDComputer<IP, F32>;
public:
    constexpr static auto Horizontal = computer::Horizontal;
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
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"
#include "superkmeans/pdx/utils.h"

namespace {

class DistanceComputerTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(DistanceComputerTest, F32_Horizontal_SIMD_MatchesScalar) {
    std::vector<size_t> dimensions = {1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 256, 384};
    const size_t n_pairs = 100;

    for (size_t d : dimensions) {
        SCOPED_TRACE("d=" + std::to_string(d));
        auto v1 = skmeans::GenerateRandomVectors(n_pairs, d, -10.0f, 10.0f, 42);
        auto v2 = skmeans::GenerateRandomVectors(n_pairs, d, -10.0f, 10.0f, 123);

        for (size_t i = 0; i < n_pairs; ++i) {
            const float* a = v1.data() + i * d;
            const float* b = v2.data() + i * d;

            float scalar =
                PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::Horizontal(a, b, d);
            float simd =
                PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::Horizontal(a, b, d);

            float rel_error = std::abs(scalar - simd) / std::max(scalar, 1e-6f);
            EXPECT_LT(rel_error, 1e-5f)
                << "pair " << i << ": scalar=" << scalar << ", simd=" << simd;
        }
    }
}

TEST_F(DistanceComputerTest, F32_Vertical_SIMD_MatchesScalar) {
    std::vector<size_t> dimensions = {64, 128, 256, 384};
    std::vector<size_t> vector_counts = {32, 64, 128};

    for (size_t d : dimensions) {
        for (size_t n : vector_counts) {
            SCOPED_TRACE("d=" + std::to_string(d) + ", n=" + std::to_string(n));

            auto query_vec = skmeans::GenerateRandomVectors(1, d, -10.0f, 10.0f, 42);
            // Transposed layout: d rows of n elements
            auto transposed = skmeans::GenerateRandomVectors(d, n, -10.0f, 10.0f, 123);

            std::vector<float> scalar_dists(n, 0.0f);
            std::vector<float> simd_dists(n, 0.0f);

            PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::Vertical<false>(
                query_vec.data(), transposed.data(), n, n, 0, d, scalar_dists.data()
            );
            PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::Vertical(
                query_vec.data(), transposed.data(), n, n, 0, d, simd_dists.data(), nullptr
            );

            for (size_t i = 0; i < n; ++i) {
                float rel_error =
                    std::abs(scalar_dists[i] - simd_dists[i]) / std::max(scalar_dists[i], 1e-6f);
                EXPECT_LT(rel_error, 1e-5f)
                    << "vec " << i << ": scalar=" << scalar_dists[i] << ", simd=" << simd_dists[i];
            }
        }
    }
}

TEST_F(DistanceComputerTest, U8_Horizontal_SIMD_MatchesScalar) {
    std::vector<size_t> dimensions = {8, 16, 32, 64, 128, 256, 384};
    const size_t n_pairs = 100;
    std::mt19937 rng(42);

    for (size_t d : dimensions) {
        SCOPED_TRACE("d=" + std::to_string(d));

        for (size_t i = 0; i < n_pairs; ++i) {
            std::vector<uint8_t> data(d);
            std::vector<uint8_t> query(d);
            for (size_t j = 0; j < d; ++j) {
                data[j] = rng() % 256;
                query[j] = rng() % 256;
            }

            auto scalar = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::U8>::Horizontal(
                query.data(), data.data(), d
            );
            auto simd = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::U8>::Horizontal(
                query.data(), data.data(), d
            );

            float rel_error = std::abs(static_cast<float>(scalar) - static_cast<float>(simd)) /
                              std::max(static_cast<float>(scalar), 1.0f);
            EXPECT_LT(rel_error, 0.01f)
                << "pair " << i << ": scalar=" << scalar << ", simd=" << simd;
        }
    }
}

TEST_F(DistanceComputerTest, U8_Vertical_SIMD_MatchesScalar) {
    std::vector<size_t> dimensions = {64, 128, 256, 384};
    std::vector<size_t> vector_counts = {32, 64, 128};
    std::mt19937 rng(42);

    for (size_t d : dimensions) {
        for (size_t n : vector_counts) {
            SCOPED_TRACE("d=" + std::to_string(d) + ", n=" + std::to_string(n));

            std::vector<uint8_t> query(d);
            for (size_t j = 0; j < d; ++j) {
                query[j] = rng() % 256;
            }
            // Interleaved layout: d * n bytes, groups of 4 dims interleaved per vector
            std::vector<uint8_t> transposed(d * n);
            for (size_t j = 0; j < d * n; ++j) {
                transposed[j] = rng() % 256;
            }

            std::vector<uint32_t> scalar_dists(n, 0);
            std::vector<uint32_t> simd_dists(n, 0);

            PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::U8>::Vertical<false>(
                query.data(), transposed.data(), n, n, 0, d, scalar_dists.data(), nullptr
            );
            PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::U8>::Vertical(
                query.data(), transposed.data(), n, n, 0, d, simd_dists.data(), nullptr
            );

            for (size_t i = 0; i < n; ++i) {
                float rel_error =
                    std::abs(
                        static_cast<float>(scalar_dists[i]) - static_cast<float>(simd_dists[i])
                    ) /
                    std::max(static_cast<float>(scalar_dists[i]), 1.0f);
                EXPECT_LT(rel_error, 0.01f)
                    << "vec " << i << ": scalar=" << scalar_dists[i] << ", simd=" << simd_dists[i];
            }
        }
    }
}

TEST_F(DistanceComputerTest, FlipSign_SIMD_MatchesScalar) {
    std::vector<size_t> dimensions = {1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 256, 384};

    for (size_t d : dimensions) {
        SCOPED_TRACE("d=" + std::to_string(d));

        std::vector<float> data(d);
        std::vector<uint32_t> masks(d);
        skmeans::GenerateRandomDataWithMasks(data.data(), masks.data(), d, 0.5f, 42);

        std::vector<float> scalar_out(d);
        std::vector<float> simd_out(d);

        PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::FlipSign(
            data.data(), scalar_out.data(), masks.data(), d
        );
        PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::FlipSign(
            data.data(), simd_out.data(), masks.data(), d
        );

        for (size_t i = 0; i < d; ++i) {
            float rel_error =
                std::abs(scalar_out[i] - simd_out[i]) / std::max(std::abs(scalar_out[i]), 1e-6f);
            EXPECT_LT(rel_error, 0.01f)
                << "idx " << i << ": scalar=" << scalar_out[i] << ", simd=" << simd_out[i];
        }
    }
}

TEST_F(DistanceComputerTest, F32_SelfDistanceIsZero) {
    std::vector<size_t> dimensions = {64, 128, 256, 384};

    for (size_t d : dimensions) {
        SCOPED_TRACE("d=" + std::to_string(d));
        auto vectors = skmeans::GenerateRandomVectors(10, d, -10.0f, 10.0f, 42);

        for (size_t i = 0; i < 10; ++i) {
            const float* v = vectors.data() + i * d;
            float dist =
                PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::Horizontal(v, v, d);
            EXPECT_LT(dist, 1e-6f) << "self-distance should be ~0 at d=" << d;
        }
    }
}

} // namespace

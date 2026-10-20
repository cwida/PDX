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

TEST_F(DistanceComputerTest, F32_HorizontalDotWithU8_SIMD_MatchesScalar) {
    std::vector<size_t> dimensions = {1, 7, 8, 15, 16, 17, 31, 32, 33, 64, 384, 1024};
    const size_t n_pairs = 100;
    std::mt19937 rng(42);

    for (size_t d : dimensions) {
        SCOPED_TRACE("d=" + std::to_string(d));
        auto queries = skmeans::GenerateRandomVectors(n_pairs, d, 0.0f, 10.0f, 42);

        for (size_t i = 0; i < n_pairs; ++i) {
            const float* query = queries.data() + i * d;
            std::vector<uint8_t> codes(d);
            for (size_t j = 0; j < d; ++j) {
                codes[j] = rng() % 256;
            }

            float scalar =
                PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::HorizontalDotWithU8(
                    query, codes.data(), d
                );
            float simd =
                PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::HorizontalDotWithU8(
                    query, codes.data(), d
                );

            float rel_error = std::abs(scalar - simd) / std::max(scalar, 1e-6f);
            EXPECT_LT(rel_error, 1e-5f)
                << "pair " << i << ": scalar=" << scalar << ", simd=" << simd;
        }
    }
}

TEST_F(DistanceComputerTest, F32_HorizontalDotWithU4_SIMD_MatchesScalar) {
    std::vector<size_t> dimensions = {1, 2, 7, 15, 16, 17, 31, 32, 33, 47, 64, 384, 1024};
    const size_t n_pairs = 100;
    std::mt19937 rng(42);

    for (size_t d : dimensions) {
        SCOPED_TRACE("d=" + std::to_string(d));
        auto queries = skmeans::GenerateRandomVectors(n_pairs, d, 0.0f, 10.0f, 42);

        for (size_t i = 0; i < n_pairs; ++i) {
            const float* query = queries.data() + i * d;
            std::vector<uint8_t> codes((d + 1) / 2);
            for (auto& code_pair : codes) {
                code_pair = rng() % 256;
            }

            float scalar =
                PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::HorizontalDotWithU4(
                    query, codes.data(), d
                );
            float simd =
                PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::HorizontalDotWithU4(
                    query, codes.data(), d
                );

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

using ScalarF32Computer = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::F32>;
using F32Computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::F32>;

const std::vector<size_t> LVQ_KERNEL_DIMENSIONS =
    {1, 2, 7, 15, 16, 17, 31, 33, 60, 64, 65, 127, 384, 500, 1024};

// MinMax, the LVQ encoders and the LVQ4 decoder match scalar, and the encoders write no byte past
// the codes.
TEST_F(DistanceComputerTest, F32_LVQEncodeDecode_SIMD_MatchesScalar) {
    constexpr uint8_t CANARY = 0xA5;
    for (size_t d : LVQ_KERNEL_DIMENSIONS) {
        SCOPED_TRACE("d=" + std::to_string(d));
        const auto values = skmeans::GenerateRandomVectors(1, d, -3.0f, 5.0f, 42);
        float scalar_lower, scalar_upper, simd_lower, simd_upper;
        ScalarF32Computer::MinMax(values.data(), d, scalar_lower, scalar_upper);
        F32Computer::MinMax(values.data(), d, simd_lower, simd_upper);
        EXPECT_EQ(simd_lower, scalar_lower);
        EXPECT_EQ(simd_upper, scalar_upper);
        const float range = std::max(scalar_upper - scalar_lower, 1e-6f);

        std::vector<uint8_t> scalar_codes(d + 1, CANARY);
        std::vector<uint8_t> simd_codes(d + 1, CANARY);
        const float scalar_norm_u8 = ScalarF32Computer::QuantizeToU8(
            values.data(), d, scalar_lower, range / 255.0f, scalar_codes.data()
        );
        const float simd_norm_u8 = F32Computer::QuantizeToU8(
            values.data(), d, scalar_lower, range / 255.0f, simd_codes.data()
        );
        EXPECT_EQ(simd_codes, scalar_codes);
        EXPECT_NEAR(simd_norm_u8, scalar_norm_u8, 1e-5f * scalar_norm_u8);

        const float step_u4 = range / 15.0f;
        std::vector<uint8_t> scalar_code_pairs((d + 1) / 2 + 1, CANARY);
        std::vector<uint8_t> simd_code_pairs((d + 1) / 2 + 1, CANARY);
        const float scalar_norm_u4 = ScalarF32Computer::QuantizeToU4(
            values.data(), d, scalar_lower, step_u4, scalar_code_pairs.data()
        );
        const float simd_norm_u4 = F32Computer::QuantizeToU4(
            values.data(), d, scalar_lower, step_u4, simd_code_pairs.data()
        );
        EXPECT_EQ(simd_code_pairs, scalar_code_pairs);
        EXPECT_NEAR(simd_norm_u4, scalar_norm_u4, 1e-5f * scalar_norm_u4);

        std::vector<float> scalar_decoded(d);
        std::vector<float> simd_decoded(d);
        ScalarF32Computer::DequantizeFromU4(
            scalar_code_pairs.data(), d, scalar_lower, step_u4, scalar_decoded.data()
        );
        F32Computer::DequantizeFromU4(
            scalar_code_pairs.data(), d, scalar_lower, step_u4, simd_decoded.data()
        );
        // The SIMD decode fuses lower + step * code; a compiler may not (MSVC), and near zero the
        // two roundings differ by more ulps of the result than of the operands
        for (size_t i = 0; i < d; ++i) {
            EXPECT_NEAR(
                simd_decoded[i],
                scalar_decoded[i],
                1e-6f * (std::abs(scalar_lower) + 15.0f * step_u4)
            ) << "dimension "
              << i;
        }
    }
}

// QuantizeToInt8 matches scalar, and the int8 dots equal the dot over the d dimensions whatever
// the bytes past the codes hold.
TEST_F(DistanceComputerTest, F32_Int8Query_SIMD_MatchesScalar) {
    std::mt19937 rng(42);
    for (size_t d : LVQ_KERNEL_DIMENSIONS) {
        SCOPED_TRACE("d=" + std::to_string(d));
        const size_t padded_size =
            PDX::AlignValue<size_t, PDX::LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES>(d);
        const size_t padded_pairs_size =
            PDX::AlignValue<size_t, PDX::LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES>((d + 1) / 2);
        const auto values = skmeans::GenerateRandomVectors(1, d, -1.0f, 1.0f, 7);
        std::vector<int8_t> scalar_query(padded_size, 0);
        std::vector<int8_t> query(padded_size, 0);
        ScalarF32Computer::QuantizeToInt8(values.data(), d, 127.0f, scalar_query.data());
        F32Computer::QuantizeToInt8(values.data(), d, 127.0f, query.data());
        EXPECT_EQ(query, scalar_query);

        std::vector<uint8_t> codes(padded_size);
        for (auto& code : codes) {
            code = static_cast<uint8_t>(rng() % 256);
        }
        int32_t expected_dot_u8 = 0;
        int32_t query_codes_sum = 0;
        for (size_t j = 0; j < d; ++j) {
            expected_dot_u8 += static_cast<int32_t>(query[j]) * static_cast<int32_t>(codes[j]);
            query_codes_sum += query[j];
        }
        EXPECT_EQ(
            ScalarF32Computer::HorizontalDotInt8WithU8(
                query.data(), codes.data(), padded_size, query_codes_sum
            ),
            expected_dot_u8
        );
        EXPECT_EQ(
            F32Computer::HorizontalDotInt8WithU8(
                query.data(), codes.data(), padded_size, query_codes_sum
            ),
            expected_dot_u8
        );

        std::vector<int8_t> even_query(padded_pairs_size, 0);
        std::vector<int8_t> odd_query(padded_pairs_size, 0);
        for (size_t j = 0; j < d; ++j) {
            (j % 2 == 0 ? even_query : odd_query)[j / 2] = query[j];
        }
        std::vector<uint8_t> code_pairs(padded_pairs_size);
        for (auto& code_pair : code_pairs) {
            code_pair = static_cast<uint8_t>(rng() % 256);
        }
        int32_t expected_dot_u4 = 0;
        for (size_t j = 0; j < d; ++j) {
            const int32_t code =
                (j % 2 == 0) ? (code_pairs[j / 2] & 0x0F) : (code_pairs[j / 2] >> 4);
            expected_dot_u4 += static_cast<int32_t>(query[j]) * code;
        }
        EXPECT_EQ(
            ScalarF32Computer::HorizontalDotInt8WithU4(
                even_query.data(), odd_query.data(), code_pairs.data(), padded_pairs_size
            ),
            expected_dot_u4
        );
        EXPECT_EQ(
            F32Computer::HorizontalDotInt8WithU4(
                even_query.data(), odd_query.data(), code_pairs.data(), padded_pairs_size
            ),
            expected_dot_u4
        );
    }
}

} // namespace

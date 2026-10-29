#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/distance_computers/scalar_computers.hpp"
#include "pdx/quantizers/cluster_sketch.hpp"
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

const std::vector<size_t> SKETCH_KERNEL_DIMENSIONS = {2, 16, 17, 64, 384, 1000, 1024, 1040, 2064};

// The sketch scan answers like a bound computed from the rows, in scalar and SIMD, and cleared
// lanes never hold a neighbour.
TEST_F(DistanceComputerTest, F32_SketchBatchMayHoldNeighbor_SIMD_MatchesScalar) {
    constexpr size_t NUM_ROWS = 40;
    for (size_t d : SKETCH_KERNEL_DIMENSIONS) {
        SCOPED_TRACE("d=" + std::to_string(d));
        const PDX::ClusterSketchQuantizer quantizer(static_cast<uint32_t>(d));
        const auto rows = skmeans::GenerateRandomVectors(NUM_ROWS, d, -1.0f, 1.0f, 42);
        const auto mean = skmeans::GenerateRandomVectors(1, d, -0.1f, 0.1f, 7);
        const auto query = skmeans::GenerateRandomVectors(1, d, -1.0f, 1.0f, 13);
        const size_t num_batches = PDX::ClusterSketchQuantizer::GetNumBatches(NUM_ROWS);
        std::vector<uint8_t> sketch_rows(NUM_ROWS * quantizer.row_size_in_bytes);
        std::vector<uint8_t> batches(num_batches * quantizer.batch_size_in_bytes);
        quantizer.ClearBatches(batches.data(), num_batches);
        for (size_t i = 0; i < NUM_ROWS; ++i) {
            uint8_t* sketch_row = sketch_rows.data() + i * quantizer.row_size_in_bytes;
            quantizer.EncodeRow(rows.data() + i * d, mean.data(), sketch_row);
            quantizer.StoreRowInBatch(
                sketch_row,
                batches.data() + quantizer.GetBatchOffsetInBytes(i),
                static_cast<uint32_t>(i % PDX::CLUSTER_SKETCH_BATCH_SIZE)
            );
        }
        const PDX::ClusterSketchQuery sketch_query(query.data(), mean.data(), quantizer);
        auto may_hold_neighbor = [&](auto kernel, size_t batch, float threshold, uint32_t lane_mask
                                 ) {
            return kernel(
                batches.data() + batch * quantizer.batch_size_in_bytes,
                quantizer.code_size_in_bytes,
                sketch_query.table.lookup_table.get(),
                sketch_query.table.lookup_table_step,
                sketch_query.table.lookup_table_base_sum,
                sketch_query.table.centering_correction,
                sketch_query.squared_distance_to_mean,
                sketch_query.error_scale,
                threshold,
                lane_mask
            );
        };
        for (size_t batch = 0; batch < num_batches; ++batch) {
            std::vector<double> lane_bounds(
                PDX::CLUSTER_SKETCH_BATCH_SIZE, std::numeric_limits<double>::infinity()
            );
            const size_t end = std::min(NUM_ROWS, (batch + 1) * PDX::CLUSTER_SKETCH_BATCH_SIZE);
            for (size_t i = batch * PDX::CLUSTER_SKETCH_BATCH_SIZE; i < end; ++i) {
                const uint8_t* sketch_row = sketch_rows.data() + i * quantizer.row_size_in_bytes;
                double centered_dot = 0.0;
                double squared_distance_to_mean = 0.0;
                for (size_t j = 0; j < d; ++j) {
                    const int bit = (sketch_row[j / 8] >> (7 - j % 8)) & 1;
                    centered_dot += (bit - 0.5) * query[j];
                    squared_distance_to_mean += (query[j] - mean[j]) * (query[j] - mean[j]);
                }
                float factors[3];
                std::memcpy(factors, sketch_row + quantizer.code_size_in_bytes, sizeof(factors));
                const double estimate =
                    factors[0] + squared_distance_to_mean + factors[1] * centered_dot;
                lane_bounds[i % PDX::CLUSTER_SKETCH_BATCH_SIZE] =
                    estimate -
                    factors[2] * PDX::CLUSTER_SKETCH_EPSILON * std::sqrt(squared_distance_to_mean);
            }
            for (const uint32_t lane_mask :
                 {PDX::CLUSTER_SKETCH_ALL_LANES, 0x55555555u, 0xAAAAAAAAu, 1u << 3}) {
                double lowest_bound = std::numeric_limits<double>::infinity();
                for (size_t lane = 0; lane < PDX::CLUSTER_SKETCH_BATCH_SIZE; ++lane) {
                    if ((lane_mask >> lane) & 1u) {
                        lowest_bound = std::min(lowest_bound, lane_bounds[lane]);
                    }
                }
                const double margin = 1e-2 * (std::abs(lowest_bound) + 1.0);
                for (const bool above_lowest_bound : {false, true}) {
                    const auto threshold = static_cast<float>(
                        above_lowest_bound ? lowest_bound + margin : lowest_bound - margin
                    );
                    const auto expect_answer = [&](auto kernel, const char* kernel_name) {
                        EXPECT_EQ(
                            may_hold_neighbor(kernel, batch, threshold, lane_mask),
                            above_lowest_bound
                        ) << kernel_name
                          << ", batch " << batch << ", lane mask " << lane_mask;
                    };
                    expect_answer(
                        ScalarF32Computer::SketchBatchMayHoldNeighbor<true>, "scalar masked"
                    );
                    expect_answer(F32Computer::SketchBatchMayHoldNeighborMasked, "masked");
                    if (lane_mask == PDX::CLUSTER_SKETCH_ALL_LANES) {
                        expect_answer(
                            ScalarF32Computer::SketchBatchMayHoldNeighbor<false>, "scalar"
                        );
                        expect_answer(F32Computer::SketchBatchMayHoldNeighbor, "unmasked");
                    }
                }
            }
        }
        quantizer.ClearBatches(batches.data(), 1);
        EXPECT_FALSE(may_hold_neighbor(
            F32Computer::SketchBatchMayHoldNeighbor,
            0,
            std::numeric_limits<float>::max(),
            PDX::CLUSTER_SKETCH_ALL_LANES
        ));
    }
}

// The SIMD row pass gives the scalar code and sums
TEST_F(DistanceComputerTest, F32_ComputeSketchCodeAndSums_SIMD_MatchesScalar) {
    constexpr size_t NUM_ROWS = 20;
    for (size_t d : SKETCH_KERNEL_DIMENSIONS) {
        SCOPED_TRACE("d=" + std::to_string(d));
        const auto rows = skmeans::GenerateRandomVectors(NUM_ROWS, d, -1.0f, 1.0f, 42);
        const auto mean = skmeans::GenerateRandomVectors(1, d, -0.1f, 0.1f, 7);
        for (size_t i = 0; i < NUM_ROWS; ++i) {
            std::vector<uint8_t> scalar_code((d + 7) / 8, 0);
            std::vector<uint8_t> simd_code((d + 7) / 8, 0);
            float scalar_sums[3] = {};
            float simd_sums[3] = {};
            ScalarF32Computer::ComputeSketchCodeAndSums(
                rows.data() + i * d,
                mean.data(),
                d,
                scalar_code.data(),
                scalar_sums[0],
                scalar_sums[1],
                scalar_sums[2]
            );
            F32Computer::ComputeSketchCodeAndSums(
                rows.data() + i * d,
                mean.data(),
                d,
                simd_code.data(),
                simd_sums[0],
                simd_sums[1],
                simd_sums[2]
            );
            EXPECT_EQ(simd_code, scalar_code) << "row " << i;
            for (size_t sum = 0; sum < 3; ++sum) {
                EXPECT_NEAR(
                    simd_sums[sum], scalar_sums[sum], 1e-4f * (std::abs(scalar_sums[sum]) + 1.0f)
                ) << "row "
                  << i << ", sum " << sum;
            }
        }
    }
}

// The SIMD pack writes the scalar columns, and both unpacks write back those columns of the rows
// and nothing else
TEST_F(DistanceComputerTest, F32_PackSketchColumns_SIMD_MatchesScalar) {
    constexpr size_t ROW_STRIDE = 40;
    constexpr size_t BATCH_SIZE = PDX::CLUSTER_SKETCH_BATCH_SIZE;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    std::vector<uint8_t> rows(BATCH_SIZE * ROW_STRIDE);
    for (auto& byte : rows) {
        byte = static_cast<uint8_t>(byte_distribution(rng));
    }
    for (size_t first_column : {size_t{0}, PDX::CLUSTER_SKETCH_TRANSPOSE_COLUMNS}) {
        for (size_t num_columns = 1; num_columns <= PDX::CLUSTER_SKETCH_TRANSPOSE_COLUMNS;
             ++num_columns) {
            SCOPED_TRACE(
                "first_column=" + std::to_string(first_column) +
                " num_columns=" + std::to_string(num_columns)
            );
            const size_t batch_size = (first_column + num_columns) * BATCH_SIZE;
            std::vector<uint8_t> scalar_batch(batch_size, 0);
            std::vector<uint8_t> simd_batch(batch_size, 0);
            ScalarF32Computer::PackSketchColumns(
                rows.data(), ROW_STRIDE, first_column, num_columns, scalar_batch.data()
            );
            F32Computer::PackSketchColumns(
                rows.data(), ROW_STRIDE, first_column, num_columns, simd_batch.data()
            );
            EXPECT_EQ(simd_batch, scalar_batch);
            for (size_t num_rows : {size_t{1}, size_t{17}, BATCH_SIZE}) {
                std::vector<uint8_t> scalar_rows(rows.size(), 0xAA);
                std::vector<uint8_t> simd_rows(rows.size(), 0xAA);
                ScalarF32Computer::UnpackSketchColumns(
                    scalar_batch.data(),
                    first_column,
                    num_columns,
                    num_rows,
                    scalar_rows.data(),
                    ROW_STRIDE
                );
                F32Computer::UnpackSketchColumns(
                    scalar_batch.data(),
                    first_column,
                    num_columns,
                    num_rows,
                    simd_rows.data(),
                    ROW_STRIDE
                );
                EXPECT_EQ(simd_rows, scalar_rows) << "num_rows " << num_rows;
                for (size_t row = 0; row < num_rows; ++row) {
                    for (size_t column = first_column; column < first_column + num_columns;
                         ++column) {
                        ASSERT_EQ(
                            scalar_rows[row * ROW_STRIDE + column], rows[row * ROW_STRIDE + column]
                        ) << "row "
                          << row << ", column " << column;
                    }
                }
            }
        }
    }
}

// The SIMD lane store and copy write the scalar bytes (only the target lane's nibbles change), and
// both lane reads give the code back, for every lane
TEST_F(DistanceComputerTest, F32_StoreSketchLane_SIMD_MatchesScalar) {
    constexpr size_t BATCH_SIZE = PDX::CLUSTER_SKETCH_BATCH_SIZE;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    for (size_t num_columns :
         {size_t{1}, size_t{15}, size_t{16}, size_t{17}, size_t{130}, size_t{258}}) {
        std::vector<uint8_t> code(num_columns);
        std::vector<uint8_t> batch(num_columns * BATCH_SIZE);
        for (auto& byte : code) {
            byte = static_cast<uint8_t>(byte_distribution(rng));
        }
        for (auto& byte : batch) {
            byte = static_cast<uint8_t>(byte_distribution(rng));
        }
        for (uint32_t lane = 0; lane < BATCH_SIZE; ++lane) {
            SCOPED_TRACE(
                "num_columns=" + std::to_string(num_columns) + " lane=" + std::to_string(lane)
            );
            std::vector<uint8_t> scalar_batch = batch;
            std::vector<uint8_t> simd_batch = batch;
            ScalarF32Computer::StoreSketchLane(code.data(), num_columns, scalar_batch.data(), lane);
            F32Computer::StoreSketchLane(code.data(), num_columns, simd_batch.data(), lane);
            EXPECT_EQ(simd_batch, scalar_batch);
            std::vector<uint8_t> scalar_code(num_columns);
            std::vector<uint8_t> simd_code(num_columns);
            ScalarF32Computer::ReadSketchLane(
                simd_batch.data(), num_columns, lane, scalar_code.data()
            );
            F32Computer::ReadSketchLane(simd_batch.data(), num_columns, lane, simd_code.data());
            EXPECT_EQ(scalar_code, code);
            EXPECT_EQ(simd_code, code);
            // Lane + 16 shares the lane's bytes
            const uint32_t target_lanes[2] = {(lane + 5) % 32u, (lane + 16) % 32u};
            for (const uint32_t target_lane : target_lanes) {
                for (const bool same_batch : {true, false}) {
                    std::vector<uint8_t> scalar_target = same_batch ? simd_batch : batch;
                    std::vector<uint8_t> simd_target = scalar_target;
                    ScalarF32Computer::CopySketchLane(
                        same_batch ? scalar_target.data() : simd_batch.data(),
                        lane,
                        scalar_target.data(),
                        target_lane,
                        num_columns
                    );
                    F32Computer::CopySketchLane(
                        same_batch ? simd_target.data() : simd_batch.data(),
                        lane,
                        simd_target.data(),
                        target_lane,
                        num_columns
                    );
                    EXPECT_EQ(simd_target, scalar_target) << "target lane " << target_lane;
                    F32Computer::ReadSketchLane(
                        simd_target.data(), num_columns, target_lane, simd_code.data()
                    );
                    EXPECT_EQ(simd_code, code) << "target lane " << target_lane;
                }
            }
        }
    }
}

// Storing whole batches writes the bytes of storing lane by lane, and reading them gives the rows
// back, also for a batch short of rows
TEST_F(DistanceComputerTest, F32_StoreRowsInBatches_MatchesStoreRowInBatch) {
    for (size_t d : SKETCH_KERNEL_DIMENSIONS) {
        const PDX::ClusterSketchQuantizer quantizer(static_cast<uint32_t>(d));
        for (size_t num_rows : {size_t{1}, PDX::CLUSTER_SKETCH_BATCH_SIZE, size_t{40}}) {
            SCOPED_TRACE("d=" + std::to_string(d) + " num_rows=" + std::to_string(num_rows));
            const auto rows = skmeans::GenerateRandomVectors(num_rows, d, -1.0f, 1.0f, 42);
            const auto mean = skmeans::GenerateRandomVectors(1, d, -0.1f, 0.1f, 7);
            const size_t num_batches = PDX::ClusterSketchQuantizer::GetNumBatches(num_rows);
            std::vector<uint8_t> sketch_rows(num_rows * quantizer.row_size_in_bytes);
            std::vector<uint8_t> lane_by_lane(num_batches * quantizer.batch_size_in_bytes);
            quantizer.ClearBatches(lane_by_lane.data(), num_batches);
            for (size_t i = 0; i < num_rows; ++i) {
                uint8_t* sketch_row = sketch_rows.data() + i * quantizer.row_size_in_bytes;
                quantizer.EncodeRow(rows.data() + i * d, mean.data(), sketch_row);
                quantizer.StoreRowInBatch(
                    sketch_row,
                    lane_by_lane.data() + quantizer.GetBatchOffsetInBytes(i),
                    static_cast<uint32_t>(i % PDX::CLUSTER_SKETCH_BATCH_SIZE)
                );
            }
            std::vector<uint8_t> whole_batches(lane_by_lane.size());
            quantizer.ClearBatches(whole_batches.data(), num_batches);
            quantizer.StoreRowsInBatches(sketch_rows.data(), num_rows, whole_batches.data());
            EXPECT_EQ(whole_batches, lane_by_lane);
            std::vector<uint8_t> read_rows(sketch_rows.size());
            quantizer.ReadRowsFromBatches(whole_batches.data(), num_rows, read_rows.data());
            EXPECT_EQ(read_rows, sketch_rows);
        }
    }
}

std::vector<uint16_t> RandomHalves(size_t n, float low, float high, std::mt19937& rng) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<uint16_t> halves(n + 16);
    for (auto& half : halves) {
        PDX::WriteHalf(distribution(rng), reinterpret_cast<uint8_t*>(&half));
    }
    return halves;
}

std::vector<int8_t> RandomQueryCodes(std::mt19937& rng) {
    std::vector<int8_t> codes(PDX::SAQ_CHUNK_SIZE);
    for (auto& code : codes) {
        code = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    }
    return codes;
}

// A threshold between two of the estimates, so rounding differences cannot flip a vector
float ThresholdBetween(std::vector<float> estimates, size_t rank) {
    std::sort(estimates.begin(), estimates.end());
    return (estimates[rank] + estimates[rank + 1]) / 2;
}

// The SIMD scans write and read a whole batch (up to 16 lanes) past the last survivor
struct SAQSurvivors {
    explicit SAQSurvivors(size_t capacity)
        : positions(capacity + 16), estimates(capacity + 16), rescales(capacity + 16) {}
    std::vector<uint32_t> positions;
    std::vector<float> estimates;
    std::vector<float> rescales;
};

void ExpectSameSurvivors(
    const SAQSurvivors& expected,
    size_t num_expected,
    const SAQSurvivors& actual,
    size_t num_actual
) {
    ASSERT_EQ(num_expected, num_actual);
    for (size_t i = 0; i < num_expected; ++i) {
        EXPECT_EQ(expected.positions[i], actual.positions[i]) << "survivor " << i;
        EXPECT_NEAR(
            expected.estimates[i],
            actual.estimates[i],
            1e-5f * std::max(1.0f, std::abs(expected.estimates[i]))
        ) << "survivor "
          << i;
        EXPECT_EQ(expected.rescales[i], actual.rescales[i]) << "survivor " << i;
    }
}

// Scalar and SIMD match the orthonormal Walsh-Hadamard matrix: (-1)^popcount(i & j) / sqrt(n).
TEST_F(DistanceComputerTest, F32_FastHadamardTransform_SIMD_MatchesScalar) {
    for (size_t n = 16; n <= 2048; n *= 2) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const auto values = skmeans::GenerateRandomVectors(1, n, -10.0f, 10.0f, 42);
        std::vector<float> expected(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const bool negative = PDX_POPCOUNT(static_cast<uint32_t>(i & j)) % 2 == 1;
                expected[i] += negative ? -values[j] : values[j];
            }
            expected[i] /= std::sqrt(static_cast<float>(n));
        }
        std::vector<float> scalar_out(values.begin(), values.end());
        std::vector<float> simd_out(values.begin(), values.end());
        ScalarF32Computer::FastHadamardTransform(scalar_out.data(), n);
        F32Computer::FastHadamardTransform(simd_out.data(), n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(scalar_out[i], expected[i], 1e-3f) << "idx " << i;
            EXPECT_NEAR(simd_out[i], expected[i], 1e-3f) << "idx " << i;
        }
    }
}

TEST_F(DistanceComputerTest, SAQ_ScanHead_SIMD_MatchesScalar) {
    using scalar_computer = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    std::mt19937 rng(42);
    for (const size_t num_vectors : {1, 5, 16, 33, 100}) {
        for (const bool with_selection : {false, true}) {
            SCOPED_TRACE(
                "n=" + std::to_string(num_vectors) + ", selection=" + std::to_string(with_selection)
            );
            const size_t stride = num_vectors + 7;
            std::vector<uint8_t> head(stride * PDX::SAQ_CHUNK_SIZE);
            for (auto& level : head) {
                level = static_cast<uint8_t>(rng() % 256);
            }
            const auto query_codes = RandomQueryCodes(rng);
            const auto head_rescales = RandomHalves(num_vectors, 0.001f, 0.01f, rng);
            const auto residual_norms = RandomHalves(num_vectors, 0.5f, 2.0f, rng);
            std::vector<uint8_t> selection(num_vectors);
            for (auto& selected : selection) {
                selected = static_cast<uint8_t>(rng() % 2);
            }
            const uint8_t* selection_ptr = with_selection ? selection.data() : nullptr;
            const auto scan = [&](auto scan_head, float threshold, SAQSurvivors& survivors) {
                return scan_head(
                    head.data(),
                    num_vectors,
                    stride,
                    query_codes.data(),
                    -0.02f,
                    1.5f,
                    0.7f,
                    head_rescales.data(),
                    residual_norms.data(),
                    selection_ptr,
                    threshold,
                    survivors.positions.data(),
                    survivors.estimates.data(),
                    survivors.rescales.data()
                );
            };
            SAQSurvivors all(num_vectors);
            const size_t num_all =
                scan(scalar_computer::ScanHead, std::numeric_limits<float>::infinity(), all);
            std::vector<float> all_estimates(
                all.estimates.begin(), all.estimates.begin() + num_all
            );
            std::vector<float> thresholds = {std::numeric_limits<float>::infinity()};
            if (num_all >= 2) {
                thresholds.push_back(ThresholdBetween(all_estimates, num_all / 2 - 1));
            }
            for (const float threshold : thresholds) {
                SAQSurvivors expected(num_vectors);
                SAQSurvivors actual(num_vectors);
                const size_t num_expected = scan(scalar_computer::ScanHead, threshold, expected);
                const size_t num_actual = scan(computer::ScanHead, threshold, actual);
                ExpectSameSurvivors(expected, num_expected, actual, num_actual);
            }
        }
    }
}

// The head over the passing positions keeps what ScanHead keeps with the same selection
TEST_F(DistanceComputerTest, SAQ_ScanHeadPruning_MatchesScanHead) {
    using scalar_computer = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    using PruningKernel = decltype(&scalar_computer::ScanHeadPruning);
    const PruningKernel kernels[] = {scalar_computer::ScanHeadPruning, computer::ScanHeadPruning};
    std::mt19937 rng(42);
    for (const size_t num_vectors : {1, 5, 16, 33, 100}) {
        for (const uint32_t passing_percent : {5u, 50u, 100u}) {
            SCOPED_TRACE(
                "n=" + std::to_string(num_vectors) + ", passing=" + std::to_string(passing_percent)
            );
            const size_t stride = num_vectors + 7;
            std::vector<uint8_t> head(stride * PDX::SAQ_CHUNK_SIZE);
            for (auto& level : head) {
                level = static_cast<uint8_t>(rng() % 256);
            }
            const auto query_codes = RandomQueryCodes(rng);
            const auto head_rescales = RandomHalves(num_vectors, 0.001f, 0.01f, rng);
            const auto residual_norms = RandomHalves(num_vectors, 0.5f, 2.0f, rng);
            std::vector<uint8_t> selection(num_vectors);
            std::vector<uint32_t> passing;
            for (size_t vector_idx = 0; vector_idx < num_vectors; ++vector_idx) {
                selection[vector_idx] = static_cast<uint8_t>(rng() % 100 < passing_percent);
                if (selection[vector_idx] != 0) {
                    passing.push_back(static_cast<uint32_t>(vector_idx));
                }
            }
            const auto scan_head = [&](float threshold, SAQSurvivors& survivors) {
                return scalar_computer::ScanHead(
                    head.data(),
                    num_vectors,
                    stride,
                    query_codes.data(),
                    -0.02f,
                    1.5f,
                    0.7f,
                    head_rescales.data(),
                    residual_norms.data(),
                    selection.data(),
                    threshold,
                    survivors.positions.data(),
                    survivors.estimates.data(),
                    survivors.rescales.data()
                );
            };
            const auto scan_passing =
                [&](PruningKernel kernel, float threshold, SAQSurvivors& survivors) {
                    std::copy(passing.begin(), passing.end(), survivors.positions.begin());
                    return kernel(
                        head.data(),
                        stride,
                        query_codes.data(),
                        -0.02f,
                        1.5f,
                        0.7f,
                        head_rescales.data(),
                        residual_norms.data(),
                        threshold,
                        survivors.positions.data(),
                        survivors.estimates.data(),
                        survivors.rescales.data(),
                        passing.size()
                    );
                };
            SAQSurvivors all(num_vectors);
            const size_t num_all = scan_head(std::numeric_limits<float>::infinity(), all);
            std::vector<float> all_estimates(
                all.estimates.begin(), all.estimates.begin() + num_all
            );
            std::vector<float> thresholds = {std::numeric_limits<float>::infinity()};
            if (num_all >= 2) {
                thresholds.push_back(ThresholdBetween(all_estimates, num_all / 2 - 1));
            }
            for (const float threshold : thresholds) {
                SAQSurvivors expected(num_vectors);
                const size_t num_expected = scan_head(threshold, expected);
                for (const PruningKernel kernel : kernels) {
                    SAQSurvivors actual(num_vectors);
                    const size_t num_actual = scan_passing(kernel, threshold, actual);
                    ExpectSameSurvivors(expected, num_expected, actual, num_actual);
                }
            }
        }
    }
}

std::vector<uint8_t> PackRecords(
    const std::vector<uint8_t>& levels,
    size_t num_records,
    uint32_t width
) {
    std::vector<uint8_t> records(num_records * 8 * width);
    for (size_t record_idx = 0; record_idx < num_records; ++record_idx) {
        PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>::PackRecord(
            levels.data() + record_idx * PDX::SAQ_CHUNK_SIZE,
            width,
            records.data() + record_idx * 8 * width
        );
    }
    return records;
}

// Unpacking a packed record gives its levels back, at every width.
TEST_F(DistanceComputerTest, SAQ_PackRecord_RoundTrips) {
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    std::mt19937 rng(3);
    for (uint32_t width = 1; width <= PDX::SAQ_MAX_CODE_WIDTH; ++width) {
        SCOPED_TRACE("width=" + std::to_string(width));
        std::vector<uint8_t> levels(PDX::SAQ_CHUNK_SIZE);
        for (auto& level : levels) {
            level = static_cast<uint8_t>(rng() % (1u << width));
        }
        std::vector<uint8_t> record(8 * width);
        computer::PackRecord(levels.data(), width, record.data());
        std::vector<uint8_t> unpacked(PDX::SAQ_CHUNK_SIZE);
        computer::UnpackRecord(record.data(), width, unpacked.data());
        EXPECT_EQ(levels, unpacked);
    }
}

// The backend's PackRecord writes the scalar one's bytes at every width.
TEST_F(DistanceComputerTest, SAQ_PackRecord_SIMD_MatchesScalar) {
    using scalar_computer = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    std::mt19937 rng(42);
    for (uint32_t width = 1; width <= PDX::SAQ_MAX_CODE_WIDTH; ++width) {
        SCOPED_TRACE("width=" + std::to_string(width));
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<uint8_t> levels(PDX::SAQ_CHUNK_SIZE);
            for (auto& level : levels) {
                level = static_cast<uint8_t>(rng() % (1u << width));
            }
            std::vector<uint8_t> expected(8 * width);
            std::vector<uint8_t> actual(8 * width);
            scalar_computer::PackRecord(levels.data(), width, expected.data());
            computer::PackRecord(levels.data(), width, actual.data());
            EXPECT_EQ(expected, actual);
        }
    }
}

// The backend's DecodeRecord returns the scalar one's floats, for whole and partial chunks.
TEST_F(DistanceComputerTest, SAQ_DecodeRecord_SIMD_MatchesScalar) {
    using scalar_computer = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (uint32_t width = 1; width <= PDX::SAQ_MAX_CODE_WIDTH; ++width) {
        for (const size_t num_values : {size_t{PDX::SAQ_CHUNK_SIZE}, size_t{40}}) {
            SCOPED_TRACE("width=" + std::to_string(width) + ", n=" + std::to_string(num_values));
            std::vector<uint8_t> record(8 * width);
            for (auto& byte : record) {
                byte = static_cast<uint8_t>(rng() % 256);
            }
            std::vector<float> mean(PDX::SAQ_CHUNK_SIZE);
            for (auto& value : mean) {
                value = uniform(rng);
            }
            const float center = 0.5f * static_cast<float>((1u << width) - 1);
            std::vector<float> expected(num_values);
            std::vector<float> actual(num_values);
            scalar_computer::DecodeRecord(
                record.data(), width, 0.0123f, center, mean.data(), num_values, expected.data()
            );
            computer::DecodeRecord(
                record.data(), width, 0.0123f, center, mean.data(), num_values, actual.data()
            );
            for (size_t j = 0; j < num_values; ++j) {
                EXPECT_NEAR(expected[j], actual[j], 1e-6f * std::max(1.0f, std::abs(expected[j])))
                    << "value " << j;
            }
        }
    }
}

// The backend's EncodeSegment keeps its codes in range with the rescale of those codes, as close
// to the values as the scalar one's, and mostly the same codes.
TEST_F(DistanceComputerTest, SAQ_EncodeSegment_SIMD_MatchesScalar) {
    using scalar_computer = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    std::mt19937 rng(42);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    size_t num_identical = 0;
    size_t num_segments = 0;
    for (uint32_t width = 1; width <= PDX::SAQ_MAX_CODE_WIDTH; ++width) {
        const float max_level = static_cast<float>((1u << (width - 1)) - 1);
        const float center = 0.5f * static_cast<float>((1u << width) - 1);
        for (const size_t num_values : {size_t{64}, size_t{128}, size_t{448}, size_t{40}}) {
            SCOPED_TRACE("width=" + std::to_string(width) + ", n=" + std::to_string(num_values));
            // A scale whose levels reach max_level near the largest normalized values
            const float scale =
                std::max(1.0f, max_level) * std::sqrt(static_cast<float>(num_values)) / 2.5f;
            for (int trial = 0; trial < 10; ++trial) {
                std::vector<float> values(num_values);
                for (auto& value : values) {
                    value = gaussian(rng);
                }
                std::vector<uint8_t> expected(num_values);
                std::vector<uint8_t> actual(num_values);
                scalar_computer::EncodeSegment(
                    values.data(), num_values, width, scale, expected.data()
                );
                const float rescale =
                    computer::EncodeSegment(values.data(), num_values, width, scale, actual.data());
                const auto cosine_and_rescale = [&](const std::vector<uint8_t>& codes) {
                    double inner_product = 0.0;
                    double squared_norm = 0.0;
                    double code_squared_norm = 0.0;
                    for (size_t j = 0; j < num_values; ++j) {
                        const double code = static_cast<double>(codes[j]) - center;
                        inner_product += values[j] * code;
                        squared_norm += static_cast<double>(values[j]) * values[j];
                        code_squared_norm += code * code;
                    }
                    return std::make_pair(
                        inner_product / std::sqrt(squared_norm * code_squared_norm),
                        squared_norm / inner_product
                    );
                };
                for (const uint8_t code : actual) {
                    ASSERT_LT(code, 1u << width);
                }
                const auto [expected_cosine, expected_rescale] = cosine_and_rescale(expected);
                const auto [actual_cosine, actual_rescale] = cosine_and_rescale(actual);
                EXPECT_NEAR(rescale, actual_rescale, 1e-4 * actual_rescale);
                EXPECT_GE(actual_cosine, expected_cosine - 1e-4);
                num_identical += expected == actual;
                ++num_segments;
            }
        }
    }
    EXPECT_GE(num_identical * 10, num_segments * 9);
}

TEST_F(DistanceComputerTest, SAQ_ScanTail_SIMD_MatchesScalar) {
    using scalar_computer = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    const computer::ScanTailFunction scalar_scan_tails[PDX::SAQ_MAX_CODE_WIDTH + 1] =
        {nullptr,
         scalar_computer::ScanTail<1>,
         scalar_computer::ScanTail<2>,
         scalar_computer::ScanTail<3>,
         scalar_computer::ScanTail<4>,
         scalar_computer::ScanTail<5>,
         scalar_computer::ScanTail<6>,
         scalar_computer::ScanTail<7>,
         scalar_computer::ScanTail<8>};
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> estimate_distribution(0.5f, 2.0f);
    std::uniform_real_distribution<float> rescale_distribution(0.001f, 0.01f);
    const size_t num_records = 100;
    for (uint32_t width = 1; width <= PDX::SAQ_MAX_CODE_WIDTH; ++width) {
        std::vector<uint8_t> levels(num_records * 64);
        for (auto& level : levels) {
            level = static_cast<uint8_t>(rng() % (1u << width));
        }
        const auto records = PackRecords(levels, num_records, width);
        const auto query_codes = RandomQueryCodes(rng);
        const auto segment_rescales = RandomHalves(num_records, 0.001f, 0.01f, rng);
        for (const size_t num_survivors : {1, 3, 4, 7, 40}) {
            for (const bool new_segment : {false, true}) {
                SCOPED_TRACE(
                    "width=" + std::to_string(width) + ", survivors=" +
                    std::to_string(num_survivors) + ", new_segment=" + std::to_string(new_segment)
                );
                SAQSurvivors input(num_survivors);
                std::vector<uint32_t> order(num_records);
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                for (size_t i = 0; i < num_survivors; ++i) {
                    input.positions[i] = order[i];
                    input.estimates[i] = estimate_distribution(rng);
                    input.rescales[i] = rescale_distribution(rng);
                }
                const uint16_t* segment_rescales_ptr =
                    new_segment ? segment_rescales.data() : nullptr;
                const auto scan = [&](computer::ScanTailFunction scan_tail,
                                      float threshold,
                                      SAQSurvivors& survivors) {
                    survivors = input;
                    return scan_tail(
                        records.data(),
                        query_codes.data(),
                        -0.02f,
                        0.3f,
                        segment_rescales_ptr,
                        threshold,
                        survivors.positions.data(),
                        survivors.estimates.data(),
                        survivors.rescales.data(),
                        num_survivors
                    );
                };
                SAQSurvivors all(num_survivors);
                const size_t num_all =
                    scan(scalar_scan_tails[width], std::numeric_limits<float>::infinity(), all);
                std::vector<float> all_estimates(
                    all.estimates.begin(), all.estimates.begin() + num_all
                );
                std::vector<float> thresholds = {std::numeric_limits<float>::infinity()};
                if (num_all >= 2) {
                    thresholds.push_back(ThresholdBetween(all_estimates, num_all / 2 - 1));
                }
                for (const float threshold : thresholds) {
                    SAQSurvivors expected(num_survivors);
                    SAQSurvivors actual(num_survivors);
                    const size_t num_expected = scan(scalar_scan_tails[width], threshold, expected);
                    const size_t num_actual = scan(computer::GetScanTail(width), threshold, actual);
                    ExpectSameSurvivors(expected, num_expected, actual, num_actual);
                }
            }
        }
    }
}

// The no-pruning scan's estimates match the scalar ones, and the pruning kernels' at no threshold.
TEST_F(DistanceComputerTest, SAQ_NoPruningScan_SIMD_MatchesScalar) {
    using scalar_computer = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    const computer::AccumulateTailEstimatesFunction
        scalar_accumulates[PDX::SAQ_MAX_CODE_WIDTH + 1] =
            {nullptr,
             scalar_computer::AccumulateTailEstimates<1>,
             scalar_computer::AccumulateTailEstimates<2>,
             scalar_computer::AccumulateTailEstimates<3>,
             scalar_computer::AccumulateTailEstimates<4>,
             scalar_computer::AccumulateTailEstimates<5>,
             scalar_computer::AccumulateTailEstimates<6>,
             scalar_computer::AccumulateTailEstimates<7>,
             scalar_computer::AccumulateTailEstimates<8>};
    const computer::ScanTailFunction scalar_scan_tails[PDX::SAQ_MAX_CODE_WIDTH + 1] =
        {nullptr,
         scalar_computer::ScanTail<1>,
         scalar_computer::ScanTail<2>,
         scalar_computer::ScanTail<3>,
         scalar_computer::ScanTail<4>,
         scalar_computer::ScanTail<5>,
         scalar_computer::ScanTail<6>,
         scalar_computer::ScanTail<7>,
         scalar_computer::ScanTail<8>};
    const auto expect_near =
        [](const std::vector<float>& expected, const std::vector<float>& actual, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                EXPECT_NEAR(expected[i], actual[i], 1e-5f * std::max(1.0f, std::abs(expected[i])))
                    << "vector " << i;
            }
        };
    std::mt19937 rng(11);
    for (const size_t num_vectors : {1, 5, 16, 33, 100}) {
        SCOPED_TRACE("n=" + std::to_string(num_vectors));
        const size_t stride = num_vectors + 7;
        std::vector<uint8_t> head(stride * PDX::SAQ_CHUNK_SIZE);
        for (auto& level : head) {
            level = static_cast<uint8_t>(rng() % 256);
        }
        const auto head_codes = RandomQueryCodes(rng);
        const auto head_rescales = RandomHalves(num_vectors, 0.001f, 0.01f, rng);
        const auto residual_norms = RandomHalves(num_vectors, 0.5f, 2.0f, rng);
        const auto head_estimates = [&](auto compute_head_estimates) {
            std::vector<float> estimates(num_vectors);
            compute_head_estimates(
                head.data(),
                num_vectors,
                stride,
                head_codes.data(),
                -0.02f,
                1.5f,
                0.7f,
                head_rescales.data(),
                residual_norms.data(),
                estimates.data()
            );
            return estimates;
        };
        const std::vector<float> expected_head =
            head_estimates(scalar_computer::ComputeHeadEstimates);
        expect_near(expected_head, head_estimates(computer::ComputeHeadEstimates), num_vectors);
        SAQSurvivors all(num_vectors);
        ASSERT_EQ(
            num_vectors,
            scalar_computer::ScanHead(
                head.data(),
                num_vectors,
                stride,
                head_codes.data(),
                -0.02f,
                1.5f,
                0.7f,
                head_rescales.data(),
                residual_norms.data(),
                nullptr,
                std::numeric_limits<float>::infinity(),
                all.positions.data(),
                all.estimates.data(),
                all.rescales.data()
            )
        );
        expect_near(expected_head, all.estimates, num_vectors);

        for (uint32_t width = 1; width <= PDX::SAQ_MAX_CODE_WIDTH; ++width) {
            SCOPED_TRACE("width=" + std::to_string(width));
            std::vector<uint8_t> levels(num_vectors * PDX::SAQ_CHUNK_SIZE);
            for (auto& level : levels) {
                level = static_cast<uint8_t>(rng() % (1u << width));
            }
            const auto records = PackRecords(levels, num_vectors, width);
            const auto tail_codes = RandomQueryCodes(rng);
            const auto segment_rescales = RandomHalves(num_vectors, 0.001f, 0.01f, rng);
            const auto accumulated = [&](computer::AccumulateTailEstimatesFunction accumulate) {
                std::vector<float> estimates = expected_head;
                accumulate(
                    records.data(),
                    num_vectors,
                    tail_codes.data(),
                    -0.02f,
                    0.3f,
                    segment_rescales.data(),
                    estimates.data()
                );
                return estimates;
            };
            const std::vector<float> expected_tail = accumulated(scalar_accumulates[width]);
            expect_near(
                expected_tail, accumulated(computer::GetAccumulateTailEstimates(width)), num_vectors
            );
            SAQSurvivors survivors = all;
            ASSERT_EQ(
                num_vectors,
                scalar_scan_tails[width](
                    records.data(),
                    tail_codes.data(),
                    -0.02f,
                    0.3f,
                    segment_rescales.data(),
                    std::numeric_limits<float>::infinity(),
                    survivors.positions.data(),
                    survivors.estimates.data(),
                    survivors.rescales.data(),
                    num_vectors
                )
            );
            expect_near(expected_tail, survivors.estimates, num_vectors);
        }
    }
}

} // namespace

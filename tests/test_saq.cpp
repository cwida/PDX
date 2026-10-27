#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "pdx/indexes/cluster.hpp"
#include "pdx/quantizers/saq.hpp"
#include "test_utils.hpp"

namespace {

class SAQQuantizerTest : public ::testing::Test {
  protected:
    static constexpr uint32_t D = TestUtils::MAX_D;
    // Below SAQ_DP_PLAN_MIN_DIMENSIONS: one segment, so the head chunk is a record; and a partial
    // last chunk
    static constexpr uint32_t T2_HEAD_D = 100;
    // The head chunk in the PDX U8 layout; the test data stops at MAX_D, so these rows are
    // synthetic
    static constexpr uint32_t U8_HEAD_D = PDX::SAQ_PDX_U8_HEAD_MIN_DIMENSIONS;

    void SetUp() override {
        data = TestUtils::LoadTestData();
        t2_head_data = TestUtils::LoadTestData(T2_HEAD_D);
        std::mt19937 generator(TestUtils::SEED);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        u8_head_data.train.resize(TestUtils::N_TRAIN * U8_HEAD_D);
        for (size_t i = 0; i < u8_head_data.train.size(); ++i) {
            u8_head_data.train[i] =
                normal(generator) * std::pow(0.995f, static_cast<float>(i % U8_HEAD_D));
        }
    }

    const TestUtils::TestData& DataAt(uint32_t d) const {
        if (d == D) {
            return data;
        }
        return d == T2_HEAD_D ? t2_head_data : u8_head_data;
    }

    PDX::SAQFit Fit(uint32_t bits_per_dimension, uint32_t d = D) {
        return PDX::SAQQuantizer::Fit(
            DataAt(d).train.data(),
            TestUtils::N_TRAIN,
            d,
            bits_per_dimension,
            TestUtils::SEED,
            *executor,
            allocator
        );
    }

    static std::vector<float> Rotate(const PDX::SAQFit& fit, const float* rows, size_t num_rows) {
        const uint32_t d = fit.model.num_dimensions;
        std::vector<float> rotated(num_rows * d);
        Eigen::Map<PDX::eigen_matrix_t>(rotated.data(), num_rows, d).noalias() =
            Eigen::Map<const PDX::eigen_matrix_t>(rows, num_rows, d) *
            Eigen::Map<const PDX::eigen_matrix_t>(fit.rotation.get(), d, d);
        return rotated;
    }

    std::unique_ptr<skmeans::ParallelExecutor> executor = skmeans::MakeDefaultExecutor(0);
    skmeans::VanillaAllocator allocator;
    TestUtils::TestData data;
    TestUtils::TestData t2_head_data;
    TestUtils::TestData u8_head_data;
};

// Below SAQ_DP_PLAN_MIN_DIMENSIONS the plan is one segment at the declared bits. From there the
// segments tile the dimensions in 64-dimension cuts, the stored codes plus fp16 factors fit the
// budget, and from SAQ_PDX_U8_HEAD_MIN_DIMENSIONS the head chunk is its own 8-bit segment.
TEST_F(SAQQuantizerTest, PlanFitsTheBudget) {
    for (const uint32_t d : {T2_HEAD_D, D, U8_HEAD_D}) {
        for (const uint32_t bits_per_dimension : {3u, 4u}) {
            SCOPED_TRACE(std::to_string(d) + " dimensions, " + std::to_string(bits_per_dimension));
            const PDX::SAQFit fit = Fit(bits_per_dimension, d);
            const PDX::SAQModel& model = fit.model;
            ASSERT_GT(model.num_segments, 0u);
            if (d < PDX::SAQ_DP_PLAN_MIN_DIMENSIONS) {
                ASSERT_EQ(model.num_segments, 1u);
                EXPECT_EQ(model.segments[0].end, d);
                EXPECT_EQ(model.segments[0].width, bits_per_dimension);
                continue;
            }
            if (d >= PDX::SAQ_PDX_U8_HEAD_MIN_DIMENSIONS) {
                EXPECT_TRUE(model.HasPDXU8Head());
                EXPECT_EQ(model.segments[0].end, PDX::SAQ_CHUNK_SIZE);
            }
            uint32_t next_start = 0;
            for (uint32_t segment_idx = 0; segment_idx < model.num_segments; ++segment_idx) {
                const PDX::SAQSegment& segment = model.segments[segment_idx];
                EXPECT_EQ(segment.start, next_start);
                EXPECT_EQ(segment.start % PDX::SAQ_CHUNK_SIZE, 0u);
                EXPECT_GE(segment.width, 1u);
                EXPECT_LE(segment.width, PDX::SAQ_MAX_CODE_WIDTH);
                next_start = segment.end;
            }
            EXPECT_EQ(next_start, d);
            EXPECT_LE(8 * model.GetEncodedEmbeddingSize(), bits_per_dimension * d);
        }
    }
}

// The fitted rotation keeps distances: it is orthogonal.
TEST_F(SAQQuantizerTest, RotationIsOrthogonal) {
    const PDX::SAQFit fit = Fit(3);
    const Eigen::Map<const PDX::eigen_matrix_t> rotation(fit.rotation.get(), D, D);
    const float max_error =
        (rotation * rotation.transpose() - Eigen::MatrixXf::Identity(D, D)).cwiseAbs().maxCoeff();
    EXPECT_LT(max_error, 1e-4f);
}

// Encoding then decoding gets closer to the rotated embeddings with 4 bits than with 3.
TEST_F(SAQQuantizerTest, MoreBitsDecodeCloser) {
    const size_t num_rows = 500;
    double squared_errors[2] = {0.0, 0.0};
    for (const uint32_t bits_per_dimension : {3u, 4u}) {
        const PDX::SAQFit fit = Fit(bits_per_dimension);
        const std::vector<float> rotated = Rotate(fit, data.train.data(), num_rows);
        const PDX::SAQQuantizer quantizer(D);
        PDX::SAQEncodingScratch scratch(D);
        std::vector<uint8_t> encoded(fit.model.GetEncodedEmbeddingSize());
        std::vector<float> decoded(D);
        for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            const float* row = rotated.data() + row_idx * D;
            quantizer.EncodeEmbedding(row, fit.model, encoded.data(), scratch);
            quantizer.DecodeEmbedding(encoded.data(), fit.model, decoded.data());
            for (uint32_t j = 0; j < D; ++j) {
                const double error = decoded[j] - row[j];
                squared_errors[bits_per_dimension - 3] += error * error;
            }
        }
    }
    EXPECT_LT(squared_errors[1], squared_errors[0]);
}

// The encoder's codes come within 3% of the estimator error (1/cos^2 - 1) of the best scale a fine
// scan finds (on this data, 1.5% for its 4-scale grid and 4% for the grid's center alone).
TEST_F(SAQQuantizerTest, EncoderNearsTheBestScale) {
    using computer = PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>;
    constexpr size_t NUM_ROWS = 200;
    constexpr size_t NUM_SCANNED_SCALES = 400;
    for (const uint32_t bits_per_dimension : {3u, 4u}) {
        const PDX::SAQFit fit = Fit(bits_per_dimension);
        const PDX::SAQModel& model = fit.model;
        const std::vector<float> rotated = Rotate(fit, data.train.data(), NUM_ROWS);
        const PDX::SAQQuantizer quantizer(D);
        PDX::SAQEncodingScratch scratch(D);
        std::vector<uint8_t> encoded(model.GetEncodedEmbeddingSize());
        std::vector<uint8_t> levels(PDX::AlignValue<uint32_t, PDX::SAQ_CHUNK_SIZE>(D));
        double encoder_error = 0.0;
        double best_error = 0.0;
        for (size_t row_idx = 0; row_idx < NUM_ROWS; ++row_idx) {
            const float* row = rotated.data() + row_idx * D;
            quantizer.EncodeEmbedding(row, model, encoded.data(), scratch);
            for (uint32_t chunk_idx = 0; chunk_idx < model.num_chunks; ++chunk_idx) {
                computer::UnpackRecord(
                    encoded.data() + model.chunk_record_offsets[chunk_idx],
                    model.chunk_widths[chunk_idx],
                    levels.data() + static_cast<size_t>(chunk_idx) * PDX::SAQ_CHUNK_SIZE
                );
            }
            for (uint32_t segment_idx = 0; segment_idx < model.num_segments; ++segment_idx) {
                const PDX::SAQSegment& segment = model.segments[segment_idx];
                if (segment.width == 1) {
                    continue;
                }
                const double center = (static_cast<double>(1u << segment.width) - 1.0) / 2.0;
                const int32_t max_level = (int32_t{1} << (segment.width - 1)) - 1;
                std::vector<double> values;
                double squared_norm = 0.0;
                double inner_product = 0.0;
                double code_squared_norm = 0.0;
                for (uint32_t j = segment.start; j < segment.end; ++j) {
                    values.push_back(static_cast<double>(row[j]) - model.rotated_mean[j]);
                    const double code = static_cast<double>(levels[j]) - center;
                    squared_norm += values.back() * values.back();
                    inner_product += values.back() * code;
                    code_squared_norm += code * code;
                }
                encoder_error +=
                    squared_norm * code_squared_norm / (inner_product * inner_product) - 1.0;
                const double norm = std::sqrt(squared_norm);
                double best = std::numeric_limits<double>::infinity();
                for (size_t scale_idx = 0; scale_idx < NUM_SCANNED_SCALES; ++scale_idx) {
                    const double scale =
                        segment.scale *
                        std::pow(16.0, static_cast<double>(scale_idx) / NUM_SCANNED_SCALES - 0.5);
                    double numerator = 0.0;
                    double denominator = 0.0;
                    for (const double value : values) {
                        const double magnitude = std::abs(value) / norm;
                        const double level =
                            std::min(
                                static_cast<double>(static_cast<int32_t>(scale * magnitude + 1e-5)),
                                static_cast<double>(max_level)
                            ) +
                            0.5;
                        numerator += level * magnitude;
                        denominator += level * level;
                    }
                    best = std::min(best, denominator / (numerator * numerator) - 1.0);
                }
                best_error += best;
            }
        }
        EXPECT_LT(encoder_error, best_error * 1.03) << bits_per_dimension << " bits";
    }
}

void ExpectSameModel(const PDX::SAQModel& expected, const PDX::SAQModel& actual) {
    ASSERT_EQ(expected.num_dimensions, actual.num_dimensions);
    EXPECT_EQ(expected.bits_per_dimension, actual.bits_per_dimension);
    ASSERT_EQ(expected.num_segments, actual.num_segments);
    ASSERT_EQ(expected.num_chunks, actual.num_chunks);
    for (uint32_t j = 0; j < expected.num_dimensions; ++j) {
        EXPECT_EQ(expected.rotated_mean[j], actual.rotated_mean[j]);
    }
    for (uint32_t segment_idx = 0; segment_idx < expected.num_segments; ++segment_idx) {
        EXPECT_EQ(expected.segments[segment_idx].start, actual.segments[segment_idx].start);
        EXPECT_EQ(expected.segments[segment_idx].end, actual.segments[segment_idx].end);
        EXPECT_EQ(expected.segments[segment_idx].width, actual.segments[segment_idx].width);
        EXPECT_EQ(expected.segments[segment_idx].scale, actual.segments[segment_idx].scale);
    }
    for (uint32_t chunk_idx = 0; chunk_idx < expected.num_chunks; ++chunk_idx) {
        EXPECT_EQ(expected.chunk_widths[chunk_idx], actual.chunk_widths[chunk_idx]);
        EXPECT_EQ(expected.chunk_segments[chunk_idx], actual.chunk_segments[chunk_idx]);
        EXPECT_EQ(expected.chunk_code_centers[chunk_idx], actual.chunk_code_centers[chunk_idx]);
        EXPECT_EQ(
            expected.chunk_remaining_variances[chunk_idx],
            actual.chunk_remaining_variances[chunk_idx]
        );
    }
}

// A copied model and a saved then loaded one are the fitted model.
TEST_F(SAQQuantizerTest, ModelSurvivesCopyAndSaveLoad) {
    const PDX::SAQFit fit = Fit(3);
    const PDX::SAQModel copied = fit.model;
    ExpectSameModel(fit.model, copied);

    std::stringstream stream;
    fit.model.Save(stream);
    PDX::SAQModel loaded;
    PDX::StreamReader reader{stream};
    loaded.Load(reader);
    ExpectSameModel(fit.model, loaded);
}

// The query's int8 chunks, scales and offsets turn integer dots with the levels into
// -2 <query - mean, decoded - mean>, up to the int8 rounding of the query.
TEST_F(SAQQuantizerTest, QueryDotsMatchDecodedInnerProducts) {
    const PDX::SAQFit fit = Fit(3);
    const PDX::SAQModel& model = fit.model;
    const size_t num_rows = 100;
    const std::vector<float> rows = Rotate(fit, data.train.data(), num_rows);
    const std::vector<float> queries = Rotate(fit, data.queries.data(), num_rows);
    const PDX::SAQQuantizer quantizer(D);
    PDX::SAQEncodingScratch scratch(D);
    std::vector<uint8_t> encoded(model.GetEncodedEmbeddingSize());
    std::vector<float> decoded(D);
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        quantizer.EncodeEmbedding(rows.data() + row_idx * D, model, encoded.data(), scratch);
        quantizer.DecodeEmbedding(encoded.data(), model, decoded.data());
        const float* query = queries.data() + row_idx * D;
        const PDX::SAQQuery saq_query(query, model);
        double expected = 0.0, query_norm = 0.0, decoded_norm = 0.0;
        for (uint32_t j = 0; j < D; ++j) {
            const double query_residual = query[j] - model.rotated_mean[j];
            const double decoded_residual = decoded[j] - model.rotated_mean[j];
            expected += -2.0 * query_residual * decoded_residual;
            query_norm += query_residual * query_residual;
            decoded_norm += decoded_residual * decoded_residual;
        }
        double actual = 0.0;
        const uint8_t* factors = encoded.data() + model.chunk_record_offsets[model.num_chunks];
        for (uint32_t chunk_idx = 0; chunk_idx < model.num_chunks; ++chunk_idx) {
            uint8_t levels[PDX::SAQ_CHUNK_SIZE];
            PDX::DistanceComputer<PDX::DistanceMetric::L2SQ, PDX::SAQ>::UnpackRecord(
                encoded.data() + model.chunk_record_offsets[chunk_idx],
                model.chunk_widths[chunk_idx],
                levels
            );
            int32_t dot = 0;
            for (uint32_t j = chunk_idx * PDX::SAQ_CHUNK_SIZE;
                 j < std::min((chunk_idx + 1) * PDX::SAQ_CHUNK_SIZE, D);
                 ++j) {
                dot += saq_query.GetCodesPtr()[j] * levels[j - chunk_idx * PDX::SAQ_CHUNK_SIZE];
            }
            const float rescale =
                PDX::ReadHalf(factors + model.chunk_segments[chunk_idx] * sizeof(uint16_t));
            actual += rescale * (saq_query.GetChunkDotScalesPtr()[chunk_idx] * dot -
                                 saq_query.GetChunkDotOffsetsPtr()[chunk_idx]);
        }
        EXPECT_NEAR(actual, expected, 0.02 * 2.0 * std::sqrt(query_norm * decoded_norm))
            << "row " << row_idx;
    }
}

// A SAQ cluster keeps each part of an encoded embedding in its own section, the head chunk in the
// PDX U8 layout or as a record, and gives the embeddings back after appends, a compaction, a
// growth and a save and load.
TEST_F(SAQQuantizerTest, ClusterKeepsEncodedEmbeddings) {
    for (const uint32_t d : {T2_HEAD_D, D, U8_HEAD_D}) {
        SCOPED_TRACE(std::to_string(d) + " dimensions");
        const PDX::SAQFit fit = Fit(3, d);
        const PDX::SAQModel& model = fit.model;
        const uint32_t num_rows = 80;
        const std::vector<float> rows = Rotate(fit, DataAt(d).train.data(), num_rows);
        const PDX::SAQQuantizer quantizer(d);
        PDX::SAQEncodingScratch scratch(d);
        const size_t row_size = model.GetEncodedEmbeddingSize();
        std::vector<uint8_t> encoded(num_rows * row_size);
        for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            quantizer.EncodeEmbedding(
                rows.data() + row_idx * d, model, encoded.data() + row_idx * row_size, scratch
            );
        }
        const PDX::ClusterFormat<PDX::SAQ> format(
            model.chunk_record_offsets.get(),
            model.num_chunks,
            model.num_segments,
            model.HasPDXU8Head()
        );
        ASSERT_EQ(format.GetValuesPerEmbedding(), row_size);

        const uint32_t num_stored = 50;
        PDX::Cluster<PDX::SAQ> cluster(num_stored, 128, format);
        cluster.StoreEmbeddings(encoded.data(), num_stored);
        for (uint32_t position = 0; position < num_stored; ++position) {
            cluster.indices[position] = position;
        }
        for (uint32_t row_id = num_stored; row_id < num_rows; ++row_id) {
            ASSERT_EQ(cluster.AppendEmbedding(row_id, encoded.data() + row_id * row_size), row_id);
        }

        const size_t stride = cluster.max_capacity;
        for (const uint32_t position : {7u, 63u}) {
            const uint8_t* row = encoded.data() + position * row_size;
            if (format.has_pdx_u8_head) {
                for (size_t group_idx = 0;
                     group_idx < PDX::SAQ_CHUNK_SIZE / PDX::U8_INTERLEAVE_SIZE;
                     ++group_idx) {
                    const size_t offset = group_idx * PDX::U8_INTERLEAVE_SIZE;
                    EXPECT_EQ(
                        0,
                        std::memcmp(
                            cluster.data.get() + stride * offset +
                                position * PDX::U8_INTERLEAVE_SIZE,
                            row + offset,
                            PDX::U8_INTERLEAVE_SIZE
                        )
                    );
                }
            }
            for (uint32_t chunk_idx = format.GetFirstRecordChunk(); chunk_idx < model.num_chunks;
                 ++chunk_idx) {
                const size_t offset = model.chunk_record_offsets[chunk_idx];
                const size_t size = model.chunk_record_offsets[chunk_idx + 1] - offset;
                EXPECT_EQ(
                    0,
                    std::memcmp(
                        cluster.data.get() + stride * offset + position * size, row + offset, size
                    )
                );
            }
            for (uint32_t factor_idx = 0; factor_idx <= model.num_segments; ++factor_idx) {
                const size_t offset =
                    model.chunk_record_offsets[model.num_chunks] + factor_idx * sizeof(uint16_t);
                EXPECT_EQ(
                    0,
                    std::memcmp(
                        cluster.data.get() + stride * offset + position * sizeof(uint16_t),
                        row + offset,
                        sizeof(uint16_t)
                    )
                );
            }
        }

        const auto expect_embeddings = [&](const PDX::Cluster<PDX::SAQ>& checked) {
            for (uint32_t position = 0; position < checked.num_embeddings; ++position) {
                const auto embedding = checked.GetHorizontalEmbeddingFromPDXBuffer(position);
                EXPECT_EQ(
                    0,
                    std::memcmp(
                        embedding.get(),
                        encoded.data() + checked.indices[position] * row_size,
                        row_size
                    )
                ) << "position "
                  << position;
            }
        };
        expect_embeddings(cluster);
        cluster.DeleteEmbedding(5);
        cluster.CompactCluster();
        ASSERT_EQ(cluster.num_embeddings, num_rows - 1);
        expect_embeddings(cluster);
        cluster.GrowTo(200);
        expect_embeddings(cluster);

        std::stringstream stream;
        cluster.SavePDXData(stream);
        PDX::Cluster<PDX::SAQ> loaded(cluster.num_embeddings, cluster.num_embeddings + 1, format);
        PDX::StreamReader reader{stream};
        loaded.LoadPDXData(reader);
        std::memcpy(
            loaded.indices.get(), cluster.indices.get(), cluster.num_embeddings * sizeof(uint32_t)
        );
        expect_embeddings(loaded);
    }
}

// With the PDX U8 head, a row's own match comes first, near distance 0, even when its cluster is
// probed last, through the head's pruning kernel
TEST_F(SAQQuantizerTest, U8HeadFindsEachRowWhenItsClusterIsProbedLast) {
    constexpr size_t NUM_QUERY_ROWS = 50;
    PDX::PDXIndexConfig config{
        .num_dimensions = U8_HEAD_D,
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .sampling_fraction = 1.0f,
    };
    PDX::PDXIndexSAQ index(config);
    index.BuildIndex(u8_head_data.train.data(), TestUtils::N_TRAIN);
    for (size_t row_id = 0; row_id < NUM_QUERY_ROWS; ++row_id) {
        const float* row = u8_head_data.train.data() + row_id * U8_HEAD_D;
        std::vector<uint32_t> clusters_access_order = index.GetClustersAccessOrder(row);
        std::reverse(clusters_access_order.begin(), clusters_access_order.end());
        PDX::TopKHeap heap;
        auto search = index.BeginIterativeSearch(
            row, TestUtils::KNN, heap, nullptr, false, &clusters_access_order
        );
        search->Next(clusters_access_order.size());
        const auto results = PDX::BuildResultSetFromHeap(TestUtils::KNN, heap.heap);
        ASSERT_EQ(results.size(), TestUtils::KNN);
        EXPECT_EQ(results[0].index, row_id);
        EXPECT_LT(std::abs(results[0].distance), 0.5f * results[1].distance);
    }
}

static constexpr uint32_t PRUNING_TEST_N_PROBE = 16;
static constexpr float PRUNING_RECALL_TOLERANCE = 0.01f;

// The bound calibrated at the default P_s 0.1 keeps the recall of one that barely prunes
// (P_s 1e-6), and a loose one (P_s 0.5) prunes true neighbours
TEST(SAQPruningBoundTest, CalibratedBoundKeepsRecallAndPrunes) {
    constexpr size_t d = TestUtils::MAX_D;
    const auto data = TestUtils::LoadTestData(d);
    const auto gt = TestUtils::ComputeBruteForceKNN(
        data.train.data(),
        data.queries.data(),
        TestUtils::N_TRAIN,
        TestUtils::N_QUERIES,
        d,
        TestUtils::KNN
    );
    const auto recall_at = [&](float significance) {
        PDX::PDXIndexConfig config{
            .num_dimensions = static_cast<uint32_t>(d),
            .distance_metric = PDX::DistanceMetric::L2SQ,
            .seed = TestUtils::SEED,
            .normalize = true,
            .sampling_fraction = 1.0f,
            .hierarchical_indexing = true,
        };
        config.saq_pruning_significance = significance;
        auto index = TestUtils::BuildIndexWithConfig(
            "pdx_saq3", config, data.train.data(), TestUtils::N_TRAIN
        );
        index->SetNProbe(PRUNING_TEST_N_PROBE);
        return TestUtils::ComputeAverageRecall(
            *index, data.queries.data(), TestUtils::N_QUERIES, d, TestUtils::KNN, gt
        );
    };
    const float barely_pruned_recall = recall_at(1e-6f);
    EXPECT_GE(recall_at(0.1f), barely_pruned_recall - PRUNING_RECALL_TOLERANCE);
    EXPECT_LT(recall_at(0.5f), barely_pruned_recall);
}

// Recalibrating an index on its own rows keeps its recall
TEST(SAQPruningBoundTest, RecalibrationOnTheIndexRowsKeepsRecall) {
    constexpr size_t d = TestUtils::MAX_D;
    const auto data = TestUtils::LoadTestData(d);
    const auto gt = TestUtils::ComputeBruteForceKNN(
        data.train.data(),
        data.queries.data(),
        TestUtils::N_TRAIN,
        TestUtils::N_QUERIES,
        d,
        TestUtils::KNN
    );
    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    PDX::PDXIndexSAQ index(config);
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    index.SetNProbe(PRUNING_TEST_N_PROBE);
    const float built_recall = TestUtils::ComputeAverageRecall(
        index, data.queries.data(), TestUtils::N_QUERIES, d, TestUtils::KNN, gt
    );

    std::vector<size_t> row_ids(TestUtils::N_TRAIN);
    std::iota(row_ids.begin(), row_ids.end(), 0);
    index.CalibratePruningBound(row_ids.data(), data.train.data(), TestUtils::N_TRAIN);
    EXPECT_NEAR(
        TestUtils::ComputeAverageRecall(
            index, data.queries.data(), TestUtils::N_QUERIES, d, TestUtils::KNN, gt
        ),
        built_recall,
        PRUNING_RECALL_TOLERANCE
    );

    index.CalibratePruningBound();
    EXPECT_NEAR(
        TestUtils::ComputeAverageRecall(
            index, data.queries.data(), TestUtils::N_QUERIES, d, TestUtils::KNN, gt
        ),
        built_recall,
        PRUNING_RECALL_TOLERANCE
    );
}

// Recalibrating on rows decoded from the index samples only the rows it still holds
TEST(SAQPruningBoundTest, RecalibrationOnDecodedRowsSkipsDeletedRows) {
    constexpr size_t d = TestUtils::MAX_D;
    const auto data = TestUtils::LoadTestData(d);
    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    PDX::PDXIndexSAQ index(config);
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; ++row_id) {
        if (row_id % 10 != 0) {
            index.Delete(row_id);
        }
    }
    EXPECT_NO_THROW(index.CalibratePruningBound());
}

// CalibrateIndex recalibrates the pruning bound only after rows were appended
TEST(SAQPruningBoundTest, CalibrateIndexRecalibratesOnlyAfterAppends) {
    constexpr size_t d = TestUtils::MAX_D;
    const auto data = TestUtils::LoadTestData(d);
    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    PDX::PDXIndexSAQ index(config);
    const size_t num_built = TestUtils::N_TRAIN / 2;
    index.BuildIndex(data.train.data(), num_built);
    const auto epsilons = [&]() {
        const auto& pdx_data = index.GetSearcher().pdx_data;
        return std::vector<float>(
            pdx_data.saq_chunk_epsilons.get(),
            pdx_data.saq_chunk_epsilons.get() + pdx_data.saq_model.num_chunks
        );
    };
    const auto built_epsilons = epsilons();
    index.CalibrateIndex();
    EXPECT_EQ(epsilons(), built_epsilons);

    std::vector<size_t> row_ids(TestUtils::N_TRAIN - num_built);
    std::iota(row_ids.begin(), row_ids.end(), num_built);
    index.BulkAppend(row_ids.data(), data.train.data() + num_built * d, row_ids.size());
    index.CalibrateIndex();
    const auto recalibrated_epsilons = epsilons();
    EXPECT_NE(recalibrated_epsilons, built_epsilons);
    index.CalibratePruningBound();
    EXPECT_EQ(epsilons(), recalibrated_epsilons);
}

} // namespace

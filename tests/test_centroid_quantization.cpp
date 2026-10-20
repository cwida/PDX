#undef HAS_FFTW

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <vector>

#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/quantizers/lvq.hpp"
#include "superkmeans/pdx/utils.h"
#include "test_utils.hpp"

namespace {

static constexpr size_t D = 128;
static constexpr size_t N_QUERIES = 100;
static constexpr uint32_t N_PROBE = 16;
static constexpr float MAINTENANCE_RECALL_TOLERANCE = 0.03f;
static constexpr uint32_t MAX_CLUSTER_SIZE = 32;

PDX::PDXIndexConfig MakeConfig(
    PDX::CentroidQuantization centroid_quantization,
    uint32_t max_cluster_size = 0,
    PDX::MemoryAllocator* allocator = nullptr
) {
    return PDX::PDXIndexConfig{
        .num_dimensions = static_cast<uint32_t>(D),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
        .max_cluster_size = max_cluster_size,
        .centroid_quantization = centroid_quantization,
        .allocator = allocator,
    };
}

float ComputeCursorRecall(
    const PDX::IPDXIndex& index,
    const float* queries,
    const TestUtils::BruteForceResult& gt
) {
    float total_recall = 0.0f;
    for (size_t q = 0; q < N_QUERIES; ++q) {
        PDX::TopKHeap top_k_heap;
        auto search_cursor =
            index.BeginIterativeSearch(queries + q * D, TestUtils::KNN, top_k_heap, nullptr);
        search_cursor->Next(N_PROBE);
        total_recall += TestUtils::ComputeRecall(
            PDX::BuildResultSetFromHeap(TestUtils::KNN, top_k_heap.heap),
            gt.indices[q],
            TestUtils::KNN
        );
    }
    return total_recall / static_cast<float>(N_QUERIES);
}

template <uint8_t BIT_WIDTH>
void ExpectComputeL2SQMatchesTheDequantizedEmbedding() {
    for (size_t d : {1, 7, 16, 33, 128, 384}) {
        SCOPED_TRACE("bit_width=" + std::to_string(BIT_WIDTH) + ", d=" + std::to_string(d));
        const auto vectors = skmeans::GenerateRandomVectors(2, d, -1.0f, 1.0f, 42);
        const float* embedding = vectors.data();
        const float* query = vectors.data() + d;
        const PDX::LVQQuantizer<BIT_WIDTH> quantizer(d);
        std::vector<uint8_t> codes(PDX::LVQQuantizer<BIT_WIDTH>::GetCodesSizeInBytes(d));
        const PDX::LVQParams params = quantizer.QuantizeEmbedding(embedding, codes.data());
        std::vector<float> dequantized(d);
        quantizer.DequantizeEmbedding(codes.data(), params, dequantized.data());

        float query_sum = 0.0f;
        float query_squared_norm = 0.0f;
        float expected_distance = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            EXPECT_LE(
                std::abs(embedding[j] - dequantized[j]), 0.5f * params.quantization_step + 1e-6f
            );
            query_sum += query[j];
            query_squared_norm += query[j] * query[j];
            const float diff = query[j] - dequantized[j];
            expected_distance += diff * diff;
        }
        const float distance = quantizer.ComputeL2SQ(
            query, query_sum, query_squared_norm, codes.data(), params
        );
        EXPECT_NEAR(
            distance,
            expected_distance,
            1e-5f * (query_squared_norm + params.dequantized_squared_norm)
        );
    }
}

template <uint8_t BIT_WIDTH>
void ExpectConstantEmbeddingDequantizesExactly() {
    const std::vector<float> embedding(D, 0.25f);
    const PDX::LVQQuantizer<BIT_WIDTH> quantizer(D);
    std::vector<uint8_t> codes(PDX::LVQQuantizer<BIT_WIDTH>::GetCodesSizeInBytes(D));
    const PDX::LVQParams params = quantizer.QuantizeEmbedding(embedding.data(), codes.data());
    std::vector<float> dequantized(D);
    quantizer.DequantizeEmbedding(codes.data(), params, dequantized.data());
    EXPECT_EQ(dequantized, embedding);
}

// ComputeL2SQ is the squared distance to the dequantized embedding, which is within half a step of
// the embedding.
TEST(LVQQuantizerTest, ComputeL2SQMatchesTheDequantizedEmbedding) {
    ExpectComputeL2SQMatchesTheDequantizedEmbedding<8>();
    ExpectComputeL2SQMatchesTheDequantizedEmbedding<4>();
}

// A constant embedding has no range to spread over the codes: it dequantizes exactly.
TEST(LVQQuantizerTest, ConstantEmbeddingDequantizesExactly) {
    ExpectConstantEmbeddingDequantizesExactly<8>();
    ExpectConstantEmbeddingDequantizesExactly<4>();
}

class CentroidQuantizationTest : public ::testing::TestWithParam<std::string> {};

// Ranking by the quantized centroids keeps the recall of ranking by the float32 ones, single-shot
// and on a cursor.
TEST_P(CentroidQuantizationTest, RecallMatchesF32) {
    auto data = TestUtils::LoadTestData(D);
    const auto gt = TestUtils::ComputeBruteForceKNN(
        data.train.data(), data.queries.data(), TestUtils::N_TRAIN, N_QUERIES, D, TestUtils::KNN
    );
    auto f32_index = TestUtils::BuildIndexWithConfig(
        GetParam(),
        MakeConfig(PDX::CentroidQuantization::F32),
        data.train.data(),
        TestUtils::N_TRAIN
    );
    f32_index->SetNProbe(N_PROBE);
    const float f32_recall = TestUtils::ComputeAverageRecall(
        *f32_index, data.queries.data(), N_QUERIES, D, TestUtils::KNN, gt
    );
    const float f32_cursor_recall = ComputeCursorRecall(*f32_index, data.queries.data(), gt);

    for (const auto& test_case : TestUtils::QUANTIZED_CENTROIDS) {
        SCOPED_TRACE(
            "centroid_quantization=" +
            std::to_string(static_cast<int>(test_case.centroid_quantization))
        );
        auto quantized_index = TestUtils::BuildIndexWithConfig(
            GetParam(),
            MakeConfig(test_case.centroid_quantization),
            data.train.data(),
            TestUtils::N_TRAIN
        );
        quantized_index->SetNProbe(N_PROBE);
        EXPECT_GE(
            TestUtils::ComputeAverageRecall(
                *quantized_index, data.queries.data(), N_QUERIES, D, TestUtils::KNN, gt
            ),
            f32_recall - test_case.recall_tolerance
        );
        EXPECT_GE(
            ComputeCursorRecall(*quantized_index, data.queries.data(), gt),
            f32_cursor_recall - test_case.recall_tolerance
        );
    }
}

// Splits (appends past MAX_CLUSTER_SIZE), merges (deleting four rows in five) and bulk appends
// place rows by the decoded centroids: every live row stays indexed, recall stays close to
// float32's and the allocator counts the encoded centroids.
TEST_P(CentroidQuantizationTest, MaintenanceKeepsRowsRecallAndAccounting) {
    auto data = TestUtils::LoadTestData(D);
    constexpr size_t n_build = TestUtils::N_TRAIN / 2;
    constexpr size_t n_bulk_append = TestUtils::N_TRAIN / 4;
    constexpr size_t row_id_step_kept = 5;
    auto is_deleted = [](size_t row_id) { return row_id % row_id_step_kept != 0; };
    std::vector<float> kept_rows(TestUtils::N_TRAIN / row_id_step_kept * D);
    for (size_t i = 0; i < TestUtils::N_TRAIN / row_id_step_kept; ++i) {
        std::copy_n(data.train.data() + i * row_id_step_kept * D, D, kept_rows.data() + i * D);
    }
    auto gt = TestUtils::ComputeBruteForceKNN(
        kept_rows.data(),
        data.queries.data(),
        TestUtils::N_TRAIN / row_id_step_kept,
        N_QUERIES,
        D,
        TestUtils::KNN
    );
    for (auto& neighbor_ids : gt.indices) {
        for (auto& neighbor_id : neighbor_ids) {
            neighbor_id *= row_id_step_kept;
        }
    }

    auto run_maintenance = [&](PDX::CentroidQuantization centroid_quantization) {
        SCOPED_TRACE(
            "centroid_quantization=" + std::to_string(static_cast<int>(centroid_quantization))
        );
        TestUtils::CountingAllocator allocator;
        auto index = TestUtils::BuildIndexWithConfig(
            GetParam(),
            MakeConfig(centroid_quantization, MAX_CLUSTER_SIZE, &allocator),
            data.train.data(),
            n_build
        );
        std::vector<size_t> row_ids(n_bulk_append);
        std::iota(row_ids.begin(), row_ids.end(), n_build);
        index->BulkAppend(row_ids.data(), data.train.data() + n_build * D, n_bulk_append);
        for (size_t row_id = n_build + n_bulk_append; row_id < TestUtils::N_TRAIN; ++row_id) {
            index->Append(row_id, data.train.data() + row_id * D);
        }
        for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; ++row_id) {
            if (is_deleted(row_id)) {
                index->Delete(row_id);
            }
        }

        for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; ++row_id) {
            EXPECT_EQ(index->Contains(row_id), !is_deleted(row_id)) << "row_id " << row_id;
        }
        EXPECT_EQ(
            index->GetSelfManagedMemorySizeInBytes() + allocator.live_bytes,
            index->GetInMemorySizeInBytes()
        );
        index->SetNProbe(N_PROBE);
        return TestUtils::ComputeAverageRecall(
            *index, data.queries.data(), N_QUERIES, D, TestUtils::KNN, gt
        );
    };
    const float f32_recall = run_maintenance(PDX::CentroidQuantization::F32);
    for (const auto& test_case : TestUtils::QUANTIZED_CENTROIDS) {
        EXPECT_GE(
            run_maintenance(test_case.centroid_quantization),
            f32_recall - MAINTENANCE_RECALL_TOLERANCE
        );
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllIndexTypes,
    CentroidQuantizationTest,
    ::testing::Values("pdx_f32", "pdx_u8", "pdx_tree_f32", "pdx_tree_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

// The tree's L0 is clustered over the decoded centroids, so only vanilla indexes share every other
// byte.
class VanillaCentroidQuantizationTest : public ::testing::TestWithParam<std::string> {};

// One clustering, so the indexes differ only by their centroids: d floats each in F32, the codes
// and their LVQParams in LVQ.
TEST_P(VanillaCentroidQuantizationTest, QuantizedCentroidsSaveTheirBytes) {
    auto data = TestUtils::LoadTestData(D);
    auto f32_index = TestUtils::BuildIndexWithConfig(
        GetParam(),
        MakeConfig(PDX::CentroidQuantization::F32),
        data.train.data(),
        TestUtils::N_TRAIN
    );
    for (const auto& test_case : TestUtils::QUANTIZED_CENTROIDS) {
        SCOPED_TRACE(
            "centroid_quantization=" +
            std::to_string(static_cast<int>(test_case.centroid_quantization))
        );
        auto quantized_index = TestUtils::BuildIndexWithConfig(
            GetParam(),
            MakeConfig(test_case.centroid_quantization),
            data.train.data(),
            TestUtils::N_TRAIN
        );
        ASSERT_EQ(f32_index->GetNumClusters(), quantized_index->GetNumClusters());
        const size_t saved_bytes_per_centroid =
            D * sizeof(float) - test_case.encoded_centroid_size_in_bytes(D);
        EXPECT_EQ(
            f32_index->GetInMemorySizeInBytes() - quantized_index->GetInMemorySizeInBytes(),
            f32_index->GetNumClusters() * saved_bytes_per_centroid
        );
    }
}

INSTANTIATE_TEST_SUITE_P(
    VanillaIndexTypes,
    VanillaCentroidQuantizationTest,
    ::testing::Values("pdx_f32", "pdx_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

} // namespace

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "superkmeans/pdx/utils.h"
#include "test_utils.hpp"

namespace {

class IndexPropertiesTest : public ::testing::TestWithParam<std::string> {
  protected:
    void SetUp() override {}
};

TEST_P(IndexPropertiesTest, ClusterRowIdsCoverAllPoints) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);

    std::unordered_set<uint32_t> all_ids;
    uint32_t num_clusters = index->GetNumClusters();
    for (uint32_t c = 0; c < num_clusters; ++c) {
        auto ids = index->GetClusterRowIds(c);
        for (auto id : ids) {
            EXPECT_TRUE(all_ids.insert(id).second)
                << "Duplicate row ID " << id << " in cluster " << c;
        }
    }
    EXPECT_EQ(all_ids.size(), TestUtils::N_TRAIN);

    for (size_t i = 0; i < TestUtils::N_TRAIN; ++i) {
        EXPECT_TRUE(all_ids.count(static_cast<uint32_t>(i))) << "Missing row ID " << i;
    }
}

TEST_P(IndexPropertiesTest, ClusterSizeMatchesRowIdCount) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);

    uint32_t num_clusters = index->GetNumClusters();
    for (uint32_t c = 0; c < num_clusters; ++c) {
        auto ids = index->GetClusterRowIds(c);
        EXPECT_EQ(index->GetClusterSize(c), ids.size()) << "Size mismatch for cluster " << c;
    }
}

TEST_P(IndexPropertiesTest, GetNumDimensionsMatchesInput) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    EXPECT_EQ(index->GetNumDimensions(), d);
}

TEST_P(IndexPropertiesTest, InMemorySizeIsPositive) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    EXPECT_GT(index->GetInMemorySizeInBytes(), 0u);
}

TEST_P(IndexPropertiesTest, KnnLargerThanDataReturnsAvailable) {
    std::string index_type = GetParam();
    size_t d = 128;

    // Build a tiny index with 5 points
    auto data = TestUtils::LoadTestData(d);
    size_t tiny_n = 5;
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), tiny_n, d);
    index->SetNProbe(index->GetNumClusters());

    auto results = index->Search(data.queries.data(), 100);
    EXPECT_LE(results.size(), tiny_n);
    EXPECT_GT(results.size(), 0u);
}

TEST_P(IndexPropertiesTest, NumClustersMatchesConfig) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);

    for (uint32_t requested : {10, 50, 100}) {
        SCOPED_TRACE("num_clusters=" + std::to_string(requested));

        PDX::PDXIndexConfig config{
            .num_dimensions = static_cast<uint32_t>(d),
            .distance_metric = PDX::DistanceMetric::L2SQ,
            .seed = TestUtils::SEED,
            .num_clusters = requested,
            .normalize = true,
            .sampling_fraction = 1.0f,
            .hierarchical_indexing = true,
        };

        std::unique_ptr<PDX::IPDXIndex> index;
        if (index_type == "pdx_f32") {
            auto p = std::make_unique<PDX::PDXIndexF32>(config);
            p->BuildIndex(data.train.data(), TestUtils::N_TRAIN);
            index = std::move(p);
        } else {
            auto p = std::make_unique<PDX::PDXIndexU8>(config);
            p->BuildIndex(data.train.data(), TestUtils::N_TRAIN);
            index = std::move(p);
        }

        EXPECT_EQ(index->GetNumClusters(), requested);

        // nprobe = num_clusters, num_clusters + 1, and 0 (all) must give identical recall
        auto gt = TestUtils::ComputeBruteForceKNN(
            data.train.data(), data.queries.data(), TestUtils::N_TRAIN, 50, d, TestUtils::KNN
        );

        auto measure_recall = [&](uint32_t nprobe) {
            index->SetNProbe(nprobe);
            float total = 0.0f;
            for (size_t q = 0; q < 50; ++q) {
                auto results = index->Search(data.queries.data() + q * d, TestUtils::KNN);
                total += TestUtils::ComputeRecall(results, gt.indices[q], TestUtils::KNN);
            }
            return total;
        };

        float recall_exact = measure_recall(requested);
        float recall_over = measure_recall(requested + 1);
        float recall_zero = measure_recall(0);

        EXPECT_FLOAT_EQ(recall_exact, recall_over)
            << "nprobe=" << requested << " vs nprobe=" << requested + 1;
        EXPECT_FLOAT_EQ(recall_exact, recall_zero) << "nprobe=" << requested << " vs nprobe=0";
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllIndexTypes,
    IndexPropertiesTest,
    ::testing::Values("pdx_f32", "pdx_u8", "pdx_tree_f32", "pdx_tree_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

// --- Edge Case Tests ---

class EdgeCaseTest : public ::testing::TestWithParam<std::string> {};

TEST_P(EdgeCaseTest, SinglePointIndex) {
    std::string index_type = GetParam();
    constexpr size_t d = 128;

    auto all_data = skmeans::MakeBlobs(1, d, 1, true, 5.0f, 2.0f, 42);

    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .num_clusters = 1,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    auto index = TestUtils::BuildIndexWithConfig(index_type, config, all_data.data(), 1);
    ASSERT_NE(index, nullptr);

    EXPECT_EQ(index->GetNumClusters(), 1u);
    EXPECT_EQ(index->GetClusterSize(0), 1u);

    index->SetNProbe(0);
    auto results = index->Search(all_data.data(), 10);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].index, 0u);
}

TEST_P(EdgeCaseTest, TwoPointIndex) {
    std::string index_type = GetParam();
    constexpr size_t d = 128;

    auto all_data = skmeans::MakeBlobs(2, d, 2, true, 5.0f, 2.0f, 42);

    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .num_clusters = 1,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    auto index = TestUtils::BuildIndexWithConfig(index_type, config, all_data.data(), 2);
    ASSERT_NE(index, nullptr);

    index->SetNProbe(0);
    // Use first point as query
    auto results = index->Search(all_data.data(), 2);
    EXPECT_EQ(results.size(), 2u);

    // Results should be sorted by distance
    if (results.size() == 2) {
        EXPECT_LE(results[0].distance, results[1].distance + 1e-6f);
    }
}

TEST_P(EdgeCaseTest, NumClustersExceedsNumPointsThrows) {
    std::string index_type = GetParam();
    constexpr size_t d = 128;
    constexpr size_t n = 10;

    auto all_data = skmeans::MakeBlobs(n, d, 5, true, 5.0f, 2.0f, 42);

    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .num_clusters = 100,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };

    EXPECT_THROW(
        TestUtils::BuildIndexWithConfig(index_type, config, all_data.data(), n),
        std::invalid_argument
    );
}

INSTANTIATE_TEST_SUITE_P(
    EdgeCases,
    EdgeCaseTest,
    ::testing::Values("pdx_f32", "pdx_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

// --- Config Validation Tests ---

TEST(ConfigValidationTest, ZeroDimensionsThrows) {
    PDX::PDXIndexConfig config{.num_dimensions = 0};
    EXPECT_THROW(([&]() { auto idx = PDX::PDXIndexF32(config); }()), std::invalid_argument);
}

TEST(ConfigValidationTest, DimensionsExceedsMaxThrows) {
    PDX::PDXIndexConfig config{.num_dimensions = static_cast<uint32_t>(PDX::PDX_MAX_DIMS + 1)};
    EXPECT_THROW(([&]() { auto idx = PDX::PDXIndexF32(config); }()), std::invalid_argument);
}

TEST(ConfigValidationTest, InvalidSamplingFractionThrows) {
    PDX::PDXIndexConfig config_neg{.num_dimensions = 128, .sampling_fraction = -0.1f};
    EXPECT_THROW(([&]() { auto idx = PDX::PDXIndexF32(config_neg); }()), std::invalid_argument);

    PDX::PDXIndexConfig config_over{.num_dimensions = 128, .sampling_fraction = 1.5f};
    EXPECT_THROW(([&]() { auto idx = PDX::PDXIndexF32(config_over); }()), std::invalid_argument);
}

TEST(ConfigValidationTest, MesoClustersNotSmallerThanClustersThrows) {
    PDX::PDXIndexConfig config{
        .num_dimensions = 128,
        .num_clusters = 10,
        .num_meso_clusters = 10,
    };
    EXPECT_THROW(([&]() { auto idx = PDX::PDXIndexF32(config); }()), std::invalid_argument);

    PDX::PDXIndexConfig config2{
        .num_dimensions = 128,
        .num_clusters = 10,
        .num_meso_clusters = 15,
    };
    EXPECT_THROW(([&]() { auto idx = PDX::PDXIndexF32(config2); }()), std::invalid_argument);
}

TEST(ConfigValidationTest, ZeroKmeansItersThrows) {
    PDX::PDXIndexConfig config{.num_dimensions = 128, .kmeans_iters = 0};
    EXPECT_THROW(([&]() { auto idx = PDX::PDXIndexF32(config); }()), std::invalid_argument);
}

TEST(ConfigValidationTest, KmeansIters100OrMoreThrows) {
    PDX::PDXIndexConfig config{.num_dimensions = 128, .kmeans_iters = 100};
    EXPECT_THROW(([&]() { auto idx = PDX::PDXIndexF32(config); }()), std::invalid_argument);
}

TEST(ConfigValidationTest, ValidConfigDoesNotThrow) {
    PDX::PDXIndexConfig config{
        .num_dimensions = 128,
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = 42,
        .num_clusters = 10,
        .num_meso_clusters = 3,
        .normalize = true,
        .sampling_fraction = 0.5f,
        .kmeans_iters = 10,
        .hierarchical_indexing = true,
    };
    EXPECT_NO_THROW(([&]() { auto idx = PDX::PDXIndexF32(config); }()));
}

} // namespace

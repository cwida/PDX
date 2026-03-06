#undef HAS_FFTW

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "test_utils.hpp"

namespace {

class FilteredSearchTest : public ::testing::TestWithParam<std::string> {
  protected:
    void SetUp() override {}
};

TEST_P(FilteredSearchTest, FilteredResultsAreSubsetOfPassingIds) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    index->SetNProbe(16);

    // Random 50% filter
    std::mt19937 rng(42);
    std::vector<size_t> passing_ids;
    for (size_t i = 0; i < TestUtils::N_TRAIN; ++i) {
        if (rng() % 2 == 0) {
            passing_ids.push_back(i);
        }
    }
    std::unordered_set<uint32_t> passing_set(passing_ids.begin(), passing_ids.end());

    for (size_t q = 0; q < 50; ++q) {
        auto results =
            index->FilteredSearch(data.queries.data() + q * d, TestUtils::KNN, passing_ids);
        for (const auto& r : results) {
            EXPECT_TRUE(passing_set.count(r.index))
                << "Result ID " << r.index << " not in passing set (query " << q << ")";
        }
    }
}

TEST_P(FilteredSearchTest, FilteredSearchMatchesUnfilteredWhenAllPass) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    index->SetNProbe(16);

    // All IDs pass
    std::vector<size_t> all_ids(TestUtils::N_TRAIN);
    std::iota(all_ids.begin(), all_ids.end(), 0);

    for (size_t q = 0; q < 20; ++q) {
        auto unfiltered = index->Search(data.queries.data() + q * d, TestUtils::KNN);
        auto filtered = index->FilteredSearch(data.queries.data() + q * d, TestUtils::KNN, all_ids);

        ASSERT_EQ(unfiltered.size(), filtered.size()) << "Size mismatch for query " << q;
        for (size_t i = 0; i < unfiltered.size(); ++i) {
            EXPECT_EQ(unfiltered[i].index, filtered[i].index)
                << "ID mismatch at query " << q << " position " << i;
        }
    }
}

TEST_P(FilteredSearchTest, EmptyFilterReturnsEmpty) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    index->SetNProbe(16);

    std::vector<size_t> empty_ids;
    auto results = index->FilteredSearch(data.queries.data(), TestUtils::KNN, empty_ids);
    EXPECT_TRUE(results.empty());
}

TEST_P(FilteredSearchTest, FilteredRecallMonotonicallyIncreasesWithNProbe) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);

    // 50% random filter
    std::mt19937 rng(42);
    std::vector<size_t> passing_ids;
    for (size_t i = 0; i < TestUtils::N_TRAIN; ++i) {
        if (rng() % 2 == 0) {
            passing_ids.push_back(i);
        }
    }

    // Brute-force GT on filtered subset
    std::vector<float> filtered_train(passing_ids.size() * d);
    for (size_t i = 0; i < passing_ids.size(); ++i) {
        std::memcpy(&filtered_train[i * d], &data.train[passing_ids[i] * d], d * sizeof(float));
    }
    auto gt = TestUtils::ComputeBruteForceKNN(
        filtered_train.data(),
        data.queries.data(),
        passing_ids.size(),
        TestUtils::N_QUERIES,
        d,
        TestUtils::KNN
    );
    // Map GT indices back to original IDs
    for (auto& query_gt : gt.indices) {
        for (auto& idx : query_gt) {
            idx = static_cast<uint32_t>(passing_ids[idx]);
        }
    }

    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);

    uint32_t num_clusters = index->GetNumClusters();
    std::vector<size_t> nprobes = {1, 4, 16, 64};
    nprobes.push_back(num_clusters);
    nprobes.erase(
        std::remove_if(
            nprobes.begin(), nprobes.end(), [num_clusters](size_t p) { return p > num_clusters; }
        ),
        nprobes.end()
    );
    std::sort(nprobes.begin(), nprobes.end());
    nprobes.erase(std::unique(nprobes.begin(), nprobes.end()), nprobes.end());

    float prev_recall = 0.0f;
    for (size_t nprobe : nprobes) {
        index->SetNProbe(nprobe);
        float total_recall = 0.0f;
        for (size_t q = 0; q < TestUtils::N_QUERIES; ++q) {
            auto results =
                index->FilteredSearch(data.queries.data() + q * d, TestUtils::KNN, passing_ids);
            total_recall += TestUtils::ComputeRecall(results, gt.indices[q], TestUtils::KNN);
        }
        float recall = total_recall / static_cast<float>(TestUtils::N_QUERIES);

        EXPECT_GE(recall, prev_recall - 0.01f) << "Filtered recall decreased from " << prev_recall
                                               << " to " << recall << " at nprobe=" << nprobe;
        prev_recall = recall;
    }
}

TEST_P(FilteredSearchTest, SingleIdFilterReturnsTheId) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    index->SetNProbe(0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, TestUtils::N_TRAIN - 1);

    for (size_t q = 0; q < 50; ++q) {
        size_t target_id = dist(rng);
        std::vector<size_t> passing_ids = {target_id};
        auto results = index->FilteredSearch(data.queries.data() + q * d, 20, passing_ids);
        ASSERT_EQ(results.size(), 1u) << "Expected exactly 1 result for query " << q;
        EXPECT_EQ(results[0].index, static_cast<uint32_t>(target_id)) << "Wrong ID for query " << q;
    }
}

TEST_P(FilteredSearchTest, TinyFilterExhaustiveReturnsAllPassingIds) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    index->SetNProbe(0);

    std::mt19937 rng(42);
    std::vector<size_t> passing_ids;
    std::uniform_int_distribution<size_t> dist(0, TestUtils::N_TRAIN - 1);
    std::unordered_set<size_t> seen;
    while (passing_ids.size() < 20) {
        size_t id = dist(rng);
        if (seen.insert(id).second) {
            passing_ids.push_back(id);
        }
    }
    std::unordered_set<uint32_t> passing_set(passing_ids.begin(), passing_ids.end());

    for (size_t q = 0; q < 50; ++q) {
        auto results = index->FilteredSearch(data.queries.data() + q * d, 20, passing_ids);
        EXPECT_EQ(results.size(), 20u) << "Expected all 20 passing IDs for query " << q;
        for (const auto& r : results) {
            EXPECT_TRUE(passing_set.count(r.index))
                << "Result ID " << r.index << " not in passing set (query " << q << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllIndexTypes,
    FilteredSearchTest,
    // TODO: add tree indexes once crash is fixed
    ::testing::Values("pdx_f32", "pdx_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

} // namespace

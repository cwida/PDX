#undef HAS_FFTW

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "test_utils.hpp"

namespace {

static constexpr size_t D = 128;
static constexpr size_t N_QUERIES = 20;

template <typename T>
struct Tag {
    using type = T;
};

template <typename F>
void ForIndexType(const std::string& index_type, F&& f) {
    if (index_type == "pdx_f32") {
        f(Tag<PDX::PDXIndexF32>{});
    } else if (index_type == "pdx_u8") {
        f(Tag<PDX::PDXIndexU8>{});
    } else if (index_type == "pdx_tree_f32") {
        f(Tag<PDX::PDXTreeIndexF32>{});
    } else if (index_type == "pdx_tree_u8") {
        f(Tag<PDX::PDXTreeIndexU8>{});
    } else {
        FAIL() << "Unknown index type: " << index_type;
    }
}

std::vector<PDX::KNNCandidate> Drain(
    PDX::IIterativeSearch& search_cursor,
    PDX::TopKHeap& top_k_heap,
    uint32_t k,
    size_t chunk
) {
    while (!search_cursor.Done()) {
        EXPECT_GT(search_cursor.Next(chunk), 0u);
    }
    EXPECT_EQ(search_cursor.Next(chunk), 0u);
    EXPECT_EQ(search_cursor.ClustersRemaining(), 0u);
    return PDX::BuildResultSetFromHeap(k, top_k_heap.heap);
}

std::unordered_set<uint32_t> Ids(const std::vector<PDX::KNNCandidate>& results) {
    std::unordered_set<uint32_t> ids;
    for (const auto& r : results) {
        ids.insert(r.index);
    }
    return ids;
}

void ExpectSameResults(
    const std::vector<PDX::KNNCandidate>& expected,
    const std::vector<PDX::KNNCandidate>& actual
) {
    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_EQ(Ids(expected), Ids(actual));
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(expected[i].distance, actual[i].distance) << "position " << i;
    }
}

std::vector<size_t> EveryThirdId() {
    std::vector<size_t> ids;
    for (size_t i = 0; i < TestUtils::N_TRAIN; i += 3) {
        ids.push_back(i);
    }
    return ids;
}

class IterativeSearchTest : public ::testing::TestWithParam<std::string> {};

TEST_P(IterativeSearchTest, ChunkedExhaustiveMatchesSingleShot) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    index->SetNProbe(0);
    for (size_t q = 0; q < N_QUERIES; ++q) {
        const float* query = data.queries.data() + q * D;
        auto expected = index->Search(query, TestUtils::KNN);
        PDX::TopKHeap top_k_heap;
        auto search_cursor =
            index->BeginIterativeSearch(query, TestUtils::KNN, top_k_heap, nullptr);
        ExpectSameResults(expected, Drain(*search_cursor, top_k_heap, TestUtils::KNN, 7));
    }
}

TEST_P(IterativeSearchTest, NextAccountsForEveryNonEmptyCluster) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    size_t non_empty = 0;
    for (uint32_t c = 0; c < index->GetNumClusters(); c++) {
        non_empty += index->GetClusterSize(c) > 0;
    }

    PDX::TopKHeap top_k_heap;
    auto search_cursor =
        index->BeginIterativeSearch(data.queries.data(), TestUtils::KNN, top_k_heap, nullptr);
    EXPECT_EQ(search_cursor->ClustersRemaining(), non_empty);
    EXPECT_EQ(search_cursor->Next(0), 0u);
    EXPECT_FALSE(search_cursor->Done());

    size_t total = 0;
    while (!search_cursor->Done()) {
        const size_t probed = search_cursor->Next(3);
        EXPECT_GE(probed, 1u);
        EXPECT_LE(probed, 3u);
        total += probed;
        EXPECT_EQ(search_cursor->ClustersRemaining(), non_empty - total);
    }
    EXPECT_EQ(total, non_empty);
    EXPECT_EQ(search_cursor->Next(3), 0u);
}

TEST_P(IterativeSearchTest, FilteredChunkedMatchesSingleShot) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    index->SetNProbe(0);
    const auto passing = EveryThirdId();
    const std::unordered_set<uint32_t> passing_set(passing.begin(), passing.end());
    for (size_t q = 0; q < N_QUERIES; ++q) {
        const float* query = data.queries.data() + q * D;
        auto expected = index->FilteredSearch(query, TestUtils::KNN, passing);
        PDX::TopKHeap top_k_heap;
        auto search_cursor =
            index->BeginIterativeSearch(query, TestUtils::KNN, top_k_heap, &passing);
        auto actual = Drain(*search_cursor, top_k_heap, TestUtils::KNN, 5);
        ExpectSameResults(expected, actual);
        for (const auto& r : actual) {
            EXPECT_TRUE(passing_set.count(r.index)) << "row_id " << r.index << " did not pass";
        }
    }
}

TEST_P(IterativeSearchTest, FilteredQueueOnlyHoldsClustersWithPassingTuples) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    const std::vector<size_t> passing = {1234};
    PDX::TopKHeap top_k_heap;
    auto search_cursor =
        index->BeginIterativeSearch(data.queries.data(), TestUtils::KNN, top_k_heap, &passing);
    EXPECT_EQ(search_cursor->ClustersRemaining(), 1u);
    auto results = Drain(*search_cursor, top_k_heap, TestUtils::KNN, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].index, 1234u);
}

TEST_P(IterativeSearchTest, UnderfilledHeapReturnsAllPassingTuples) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    std::mt19937 rng(TestUtils::SEED);
    std::uniform_int_distribution<size_t> dist(0, TestUtils::N_TRAIN - 1);
    std::unordered_set<size_t> chosen;
    while (chosen.size() < 10) {
        chosen.insert(dist(rng));
    }
    const std::vector<size_t> passing(chosen.begin(), chosen.end());
    const uint32_t k = 50;

    for (size_t q = 0; q < N_QUERIES; ++q) {
        PDX::TopKHeap top_k_heap;
        auto search_cursor =
            index->BeginIterativeSearch(data.queries.data() + q * D, k, top_k_heap, &passing);
        auto results = Drain(*search_cursor, top_k_heap, k, 4);
        EXPECT_EQ(Ids(results), std::unordered_set<uint32_t>(passing.begin(), passing.end()));
        for (size_t i = 1; i < results.size(); ++i) {
            EXPECT_LE(results[i - 1].distance, results[i].distance);
        }
    }
}

TEST_P(IterativeSearchTest, ChunkSizeDoesNotChangeResults) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    const float* query = data.queries.data();

    PDX::TopKHeap baseline_top_k_heap;
    auto baseline_search_cursor =
        index->BeginIterativeSearch(query, TestUtils::KNN, baseline_top_k_heap, nullptr);
    auto baseline = Drain(*baseline_search_cursor, baseline_top_k_heap, TestUtils::KNN, 1);

    for (size_t chunk : {2ul, 5ul, 64ul, static_cast<size_t>(index->GetNumClusters())}) {
        PDX::TopKHeap top_k_heap;
        auto search_cursor =
            index->BeginIterativeSearch(query, TestUtils::KNN, top_k_heap, nullptr);
        ExpectSameResults(baseline, Drain(*search_cursor, top_k_heap, TestUtils::KNN, chunk));
    }
}

TEST_P(IterativeSearchTest, ConcurrentCursorsOnOneIndex) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    index->SetNProbe(0);
    constexpr size_t N_THREADS = 4;
    constexpr size_t QUERIES_PER_THREAD = 5;

    std::vector<std::vector<PDX::KNNCandidate>> expected(N_THREADS * QUERIES_PER_THREAD);
    for (size_t q = 0; q < expected.size(); ++q) {
        expected[q] = index->Search(data.queries.data() + q * D, TestUtils::KNN);
    }

    std::vector<std::vector<PDX::KNNCandidate>> actual(expected.size());
    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (size_t t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < QUERIES_PER_THREAD; ++i) {
                const size_t q = t * QUERIES_PER_THREAD + i;
                PDX::TopKHeap top_k_heap;
                auto search_cursor = index->BeginIterativeSearch(
                    data.queries.data() + q * D, TestUtils::KNN, top_k_heap, nullptr
                );
                while (!search_cursor->Done()) {
                    search_cursor->Next(3);
                }
                actual[q] = PDX::BuildResultSetFromHeap(TestUtils::KNN, top_k_heap.heap);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (size_t q = 0; q < expected.size(); ++q) {
        ExpectSameResults(expected[q], actual[q]);
    }
}

// Four partitions of the data, one index each, one shared heap: the DuckDB row-group setting
template <typename IndexT>
void RunSharedHeapAcrossPartitions() {
    auto data = TestUtils::LoadTestData(D);
    constexpr size_t N_PARTS = 4;
    const size_t part_size = TestUtils::N_TRAIN / N_PARTS;

    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(D),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    std::vector<std::unique_ptr<IndexT>> parts;
    parts.reserve(N_PARTS);
    for (size_t p = 0; p < N_PARTS; ++p) {
        std::vector<size_t> row_ids(part_size);
        std::iota(row_ids.begin(), row_ids.end(), p * part_size);
        parts.push_back(std::make_unique<IndexT>(config));
        parts.back()->BuildIndex(row_ids.data(), data.train.data() + p * part_size * D, part_size);
    }

    // The indexes normalize, so the brute-force ground truth must too
    std::vector<float> normalized_train = data.train;
    std::vector<float> normalized_queries = data.queries;
    PDX::Quantizer normalizer(D);
    for (size_t i = 0; i < TestUtils::N_TRAIN; ++i) {
        normalizer.NormalizeQuery(&normalized_train[i * D], &normalized_train[i * D]);
    }
    for (size_t i = 0; i < N_QUERIES; ++i) {
        normalizer.NormalizeQuery(&normalized_queries[i * D], &normalized_queries[i * D]);
    }
    auto gt = TestUtils::ComputeBruteForceKNN(
        normalized_train.data(),
        normalized_queries.data(),
        TestUtils::N_TRAIN,
        N_QUERIES,
        D,
        TestUtils::KNN
    );

    float total_recall = 0.0f;
    for (size_t q = 0; q < N_QUERIES; ++q) {
        const float* query = data.queries.data() + q * D;
        PDX::TopKHeap top_k_heap(/*thread_safe=*/true);
        std::vector<std::unique_ptr<PDX::IIterativeSearch>> search_cursors;
        search_cursors.reserve(parts.size());
        for (auto& part : parts) {
            search_cursors.push_back(
                part->BeginIterativeSearch(query, TestUtils::KNN, top_k_heap, nullptr)
            );
        }
        std::vector<std::thread> threads;
        threads.reserve(search_cursors.size());
        for (auto& search_cursor : search_cursors) {
            threads.emplace_back([&search_cursor]() {
                while (!search_cursor->Done()) {
                    search_cursor->Next(3);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        auto results = PDX::BuildResultSetFromHeap(TestUtils::KNN, top_k_heap.heap);
        ASSERT_EQ(results.size(), TestUtils::KNN);
        total_recall += TestUtils::ComputeRecall(results, gt.indices[q], TestUtils::KNN);
    }
    EXPECT_GE(total_recall / static_cast<float>(N_QUERIES), 0.9f);
}

TEST_P(IterativeSearchTest, SharedHeapAcrossPartitionsMatchesBruteForce) {
    ForIndexType(GetParam(), [](auto tag) {
        RunSharedHeapAcrossPartitions<typename decltype(tag)::type>();
    });
}

static constexpr size_t SPLIT_BUDGET[] = {32, 8, 8, 16};
static constexpr size_t SPLIT_BUDGET_TOTAL = 64;

// Single-shot FilteredSearch skips clusters without passing tuples but still counts them against
// nprobe; the cursor never queues them. With every third id passing, every cluster has passing
// tuples, so both probe the same 64 clusters in the same order.
TEST_P(IterativeSearchTest, FilteredSplitProbeBudgetMatchesSingleShotNProbe) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    const auto passing = EveryThirdId();
    const std::unordered_set<uint32_t> passing_set(passing.begin(), passing.end());
    ASSERT_GE(index->GetNumClusters(), SPLIT_BUDGET_TOTAL);
    for (uint32_t c = 0; c < index->GetNumClusters(); c++) {
        const auto row_ids = index->GetClusterRowIds(c);
        ASSERT_TRUE(std::any_of(
            row_ids.begin(), row_ids.end(), [&](uint32_t id) { return passing_set.count(id) > 0; }
        )) << "cluster "
           << c << " has no passing tuple";
    }
    index->SetNProbe(SPLIT_BUDGET_TOTAL);

    for (size_t q = 0; q < N_QUERIES; ++q) {
        const float* query = data.queries.data() + q * D;
        auto expected = index->FilteredSearch(query, TestUtils::KNN, passing);
        PDX::TopKHeap top_k_heap;
        auto search_cursor =
            index->BeginIterativeSearch(query, TestUtils::KNN, top_k_heap, &passing);
        for (size_t step : SPLIT_BUDGET) {
            EXPECT_EQ(search_cursor->Next(step), step);
        }
        EXPECT_EQ(search_cursor->ClustersRemaining(), index->GetNumClusters() - SPLIT_BUDGET_TOTAL);
        ExpectSameResults(expected, PDX::BuildResultSetFromHeap(TestUtils::KNN, top_k_heap.heap));
    }
}

// Search runs on a cursor too: with nprobe 1 it must probe exactly one whole cluster (for the
// tree, the one its meso-cluster layer ranked first) and return only rows from it.
TEST_P(IterativeSearchTest, NProbeOneProbesExactlyOneCluster) {
    ForIndexType(GetParam(), [](auto tag) {
        using IndexType = typename decltype(tag)::type;
        auto data = TestUtils::LoadTestData(D);
        auto base = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
        auto& index = dynamic_cast<IndexType&>(*base);
        std::vector<uint32_t> cluster_of_row(TestUtils::N_TRAIN);
        for (uint32_t c = 0; c < index.GetNumClusters(); c++) {
            for (uint32_t row_id : index.GetClusterRowIds(c)) {
                cluster_of_row[row_id] = c;
            }
        }
        index.SetNProbe(1);
        for (size_t q = 0; q < N_QUERIES; ++q) {
            const size_t accessed_before = index.GetNumVectorsAccessed();
            auto results = index.Search(data.queries.data() + q * D, TestUtils::KNN);
            ASSERT_FALSE(results.empty());
            const uint32_t probed = cluster_of_row[results[0].index];
            for (const auto& r : results) {
                EXPECT_EQ(cluster_of_row[r.index], probed);
            }
            EXPECT_EQ(
                results.size(), std::min<size_t>(TestUtils::KNN, index.GetClusterSize(probed))
            );
            EXPECT_EQ(
                index.GetNumVectorsAccessed() - accessed_before, index.GetClusterSize(probed)
            );
        }
    });
}

INSTANTIATE_TEST_SUITE_P(
    AllIndexTypes,
    IterativeSearchTest,
    ::testing::Values("pdx_f32", "pdx_u8", "pdx_tree_f32", "pdx_tree_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

// The tree's single-shot Search ranks clusters through the meso-cluster layer, the cursor through
// the flat centroid ranking, so this equality only holds for the vanilla indexes.
class VanillaIterativeSearchTest : public ::testing::TestWithParam<std::string> {};

TEST_P(VanillaIterativeSearchTest, SplitProbeBudgetMatchesSingleShotNProbe) {
    auto data = TestUtils::LoadTestData(D);
    auto index = TestUtils::BuildIndex(GetParam(), data.train.data(), TestUtils::N_TRAIN, D);
    ASSERT_GE(index->GetNumClusters(), SPLIT_BUDGET_TOTAL);
    for (uint32_t c = 0; c < index->GetNumClusters(); c++) {
        ASSERT_GT(index->GetClusterSize(c), 0u)
            << "single-shot Search counts empty clusters against nprobe, the cursor does not";
    }
    index->SetNProbe(SPLIT_BUDGET_TOTAL);

    for (size_t q = 0; q < N_QUERIES; ++q) {
        const float* query = data.queries.data() + q * D;
        auto expected = index->Search(query, TestUtils::KNN);
        PDX::TopKHeap top_k_heap;
        auto search_cursor =
            index->BeginIterativeSearch(query, TestUtils::KNN, top_k_heap, nullptr);
        for (size_t step : SPLIT_BUDGET) {
            EXPECT_EQ(search_cursor->Next(step), step);
        }
        EXPECT_EQ(search_cursor->ClustersRemaining(), index->GetNumClusters() - SPLIT_BUDGET_TOTAL);
        ExpectSameResults(expected, PDX::BuildResultSetFromHeap(TestUtils::KNN, top_k_heap.heap));
    }
}

INSTANTIATE_TEST_SUITE_P(
    VanillaIndexTypes,
    VanillaIterativeSearchTest,
    ::testing::Values("pdx_f32", "pdx_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

} // namespace

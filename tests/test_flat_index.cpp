#undef HAS_FFTW

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "pdx/indexes/flat.hpp"
#include "test_utils.hpp"

namespace {

static constexpr size_t D = 128;

PDX::PDXIndexConfig MakeConfig() {
    return PDX::PDXIndexConfig{
        .num_dimensions = static_cast<uint32_t>(D),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
    };
}

std::vector<uint32_t> Ids(const std::vector<PDX::KNNCandidate>& results) {
    std::vector<uint32_t> ids;
    ids.reserve(results.size());
    for (const auto& r : results) {
        ids.push_back(r.index);
    }
    return ids;
}

TEST(FlatIndex, SearchIsExact) {
    auto data = TestUtils::LoadTestData(D);
    PDX::FlatIndex index(MakeConfig());
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    EXPECT_EQ(index.GetNumClusters(), 1u);
    EXPECT_EQ(index.GetClusterSize(0), TestUtils::N_TRAIN);

    auto gt = TestUtils::ComputeBruteForceKNN(
        data.train.data(),
        data.queries.data(),
        TestUtils::N_TRAIN,
        TestUtils::N_QUERIES,
        D,
        TestUtils::KNN
    );
    const float recall = TestUtils::ComputeAverageRecall(
        index, data.queries.data(), TestUtils::N_QUERIES, D, TestUtils::KNN, gt
    );
    EXPECT_GE(recall, 0.999f);
}

TEST(FlatIndex, CursorMatchesSearch) {
    auto data = TestUtils::LoadTestData(D);
    PDX::FlatIndex index(MakeConfig());
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);

    for (size_t q = 0; q < 20; q++) {
        const float* query = data.queries.data() + q * D;
        PDX::TopKHeap top_k_heap;
        auto cursor = index.BeginIterativeSearch(query, TestUtils::KNN, top_k_heap, nullptr);
        EXPECT_FALSE(cursor->Done());
        EXPECT_EQ(cursor->ClustersRemaining(), 1u);
        EXPECT_EQ(cursor->Next(5), 1u);
        EXPECT_TRUE(cursor->Done());
        EXPECT_EQ(cursor->Next(1), 0u);
        auto cursor_results = PDX::BuildResultSetFromHeap(TestUtils::KNN, top_k_heap.heap);
        EXPECT_EQ(Ids(cursor_results), Ids(index.Search(query, TestUtils::KNN)));
    }
}

TEST(FlatIndex, FilteredSearchOnlyReturnsPassingRowIds) {
    auto data = TestUtils::LoadTestData(D);
    PDX::FlatIndex index(MakeConfig());
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);

    std::vector<size_t> passing;
    std::vector<float> passing_train;
    for (size_t i = 0; i < TestUtils::N_TRAIN; i += 2) {
        passing.push_back(i);
        passing_train.insert(
            passing_train.end(), data.train.begin() + i * D, data.train.begin() + (i + 1) * D
        );
    }
    auto gt = TestUtils::ComputeBruteForceKNN(
        passing_train.data(),
        data.queries.data(),
        passing.size(),
        TestUtils::N_QUERIES,
        D,
        TestUtils::KNN
    );

    float total_recall = 0.0f;
    for (size_t q = 0; q < TestUtils::N_QUERIES; q++) {
        auto results = index.FilteredSearch(data.queries.data() + q * D, TestUtils::KNN, passing);
        ASSERT_EQ(results.size(), TestUtils::KNN);
        for (const auto& r : results) {
            EXPECT_EQ(r.index % 2, 0u);
        }
        std::vector<PDX::KNNCandidate> in_subset_ids = results;
        for (auto& r : in_subset_ids) {
            r.index /= 2;
        }
        total_recall += TestUtils::ComputeRecall(in_subset_ids, gt.indices[q], TestUtils::KNN);
    }
    EXPECT_GE(total_recall / static_cast<float>(TestUtils::N_QUERIES), 0.999f);
}

TEST(FlatIndex, AppendDeleteAndReappend) {
    auto data = TestUtils::LoadTestData(D);
    const size_t last = TestUtils::N_TRAIN - 1;
    const float* last_embedding = data.train.data() + last * D;

    PDX::FlatIndex index(MakeConfig());
    index.BuildIndex(data.train.data(), last);
    EXPECT_EQ(index.GetRowIdMapping(last).first, PDX::DELETED_MARKER);

    index.Append(last, last_embedding);
    auto results = index.Search(last_embedding, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].index, last);
    EXPECT_EQ(index.GetClusterSize(0), TestUtils::N_TRAIN);
    EXPECT_THROW(index.Append(last, last_embedding), std::invalid_argument);

    index.Delete(last);
    EXPECT_NE(index.Search(last_embedding, 1)[0].index, last);
    EXPECT_EQ(index.GetClusterSize(0), last);
    EXPECT_NO_THROW(index.Delete(last));
    EXPECT_NO_THROW(index.Delete(TestUtils::N_TRAIN + 12345));

    index.Append(last, last_embedding);
    EXPECT_EQ(index.Search(last_embedding, 1)[0].index, last);
    EXPECT_EQ(index.GetRowIdMapping(last).first, 0u);
}

TEST(FlatIndex, TransformedInputAndPromotionToIVF) {
    auto data = TestUtils::LoadTestData(D);
    PDX::ADSamplingPruner pruner(D, TestUtils::SEED);
    auto transformed =
        PDX::NormalizeAndRotate(data.train.data(), TestUtils::N_TRAIN, D, false, pruner);

    PDX::FlatIndex reference(MakeConfig());
    reference.BuildIndex(data.train.data(), TestUtils::N_TRAIN);

    auto config = MakeConfig();
    config.is_data_transformed = true;
    PDX::FlatIndex flat(config, pruner);
    flat.BuildIndex(transformed.get(), TestUtils::N_TRAIN);

    std::unique_ptr<float[]> transformed_query(new float[D]);
    for (size_t q = 0; q < 20; q++) {
        const float* query = data.queries.data() + q * D;
        pruner.PreprocessQuery(query, transformed_query.get());
        PDX::TopKHeap top_k_heap;
        flat.BeginIterativeSearch(
                transformed_query.get(), TestUtils::KNN, top_k_heap, nullptr, true
        )
            ->Next(1);
        auto results = PDX::BuildResultSetFromHeap(TestUtils::KNN, top_k_heap.heap);
        EXPECT_EQ(Ids(results), Ids(reference.Search(query, TestUtils::KNN)));
    }

    const auto row_ids = flat.GetRowIds();
    const auto embeddings = flat.GetEmbeddings();
    ASSERT_EQ(row_ids.size(), TestUtils::N_TRAIN);
    PDX::PDXIndex<PDX::F32> ivf(config, pruner);
    ivf.BuildIndex(row_ids.data(), embeddings.get(), row_ids.size());

    auto gt = TestUtils::ComputeBruteForceKNN(
        data.train.data(),
        data.queries.data(),
        TestUtils::N_TRAIN,
        TestUtils::N_QUERIES,
        D,
        TestUtils::KNN
    );
    const float recall = TestUtils::ComputeAverageRecall(
        ivf, data.queries.data(), TestUtils::N_QUERIES, D, TestUtils::KNN, gt
    );
    EXPECT_GE(recall, 0.99f);
}

} // namespace

#undef HAS_FFTW

#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <vector>

#include "pdx/indexes/flat.hpp"
#include "pdx/indexes/ivf_tree.hpp"
#include "test_utils.hpp"

namespace {

static constexpr size_t D = 128;

PDX::PDXIndexConfig MakeConfig() {
    return PDX::PDXIndexConfig{
        .num_dimensions = static_cast<uint32_t>(D),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
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

// Building from raw data, or from the same data transformed with the same rotation matrix and
// passed in with is_data_transformed, must give the same index
template <typename IndexT>
void RunTransformedBuildMatchesRawBuild() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_build = TestUtils::N_TRAIN - 1;
    const size_t last = TestUtils::N_TRAIN - 1;

    IndexT raw_index(MakeConfig());
    raw_index.BuildIndex(data.train.data(), n_build);

    PDX::ADSamplingPruner pruner(D, TestUtils::SEED);
    auto transformed = PDX::NormalizeAndRotate(data.train.data(), n_build, D, true, pruner);
    auto config = MakeConfig();
    config.is_data_transformed = true;
    IndexT transformed_index(config, pruner);
    transformed_index.BuildIndex(transformed.get(), n_build);

    for (size_t q = 0; q < 50; q++) {
        const float* query = data.queries.data() + q * D;
        EXPECT_EQ(
            Ids(raw_index.Search(query, TestUtils::KNN)),
            Ids(transformed_index.Search(query, TestUtils::KNN))
        );
    }

    auto transformed_last =
        PDX::NormalizeAndRotate(data.train.data() + last * D, 1, D, true, pruner);
    raw_index.Append(last, data.train.data() + last * D);
    transformed_index.Append(last, transformed_last.get());
    const float* query = data.train.data() + last * D;
    EXPECT_EQ(raw_index.Search(query, 1)[0].index, last);
    EXPECT_EQ(transformed_index.Search(query, 1)[0].index, last);
}

TEST(TransformedInput, PDXIndexF32) {
    RunTransformedBuildMatchesRawBuild<PDX::PDXIndexF32>();
}

TEST(TransformedInput, PDXIndexU8) {
    RunTransformedBuildMatchesRawBuild<PDX::PDXIndexU8>();
}

TEST(TransformedInput, PDXTreeIndexF32) {
    RunTransformedBuildMatchesRawBuild<PDX::PDXTreeIndexF32>();
}

// GetEmbeddingsFromIndexByRowIds gives back the transformed embeddings the index was built from
// (u8: within its quantization error), in the order of the requested row ids.
template <typename IndexT>
void RunEmbeddingsFromIndexByRowIdsComeBack(const float tolerance) {
    auto data = TestUtils::LoadTestData(D);
    PDX::ADSamplingPruner pruner(D, TestUtils::SEED);
    auto transformed =
        PDX::NormalizeAndRotate(data.train.data(), TestUtils::N_TRAIN, D, true, pruner);
    auto config = MakeConfig();
    config.is_data_transformed = true;
    IndexT index(config, pruner);
    index.BuildIndex(transformed.get(), TestUtils::N_TRAIN);

    std::vector<size_t> row_ids;
    for (size_t row_id = TestUtils::N_TRAIN; row_id-- > 0;) {
        if (row_id % 3 == 0) {
            row_ids.push_back(row_id);
        }
    }
    std::vector<float> embeddings(row_ids.size() * D);
    index.GetEmbeddingsFromIndexByRowIds(row_ids, embeddings.data());
    float max_error = 0.0f;
    for (size_t i = 0; i < row_ids.size(); i++) {
        for (size_t j = 0; j < D; j++) {
            max_error = std::max(
                max_error, std::abs(embeddings[i * D + j] - transformed[row_ids[i] * D + j])
            );
        }
    }
    EXPECT_LE(max_error, tolerance);
}

TEST(EmbeddingsFromIndexByRowIds, PDXIndexF32) {
    RunEmbeddingsFromIndexByRowIdsComeBack<PDX::PDXIndexF32>(0.0f);
}

TEST(EmbeddingsFromIndexByRowIds, PDXIndexU8) {
    RunEmbeddingsFromIndexByRowIdsComeBack<PDX::PDXIndexU8>(0.01f);
}

TEST(EmbeddingsFromIndexByRowIds, PDXTreeIndexF32) {
    RunEmbeddingsFromIndexByRowIdsComeBack<PDX::PDXTreeIndexF32>(0.0f);
}

TEST(EmbeddingsFromIndexByRowIds, FlatIndex) {
    RunEmbeddingsFromIndexByRowIdsComeBack<PDX::FlatIndex>(0.0f);
}

TEST(RowIdMapping, GrowsAndReusesIds) {
    auto data = TestUtils::LoadTestData(D);
    PDX::PDXIndexF32 index(MakeConfig());
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);

    const size_t far_row_id = TestUtils::N_TRAIN + 100000;
    const float* embedding = data.queries.data();
    EXPECT_EQ(index.GetRowIdMapping(far_row_id).first, PDX::DELETED_MARKER);
    index.Append(far_row_id, embedding);
    EXPECT_NE(index.GetRowIdMapping(far_row_id).first, PDX::DELETED_MARKER);
    EXPECT_EQ(index.Search(embedding, 1)[0].index, far_row_id);

    EXPECT_NO_THROW(index.Delete(far_row_id + 1));
    index.Delete(far_row_id);
    EXPECT_EQ(index.GetRowIdMapping(far_row_id).first, PDX::DELETED_MARKER);
    EXPECT_NO_THROW(index.Delete(far_row_id));

    index.Append(far_row_id, embedding);
    EXPECT_EQ(index.Search(embedding, 1)[0].index, far_row_id);
    EXPECT_THROW(index.Append(far_row_id, embedding), std::invalid_argument);
}

TEST(RowIdMapping, BaseRowIdKeepsTheMappingLocal) {
    auto data = TestUtils::LoadTestData(D);
    const size_t base_row_id = 3000000;
    std::vector<size_t> row_ids(TestUtils::N_TRAIN);
    std::iota(row_ids.begin(), row_ids.end(), base_row_id);

    PDX::PDXIndexF32 reference(MakeConfig());
    reference.BuildIndex(data.train.data(), TestUtils::N_TRAIN);

    auto config = MakeConfig();
    config.base_row_id = base_row_id;
    PDX::PDXIndexF32 index(config);
    index.BuildIndex(row_ids.data(), data.train.data(), TestUtils::N_TRAIN);

    EXPECT_EQ(index.GetInMemorySizeInBytes(), reference.GetInMemorySizeInBytes());
    for (size_t q = 0; q < 20; q++) {
        const float* query = data.queries.data() + q * D;
        auto expected = Ids(reference.Search(query, TestUtils::KNN));
        for (auto& id : expected) {
            id += base_row_id;
        }
        EXPECT_EQ(Ids(index.Search(query, TestUtils::KNN)), expected);
    }

    EXPECT_EQ(index.GetRowIdMapping(base_row_id - 1).first, PDX::DELETED_MARKER);
    EXPECT_NE(index.GetRowIdMapping(base_row_id).first, PDX::DELETED_MARKER);
    EXPECT_NO_THROW(index.Delete(base_row_id - 1));
    index.Delete(base_row_id);
    EXPECT_EQ(index.GetRowIdMapping(base_row_id).first, PDX::DELETED_MARKER);
    index.Append(base_row_id, data.train.data());
    EXPECT_EQ(index.Search(data.train.data(), 1)[0].index, base_row_id);
}

// An index on a shared pruner does not count the rotation matrix it does not own
TEST(InMemorySize, SharedPrunerIsNotCounted) {
    auto data = TestUtils::LoadTestData(D);
    PDX::PDXIndexF32 owned(MakeConfig());
    owned.BuildIndex(data.train.data(), TestUtils::N_TRAIN);

    PDX::ADSamplingPruner pruner(D, TestUtils::SEED);
    auto transformed =
        PDX::NormalizeAndRotate(data.train.data(), TestUtils::N_TRAIN, D, true, pruner);
    auto config = MakeConfig();
    config.is_data_transformed = true;
    PDX::PDXIndexF32 shared(config, pruner);
    shared.BuildIndex(transformed.get(), TestUtils::N_TRAIN);

    EXPECT_GE(
        owned.GetInMemorySizeInBytes() - shared.GetInMemorySizeInBytes(), D * D * sizeof(float)
    );
}

} // namespace

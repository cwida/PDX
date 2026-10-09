#undef HAS_FFTW

#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "test_utils.hpp"

namespace {

class SerializationTest : public ::testing::TestWithParam<std::string> {
  protected:
    void SetUp() override {}
};

TEST_P(SerializationTest, SaveLoadProducesSameSearchResults) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);

    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    ASSERT_NE(index, nullptr);
    index->SetNProbe(16);

    // Search before save
    std::vector<std::vector<PDX::KNNCandidate>> original_results;
    for (size_t q = 0; q < 50; ++q) {
        original_results.push_back(index->Search(data.queries.data() + q * d, TestUtils::KNN));
    }

    // Save and reload
    std::string path = TestUtils::TempPath("pdx_test_" + index_type);
    index->Save(path);
    auto loaded = PDX::LoadPDXIndex(path);
    ASSERT_NE(loaded, nullptr);
    loaded->SetNProbe(16);

    // Search after load
    for (size_t q = 0; q < 50; ++q) {
        auto loaded_results = loaded->Search(data.queries.data() + q * d, TestUtils::KNN);
        ASSERT_EQ(original_results[q].size(), loaded_results.size())
            << "Result count mismatch for query " << q;

        for (size_t i = 0; i < original_results[q].size(); ++i) {
            EXPECT_EQ(original_results[q][i].index, loaded_results[i].index)
                << "ID mismatch at query " << q << " position " << i;

            float rel_error =
                std::abs(original_results[q][i].distance - loaded_results[i].distance) /
                std::max(original_results[q][i].distance, 1e-6f);
            EXPECT_LT(rel_error, 1e-5f) << "Distance mismatch at query " << q << " position " << i;
        }
    }

    std::remove(path.c_str());
}

// Filtered search ranks clusters through the leaf centroids, which Search does not touch on the
// tree
TEST_P(SerializationTest, SaveLoadProducesSameFilteredSearchResults) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);

    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    index->SetNProbe(16);

    std::vector<size_t> passing_ids;
    for (size_t i = 0; i < TestUtils::N_TRAIN; i += 3) {
        passing_ids.push_back(i);
    }

    std::vector<std::vector<PDX::KNNCandidate>> original_results;
    for (size_t q = 0; q < 50; ++q) {
        original_results.push_back(
            index->FilteredSearch(data.queries.data() + q * d, TestUtils::KNN, passing_ids)
        );
    }

    std::string path = TestUtils::TempPath("pdx_test_filtered_" + index_type);
    index->Save(path);
    auto loaded = PDX::LoadPDXIndex(path);
    ASSERT_NE(loaded, nullptr);
    loaded->SetNProbe(16);

    for (size_t q = 0; q < 50; ++q) {
        auto loaded_results =
            loaded->FilteredSearch(data.queries.data() + q * d, TestUtils::KNN, passing_ids);
        ASSERT_EQ(original_results[q].size(), loaded_results.size())
            << "Result count mismatch for query " << q;
        for (size_t i = 0; i < original_results[q].size(); ++i) {
            EXPECT_EQ(original_results[q][i].index, loaded_results[i].index)
                << "ID mismatch at query " << q << " position " << i;
        }
    }

    std::remove(path.c_str());
}

TEST_P(SerializationTest, LoadedIndexProperties) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);

    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    uint32_t orig_dims = index->GetNumDimensions();
    uint32_t orig_clusters = index->GetNumClusters();
    size_t orig_mem = index->GetInMemorySizeInBytes();

    std::string path = TestUtils::TempPath("pdx_test_props_" + index_type);
    index->Save(path);
    auto loaded = PDX::LoadPDXIndex(path);

    EXPECT_EQ(loaded->GetNumDimensions(), orig_dims);
    EXPECT_EQ(loaded->GetNumClusters(), orig_clusters);

    float mem_ratio =
        static_cast<float>(loaded->GetInMemorySizeInBytes()) / static_cast<float>(orig_mem);
    EXPECT_GT(mem_ratio, 0.99f);
    EXPECT_LT(mem_ratio, 1.01f);

    std::remove(path.c_str());
}

TEST_P(SerializationTest, LoadAutoDetectsType) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);

    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    std::string path = TestUtils::TempPath("pdx_test_autodetect_" + index_type);
    index->Save(path);

    // LoadPDXIndex should auto-detect the type from the header byte
    auto loaded = PDX::LoadPDXIndex(path);
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->GetNumDimensions(), index->GetNumDimensions());
    EXPECT_EQ(loaded->GetNumClusters(), index->GetNumClusters());

    std::remove(path.c_str());
}

INSTANTIATE_TEST_SUITE_P(
    AllIndexTypes,
    SerializationTest,
    ::testing::Values("pdx_f32", "pdx_u8", "pdx_tree_f32", "pdx_tree_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

// On an external pruner, as indexes that share one rotation are built.
std::unique_ptr<PDX::IPDXIndex> BuildOnPruner(
    const std::string& index_type,
    const PDX::PDXIndexConfig& config,
    PDX::ADSamplingPruner& pruner
) {
    if (index_type == "pdx_f32") {
        return std::make_unique<PDX::PDXIndexF32>(config, pruner);
    }
    if (index_type == "pdx_u8") {
        return std::make_unique<PDX::PDXIndexU8>(config, pruner);
    }
    if (index_type == "pdx_tree_f32") {
        return std::make_unique<PDX::PDXTreeIndexF32>(config, pruner);
    }
    if (index_type == "pdx_tree_u8") {
        return std::make_unique<PDX::PDXTreeIndexU8>(config, pruner);
    }
    return std::make_unique<PDX::FlatIndex>(config, pruner);
}

void ExpectSameResults(
    const PDX::IPDXIndex& expected,
    const PDX::IPDXIndex& actual,
    const float* queries,
    size_t d,
    const std::vector<size_t>& passing_ids
) {
    for (size_t q = 0; q < 50; ++q) {
        const float* query = queries + q * d;
        const auto expected_results = expected.Search(query, TestUtils::KNN);
        const auto actual_results = actual.Search(query, TestUtils::KNN);
        ASSERT_EQ(expected_results.size(), actual_results.size()) << "query " << q;
        for (size_t i = 0; i < expected_results.size(); ++i) {
            EXPECT_EQ(expected_results[i].index, actual_results[i].index) << "query " << q;
        }
        const auto expected_filtered = expected.FilteredSearch(query, TestUtils::KNN, passing_ids);
        const auto actual_filtered = actual.FilteredSearch(query, TestUtils::KNN, passing_ids);
        ASSERT_EQ(expected_filtered.size(), actual_filtered.size()) << "filtered query " << q;
        for (size_t i = 0; i < expected_filtered.size(); ++i) {
            EXPECT_EQ(expected_filtered[i].index, actual_filtered[i].index)
                << "filtered query " << q;
        }
    }
}

class StreamSerializationTest : public ::testing::TestWithParam<std::string> {};

TEST_P(StreamSerializationTest, LoadsTheSameIndexOnTheSamePruner) {
    const std::string& index_type = GetParam();
    const size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    const PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    PDX::ADSamplingPruner pruner(static_cast<uint32_t>(d), TestUtils::SEED);
    auto index = BuildOnPruner(index_type, config, pruner);
    index->BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    index->SetNProbe(16);
    // The save compacts these tombstones away.
    for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; row_id += 7) {
        index->Delete(row_id);
    }

    std::stringstream stream;
    index->SaveToStream(stream);
    auto loaded = PDX::LoadPDXIndexFromStream(stream, pruner);
    ASSERT_NE(loaded, nullptr);
    loaded->SetNProbe(16);

    std::vector<size_t> passing_ids;
    for (size_t i = 0; i < TestUtils::N_TRAIN; i += 3) {
        passing_ids.push_back(i);
    }
    ExpectSameResults(*index, *loaded, data.queries.data(), d, passing_ids);

    // The loaded index keeps its config and row id mapping, so maintenance continues the same way.
    index->Append(TestUtils::N_TRAIN, data.queries.data());
    loaded->Append(TestUtils::N_TRAIN, data.queries.data());
    passing_ids.push_back(TestUtils::N_TRAIN);
    ExpectSameResults(*index, *loaded, data.queries.data(), d, passing_ids);
}

// Serves the clusters of a saved stream, each copied into a buffer of its own as a cache would.
class SavedStreamClusterSource : public PDX::IClusterSource {
  public:
    explicit SavedStreamClusterSource(std::string saved_stream)
        : saved_stream(std::move(saved_stream)) {}

    // The cluster data starts where loading the resident data stopped.
    void Attach(const PDX::IPDXIndex& loaded_index, size_t loaded_cluster_data_start) {
        index = &loaded_index;
        cluster_data_start = loaded_cluster_data_start;
    }

    const char* Acquire(uint32_t cluster_id) override {
        const auto [offset, size] = index->GetClusterDataRange(cluster_id);
        auto& buffer = acquired_clusters[cluster_id];
        buffer.resize((size + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        std::memcpy(buffer.data(), saved_stream.data() + cluster_data_start + offset, size);
        num_acquired++;
        return reinterpret_cast<const char*>(buffer.data());
    }

    void Release(uint32_t cluster_id) override {
        EXPECT_EQ(acquired_clusters.erase(cluster_id), 1u);
    }

    size_t num_acquired = 0;
    std::unordered_map<uint32_t, std::vector<uint32_t>> acquired_clusters;

  private:
    std::string saved_stream;
    const PDX::IPDXIndex* index = nullptr;
    size_t cluster_data_start = 0;
};

// Without its clusters' data, the index searches them through the cluster source, also after
// deletes
TEST_P(StreamSerializationTest, ResidentDataLoadSearchesThroughTheClusterSource) {
    const std::string& index_type = GetParam();
    const size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    const PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    PDX::ADSamplingPruner pruner(static_cast<uint32_t>(d), TestUtils::SEED);
    auto index = BuildOnPruner(index_type, config, pruner);
    index->BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    index->SetNProbe(16);

    std::stringstream stream;
    index->SaveToStream(stream);
    SavedStreamClusterSource source(stream.str());
    auto loaded = PDX::LoadPDXIndexFromStream(stream, pruner, &source);
    source.Attach(*loaded, static_cast<size_t>(stream.tellg()));
    loaded->SetNProbe(16);

    std::vector<size_t> passing_ids;
    for (size_t i = 0; i < TestUtils::N_TRAIN; i += 3) {
        passing_ids.push_back(i);
    }
    ExpectSameResults(*index, *loaded, data.queries.data(), d, passing_ids);

    // The gather reads the same embeddings through the cluster source.
    std::vector<float> expected_embeddings(passing_ids.size() * d);
    std::vector<float> actual_embeddings(passing_ids.size() * d);
    index->GetEmbeddingsFromIndexByRowIds(passing_ids, expected_embeddings.data());
    loaded->GetEmbeddingsFromIndexByRowIds(passing_ids, actual_embeddings.data());
    EXPECT_EQ(expected_embeddings, actual_embeddings);

    for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; row_id += 500) {
        index->Delete(row_id);
        loaded->Delete(row_id);
    }
    ExpectSameResults(*index, *loaded, data.queries.data(), d, passing_ids);

    // Flat and the tree load everything.
    const bool pages_clusters = index_type == "pdx_f32" || index_type == "pdx_u8";
    EXPECT_EQ(source.num_acquired > 0, pages_clusters);
    EXPECT_TRUE(source.acquired_clusters.empty());
    if (pages_clusters) {
        EXPECT_LT(loaded->GetInMemorySizeInBytes(), index->GetInMemorySizeInBytes() / 2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllIndexTypes,
    StreamSerializationTest,
    ::testing::Values("pdx_f32", "pdx_u8", "pdx_tree_f32", "pdx_tree_u8", "flat"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

TEST(FileSerialization, FlatSaveLoadProducesSameResults) {
    const size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    const PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .seed = TestUtils::SEED,
        .normalize = true,
    };
    PDX::FlatIndex index(config);
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; row_id += 7) {
        index.Delete(row_id);
    }

    const std::string path = TestUtils::TempPath("pdx_test_flat");
    index.Save(path);
    auto loaded = PDX::LoadPDXIndex(path);
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->GetNumDimensions(), index.GetNumDimensions());

    std::vector<size_t> passing_ids;
    for (size_t i = 0; i < TestUtils::N_TRAIN; i += 3) {
        passing_ids.push_back(i);
    }
    ExpectSameResults(index, *loaded, data.queries.data(), d, passing_ids);
    std::remove(path.c_str());
}

TEST(StreamSerialization, OtherVersionThrows) {
    const size_t d = 128;
    auto data = TestUtils::LoadTestData(d);
    const PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d), .seed = TestUtils::SEED
    };
    PDX::ADSamplingPruner pruner(static_cast<uint32_t>(d), TestUtils::SEED);
    PDX::FlatIndex index(config, pruner);
    index.BuildIndex(data.train.data(), 100);

    std::stringstream stream;
    index.SaveToStream(stream);
    auto bytes = stream.str();
    bytes[0] = static_cast<char>(PDX::PDX_SERIALIZATION_VERSION + 1);
    std::stringstream other_version(bytes);
    EXPECT_THROW(PDX::LoadPDXIndexFromStream(other_version, pruner), std::runtime_error);
}

} // namespace

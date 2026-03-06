#undef HAS_FFTW

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
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
    std::string path = "/tmp/pdx_test_" + index_type;
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

TEST_P(SerializationTest, LoadedIndexProperties) {
    std::string index_type = GetParam();
    size_t d = 128;
    auto data = TestUtils::LoadTestData(d);

    auto index = TestUtils::BuildIndex(index_type, data.train.data(), TestUtils::N_TRAIN, d);
    uint32_t orig_dims = index->GetNumDimensions();
    uint32_t orig_clusters = index->GetNumClusters();
    size_t orig_mem = index->GetInMemorySizeInBytes();

    std::string path = "/tmp/pdx_test_props_" + index_type;
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
    std::string path = "/tmp/pdx_test_autodetect_" + index_type;
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
    // TODO: add tree indexes once crash is fixed
    ::testing::Values("pdx_f32", "pdx_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

} // namespace

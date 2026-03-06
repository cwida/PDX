#undef HAS_FFTW

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "test_utils.hpp"

namespace {

static constexpr size_t D = 384;

template <typename IndexT>
IndexT BuildTreeIndex(const float* data, size_t n, size_t d) {
    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
    IndexT index(config);
    index.BuildIndex(data, n);
    return index;
}

// Test 1: Build with N-1 points, insert the last one, search for it
template <typename IndexT>
void RunInsertSingleAndSearch() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_build = TestUtils::N_TRAIN - 1;
    const size_t inserted_row_id = n_build;

    auto index = BuildTreeIndex<IndexT>(data.train.data(), n_build, D);
    index.Append(inserted_row_id, data.train.data() + inserted_row_id * D);
    index.SetNProbe(0);

    auto results = index.Search(data.train.data() + inserted_row_id * D, TestUtils::KNN);

    bool found = false;
    for (const auto& r : results) {
        if (r.index == static_cast<uint32_t>(inserted_row_id)) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Inserted point (row_id=" << inserted_row_id
                       << ") not found in search results";
}

// Test 2: Build with N-10 points, insert 10, filtered search should return all 10
template <typename IndexT>
void RunInsertMultipleAndFilteredSearch() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_insert = 10;
    const size_t n_build = TestUtils::N_TRAIN - n_insert;

    auto index = BuildTreeIndex<IndexT>(data.train.data(), n_build, D);

    std::vector<size_t> inserted_ids;
    for (size_t i = 0; i < n_insert; ++i) {
        size_t row_id = n_build + i;
        index.Append(row_id, data.train.data() + row_id * D);
        inserted_ids.push_back(row_id);
    }

    index.SetNProbe(0);

    // Use the first inserted embedding as query
    const float* query = data.train.data() + n_build * D;
    auto results = index.FilteredSearch(query, n_insert, inserted_ids);

    std::unordered_set<uint32_t> result_ids;
    for (const auto& r : results) {
        result_ids.insert(r.index);
    }

    for (size_t id : inserted_ids) {
        EXPECT_TRUE(result_ids.count(static_cast<uint32_t>(id)))
            << "Inserted point (row_id=" << id << ") not found in filtered search results";
    }
}

// Test 3: Build with N-1 points, insert 1, delete it, search should not find it
template <typename IndexT>
void RunInsertDeleteAndSearch() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_build = TestUtils::N_TRAIN - 1;
    const size_t inserted_row_id = n_build;

    auto index = BuildTreeIndex<IndexT>(data.train.data(), n_build, D);
    index.Append(inserted_row_id, data.train.data() + inserted_row_id * D);
    index.Delete(inserted_row_id);
    index.SetNProbe(0);

    auto results = index.Search(data.train.data() + inserted_row_id * D, TestUtils::KNN);

    for (const auto& r : results) {
        EXPECT_NE(r.index, static_cast<uint32_t>(inserted_row_id))
            << "Deleted point (row_id=" << inserted_row_id
            << ") should not appear in search results";
    }
}

class MaintenanceTest : public ::testing::TestWithParam<std::string> {};

TEST_P(MaintenanceTest, InsertSingleAndSearch) {
    std::string index_type = GetParam();
    if (index_type == "pdx_tree_f32") {
        RunInsertSingleAndSearch<PDX::PDXTreeIndexF32>();
    } else {
        RunInsertSingleAndSearch<PDX::PDXTreeIndexU8>();
    }
}

TEST_P(MaintenanceTest, InsertMultipleAndFilteredSearch) {
    std::string index_type = GetParam();
    if (index_type == "pdx_tree_f32") {
        RunInsertMultipleAndFilteredSearch<PDX::PDXTreeIndexF32>();
    } else {
        RunInsertMultipleAndFilteredSearch<PDX::PDXTreeIndexU8>();
    }
}

TEST_P(MaintenanceTest, InsertDeleteAndSearch) {
    std::string index_type = GetParam();
    if (index_type == "pdx_tree_f32") {
        RunInsertDeleteAndSearch<PDX::PDXTreeIndexF32>();
    } else {
        RunInsertDeleteAndSearch<PDX::PDXTreeIndexU8>();
    }
}

INSTANTIATE_TEST_SUITE_P(
    TreeIndexTypes,
    MaintenanceTest,
    ::testing::Values("pdx_tree_f32", "pdx_tree_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

} // namespace

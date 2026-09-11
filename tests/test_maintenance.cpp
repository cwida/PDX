#undef HAS_FFTW

#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "test_utils.hpp"

namespace {

static constexpr size_t D = 384;

template <typename T>
struct Tag {
    using type = T;
};

// Calls f(Tag<IndexT>{}) with the index class named by index_type
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

PDX::PDXIndexConfig MakeConfig(size_t d) {
    return PDX::PDXIndexConfig{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };
}

bool Contains(const std::vector<PDX::KNNCandidate>& results, size_t row_id) {
    return std::any_of(results.begin(), results.end(), [row_id](const PDX::KNNCandidate& r) {
        return r.index == static_cast<uint32_t>(row_id);
    });
}

// Every live row id is stored exactly once across all clusters and nothing was lost
void ExpectIndexHoldsExactly(const PDX::IPDXIndex& index, size_t expected_count) {
    size_t total = 0;
    std::unordered_set<uint32_t> seen;
    for (uint32_t c = 0; c < index.GetNumClusters(); c++) {
        total += index.GetClusterSize(c);
        for (uint32_t row_id : index.GetClusterRowIds(c)) {
            EXPECT_TRUE(seen.insert(row_id).second) << "row_id " << row_id << " stored twice";
        }
    }
    EXPECT_EQ(total, expected_count);
    EXPECT_EQ(seen.size(), expected_count);
}

// Test 1: Build with N-1 points, insert the last one, search for it
template <typename IndexT>
void RunInsertSingleAndSearch() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_build = TestUtils::N_TRAIN - 1;
    const size_t inserted_row_id = n_build;

    IndexT index(MakeConfig(D));
    index.BuildIndex(data.train.data(), n_build);
    index.Append(inserted_row_id, data.train.data() + inserted_row_id * D);
    index.SetNProbe(0);

    auto results = index.Search(data.train.data() + inserted_row_id * D, TestUtils::KNN);
    EXPECT_TRUE(Contains(results, inserted_row_id))
        << "Inserted point (row_id=" << inserted_row_id << ") not found in search results";
}

// Test 2: Build with N-10 points, insert 10, filtered search should return all 10
template <typename IndexT>
void RunInsertMultipleAndFilteredSearch() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_insert = 10;
    const size_t n_build = TestUtils::N_TRAIN - n_insert;

    IndexT index(MakeConfig(D));
    index.BuildIndex(data.train.data(), n_build);

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

    IndexT index(MakeConfig(D));
    index.BuildIndex(data.train.data(), n_build);
    index.Append(inserted_row_id, data.train.data() + inserted_row_id * D);
    index.Delete(inserted_row_id);
    index.SetNProbe(0);

    auto results = index.Search(data.train.data() + inserted_row_id * D, TestUtils::KNN);
    EXPECT_FALSE(Contains(results, inserted_row_id))
        << "Deleted point (row_id=" << inserted_row_id << ") should not appear in search results";
}

// Test 4: Insert 5, delete 5 of 10 inserted, filtered search over all 10 returns only the kept 5
template <typename IndexT>
void RunFilteredSearchExcludesDeleted() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_insert = 10;
    const size_t n_delete = 5;
    const size_t n_build = TestUtils::N_TRAIN - n_insert;

    IndexT index(MakeConfig(D));
    index.BuildIndex(data.train.data(), n_build);

    std::vector<size_t> inserted_ids;
    for (size_t i = 0; i < n_insert; ++i) {
        size_t row_id = n_build + i;
        index.Append(row_id, data.train.data() + row_id * D);
        inserted_ids.push_back(row_id);
    }
    for (size_t i = 0; i < n_delete; ++i) {
        index.Delete(inserted_ids[i]);
    }
    ExpectIndexHoldsExactly(index, n_build + n_insert - n_delete);

    index.SetNProbe(0);
    const float* query = data.train.data() + n_build * D;
    auto results = index.FilteredSearch(query, n_insert, inserted_ids);
    EXPECT_EQ(results.size(), n_insert - n_delete);
    for (size_t i = 0; i < n_insert; ++i) {
        const bool deleted = i < n_delete;
        EXPECT_EQ(Contains(results, inserted_ids[i]), !deleted)
            << "row_id=" << inserted_ids[i] << (deleted ? " was deleted" : " was kept");
    }
}

// Test 5: A dense cloud of near-duplicates lands in one cluster. Small clusters get
// MIN_MAX_CAPACITY slots, so inserting more than that must trigger at least one split, after
// which every point (inserted or original) must still be stored exactly once and be reachable.
template <typename IndexT>
void RunInsertDenseRegionForcesSplit() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_build = TestUtils::N_TRAIN;

    IndexT index(MakeConfig(D));
    index.BuildIndex(data.train.data(), n_build);
    const uint32_t clusters_before = index.GetNumClusters();

    const size_t n_insert = PDX::Cluster<PDX::F32>::MIN_MAX_CAPACITY + 64;
    const float* base = data.train.data();
    std::mt19937 rng(TestUtils::SEED);
    std::normal_distribution<float> noise(0.0f, 1e-3f);
    std::vector<float> inserted(n_insert * D);
    std::vector<size_t> inserted_ids;
    for (size_t i = 0; i < n_insert; ++i) {
        for (size_t j = 0; j < D; ++j) {
            inserted[i * D + j] = base[j] + noise(rng);
        }
        size_t row_id = n_build + i;
        index.Append(row_id, inserted.data() + i * D);
        inserted_ids.push_back(row_id);
    }

    EXPECT_GT(index.GetNumClusters(), clusters_before) << "No cluster split happened";
    ExpectIndexHoldsExactly(index, n_build + n_insert);

    index.SetNProbe(0);
    auto results = index.FilteredSearch(inserted.data(), n_insert, inserted_ids);
    EXPECT_EQ(results.size(), n_insert);
    std::unordered_set<uint32_t> result_ids;
    for (const auto& r : results) {
        result_ids.insert(r.index);
    }
    for (size_t id : inserted_ids) {
        EXPECT_TRUE(result_ids.count(static_cast<uint32_t>(id)))
            << "Inserted point (row_id=" << id << ") lost after the split";
    }
}

// Test 6: Deleting just over half of a cluster drives it under min_capacity, so it must be
// destroyed and its survivors reassigned to other clusters, where they stay searchable.
template <typename IndexT>
void RunDeleteUntilMerge() {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_build = TestUtils::N_TRAIN;

    IndexT index(MakeConfig(D));
    index.BuildIndex(data.train.data(), n_build);
    const uint32_t clusters_before = index.GetNumClusters();
    ASSERT_GT(clusters_before, 1u);

    // The largest cluster, so that the merge reassigns a meaningful number of points
    uint32_t victim = 0;
    for (uint32_t c = 1; c < clusters_before; c++) {
        if (index.GetClusterSize(c) > index.GetClusterSize(victim)) {
            victim = c;
        }
    }
    auto victim_ids = index.GetClusterRowIds(victim);
    ASSERT_GE(victim_ids.size(), 2u);
    const size_t n_delete = victim_ids.size() / 2 + 1;
    std::vector<uint32_t> deleted(victim_ids.begin(), victim_ids.begin() + n_delete);
    std::vector<uint32_t> kept(victim_ids.begin() + n_delete, victim_ids.end());

    for (uint32_t row_id : deleted) {
        index.Delete(row_id);
    }

    EXPECT_EQ(index.GetNumClusters(), clusters_before - 1) << "Cluster was not merged away";
    ExpectIndexHoldsExactly(index, n_build - n_delete);

    index.SetNProbe(0);
    for (uint32_t row_id : kept) {
        auto results = index.Search(data.train.data() + row_id * D, TestUtils::KNN);
        EXPECT_TRUE(Contains(results, row_id))
            << "Reassigned point (row_id=" << row_id << ") not found after the merge";
    }
    for (uint32_t row_id : deleted) {
        auto results = index.Search(data.train.data() + row_id * D, TestUtils::KNN);
        EXPECT_FALSE(Contains(results, row_id))
            << "Deleted point (row_id=" << row_id << ") found after the merge";
    }
}

// Test 7: Append/Delete on an index that went through Save + LoadPDXIndex (no config on disk)
template <typename IndexT>
void RunMaintenanceAfterRestore(const std::string& index_type) {
    auto data = TestUtils::LoadTestData(D);
    const size_t n_build = TestUtils::N_TRAIN - 2;
    const size_t kept_row_id = n_build;
    const size_t deleted_row_id = n_build + 1;

    std::string path = "/tmp/pdx_test_maintenance_" + index_type;
    {
        IndexT index(MakeConfig(D));
        index.BuildIndex(data.train.data(), n_build);
        index.Save(path);
    }
    auto loaded = PDX::LoadPDXIndex(path);
    std::remove(path.c_str());
    ASSERT_NE(loaded, nullptr);

    loaded->Append(kept_row_id, data.train.data() + kept_row_id * D);
    loaded->Append(deleted_row_id, data.train.data() + deleted_row_id * D);
    loaded->Delete(deleted_row_id);
    ExpectIndexHoldsExactly(*loaded, n_build + 1);

    loaded->SetNProbe(0);
    auto results = loaded->Search(data.train.data() + kept_row_id * D, TestUtils::KNN);
    EXPECT_TRUE(Contains(results, kept_row_id))
        << "Point appended after restore (row_id=" << kept_row_id << ") not found";
    results = loaded->Search(data.train.data() + deleted_row_id * D, TestUtils::KNN);
    EXPECT_FALSE(Contains(results, deleted_row_id))
        << "Point deleted after restore (row_id=" << deleted_row_id << ") still found";
}

class MaintenanceTest : public ::testing::TestWithParam<std::string> {};

TEST_P(MaintenanceTest, InsertSingleAndSearch) {
    ForIndexType(GetParam(), [](auto tag) {
        RunInsertSingleAndSearch<typename decltype(tag)::type>();
    });
}

TEST_P(MaintenanceTest, InsertMultipleAndFilteredSearch) {
    ForIndexType(GetParam(), [](auto tag) {
        RunInsertMultipleAndFilteredSearch<typename decltype(tag)::type>();
    });
}

TEST_P(MaintenanceTest, InsertDeleteAndSearch) {
    ForIndexType(GetParam(), [](auto tag) {
        RunInsertDeleteAndSearch<typename decltype(tag)::type>();
    });
}

TEST_P(MaintenanceTest, FilteredSearchExcludesDeleted) {
    ForIndexType(GetParam(), [](auto tag) {
        RunFilteredSearchExcludesDeleted<typename decltype(tag)::type>();
    });
}

TEST_P(MaintenanceTest, InsertDenseRegionForcesSplit) {
    ForIndexType(GetParam(), [](auto tag) {
        RunInsertDenseRegionForcesSplit<typename decltype(tag)::type>();
    });
}

TEST_P(MaintenanceTest, DeleteUntilMerge) {
    ForIndexType(GetParam(), [](auto tag) { RunDeleteUntilMerge<typename decltype(tag)::type>(); });
}

TEST_P(MaintenanceTest, MaintenanceAfterRestore) {
    const std::string& index_type = GetParam();
    ForIndexType(index_type, [&index_type](auto tag) {
        RunMaintenanceAfterRestore<typename decltype(tag)::type>(index_type);
    });
}

INSTANTIATE_TEST_SUITE_P(
    AllIndexTypes,
    MaintenanceTest,
    ::testing::Values("pdx_f32", "pdx_u8", "pdx_tree_f32", "pdx_tree_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

} // namespace

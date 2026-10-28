#undef HAS_FFTW

#include <gtest/gtest.h>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "test_utils.hpp"

namespace {

static constexpr size_t D = TestUtils::MAX_D;
static constexpr uint32_t N_PROBE = 16;
static constexpr size_t NUM_QUERIES = 100;
static constexpr float MIN_TOP_K_OVERLAP = 0.99f;

template <class Index>
class ClusterSketchTest : public ::testing::Test {};

using SketchIndexTypes = ::testing::Types<PDX::PDXIndexF32, PDX::PDXIndexU8>;
TYPED_TEST_SUITE(ClusterSketchTest, SketchIndexTypes);

PDX::PDXIndexConfig SketchConfig() {
    return PDX::PDXIndexConfig{
        .num_dimensions = static_cast<uint32_t>(D),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .sampling_fraction = 1.0f,
        .build_cluster_sketches = true,
    };
}

std::vector<std::vector<uint32_t>> SearchTopKIds(const PDX::IPDXIndex& index, const float* queries) {
    std::vector<std::vector<uint32_t>> ids(NUM_QUERIES);
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        for (const auto& candidate : index.Search(queries + q * D, TestUtils::KNN)) {
            ids[q].push_back(candidate.index);
        }
    }
    return ids;
}

float TopKOverlap(
    const std::vector<std::vector<uint32_t>>& expected,
    const std::vector<std::vector<uint32_t>>& actual
) {
    size_t shared = 0;
    size_t total = 0;
    for (size_t q = 0; q < expected.size(); ++q) {
        const std::unordered_set<uint32_t> expected_ids(expected[q].begin(), expected[q].end());
        for (const uint32_t id : actual[q]) {
            shared += expected_ids.count(id);
        }
        total += expected[q].size();
    }
    return static_cast<float>(shared) / static_cast<float>(total);
}

// Serves the clusters and the sketches of a saved stream, each copied into a buffer of its own as a
// cache would.
class SavedStreamClusterAndSketchSource : public PDX::IClusterSource {
  public:
    explicit SavedStreamClusterAndSketchSource(std::string saved_stream)
        : saved_stream(std::move(saved_stream)) {}

    // The cluster data starts where loading the resident data stopped.
    void Attach(const PDX::IPDXIndex& loaded_index, size_t loaded_cluster_data_start) {
        index = &loaded_index;
        cluster_data_start = loaded_cluster_data_start;
    }

    const char* Acquire(uint32_t cluster_id) override {
        num_acquired_clusters++;
        return CopyRange(index->GetClusterDataRange(cluster_id), acquired_clusters[cluster_id]);
    }

    void Release(uint32_t cluster_id) override {
        EXPECT_EQ(acquired_clusters.erase(cluster_id), 1u);
    }

    const char* AcquireSketch(uint32_t cluster_id) override {
        num_acquired_sketches++;
        return CopyRange(index->GetClusterSketchRange(cluster_id), acquired_sketches[cluster_id]);
    }

    void ReleaseSketch(uint32_t cluster_id) override {
        EXPECT_EQ(acquired_sketches.erase(cluster_id), 1u);
    }

    size_t num_acquired_clusters = 0;
    size_t num_acquired_sketches = 0;
    std::unordered_map<uint32_t, std::vector<uint32_t>> acquired_clusters;
    std::unordered_map<uint32_t, std::vector<uint32_t>> acquired_sketches;

  private:
    const char* CopyRange(std::pair<uint64_t, uint64_t> range, std::vector<uint32_t>& buffer) {
        const auto [offset, size] = range;
        buffer.resize((size + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        std::memcpy(buffer.data(), saved_stream.data() + cluster_data_start + offset, size);
        return reinterpret_cast<const char*>(buffer.data());
    }

    std::string saved_stream;
    const PDX::IPDXIndex* index = nullptr;
    size_t cluster_data_start = 0;
};

template <class Index>
void ExpectSketchCheckSkipsWithoutChangingResults(Index& index, const float* queries) {
    index.SetClusterSketchCheck(PDX::ClusterSketchCheck::NEVER);
    size_t accessed_before = index.GetNumVectorsAccessed();
    const auto unchecked_ids = SearchTopKIds(index, queries);
    const size_t unchecked_accessed = index.GetNumVectorsAccessed() - accessed_before;

    index.SetClusterSketchCheck(PDX::ClusterSketchCheck::ALWAYS);
    accessed_before = index.GetNumVectorsAccessed();
    const auto checked_ids = SearchTopKIds(index, queries);
    const size_t checked_accessed = index.GetNumVectorsAccessed() - accessed_before;

    EXPECT_GE(TopKOverlap(unchecked_ids, checked_ids), MIN_TOP_K_OVERLAP);
    EXPECT_LT(checked_accessed, unchecked_accessed);
}

// Checking every sketch skips clusters and keeps the index's own top-k
TYPED_TEST(ClusterSketchTest, SkipsClustersAndKeepsTheTopK) {
    const auto data = TestUtils::LoadTestData(D);
    TypeParam index(SketchConfig());
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    index.SetNProbe(N_PROBE);
    ExpectSketchCheckSkipsWithoutChangingResults(index, data.queries.data());
}

// The sketches follow their rows through appends that split clusters and deletes that merge them
TYPED_TEST(ClusterSketchTest, SketchesFollowTheirRowsThroughMaintenance) {
    constexpr size_t NUM_BUILT = 3000;
    constexpr size_t NUM_BULK_APPENDED = 1000;
    const auto data = TestUtils::LoadTestData(D);
    auto config = SketchConfig();
    config.max_cluster_size = 64;
    TypeParam index(config);
    index.BuildIndex(data.train.data(), NUM_BUILT);
    std::vector<size_t> bulk_row_ids(NUM_BULK_APPENDED);
    std::iota(bulk_row_ids.begin(), bulk_row_ids.end(), NUM_BUILT);
    index.BulkAppend(bulk_row_ids.data(), data.train.data() + NUM_BUILT * D, NUM_BULK_APPENDED);
    for (size_t row_id = NUM_BUILT + NUM_BULK_APPENDED; row_id < TestUtils::N_TRAIN; ++row_id) {
        index.Append(row_id, data.train.data() + row_id * D);
    }
    for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; row_id += 3) {
        index.Delete(row_id);
    }
    index.SetNProbe(N_PROBE);
    ExpectSketchCheckSkipsWithoutChangingResults(index, data.queries.data());
}

// A full load of a stream prunes as the saved index did, and a paged load reads the sketches through
// its cluster source instead of the clusters they rule out
TYPED_TEST(ClusterSketchTest, StreamsKeepTheSketchesAndPageThem) {
    const auto data = TestUtils::LoadTestData(D);
    const auto config = SketchConfig();
    PDX::ADSamplingPruner pruner(config.num_dimensions, config.seed);
    TypeParam index(config, pruner);
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    index.SetNProbe(N_PROBE);
    std::stringstream stream;
    index.SaveToStream(stream);
    const std::string saved_stream = stream.str();

    std::stringstream full_stream(saved_stream);
    auto loaded = PDX::LoadPDXIndexFromStream(full_stream, pruner);
    loaded->SetNProbe(N_PROBE);
    loaded->SetClusterSketchCheck(PDX::ClusterSketchCheck::ALWAYS);
    index.SetClusterSketchCheck(PDX::ClusterSketchCheck::ALWAYS);
    EXPECT_EQ(SearchTopKIds(*loaded, data.queries.data()), SearchTopKIds(index, data.queries.data()));

    std::stringstream paged_stream(saved_stream);
    SavedStreamClusterAndSketchSource source(saved_stream);
    auto paged = PDX::LoadPDXIndexFromStream(paged_stream, pruner, &source);
    source.Attach(*paged, static_cast<size_t>(paged_stream.tellg()));
    paged->SetNProbe(N_PROBE);
    paged->SetClusterSketchCheck(PDX::ClusterSketchCheck::NEVER);
    const auto unchecked_ids = SearchTopKIds(*paged, data.queries.data());
    const size_t unchecked_acquired_clusters = source.num_acquired_clusters;
    source.num_acquired_clusters = 0;
    paged->SetClusterSketchCheck(PDX::ClusterSketchCheck::WHEN_CLUSTER_NOT_CACHED);
    const auto checked_ids = SearchTopKIds(*paged, data.queries.data());
    EXPECT_GE(TopKOverlap(unchecked_ids, checked_ids), MIN_TOP_K_OVERLAP);
    EXPECT_LT(source.num_acquired_clusters, unchecked_acquired_clusters);
    EXPECT_GT(source.num_acquired_sketches, 0u);
    EXPECT_TRUE(source.acquired_clusters.empty());
    EXPECT_TRUE(source.acquired_sketches.empty());
}

} // namespace

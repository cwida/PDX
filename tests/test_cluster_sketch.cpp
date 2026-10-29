#undef HAS_FFTW

#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
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

using SketchIndexTypes = ::testing::Types<PDX::PDXIndexF32, PDX::PDXIndexU8, PDX::PDXIndexSAQ>;
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

// The rotation that indexes of one type share, and an index on it. SAQ's is a fit of the rows.
template <class Index>
struct SharedRotation {
    explicit SharedRotation(const float* /*rows*/, size_t /*num_rows*/)
        : pruner(static_cast<uint32_t>(D), TestUtils::SEED) {}

    std::unique_ptr<Index> MakeIndex(const PDX::PDXIndexConfig& config) {
        return std::make_unique<Index>(config, pruner);
    }

    PDX::ADSamplingPruner pruner;
};

template <>
struct SharedRotation<PDX::PDXIndexSAQ> {
    SharedRotation(const float* rows, size_t num_rows)
        : fit(PDX::SAQQuantizer::Fit(
              rows,
              num_rows,
              static_cast<uint32_t>(D),
              SketchConfig().saq_bits_per_dimension,
              TestUtils::SEED,
              *executor,
              allocator
          )),
          pruner(static_cast<uint32_t>(D), fit.rotation.get(), static_cast<uint32_t>(D)) {}

    std::unique_ptr<PDX::PDXIndexSAQ> MakeIndex(const PDX::PDXIndexConfig& config) {
        return std::make_unique<PDX::PDXIndexSAQ>(config, pruner, fit.model);
    }

    std::unique_ptr<skmeans::ParallelExecutor> executor = skmeans::MakeDefaultExecutor(0);
    skmeans::VanillaAllocator allocator;
    PDX::SAQFit fit;
    PDX::ADSamplingPruner pruner;
};

std::vector<std::vector<uint32_t>> SearchTopKIds(
    const PDX::IPDXIndex& index,
    const float* queries,
    const std::vector<size_t>* passing_row_ids = nullptr
) {
    std::vector<std::vector<uint32_t>> ids(NUM_QUERIES);
    for (size_t q = 0; q < NUM_QUERIES; ++q) {
        const float* query = queries + q * D;
        const auto candidates = passing_row_ids
                                    ? index.FilteredSearch(query, TestUtils::KNN, *passing_row_ids)
                                    : index.Search(query, TestUtils::KNN);
        for (const auto& candidate : candidates) {
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
void ExpectSketchCheckSkipsWithoutChangingResults(
    Index& index,
    const float* queries,
    const std::vector<size_t>* passing_row_ids = nullptr
) {
    index.SetClusterSketchCheck(PDX::ClusterSketchCheck::NEVER);
    size_t accessed_before = index.GetNumVectorsAccessed();
    const auto unchecked_ids = SearchTopKIds(index, queries, passing_row_ids);
    const size_t unchecked_accessed = index.GetNumVectorsAccessed() - accessed_before;

    index.SetClusterSketchCheck(PDX::ClusterSketchCheck::ALWAYS);
    accessed_before = index.GetNumVectorsAccessed();
    const auto checked_ids = SearchTopKIds(index, queries, passing_row_ids);
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

// Under a filter, where the check bounds only the passing rows, it skips clusters and keeps the
// filtered top-k
TYPED_TEST(ClusterSketchTest, FilteredSearchesSkipClustersAndKeepTheTopK) {
    constexpr size_t PASSING_ROW_ID_STRIDE = 20;
    const auto data = TestUtils::LoadTestData(D);
    TypeParam index(SketchConfig());
    index.BuildIndex(data.train.data(), TestUtils::N_TRAIN);
    index.SetNProbe(N_PROBE);
    std::vector<size_t> passing_row_ids;
    for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; row_id += PASSING_ROW_ID_STRIDE) {
        passing_row_ids.push_back(row_id);
    }
    ExpectSketchCheckSkipsWithoutChangingResults(index, data.queries.data(), &passing_row_ids);
}

// Every saved sketch lane is expected_sketch_row of the row in its slot, and the free lanes are
// cleared. The index holds every row but those with an id divisible by 3.
void ExpectSketchesFollowTheirRows(
    PDX::IPDXIndex& index,
    PDX::ADSamplingPruner& pruner,
    uint32_t sketch_num_dimensions,
    const std::function<void(size_t row_id, uint8_t* sketch_row)>& expected_sketch_row
) {
    std::stringstream stream;
    index.SaveToStream(stream);
    SavedStreamClusterAndSketchSource source(stream.str());
    auto paged = PDX::LoadPDXIndexFromStream(stream, pruner, &source);
    source.Attach(*paged, static_cast<size_t>(stream.tellg()));

    const PDX::ClusterSketchQuantizer quantizer(sketch_num_dimensions);
    std::vector<uint8_t> saved_row(quantizer.row_size_in_bytes);
    std::vector<uint8_t> encoded_row(quantizer.row_size_in_bytes);
    size_t num_rows_checked = 0;
    size_t num_mismatched_rows = 0;
    size_t num_uncleared_free_lanes = 0;
    for (uint32_t cluster_id = 0; cluster_id < index.GetNumClusters(); ++cluster_id) {
        const auto row_ids = index.GetClusterRowIds(cluster_id);
        const auto* batches = reinterpret_cast<const uint8_t*>(source.AcquireSketch(cluster_id));
        const size_t num_lanes =
            PDX::ClusterSketchQuantizer::GetNumBatches(static_cast<uint32_t>(row_ids.size())) *
            PDX::CLUSTER_SKETCH_BATCH_SIZE;
        for (size_t slot = 0; slot < num_lanes; ++slot) {
            quantizer.ReadRowFromBatch(
                batches + quantizer.GetBatchOffsetInBytes(slot),
                static_cast<uint32_t>(slot % PDX::CLUSTER_SKETCH_BATCH_SIZE),
                saved_row.data()
            );
            if (slot < row_ids.size()) {
                expected_sketch_row(row_ids[slot], encoded_row.data());
                num_mismatched_rows += saved_row != encoded_row;
                num_rows_checked++;
            } else {
                float f_add;
                std::memcpy(&f_add, saved_row.data() + quantizer.code_size_in_bytes, sizeof(float));
                num_uncleared_free_lanes += f_add != std::numeric_limits<float>::infinity();
            }
        }
        source.ReleaseSketch(cluster_id);
    }
    EXPECT_EQ(num_rows_checked, TestUtils::N_TRAIN - (TestUtils::N_TRAIN + 2) / 3);
    EXPECT_EQ(num_mismatched_rows, 0u);
    EXPECT_EQ(num_uncleared_free_lanes, 0u);
}

// After appends that split clusters and deletes that merge them, every saved sketch is the encoding
// of the row in its slot, and the free lanes are cleared (F32 stores the rows it encoded)
TEST(ClusterSketchMaintenanceTest, SketchesFollowTheirRowsThroughMaintenance) {
    constexpr size_t NUM_BUILT = 3000;
    constexpr size_t NUM_BULK_APPENDED = 1000;
    const auto data = TestUtils::LoadTestData(D);
    auto config = SketchConfig();
    config.max_cluster_size = 64;
    PDX::ADSamplingPruner pruner(config.num_dimensions, config.seed);
    PDX::PDXIndexF32 index(config, pruner);
    index.BuildIndex(data.train.data(), NUM_BUILT);
    std::vector<size_t> built_row_ids(NUM_BUILT);
    std::iota(built_row_ids.begin(), built_row_ids.end(), 0);
    std::vector<float> built_rows(NUM_BUILT * D);
    index.GetEmbeddingsFromIndexByRowIds(built_row_ids, built_rows.data());
    const auto mean = PDX::ClusterSketchQuantizer::ComputeMean(
        built_rows.data(), NUM_BUILT, static_cast<uint32_t>(D)
    );

    std::vector<size_t> bulk_row_ids(NUM_BULK_APPENDED);
    std::iota(bulk_row_ids.begin(), bulk_row_ids.end(), NUM_BUILT);
    index.BulkAppend(bulk_row_ids.data(), data.train.data() + NUM_BUILT * D, NUM_BULK_APPENDED);
    for (size_t row_id = NUM_BUILT + NUM_BULK_APPENDED; row_id < TestUtils::N_TRAIN; ++row_id) {
        index.Append(row_id, data.train.data() + row_id * D);
    }
    for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; row_id += 3) {
        index.Delete(row_id);
    }
    const PDX::ClusterSketchQuantizer quantizer(static_cast<uint32_t>(D));
    std::vector<float> row(D);
    ExpectSketchesFollowTheirRows(
        index,
        pruner,
        static_cast<uint32_t>(D),
        [&](size_t row_id, uint8_t* sketch_row) {
            index.GetEmbeddingsFromIndexByRowIds({row_id}, row.data());
            quantizer.EncodeRow(row.data(), mean.get(), sketch_row);
        }
    );
}

// The same for SAQ, whose sketches encode its rows rotated into the sketch domain, against the
// fit's mean rotated alike
TEST(ClusterSketchMaintenanceTest, SAQSketchesFollowTheirRowsThroughMaintenance) {
    constexpr size_t NUM_BUILT = 3000;
    constexpr size_t NUM_BULK_APPENDED = 1000;
    const auto data = TestUtils::LoadTestData(D);
    auto config = SketchConfig();
    config.max_cluster_size = 64;
    config.is_data_transformed = true;
    SharedRotation<PDX::PDXIndexSAQ> shared(data.train.data(), NUM_BUILT);
    std::vector<float> rows(TestUtils::N_TRAIN * D);
    shared.pruner.PreprocessEmbeddings(data.train.data(), rows.data(), TestUtils::N_TRAIN);
    std::vector<size_t> row_ids(TestUtils::N_TRAIN);
    std::iota(row_ids.begin(), row_ids.end(), 0);
    const auto index = shared.MakeIndex(config);
    index->BuildIndex(row_ids.data(), rows.data(), NUM_BUILT);
    index->BulkAppend(row_ids.data() + NUM_BUILT, rows.data() + NUM_BUILT * D, NUM_BULK_APPENDED);
    for (size_t row_id = NUM_BUILT + NUM_BULK_APPENDED; row_id < TestUtils::N_TRAIN; ++row_id) {
        index->Append(row_id, rows.data() + row_id * D);
    }
    for (size_t row_id = 0; row_id < TestUtils::N_TRAIN; row_id += 3) {
        index->Delete(row_id);
    }
    const PDX::ClusterSketchRotation rotation(static_cast<uint32_t>(D), config.seed);
    std::vector<float> rotated_mean(rotation.padded_num_dimensions);
    rotation.Rotate(shared.fit.model.rotated_mean.get(), rotated_mean.data());
    const PDX::ClusterSketchQuantizer quantizer(rotation.padded_num_dimensions);
    std::vector<float> rotated_row(rotation.padded_num_dimensions);
    ExpectSketchesFollowTheirRows(
        *index,
        shared.pruner,
        rotation.padded_num_dimensions,
        [&](size_t row_id, uint8_t* sketch_row) {
            rotation.Rotate(rows.data() + row_id * D, rotated_row.data());
            quantizer.EncodeRow(rotated_row.data(), rotated_mean.data(), sketch_row);
        }
    );
}

// A full load of a stream prunes as the saved index did, and a paged load reads the sketches
// through its cluster source instead of the clusters they rule out
TYPED_TEST(ClusterSketchTest, StreamsKeepTheSketchesAndPageThem) {
    const auto data = TestUtils::LoadTestData(D);
    const auto config = SketchConfig();
    SharedRotation<TypeParam> shared(data.train.data(), TestUtils::N_TRAIN);
    PDX::ADSamplingPruner& pruner = shared.pruner;
    const auto owned_index = shared.MakeIndex(config);
    TypeParam& index = *owned_index;
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
    EXPECT_EQ(
        SearchTopKIds(*loaded, data.queries.data()), SearchTopKIds(index, data.queries.data())
    );

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

// Indexes on one rotation search alike with one table per query as with a table each, and a table
// of other dimensions is refused
TYPED_TEST(ClusterSketchTest, IndexesShareTheQueryTable) {
    const auto data = TestUtils::LoadTestData(D);
    auto config = SketchConfig();
    config.is_data_transformed = true;
    SharedRotation<TypeParam> shared(data.train.data(), TestUtils::N_TRAIN);
    PDX::ADSamplingPruner& pruner = shared.pruner;
    const size_t rows_per_index = TestUtils::N_TRAIN / 2;
    std::vector<float> rows(TestUtils::N_TRAIN * D);
    pruner.PreprocessEmbeddings(data.train.data(), rows.data(), TestUtils::N_TRAIN);
    std::vector<float> queries(NUM_QUERIES * D);
    pruner.PreprocessEmbeddings(data.queries.data(), queries.data(), NUM_QUERIES);
    std::vector<std::unique_ptr<TypeParam>> indexes;
    for (size_t part = 0; part < 2; ++part) {
        config.base_row_id = part * rows_per_index;
        indexes.push_back(shared.MakeIndex(config));
        std::vector<size_t> row_ids(rows_per_index);
        std::iota(row_ids.begin(), row_ids.end(), config.base_row_id);
        indexes.back()->BuildIndex(
            row_ids.data(), rows.data() + config.base_row_id * D, rows_per_index
        );
        indexes.back()->SetClusterSketchCheck(PDX::ClusterSketchCheck::ALWAYS);
    }
    const auto search = [&](bool share_table, size_t& vectors_accessed) {
        std::vector<std::vector<uint32_t>> ids(NUM_QUERIES);
        size_t accessed_before = 0;
        for (const auto& index : indexes) {
            accessed_before += index->GetNumVectorsAccessed();
        }
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            const float* query = queries.data() + q * D;
            const auto table = indexes[0]->CreateClusterSketchQueryTable(query, true);
            PDX::TopKHeap heap;
            for (const auto& index : indexes) {
                index
                    ->BeginIterativeSearch(
                        query,
                        TestUtils::KNN,
                        heap,
                        nullptr,
                        true,
                        nullptr,
                        nullptr,
                        share_table ? table.get() : nullptr
                    )
                    ->Next(N_PROBE);
            }
            for (; !heap.heap.empty(); heap.heap.pop()) {
                ids[q].push_back(heap.heap.top().index);
            }
        }
        vectors_accessed = 0;
        for (const auto& index : indexes) {
            vectors_accessed += index->GetNumVectorsAccessed();
        }
        vectors_accessed -= accessed_before;
        return ids;
    };
    size_t own_tables_accessed = 0;
    size_t shared_table_accessed = 0;
    const auto own_tables_ids = search(false, own_tables_accessed);
    EXPECT_EQ(search(true, shared_table_accessed), own_tables_ids);
    EXPECT_EQ(shared_table_accessed, own_tables_accessed);

    const PDX::ClusterSketchQueryTable other_dimensions_table(
        queries.data(), static_cast<uint32_t>(D - 1)
    );
    PDX::TopKHeap heap;
    EXPECT_THROW(
        indexes[0]->BeginIterativeSearch(
            queries.data(),
            TestUtils::KNN,
            heap,
            nullptr,
            true,
            nullptr,
            nullptr,
            &other_dimensions_table
        ),
        std::invalid_argument
    );
}

// The sketch rotation keeps dot products, with or without padding and Kac's walk; its seed alone
// decides it, and Save/Load keeps it
TEST(ClusterSketchRotationTest, IsOrthogonalSeededAndSaved) {
    constexpr size_t NUM_VECTORS = 8;
    constexpr uint32_t PADDED_D = 100;
    constexpr uint32_t WALKED_D = 384;
    constexpr uint32_t POWER_OF_TWO_D = 512;
    const auto dot = [](const float* a, const float* b, size_t n) {
        double sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            sum += static_cast<double>(a[j]) * static_cast<double>(b[j]);
        }
        return sum;
    };
    std::mt19937 generator(TestUtils::SEED);
    std::normal_distribution<float> normal;
    for (const uint32_t d : {PADDED_D, WALKED_D, POWER_OF_TWO_D}) {
        SCOPED_TRACE("d=" + std::to_string(d));
        const PDX::ClusterSketchRotation rotation(d, TestUtils::SEED);
        const PDX::ClusterSketchRotation same_seed_rotation(d, TestUtils::SEED);
        const PDX::ClusterSketchRotation other_seed_rotation(d, TestUtils::SEED + 1);
        std::stringstream saved_rotation;
        rotation.Save(saved_rotation);
        PDX::StreamReader reader{saved_rotation};
        const auto loaded_rotation = PDX::ClusterSketchRotation::Load(reader, d);
        const size_t padded_d = rotation.padded_num_dimensions;
        std::vector<float> vectors(NUM_VECTORS * d);
        for (float& value : vectors) {
            value = normal(generator);
        }
        std::vector<float> rotated(NUM_VECTORS * padded_d);
        std::vector<float> same_seed_rotated(padded_d);
        std::vector<float> other_seed_rotated(padded_d);
        std::vector<float> loaded_rotated(padded_d);
        for (size_t i = 0; i < NUM_VECTORS; ++i) {
            const auto rotated_begin = rotated.begin() + static_cast<std::ptrdiff_t>(i * padded_d);
            rotation.Rotate(vectors.data() + i * d, &*rotated_begin);
            same_seed_rotation.Rotate(vectors.data() + i * d, same_seed_rotated.data());
            other_seed_rotation.Rotate(vectors.data() + i * d, other_seed_rotated.data());
            loaded_rotation->Rotate(vectors.data() + i * d, loaded_rotated.data());
            EXPECT_TRUE(
                std::equal(same_seed_rotated.begin(), same_seed_rotated.end(), rotated_begin)
            );
            EXPECT_FALSE(
                std::equal(other_seed_rotated.begin(), other_seed_rotated.end(), rotated_begin)
            );
            EXPECT_TRUE(std::equal(loaded_rotated.begin(), loaded_rotated.end(), rotated_begin));
        }
        for (size_t i = 0; i < NUM_VECTORS; ++i) {
            for (size_t j = i; j < NUM_VECTORS; ++j) {
                EXPECT_NEAR(
                    dot(rotated.data() + i * padded_d, rotated.data() + j * padded_d, padded_d),
                    dot(vectors.data() + i * d, vectors.data() + j * d, d),
                    1e-4 * d
                ) << "vectors "
                  << i << ", " << j;
            }
        }
    }
}

} // namespace

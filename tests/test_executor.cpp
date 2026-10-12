#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "superkmeans/executor.h"
#include "test_utils.hpp"

namespace {

static constexpr size_t D = 128;

// Forwards to a default executor of two threads and counts the calls.
class CountingExecutor final : public skmeans::ParallelExecutor {
  public:
    size_t NumWorkers() const override { return inner->NumWorkers(); }
    void ParallelFor(size_t n, const std::function<void(size_t, size_t, size_t)>& fn) override {
        parallel_for_calls++;
        inner->ParallelFor(n, fn);
    }
    void ReleaseThreads() override {
        release_threads_calls++;
        inner->ReleaseThreads();
    }

    size_t parallel_for_calls = 0;
    size_t release_threads_calls = 0;

  private:
    std::unique_ptr<skmeans::ParallelExecutor> inner = skmeans::MakeDefaultExecutor(2);
};

PDX::PDXIndexConfig MakeConfig(PDX::ParallelExecutor* executor, uint32_t n_threads) {
    return PDX::PDXIndexConfig{
        .num_dimensions = static_cast<uint32_t>(D),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = TestUtils::SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
        .n_threads = n_threads,
        .executor = executor,
    };
}

class ExecutorTest : public ::testing::TestWithParam<std::string> {};

// BuildIndex runs on the injected executor and releases its threads.
TEST_P(ExecutorTest, BuildRunsOnInjectedExecutor) {
    auto data = TestUtils::LoadTestData(D);
    CountingExecutor executor;
    auto index = TestUtils::BuildIndexWithConfig(
        GetParam(), MakeConfig(&executor, 0), data.train.data(), TestUtils::N_TRAIN
    );
    EXPECT_GT(executor.parallel_for_calls, 0u);
    EXPECT_GT(executor.release_threads_calls, 0u);
}

// A serial and a threaded default executor build the same clusters.
TEST_P(ExecutorTest, SerialAndThreadedBuildsMatch) {
    auto data = TestUtils::LoadTestData(D);
    skmeans::SerialExecutor serial;
    auto serial_index = TestUtils::BuildIndexWithConfig(
        GetParam(), MakeConfig(&serial, 0), data.train.data(), TestUtils::N_TRAIN
    );
    auto threaded_index = TestUtils::BuildIndexWithConfig(
        GetParam(), MakeConfig(nullptr, 4), data.train.data(), TestUtils::N_TRAIN
    );
    ASSERT_EQ(serial_index->GetNumClusters(), threaded_index->GetNumClusters());
    for (uint32_t c = 0; c < serial_index->GetNumClusters(); ++c) {
        EXPECT_EQ(serial_index->GetClusterRowIds(c), threaded_index->GetClusterRowIds(c))
            << "cluster " << c;
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllIndexTypes,
    ExecutorTest,
    ::testing::Values("pdx_f32", "pdx_u8", "pdx_tree_f32", "pdx_tree_u8"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

} // namespace

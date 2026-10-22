#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "pdx/distance_computers/scalar_computers.hpp"
#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"

namespace TestUtils {

static constexpr size_t N_TRAIN = 5000;
static constexpr size_t N_QUERIES = 500;
static constexpr size_t MAX_D = 384;
static constexpr size_t N_TOTAL = N_TRAIN + N_QUERIES;
static constexpr unsigned int SEED = 42;
static constexpr size_t KNN = 10;

// SAQ plans fit next to their 8-bit head chunk only from about 166 (4 bits) and 248 (3 bits)
// dimensions
static constexpr size_t SAQ_TEST_DIMENSIONS = 256;

struct TestData {
    std::vector<float> train;
    std::vector<float> queries;
};

// The dimensions an index type is tested at, for a suite at d
inline size_t GetTestDimensions(const std::string& index_type, size_t d) {
    const bool is_saq = index_type == "pdx_saq3" || index_type == "pdx_saq4";
    return is_saq ? std::max(d, SAQ_TEST_DIMENSIONS) : d;
}

// The bits per dimension of an SAQ index type
inline uint32_t GetSAQBitsPerDimension(const std::string& index_type) {
    return index_type == "pdx_saq3" ? 3 : 4;
}

inline std::string TempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

inline TestData LoadTestData(size_t d = MAX_D) {
    static std::vector<float> full_data;
    if (full_data.empty()) {
        std::string path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error(
                "Could not open test_data.bin. Run generate_test_data.out first."
            );
        }
        full_data.resize(N_TOTAL * MAX_D);
        in.read(reinterpret_cast<char*>(full_data.data()), full_data.size() * sizeof(float));
    }

    TestData data;
    if (d == MAX_D) {
        data.train.assign(full_data.begin(), full_data.begin() + N_TRAIN * MAX_D);
        data.queries.assign(full_data.begin() + N_TRAIN * MAX_D, full_data.end());
    } else {
        data.train.resize(N_TRAIN * d);
        data.queries.resize(N_QUERIES * d);
        for (size_t i = 0; i < N_TRAIN; ++i) {
            std::memcpy(&data.train[i * d], &full_data[i * MAX_D], d * sizeof(float));
        }
        for (size_t i = 0; i < N_QUERIES; ++i) {
            std::memcpy(&data.queries[i * d], &full_data[(N_TRAIN + i) * MAX_D], d * sizeof(float));
        }
    }
    return data;
}

struct BruteForceResult {
    std::vector<std::vector<uint32_t>> indices;
    std::vector<std::vector<float>> distances;
};

inline BruteForceResult ComputeBruteForceKNN(
    const float* train,
    const float* queries,
    size_t n_train,
    size_t n_queries,
    size_t d,
    size_t k
) {
    BruteForceResult result;
    result.indices.resize(n_queries);
    result.distances.resize(n_queries);

    for (size_t q = 0; q < n_queries; ++q) {
        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair> max_heap;

        for (size_t i = 0; i < n_train; ++i) {
            float dist = PDX::ScalarComputer<PDX::DistanceMetric::L2SQ, PDX::F32>::Horizontal(
                queries + q * d, train + i * d, d
            );
            if (max_heap.size() < k) {
                max_heap.push({dist, static_cast<uint32_t>(i)});
            } else if (dist < max_heap.top().first) {
                max_heap.pop();
                max_heap.push({dist, static_cast<uint32_t>(i)});
            }
        }

        size_t actual_k = max_heap.size();
        result.indices[q].resize(actual_k);
        result.distances[q].resize(actual_k);
        for (size_t i = actual_k; i > 0; --i) {
            result.indices[q][i - 1] = max_heap.top().second;
            result.distances[q][i - 1] = max_heap.top().first;
            max_heap.pop();
        }
    }
    return result;
}

inline BruteForceResult ComputeBruteForceCosineKNN(
    const float* train,
    const float* queries,
    size_t n_train,
    size_t n_queries,
    size_t d,
    size_t k
) {
    BruteForceResult result;
    result.indices.resize(n_queries);
    result.distances.resize(n_queries);

    for (size_t q = 0; q < n_queries; ++q) {
        using Pair = std::pair<float, uint32_t>;
        std::priority_queue<Pair> max_heap;

        for (size_t i = 0; i < n_train; ++i) {
            float dist = PDX::ScalarComputer<PDX::DistanceMetric::COSINE, PDX::F32>::Horizontal(
                queries + q * d, train + i * d, d
            );
            if (max_heap.size() < k) {
                max_heap.push({dist, static_cast<uint32_t>(i)});
            } else if (dist < max_heap.top().first) {
                max_heap.pop();
                max_heap.push({dist, static_cast<uint32_t>(i)});
            }
        }

        size_t actual_k = max_heap.size();
        result.indices[q].resize(actual_k);
        result.distances[q].resize(actual_k);
        for (size_t i = actual_k; i > 0; --i) {
            result.indices[q][i - 1] = max_heap.top().second;
            result.distances[q][i - 1] = max_heap.top().first;
            max_heap.pop();
        }
    }
    return result;
}

inline float ComputeRecall(
    const std::vector<PDX::KNNCandidate>& results,
    const std::vector<uint32_t>& gt_ids,
    size_t k
) {
    std::unordered_set<uint32_t> gt_set(
        gt_ids.begin(), gt_ids.begin() + std::min(k, gt_ids.size())
    );
    size_t hits = 0;
    for (size_t i = 0; i < std::min(k, results.size()); ++i) {
        if (gt_set.count(results[i].index)) {
            ++hits;
        }
    }
    return static_cast<float>(hits) / static_cast<float>(std::min(k, gt_set.size()));
}

inline float ComputeAverageRecall(
    PDX::IPDXIndex& index,
    const float* queries,
    size_t n_queries,
    size_t d,
    size_t k,
    const BruteForceResult& gt
) {
    float total_recall = 0.0f;
    for (size_t q = 0; q < n_queries; ++q) {
        auto results = index.Search(queries + q * d, k);
        total_recall += ComputeRecall(results, gt.indices[q], k);
    }
    return total_recall / static_cast<float>(n_queries);
}

inline std::unique_ptr<PDX::IPDXIndex> BuildIndex(
    const std::string& index_type,
    const float* data,
    size_t n,
    size_t d
) {
    PDX::PDXIndexConfig config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = PDX::DistanceMetric::L2SQ,
        .seed = SEED,
        .normalize = true,
        .sampling_fraction = 1.0f,
        .hierarchical_indexing = true,
    };

    std::unique_ptr<PDX::IPDXIndex> index;
    if (index_type == "pdx_f32") {
        auto p = std::make_unique<PDX::PDXIndexF32>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    } else if (index_type == "pdx_u8") {
        auto p = std::make_unique<PDX::PDXIndexU8>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    } else if (index_type == "pdx_tree_f32") {
        auto p = std::make_unique<PDX::PDXTreeIndexF32>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    } else if (index_type == "pdx_tree_u8") {
        auto p = std::make_unique<PDX::PDXTreeIndexU8>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    } else if (index_type == "pdx_saq3" || index_type == "pdx_saq4") {
        config.saq_bits_per_dimension = GetSAQBitsPerDimension(index_type);
        auto p = std::make_unique<PDX::PDXIndexSAQ>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    }
    return index;
}

inline std::unique_ptr<PDX::IPDXIndex> BuildIndexWithConfig(
    const std::string& index_type,
    const PDX::PDXIndexConfig& config,
    const float* data,
    size_t n
) {
    std::unique_ptr<PDX::IPDXIndex> index;
    if (index_type == "pdx_f32") {
        auto p = std::make_unique<PDX::PDXIndexF32>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    } else if (index_type == "pdx_u8") {
        auto p = std::make_unique<PDX::PDXIndexU8>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    } else if (index_type == "pdx_tree_f32") {
        auto p = std::make_unique<PDX::PDXTreeIndexF32>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    } else if (index_type == "pdx_tree_u8") {
        auto p = std::make_unique<PDX::PDXTreeIndexU8>(config);
        p->BuildIndex(data, n);
        index = std::move(p);
    } else if (index_type == "pdx_saq3" || index_type == "pdx_saq4") {
        PDX::PDXIndexConfig saq_config = config;
        saq_config.saq_bits_per_dimension = GetSAQBitsPerDimension(index_type);
        auto p = std::make_unique<PDX::PDXIndexSAQ>(saq_config);
        p->BuildIndex(data, n);
        index = std::move(p);
    }
    return index;
}

inline size_t CountEmptyClusters(const PDX::IPDXIndex& index) {
    size_t num_empty = 0;
    for (uint32_t cluster_id = 0; cluster_id < index.GetNumClusters(); cluster_id++) {
        num_empty += index.GetClusterSize(cluster_id) == 0;
    }
    return num_empty;
}

// The centroid quantizations under test: the recall that ranking by them may lose against float32,
// the bytes of one encoded centroid, and the bytes the centroids keep after the last one
struct QuantizedCentroidsTestCase {
    PDX::CentroidQuantization centroid_quantization;
    float recall_tolerance;
    size_t (*encoded_centroid_size_in_bytes)(size_t num_dimensions);
    size_t trailing_padding_in_bytes;
};

static constexpr float LVQ8_RECALL_TOLERANCE = 0.01f;
static constexpr float LVQ4_RECALL_TOLERANCE = 0.02f;

inline constexpr QuantizedCentroidsTestCase QUANTIZED_CENTROIDS[] = {
    {PDX::CentroidQuantization::LVQ8,
     LVQ8_RECALL_TOLERANCE,
     [](size_t num_dimensions) {
         return sizeof(PDX::LVQParams) + PDX::LVQQuantizer<8>::GetCodesSizeInBytes(num_dimensions);
     },
     PDX::LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES},
    {PDX::CentroidQuantization::LVQ4,
     LVQ4_RECALL_TOLERANCE,
     [](size_t num_dimensions) {
         return sizeof(PDX::LVQParams) + PDX::LVQQuantizer<4>::GetCodesSizeInBytes(num_dimensions);
     },
     PDX::LVQ_INT8_QUERY_CHUNK_SIZE_IN_BYTES},
};

class CountingAllocator final : public PDX::MemoryAllocator {
  public:
    void* Allocate(const size_t bytes) override {
        live_bytes += bytes;
        return vanilla.Allocate(bytes);
    }
    void Free(void* const ptr, const size_t bytes) override {
        live_bytes -= bytes;
        vanilla.Free(ptr, bytes);
    }

    std::atomic<size_t> live_bytes{0};

  private:
    skmeans::VanillaAllocator vanilla;
};

} // namespace TestUtils

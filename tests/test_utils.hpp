#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "pdx/distance_computers/scalar_computers.hpp"
#include "pdx/index.hpp"

namespace TestUtils {

static constexpr size_t N_TRAIN = 5000;
static constexpr size_t N_QUERIES = 500;
static constexpr size_t MAX_D = 384;
static constexpr size_t N_TOTAL = N_TRAIN + N_QUERIES;
static constexpr unsigned int SEED = 42;
static constexpr size_t KNN = 10;

struct TestData {
    std::vector<float> train;
    std::vector<float> queries;
};

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
    }
    return index;
}

} // namespace TestUtils

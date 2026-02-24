#pragma once

#include "pdx/common.hpp"
#include "pdx/utils.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class TicToc {
  public:
    size_t accum_time = 0;
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    void Reset() {
        accum_time = 0;
        start = std::chrono::high_resolution_clock::now();
    }

    inline void Tic() { start = std::chrono::high_resolution_clock::now(); }

    inline void Toc() {
        auto end = std::chrono::high_resolution_clock::now();
        accum_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    double GetMilliseconds() const { return static_cast<double>(accum_time) / 1e6; }
};

// Raw binary data paths (SuperKMeans convention: data_<name>.bin / data_<name>_test.bin)
inline std::string RAW_DATA_DIR =
    std::string{CMAKE_SOURCE_DIR} + "/../../SuperKMeans/benchmarks/data";
inline std::string GROUND_TRUTH_JSON_DIR =
    std::string{CMAKE_SOURCE_DIR} + "/../../SuperKMeans/benchmarks/ground_truth";

struct RawDatasetInfo {
    size_t num_embeddings;
    size_t num_dimensions;
    size_t num_queries;
    PDX::DistanceMetric distance_metric;
    std::string pdx_dataset_name; // Name used in PDX ground truth / query files
};

inline const std::unordered_map<std::string, RawDatasetInfo> RAW_DATASET_PARAMS = {
    {"sift", {1000000, 128, 1000, PDX::DistanceMetric::L2SQ, "sift-128-euclidean"}},
    {"yi", {187843, 128, 1000, PDX::DistanceMetric::IP, "yi-128-ip"}},
    {"llama", {256921, 128, 1000, PDX::DistanceMetric::IP, "llama-128-ip"}},
    {"glove200", {1183514, 200, 1000, PDX::DistanceMetric::COSINE, "glove-200-angular"}},
    {"yandex", {1000000, 200, 1000, PDX::DistanceMetric::COSINE, "yandex-200-cosine"}},
    {"yahoo", {677305, 384, 1000, PDX::DistanceMetric::COSINE, "yahoo-minilm-384-normalized"}},
    {"clip", {1281167, 512, 1000, PDX::DistanceMetric::L2SQ, "imagenet-clip-512-normalized"}},
    {"contriever", {990000, 768, 1000, PDX::DistanceMetric::L2SQ, "contriever-768"}},
    {"gist", {1000000, 960, 1000, PDX::DistanceMetric::L2SQ, "gist-960-euclidean"}},
    {"mxbai", {769382, 1024, 1000, PDX::DistanceMetric::L2SQ, "agnews-mxbai-1024-euclidean"}},
    {"openai", {999000, 1536, 1000, PDX::DistanceMetric::L2SQ, "openai-1536-angular"}},
    {"arxiv", {2253000, 768, 1000, PDX::DistanceMetric::L2SQ, "arxiv-nomic-768-normalized"}},
    {"wiki", {260372, 3072, 1000, PDX::DistanceMetric::L2SQ, "simplewiki-openai-3072-normalized"}},
};

struct BenchmarkMetadata {
    std::string dataset;
    std::string algorithm;
    size_t num_measure_runs{0};
    size_t num_queries{100};
    size_t ivf_nprobe{0};
    size_t knn{10};
    float recalls{1.0};
    float selectivity_threshold{0.0};
    float epsilon{0.0};
};

struct PhasesRuntime {
    size_t end_to_end{0};
};

class BenchmarkUtils {
  public:
    inline static std::string PDX_DATA =
        std::string{CMAKE_SOURCE_DIR} + "/benchmarks/datasets/pdx/";
    inline static std::string PDX_ADSAMPLING_DATA =
        std::string{CMAKE_SOURCE_DIR} + "/benchmarks/datasets/adsampling_pdx/";
    inline static std::string GROUND_TRUTH_DATA =
        std::string{CMAKE_SOURCE_DIR} + "/benchmarks/datasets/ground_truth/";
    inline static std::string FILTERED_GROUND_TRUTH_DATA =
        std::string{CMAKE_SOURCE_DIR} + "/benchmarks/datasets/ground_truth_filtered/";
    inline static std::string PURESCAN_DATA =
        std::string{CMAKE_SOURCE_DIR} + "/benchmarks/datasets/purescan/";
    inline static std::string QUERIES_DATA =
        std::string{CMAKE_SOURCE_DIR} + "/benchmarks/datasets/queries/";
    inline static std::string SELECTION_VECTOR_DATA =
        std::string{CMAKE_SOURCE_DIR} + "/benchmarks/datasets/selection_vectors/";

    std::string CPU_ARCHITECTURE = "DEFAULT";
    std::string RESULTS_DIR_PATH =
        std::string{CMAKE_SOURCE_DIR} + "/benchmarks/results/" + CPU_ARCHITECTURE + "/";

    explicit BenchmarkUtils() {
        CPU_ARCHITECTURE = std::getenv("PDX_ARCH") ? std::getenv("PDX_ARCH") : "DEFAULT";
        RESULTS_DIR_PATH =
            std::string{CMAKE_SOURCE_DIR} + "/benchmarks/results/" + CPU_ARCHITECTURE + "/";
    }

    inline static std::string DATASETS[] = {
        "sift-128-euclidean",
        "yi-128-ip",
        "llama-128-ip",
        "glove-200-angular",
        "yandex-200-cosine",
        "word2vec-300",
        "yahoo-minilm-384-normalized",
        "msong-420",
        "imagenet-clip-512-normalized",
        "laion-clip-512-normalized",
        "imagenet-align-640-normalized",
        "codesearchnet-jina-768-cosine",
        "landmark-dino-768-cosine",
        "landmark-nomic-768-normalized",
        "arxiv-nomic-768-normalized",
        "ccnews-nomic-768-normalized",
        "coco-nomic-768-normalized",
        "contriever-768",
        "instructorxl-arxiv-768",
        "gooaq-distilroberta-768-normalized",
        "gist-960-euclidean",
        "agnews-mxbai-1024-euclidean",
        "openai-1536-angular",
        "celeba-resnet-2048-cosine",
        "simplewiki-openai-3072-normalized"
    };

    inline static std::string FILTERED_SELECTIVITIES[] = {
        "0_000135",
        "0_001",
        "0_01",
        "0_1",
        "0_2",
        "0_3",
        "0_4",
        "0_5",
        "0_75",
        "0_9",
        "0_95",
        "0_99",
        "PART_1",
        "PART_30",
        "PART+_1",
    };

    inline static size_t IVF_PROBES[] = {
        // 4000, 3980, 3967, 2048, 1024, 512, 256,224,192,160,144,128,
        2048, 1536, 1024, 512, 384, 256, 224, 192, 160, 144, 128, 112, 96, 80, 64, 56, 48,
        40,   32,   28,   26,  24,  22,  20,  18,  16,  14,  12,  10,  8,  6,  4,  2,  1
    };

    inline static int POW_10[10] =
        {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

    inline static size_t IVF_PROBES_PHASES[] = {
        512,
        256,
        128,
        64,
        32,
        16,
        8,
        4,
        2,
    };

    inline static float SELECTIVITY_THRESHOLDS[] = {0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                                    0.35,  0.4,  0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                                                    0.75,  0.8,  0.85, 0.9, 0.95, 0.99};

    inline static size_t NUM_MEASURE_RUNS = 1;
    inline static float EPSILON0 = 1.5;               // 2.1;
    inline static float SELECTIVITY_THRESHOLD = 0.80; // more than 20% pruned to pass
    inline static bool VERIFY_RESULTS = true;
    inline static uint8_t KNN = 100;

    inline static uint8_t GROUND_TRUTH_MAX_K =
        100; // To properly skip on the ground truth file (do not change)

    template <bool MEASURE_RECALL, PDX::Quantization q = PDX::F32>
    static void VerifyResult(
        float& recalls,
        const std::vector<PDX::KNNCandidate>& result,
        size_t knn,
        const uint32_t* int_ground_truth,
        size_t n_query
    ) {
        std::unordered_set<uint32_t> seen;
        for (const auto& val : result) {
            if (!seen.insert(val.index).second) {
                throw std::runtime_error(
                    "Duplicates detected in the result set! This is likely a bug on PDXearch"
                );
            }
        }
        if (result.size() < knn) {
            std::cerr << "WARNING: Result set is not complete! Set a higher `nbuckets` parameter "
                         "(Only got "
                      << result.size() << " results)" << std::endl;
        }
        if constexpr (MEASURE_RECALL) {
            size_t true_positives = 0;
            for (size_t j = 0; j < result.size(); ++j) {
                for (size_t m = 0; m < knn; ++m) {
                    if (result[j].index == int_ground_truth[m + n_query * GROUND_TRUTH_MAX_K]) {
                        true_positives++;
                        break;
                    }
                }
            }
            recalls += 1.0 * true_positives / knn;
        } else {
            for (size_t j = 0; j < knn; ++j) {
                if (result[j].index != int_ground_truth[j + n_query * GROUND_TRUTH_MAX_K]) {
                    std::cout << "WRONG RESULT!\n";
                    break;
                }
            }
        }
    }

    // We remove extreme outliers on both sides (Q3 + 1.5*IQR & Q1 - 1.5*IQR)
    static void SaveResults(
        std::vector<PhasesRuntime> runtimes,
        const std::string& results_path,
        const BenchmarkMetadata& metadata
    ) {
        bool write_header = true;
        if (std::filesystem::exists(results_path)) {
            write_header = false;
        }
        std::ofstream file{results_path, std::ios::app};
        size_t min_runtime = std::numeric_limits<size_t>::max();
        size_t max_runtime = std::numeric_limits<size_t>::min();
        size_t sum_runtimes = 0;
        size_t all_min_runtime = std::numeric_limits<size_t>::max();
        size_t all_max_runtime = std::numeric_limits<size_t>::min();
        size_t all_sum_runtimes = 0;
        auto const Q1 = runtimes.size() / 4;
        auto const Q2 = runtimes.size() / 2;
        auto const Q3 = Q1 + Q2;
        std::sort(runtimes.begin(), runtimes.end(), [](PhasesRuntime i1, PhasesRuntime i2) {
            return i1.end_to_end < i2.end_to_end;
        });
        auto const iqr = runtimes[Q3].end_to_end - runtimes[Q1].end_to_end;
        size_t accounted_queries = 0;
        for (size_t j = 0; j < metadata.num_measure_runs * metadata.num_queries; ++j) {
            all_min_runtime = std::min(all_min_runtime, runtimes[j].end_to_end);
            all_max_runtime = std::max(all_max_runtime, runtimes[j].end_to_end);
            all_sum_runtimes += runtimes[j].end_to_end;
            // Removing outliers
            if (runtimes[j].end_to_end > runtimes[Q3].end_to_end + 1.5 * iqr) {
                continue;
            }
            if (runtimes[j].end_to_end < runtimes[Q1].end_to_end - 1.5 * iqr) {
                continue;
            }
            min_runtime = std::min(min_runtime, runtimes[j].end_to_end);
            max_runtime = std::max(max_runtime, runtimes[j].end_to_end);
            sum_runtimes += runtimes[j].end_to_end;
            accounted_queries += 1;
        }
        double all_min_runtime_ms = 1.0 * all_min_runtime / 1000000;
        double all_max_runtime_ms = 1.0 * all_max_runtime / 1000000;
        double all_avg_runtime_ms =
            1.0 * all_sum_runtimes / (1000000 * (metadata.num_measure_runs * metadata.num_queries));
        double min_runtime_ms = 1.0 * min_runtime / 1000000;
        double max_runtime_ms = 1.0 * max_runtime / 1000000;
        double avg_runtime_ms = 1.0 * sum_runtimes / (1000000 * accounted_queries);
        double avg_recall = metadata.recalls / metadata.num_queries;

        std::cout << metadata.dataset << " --------------\n";
        std::cout << "n_queries: " << metadata.num_queries << "\n";
        if (metadata.ivf_nprobe > 0) {
            std::cout << "nprobe: " << metadata.ivf_nprobe << "\n";
        }
        std::cout << "avg: " << std::setprecision(6) << avg_runtime_ms << "\n";
        std::cout << "max: " << std::setprecision(6) << max_runtime_ms << "\n";
        std::cout << "min: " << std::setprecision(6) << min_runtime_ms << "\n";
        std::cout << "rec: " << std::setprecision(6) << avg_recall << "\n";

        if (write_header) {
            file << "dataset,algorithm,avg,max,min,recall,ivf_nprobe,epsilon,"
                    "knn,n_queries,selectivity,"
                    "num_measure_runs,avg_all,max_all,min_all"
                 << "\n";
        }
        file << metadata.dataset << "," << metadata.algorithm << "," << std::setprecision(6)
             << avg_runtime_ms << "," << std::setprecision(6) << max_runtime_ms << ","
             << std::setprecision(6) << min_runtime_ms << "," << avg_recall << ","
             << metadata.ivf_nprobe << "," << metadata.epsilon << "," << +metadata.knn << ","
             << metadata.num_queries << "," << std::setprecision(4)
             << metadata.selectivity_threshold << "," << metadata.num_measure_runs << ","
             << all_avg_runtime_ms << "," << all_max_runtime_ms << "," << all_min_runtime_ms
             << "\n";
        file.close();
    }
};

BenchmarkUtils BENCHMARK_UTILS;

inline std::unordered_map<int, std::vector<int>> ParseGroundTruthJson(const std::string& filename) {
    std::unordered_map<int, std::vector<int>> gt_map;
    std::ifstream file(filename);
    if (!file.is_open())
        return gt_map;

    std::string line;
    std::getline(file, line);

    size_t pos = 0;
    while ((pos = line.find("\"", pos)) != std::string::npos) {
        size_t key_start = pos + 1;
        size_t key_end = line.find("\"", key_start);
        if (key_end == std::string::npos)
            break;

        int query_idx = std::stoi(line.substr(key_start, key_end - key_start));

        size_t arr_start = line.find("[", key_end);
        size_t arr_end = line.find("]", arr_start);
        if (arr_start == std::string::npos || arr_end == std::string::npos)
            break;

        std::string arr_str = line.substr(arr_start + 1, arr_end - arr_start - 1);
        std::vector<int> ids;
        std::istringstream iss(arr_str);
        std::string token;
        while (std::getline(iss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            if (!token.empty())
                ids.push_back(std::stoi(token));
        }
        gt_map[query_idx] = ids;
        pos = arr_end + 1;
    }
    return gt_map;
}

inline float ComputeRecallFromJson(
    const std::vector<PDX::KNNCandidate>& result,
    const std::vector<int>& gt_ids,
    size_t knn
) {
    size_t hits = 0;
    size_t gt_count = std::min(knn, gt_ids.size());
    for (size_t i = 0; i < result.size(); i++) {
        for (size_t j = 0; j < gt_count; j++) {
            if (result[i].index == static_cast<uint32_t>(gt_ids[j])) {
                hits++;
                break;
            }
        }
    }
    return static_cast<float>(hits) / static_cast<float>(gt_count);
}

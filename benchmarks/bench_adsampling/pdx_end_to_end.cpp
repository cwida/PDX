#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <chrono>

#include "pdx/index.hpp"
#include "benchmark_utils.hpp"
#include "pdx/utils.hpp"

template <typename IndexT>
void RunBenchmark(const RawDatasetInfo &info, const std::string &dataset, const std::string &algorithm,
                  const float *data, const float *queries, const std::vector<size_t> &nprobes_to_use) {
    const size_t d = info.num_dimensions;
    const size_t n = info.num_embeddings;
    const size_t n_queries = info.num_queries;
    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;
    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "END_TO_END_PDX_ADSAMPLING.csv";

    PDX::PDXIndexConfig index_config {
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = info.distance_metric,
        .seed = 42,
        .normalize = true,
        .sampling_fraction = 1.0f
    };

    std::cout << "Building index (num_clusters=auto)...\n";
    auto build_start = std::chrono::high_resolution_clock::now();
    IndexT pdx_index(index_config);
    pdx_index.BuildIndex(data, n);
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << "Build time: " << build_ms << " ms\n";
    std::cout << "Clusters: " << pdx_index.GetNumClusters() << "\n";

    // Load ground truth
    bool use_skmeans_gt = false;
    std::unordered_map<int, std::vector<int>> gt_map;
    std::unique_ptr<char[]> gt_buffer;
    uint32_t *int_ground_truth = nullptr;

    if (use_skmeans_gt) {
        std::string gt_path = GROUND_TRUTH_JSON_DIR + "/" + dataset + ".json";
        gt_map = ParseGroundTruthJson(gt_path);
        if (gt_map.empty()) {
            std::cerr << "No ground truth found at " << gt_path << "\n";
            return;
        }
        std::cout << "Ground truth loaded (json): " << gt_map.size() << " queries\n";
    } else {
        std::string gt_path = BenchmarkUtils::GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm";
        gt_buffer = MmapFile(gt_path);
        int_ground_truth = reinterpret_cast<uint32_t *>(gt_buffer.get());
        std::cout << "Ground truth loaded (pdx binary): " << gt_path << "\n";
    }

    for (size_t ivf_nprobe : nprobes_to_use) {
        if (pdx_index.GetNumClusters() < ivf_nprobe) continue;

        pdx_index.SetNProbe(ivf_nprobe);

        // Recall pass
        float recalls = 0;
        if (use_skmeans_gt) {
            for (size_t l = 0; l < n_queries; ++l) {
                auto result = pdx_index.Search(queries + l * d, KNN);
                if (gt_map.count(static_cast<int>(l))) {
                    recalls += ComputeRecallFromJson(result, gt_map.at(static_cast<int>(l)), KNN);
                }
            }
        } else {
            for (size_t l = 0; l < n_queries; ++l) {
                auto result = pdx_index.Search(queries + l * d, KNN);
                BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
            }
        }

        // Runtime pass
        std::vector<PhasesRuntime> runtimes;
        runtimes.resize(NUM_MEASURE_RUNS * n_queries);
        TicToc clock;
        for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
            for (size_t l = 0; l < n_queries; ++l) {
                clock.Reset();
                clock.Tic();
                pdx_index.Search(queries + l * d, KNN);
                clock.Toc();
                runtimes[j + l * NUM_MEASURE_RUNS] = {
                    clock.accum_time
                };
            }
        }

        BenchmarkMetadata results_metadata = {
            dataset,
            algorithm,
            NUM_MEASURE_RUNS,
            n_queries,
            ivf_nprobe,
            KNN,
            recalls,
        };
        BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset_name> <index_type> [ivf_nprobe]\n";
        std::cerr << "Index types: pdx_f32, pdx_u8, pdx_tree_f32, pdx_tree_u8\n";
        std::cerr << "Available datasets:";
        for (const auto &[name, _] : RAW_DATASET_PARAMS) {
            std::cerr << " " << name;
        }
        std::cerr << "\n";
        return 1;
    }
    std::string dataset = argv[1];
    std::string index_type = argv[2];
    size_t arg_ivf_nprobe = (argc > 3) ? std::atoi(argv[3]) : 0;

    auto it = RAW_DATASET_PARAMS.find(dataset);
    if (it == RAW_DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset: " << dataset << "\n";
        return 1;
    }
    const auto &info = it->second;
    const size_t n = info.num_embeddings;
    const size_t d = info.num_dimensions;
    const size_t n_queries = info.num_queries;

    std::cout << "==> PDX End-to-End (Build + Search)\n";
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "Index type: " << index_type << "\n";

    // Read data
    std::string data_path = RAW_DATA_DIR + "/data_" + dataset + ".bin";
    std::string query_path = RAW_DATA_DIR + "/data_" + dataset + "_test.bin";

    std::vector<float> data(n * d);
    {
        std::ifstream file(data_path, std::ios::binary);
        if (!file) { std::cerr << "Failed to open " << data_path << "\n"; return 1; }
        file.read(reinterpret_cast<char *>(data.data()), n * d * sizeof(float));
    }

    std::vector<float> queries(n_queries * d);
    {
        std::ifstream file(query_path, std::ios::binary);
        if (!file) { std::cerr << "Failed to open " << query_path << "\n"; return 1; }
        file.read(reinterpret_cast<char *>(queries.data()), n_queries * d * sizeof(float));
    }

    std::vector<size_t> nprobes_to_use;
    if (arg_ivf_nprobe > 0) {
        nprobes_to_use = {arg_ivf_nprobe};
    } else {
        nprobes_to_use.assign(std::begin(BenchmarkUtils::IVF_PROBES), std::end(BenchmarkUtils::IVF_PROBES));
    }

    std::string algorithm = "end_to_end_" + index_type;

    if (index_type == "pdx_f32") {
        RunBenchmark<PDX::PDXIndexF32>(info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use);
    } else if (index_type == "pdx_u8") {
        RunBenchmark<PDX::PDXIndexU8>(info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use);
    } else if (index_type == "pdx_tree_f32") {
        RunBenchmark<PDX::PDXTreeIndexF32>(info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use);
    } else if (index_type == "pdx_tree_u8") {
        RunBenchmark<PDX::PDXTreeIndexU8>(info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use);
    } else {
        std::cerr << "Unknown index type: " << index_type << "\n";
        std::cerr << "Valid types: pdx_f32, pdx_u8, pdx_tree_f32, pdx_tree_u8\n";
        return 1;
    }

    return 0;
}

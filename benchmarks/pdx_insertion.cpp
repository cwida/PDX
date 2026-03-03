#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "benchmark_utils.hpp"
#include "pdx/index.hpp"
#include "pdx/utils.hpp"
#include "pdx/profiler.hpp"

template <typename IndexT>
void RunBenchmark(
    const RawDatasetInfo& info,
    const std::string& dataset,
    const std::string& algorithm,
    const float* data,
    const float* queries,
    const std::vector<size_t>& nprobes_to_use
) {
    const size_t d = info.num_dimensions;
    const size_t n = info.num_embeddings;
    const size_t n_queries = info.num_queries;
    const float proportion_to_build = 0.75f;
    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;
    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "INSERTION_PDX.csv";

    const size_t n_build = static_cast<size_t>(n * proportion_to_build);
    const size_t n_insert = n - n_build;

    PDX::PDXIndexConfig index_config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = info.distance_metric,
        .seed = 42,
        .normalize = true,
        .sampling_fraction = 1.0f
    };

    // Build index with 75% of the data
    TicToc clock;
    std::cout << "Building index with " << n_build << " / " << n << " embeddings...\n";
    clock.Reset();
    clock.Tic();
    IndexT pdx_index(index_config);
    pdx_index.BuildIndex(data, n_build);
    clock.Toc();
    std::cout << "Build time: " << clock.GetMilliseconds() << " ms\n";
    std::cout << "Clusters: " << pdx_index.GetNumClusters() << "\n";
    std::cout << "Index in-memory size: " << std::fixed << std::setprecision(2)
              << static_cast<double>(pdx_index.GetInMemorySizeInBytes()) / (1024.0 * 1024.0)
              << " MB\n";

    // Insert remaining 25%
    std::cout << "Inserting " << n_insert << " embeddings...\n";
    clock.Reset();
    clock.Tic();
    for (size_t i = 0; i < n_insert; ++i) {
        size_t row_id = n_build + i;
        pdx_index.Append(row_id, data + row_id * d);
    }
    clock.Toc();
    std::cout << "Insertion time: " << clock.GetMilliseconds() << " ms\n";
    std::cout << "Avg insertion time: " << clock.GetMilliseconds() / n_insert << " ms/embedding\n";
    std::cout << "Clusters after insertion: " << pdx_index.GetNumClusters() << "\n";
    std::cout << "Index in-memory size after insertion: " << std::fixed << std::setprecision(2)
              << static_cast<double>(pdx_index.GetInMemorySizeInBytes()) / (1024.0 * 1024.0)
              << " MB\n";

    PDX::Profiler::Get().PrintHierarchical();

    // Load ground truth
    std::string gt_path =
        BenchmarkUtils::GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm";
    auto gt_buffer = MmapFile(gt_path);
    uint32_t* int_ground_truth = reinterpret_cast<uint32_t*>(gt_buffer.get());
    std::cout << "Ground truth loaded: " << gt_path << "\n";

    for (size_t ivf_nprobe : nprobes_to_use) {
        if (pdx_index.GetNumClusters() < ivf_nprobe)
            continue;

        pdx_index.SetNProbe(ivf_nprobe);

        // Recall pass
        float recalls = 0;
        for (size_t l = 0; l < n_queries; ++l) {
            auto result = pdx_index.Search(queries + l * d, KNN);
            BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
        }

        // Timing pass
        std::vector<PhasesRuntime> runtimes;
        runtimes.resize(NUM_MEASURE_RUNS * n_queries);
        TicToc search_clock;
        for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
            for (size_t l = 0; l < n_queries; ++l) {
                search_clock.Reset();
                search_clock.Tic();
                pdx_index.Search(queries + l * d, KNN);
                search_clock.Toc();
                runtimes[j + l * NUM_MEASURE_RUNS] = {search_clock.accum_time};
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset> [index_type] [nprobe]\n";
        std::cerr << "Index types: pdx_tree_f32 (default), pdx_tree_u8\n";
        std::cerr << "Available datasets:";
        for (const auto& [name, _] : RAW_DATASET_PARAMS) {
            std::cerr << " " << name;
        }
        std::cerr << "\n";
        return 1;
    }
    std::string dataset = argv[1];
    std::string index_type = (argc > 2) ? argv[2] : "pdx_tree_f32";
    size_t arg_ivf_nprobe = (argc > 3) ? std::atoi(argv[3]) : 0;

    if (index_type != "pdx_tree_f32" && index_type != "pdx_tree_u8") {
        std::cerr << "Error: Only pdx_tree_f32 and pdx_tree_u8 support maintenance (insertion).\n";
        std::cerr << "Got: " << index_type << "\n";
        return 1;
    }

    auto it = RAW_DATASET_PARAMS.find(dataset);
    if (it == RAW_DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset: " << dataset << "\n";
        return 1;
    }
    const auto& info = it->second;
    const size_t n = info.num_embeddings;
    const size_t d = info.num_dimensions;
    const size_t n_queries = info.num_queries;

    std::cout << "==> PDX Insertion Benchmark (Build 75% + Insert 25% + Search)\n";
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "Index type: " << index_type << "\n";

    // Read data
    std::string data_path = RAW_DATA_DIR + "/data_" + dataset + ".bin";
    std::string query_path = RAW_DATA_DIR + "/data_" + dataset + "_test.bin";

    std::vector<float> data(n * d);
    {
        std::ifstream file(data_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << data_path << "\n";
            return 1;
        }
        file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }

    std::vector<float> queries(n_queries * d);
    {
        std::ifstream file(query_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << query_path << "\n";
            return 1;
        }
        file.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    }

    std::vector<size_t> nprobes_to_use;
    if (arg_ivf_nprobe > 0) {
        nprobes_to_use = {arg_ivf_nprobe};
    } else {
        nprobes_to_use.assign(
            std::begin(BenchmarkUtils::IVF_PROBES), std::end(BenchmarkUtils::IVF_PROBES)
        );
    }

    std::string algorithm = "insertion_" + index_type;

    if (index_type == "pdx_tree_f32") {
        RunBenchmark<PDX::PDXTreeIndexF32>(
            info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use
        );
    } else if (index_type == "pdx_tree_u8") {
        RunBenchmark<PDX::PDXTreeIndexU8>(
            info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use
        );
    }

    return 0;
}

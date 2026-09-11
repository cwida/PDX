#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "benchmark_utils.hpp"
#include "pdx/indexes/ivf_forest.hpp"
#include "pdx/profiler.hpp"
#include "pdx/utils.hpp"

std::vector<size_t> LoadPassingRowIds(const std::string& path) {
    auto buffer = MmapFile(path);
    char* ptr = buffer.get();
    uint32_t num_ids = *reinterpret_cast<uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    auto* ids = reinterpret_cast<uint32_t*>(ptr);
    std::vector<size_t> result(num_ids);
    for (uint32_t i = 0; i < num_ids; i++) {
        result[i] = ids[i];
    }
    return result;
}

// Parse "same_tree_X" selectivity: returns X, or 0 if not that pattern.
static size_t ParseSameTreeCount(const std::string& selectivity) {
    const std::string prefix = "same_tree_";
    if (selectivity.size() > prefix.size() && selectivity.substr(0, prefix.size()) == prefix) {
        return static_cast<size_t>(std::stoul(selectivity.substr(prefix.size())));
    }
    return 0;
}

// Build per-query sequential row IDs: query l gets [l*count, l*count+count-1],
// clamped to num_embeddings.
static std::vector<size_t> BuildSameTreeRowIds(
    size_t query_idx,
    size_t count,
    size_t num_embeddings
) {
    size_t start = query_idx * count;
    if (start >= num_embeddings) {
        start = start % num_embeddings;
    }
    size_t end = std::min(start + count, num_embeddings);
    std::vector<size_t> row_ids(end - start);
    for (size_t i = 0; i < row_ids.size(); i++) {
        row_ids[i] = start + i;
    }
    return row_ids;
}

template <typename IndexT>
void RunBenchmark(
    const RawDatasetInfo& info,
    const std::string& dataset,
    const std::string& algorithm,
    const float* data,
    const float* queries,
    const std::vector<size_t>& nprobes_to_use,
    const std::string& arg_selectivity
) {
    const size_t d = info.num_dimensions;
    const size_t n = info.num_embeddings;
    const size_t n_queries = info.num_queries;
    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;
    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "PDX_FOREST_FILTERED.csv";

    PDX::PDXIndexConfig index_config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = info.distance_metric,
        .seed = 42,
        .normalize = true,
        .sampling_fraction = 1.0f
    };

    std::cout << "Building forest index...\n";
    auto build_start = std::chrono::high_resolution_clock::now();
    IndexT pdx_index(index_config);
    pdx_index.BuildIndex(data, n);
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << "Build time: " << build_ms << " ms\n";
    std::cout << "Trees: " << pdx_index.GetNumTrees() << "\n";
    std::cout << "Total clusters: " << pdx_index.GetForestNClusters() << "\n";
    std::cout << "Total L0 clusters: " << pdx_index.GetForestNL0Clusters() << "\n";

    // Determine selectivity mode
    size_t same_tree_count = ParseSameTreeCount(arg_selectivity);
    bool is_same_tree = (same_tree_count > 0);

    // For regular selectivity: load shared passing row IDs and ground truth
    std::vector<size_t> shared_passing_row_ids;
    std::unique_ptr<char[]> gt_buffer;
    uint32_t* int_ground_truth = nullptr;
    float selectivity_value = 0.0f;

    if (!is_same_tree) {
        // Parse selectivity value
        try {
            std::string sel = arg_selectivity;
            std::replace(sel.begin(), sel.end(), '_', '.');
            selectivity_value = std::stof(sel);
        } catch (...) {
        }

        shared_passing_row_ids = LoadPassingRowIds(
            BenchmarkUtils::SELECTION_VECTOR_DATA + info.pdx_dataset_name + "_" + arg_selectivity +
            ".bin"
        );
        std::cout << "Passing row IDs: " << shared_passing_row_ids.size() << "\n";

        // Load filtered ground truth
        std::string gt_path = BenchmarkUtils::FILTERED_GROUND_TRUTH_DATA + info.pdx_dataset_name +
                              "_100_norm_" + arg_selectivity;
        gt_buffer = MmapFile(gt_path);
        int_ground_truth = reinterpret_cast<uint32_t*>(gt_buffer.get());
    } else {
        std::cout << "same_tree mode: " << same_tree_count << " consecutive row IDs per query\n";
        selectivity_value = static_cast<float>(same_tree_count) / static_cast<float>(n);
    }
    PDX::Profiler::Get().Reset();

    for (size_t ivf_nprobe : nprobes_to_use) {
        pdx_index.SetNProbe(ivf_nprobe);

        // Recall pass (only for regular selectivity with ground truth)
        float recalls = 0;
        if (!is_same_tree && int_ground_truth) {
            for (size_t l = 0; l < n_queries; ++l) {
                const auto& row_ids = shared_passing_row_ids;
                auto result = pdx_index.FilteredSearch(queries + l * d, KNN, row_ids);
                BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
            }
        }

        std::vector<PhasesRuntime> runtimes;
        runtimes.resize(NUM_MEASURE_RUNS * n_queries);
        TicToc clock;
        for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
            for (size_t l = 0; l < n_queries; ++l) {
                // Build per-query row IDs for same_tree mode
                std::vector<size_t> per_query_row_ids;
                if (is_same_tree) {
                    per_query_row_ids = BuildSameTreeRowIds(l, same_tree_count, n);
                }
                const auto& row_ids = is_same_tree ? per_query_row_ids : shared_passing_row_ids;

                clock.Reset();
                clock.Tic();
                pdx_index.FilteredSearch(queries + l * d, KNN, row_ids);
                clock.Toc();
                runtimes[j + l * NUM_MEASURE_RUNS] = {clock.accum_time};
            }
        }
        PDX::Profiler::Get().PrintHierarchical();
        PDX::Profiler::Get().Reset();

        BenchmarkMetadata results_metadata = {
            dataset,
            algorithm + "_" + arg_selectivity,
            NUM_MEASURE_RUNS,
            n_queries,
            ivf_nprobe,
            KNN,
            recalls,
            selectivity_value
        };
        BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset> [index_type] [nprobe] [selectivity]\n";
        std::cerr << "Index types: pdx_forest_f32 (default), pdx_forest_u8\n";
        std::cerr << "Selectivity: 0_99, 0_5, same_tree_1000, same_tree_500, ...\n";
        std::cerr << "Available datasets:";
        for (const auto& [name, _] : RAW_DATASET_PARAMS) {
            std::cerr << " " << name;
        }
        std::cerr << "\n";
        return 1;
    }

    std::string dataset = argv[1];
    std::string index_type = (argc > 2) ? argv[2] : "pdx_forest_f32";
    size_t arg_ivf_nprobe = (argc > 3) ? std::atoi(argv[3]) : 0;
    std::string arg_selectivity = (argc > 4) ? argv[4] : "0_99";

    auto it = RAW_DATASET_PARAMS.find(dataset);
    if (it == RAW_DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset: " << dataset << "\n";
        return 1;
    }
    const auto& info = it->second;
    const size_t n = info.num_embeddings;
    const size_t d = info.num_dimensions;

    std::cout << "==> PDX Forest Filtered (Build + FilteredSearch)\n";
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "Index type: " << index_type << "\n";
    std::cout << "Selectivity: " << arg_selectivity << "\n";

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

    size_t n_queries = info.num_queries;
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

    std::string algorithm = "forest_filtered_" + index_type;

    if (index_type == "pdx_forest_f32") {
        RunBenchmark<PDX::PDXForestIndexF32>(
            info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use, arg_selectivity
        );
    } else if (index_type == "pdx_forest_u8") {
        RunBenchmark<PDX::PDXForestIndexU8>(
            info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use, arg_selectivity
        );
    } else {
        std::cerr << "Unknown index type: " << index_type << "\n";
        std::cerr << "Valid types: pdx_forest_f32, pdx_forest_u8\n";
        return 1;
    }

    return 0;
}

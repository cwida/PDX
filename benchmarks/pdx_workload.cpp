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

// ---- Edit workload here ----
static const std::vector<WorkloadStep> WORKLOAD = {
    // {StepType::BUILD,  0.50f},
    // {StepType::INSERT, 0.20f},
    // {StepType::DELETE, 0.10f},
    // {StepType::INSERT, 0.30f},
    // {StepType::DELETE, 0.20f},
    // {StepType::INSERT, 0.30f},


    {StepType::BUILD,  0.50f},
    {StepType::INSERT, 0.50f}
};

template <typename IndexT>
void RunWorkload(
    const RawDatasetInfo& info,
    const std::string& dataset,
    const std::string& algorithm,
    const float* data,
    const float* queries,
    const std::vector<size_t>& nprobes_to_use,
    const std::vector<WorkloadStep>& workload
) {
    const size_t d = info.num_dimensions;
    const size_t n = info.num_embeddings;
    const size_t n_queries = info.num_queries;
    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;
    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "WORKLOAD_PDX.csv";

    PDX::PDXIndexConfig index_config{
        .num_dimensions = static_cast<uint32_t>(d),
        .distance_metric = info.distance_metric,
        .seed = 42,
        .normalize = true,
        .sampling_fraction = 1.0f
    };

    IndexT pdx_index(index_config);
    TicToc clock;

    // State tracking:
    // - next_row_id: monotonically increasing row_id counter (row_ids are never reused)
    // - live_entries: (row_id, data_index) pairs currently in the index (stack for deletes)
    // - available_data: data indices freed by deletes, available for re-insertion
    // - next_data_cursor: next fresh data index (for inserts when available_data is empty)
    // - data_to_row_id: maps data array index → current row_id (for ground truth remapping)
    size_t next_row_id = 0;
    size_t next_data_cursor = 0;
    std::vector<std::pair<size_t, size_t>> live_entries; // (row_id, data_index)
    std::vector<size_t> available_data;
    std::vector<uint32_t> data_to_row_id(n);
    live_entries.reserve(n);

    // Execute workload steps
    for (size_t step_idx = 0; step_idx < workload.size(); ++step_idx) {
        const auto& step = workload[step_idx];
        size_t count = static_cast<size_t>(n * step.proportion);

        switch (step.type) {
        case StepType::BUILD: {
            std::cout << "\n=== Step " << step_idx << ": BUILD " << count << " embeddings ===\n";
            clock.Reset();
            clock.Tic();
            pdx_index.BuildIndex(data, count);
            clock.Toc();
            for (size_t i = 0; i < count; ++i) {
                live_entries.push_back({i, i});
                data_to_row_id[i] = static_cast<uint32_t>(i);
            }
            next_row_id = count;
            next_data_cursor = count;
            std::cout << "Build time: " << clock.GetMilliseconds() << " ms\n";
            break;
        }
        case StepType::INSERT: {
            if (available_data.size() + (n - next_data_cursor) < count) {
                std::cerr << "Step " << step_idx << ": INSERT " << count
                          << " but only " << available_data.size() + (n - next_data_cursor)
                          << " data points available\n";
                return;
            }
            std::cout << "\n=== Step " << step_idx << ": INSERT " << count << " embeddings ===\n";
            clock.Reset();
            clock.Tic();
            for (size_t i = 0; i < count; ++i) {
                // Pick a data point: reuse freed ones first, then fresh
                size_t data_idx;
                if (!available_data.empty()) {
                    data_idx = available_data.back();
                    available_data.pop_back();
                } else {
                    data_idx = next_data_cursor++;
                }
                size_t row_id = next_row_id++;
                std::cout << "Inserting row_id=" << row_id << " (data=" << data_idx << ")\r" << std::flush;
                pdx_index.Append(row_id, data + data_idx * d);
                live_entries.push_back({row_id, data_idx});
                data_to_row_id[data_idx] = static_cast<uint32_t>(row_id);
            }
            clock.Toc();
            std::cout << "\nInsertion time: " << clock.GetMilliseconds() << " ms\n";
            std::cout << "Avg insertion time: " << clock.GetMilliseconds() / count << " ms/embedding\n";
            break;
        }
        case StepType::DELETE: {
            if (count > live_entries.size()) {
                std::cerr << "Step " << step_idx << ": DELETE " << count
                          << " but only " << live_entries.size() << " live entries\n";
                return;
            }
            std::cout << "\n=== Step " << step_idx << ": DELETE " << count << " embeddings ===\n";
            clock.Reset();
            clock.Tic();
            for (size_t i = 0; i < count; ++i) {
                auto [row_id, data_idx] = live_entries.back();
                live_entries.pop_back();
                std::cout << "Deleting row_id=" << row_id << " (" << i + 1 << "/" << count << ")\r" << std::flush;
                pdx_index.Delete(row_id);
                available_data.push_back(data_idx);
            }
            clock.Toc();
            std::cout << "\nDeletion time: " << clock.GetMilliseconds() << " ms\n";
            std::cout << "Avg deletion time: " << clock.GetMilliseconds() / count << " ms/embedding\n";
            break;
        }
        }

        std::cout << "Clusters: " << pdx_index.GetNumClusters() << "\n";
        std::cout << "Index in-memory size: " << std::fixed << std::setprecision(2)
                  << static_cast<double>(pdx_index.GetInMemorySizeInBytes()) / (1024.0 * 1024.0)
                  << " MB\n";
        std::cout << "Live embeddings: " << live_entries.size() << "\n";
    }

    PDX::Profiler::Get().PrintHierarchical();

    // Load ground truth and remap data indices → current row_ids.
    // Ground truth entries are data indices (0..N-1). After deletes + re-inserts,
    // some data points have new row_ids. We remap so VerifyResult can compare.
    std::string gt_path =
        BenchmarkUtils::GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm";
    auto gt_buffer = MmapFile(gt_path);
    uint32_t* original_gt = reinterpret_cast<uint32_t*>(gt_buffer.get());

    const size_t gt_max_k = BenchmarkUtils::GROUND_TRUTH_MAX_K;
    std::vector<uint32_t> remapped_gt(n_queries * gt_max_k);
    for (size_t q = 0; q < n_queries; ++q) {
        for (size_t k = 0; k < gt_max_k; ++k) {
            uint32_t data_idx = original_gt[k + q * gt_max_k];
            remapped_gt[k + q * gt_max_k] = data_to_row_id[data_idx];
        }
    }
    std::cout << "\nGround truth loaded and remapped: " << gt_path << "\n";

    for (size_t ivf_nprobe : nprobes_to_use) {
        if (pdx_index.GetNumClusters() < ivf_nprobe)
            continue;

        pdx_index.SetNProbe(ivf_nprobe);

        // Recall pass
        float recalls = 0;
        for (size_t l = 0; l < n_queries; ++l) {
            auto result = pdx_index.Search(queries + l * d, KNN);
            BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, remapped_gt.data(), l);
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
    const auto& workload = WORKLOAD;

    // Validate workload: build + inserts - deletes must equal 1.0
    float net_proportion = 0.0f;
    for (const auto& step : workload) {
        if (step.type == StepType::DELETE) {
            net_proportion -= step.proportion;
        } else {
            net_proportion += step.proportion;
        }
    }
    if (std::abs(net_proportion - 1.0f) > 1e-5f) {
        std::cerr << "Error: workload net proportion must equal 1.0 "
                  << "(build + inserts - deletes), got: " << net_proportion << "\n";
        return 1;
    }

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
        std::cerr << "Error: Only pdx_tree_f32 and pdx_tree_u8 support maintenance.\n";
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

    // Print workload summary
    std::cout << "==> PDX Workload Benchmark\n";
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "Index type: " << index_type << "\n";
    std::cout << "Workload: ";
    for (size_t i = 0; i < workload.size(); ++i) {
        if (i > 0) std::cout << " -> ";
        switch (workload[i].type) {
            case StepType::BUILD:  std::cout << "build("  << workload[i].proportion << ")"; break;
            case StepType::INSERT: std::cout << "insert(" << workload[i].proportion << ")"; break;
            case StepType::DELETE: std::cout << "delete(" << workload[i].proportion << ")"; break;
        }
    }
    std::cout << "\n";

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

    std::string algorithm = "workload_" + index_type;

    if (index_type == "pdx_tree_f32") {
        RunWorkload<PDX::PDXTreeIndexF32>(
            info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use, workload
        );
    } else if (index_type == "pdx_tree_u8") {
        RunWorkload<PDX::PDXTreeIndexU8>(
            info, dataset, algorithm, data.data(), queries.data(), nprobes_to_use, workload
        );
    }

    return 0;
}

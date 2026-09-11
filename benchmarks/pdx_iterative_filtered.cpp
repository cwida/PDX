#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include "benchmark_utils.hpp"
#include "pdx/indexes/ivf_tree.hpp"
#include "pdx/indexes/ivf_vanilla.hpp"
#include "pdx/profiler.hpp"
#include "pdx/utils.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

// Mirrors the DuckDB extension: probe n_probe clusters, then keep resuming in steps until k
// results exist or every cluster with passing tuples has been probed.
static constexpr size_t RESUME_STEP = 5;

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

struct IterativeRun {
    std::vector<PDX::KNNCandidate> result;
    size_t iterations;
    size_t clusters_remaining;
};

static IterativeRun RunIterative(
    const PDX::IPDXIndex& index,
    const float* query,
    uint8_t knn,
    size_t n_probe,
    const std::vector<size_t>& passing_row_ids
) {
    PDX::TopKHeap top_k_heap;
    auto search_cursor = index.BeginIterativeSearch(query, knn, top_k_heap, &passing_row_ids);
    search_cursor->Next(n_probe);
    size_t iterations = 1;
    while (top_k_heap.heap.size() < knn && !search_cursor->Done()) {
        search_cursor->Next(RESUME_STEP);
        iterations++;
    }
    return {
        PDX::BuildResultSetFromHeap(knn, top_k_heap.heap),
        iterations,
        search_cursor->ClustersRemaining()
    };
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset> [index_type] [nprobe] [selectivity]\n";
        std::cerr << "Index types: pdx_f32 (default), pdx_u8, pdx_tree_f32, pdx_tree_u8\n";
        std::cerr << "Selectivity: 0_99, 0_5, 0_1, ...\n";
        std::cerr << "Available datasets:";
        for (const auto& [name, _] : RAW_DATASET_PARAMS) {
            std::cerr << " " << name;
        }
        std::cerr << "\n";
        return 1;
    }

    std::string arg_dataset = argv[1];
    std::string index_type = (argc > 2) ? argv[2] : "pdx_f32";
    size_t arg_ivf_nprobe = (argc > 3) ? atoi(argv[3]) : 0;
    std::string arg_selectivity = (argc > 4) ? argv[4] : "0_99";

    std::cout << "==> PDX Iterative Filtered (" << index_type << ")\n";
    std::cout << "==> Selectivity: " << arg_selectivity << ", resume step: " << RESUME_STEP << "\n";

    std::string ALGORITHM = "iterative_filtered";
    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;
    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    std::string index_type_upper = index_type;
    for (auto& c : index_type_upper)
        c = toupper(c);
    std::string RESULTS_PATH =
        BENCHMARK_UTILS.RESULTS_DIR_PATH + index_type_upper + "_ITERATIVE_FILTERED.csv";

    float selectivity_value = 0.0f;
    try {
        std::string sel = arg_selectivity;
        std::replace(sel.begin(), sel.end(), '_', '.');
        selectivity_value = std::stof(sel);
    } catch (...) {
    }

    for (const auto& [dataset, info] : RAW_DATASET_PARAMS) {
        if (!arg_dataset.empty() && arg_dataset != dataset) {
            continue;
        }

        std::string index_path = BenchmarkUtils::PDX_DATA + dataset + "-" + index_type;
        std::cout << "Loading " << index_path << "...\n";
        auto pdx_index = PDX::LoadPDXIndex(index_path);
        const size_t d = pdx_index->GetNumDimensions();

        std::unique_ptr<char[]> query_ptr =
            MmapFile(BenchmarkUtils::QUERIES_DATA + info.pdx_dataset_name);
        auto* query = reinterpret_cast<float*>(query_ptr.get());
        const size_t NUM_QUERIES = info.num_queries;
        query += 1; // skip number of embeddings header

        auto ground_truth = MmapFile(
            BenchmarkUtils::FILTERED_GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm_" +
            arg_selectivity
        );
        auto* int_ground_truth = reinterpret_cast<uint32_t*>(ground_truth.get());
        auto passing_row_ids = LoadPassingRowIds(
            BenchmarkUtils::SELECTION_VECTOR_DATA + info.pdx_dataset_name + "_" + arg_selectivity +
            ".bin"
        );
        std::cout << "Passing row IDs: " << passing_row_ids.size() << "\n";

        std::vector<size_t> nprobes_to_use;
        if (arg_ivf_nprobe > 0) {
            nprobes_to_use = {arg_ivf_nprobe};
        } else {
            nprobes_to_use.assign(
                std::begin(BenchmarkUtils::IVF_PROBES), std::end(BenchmarkUtils::IVF_PROBES)
            );
        }

        for (size_t ivf_nprobe : nprobes_to_use) {
            if (pdx_index->GetNumClusters() < ivf_nprobe) {
                continue;
            }

            float recalls = 0;
            size_t total_iterations = 0;
            size_t total_remaining = 0;
            size_t exhausted = 0;
            if (VERIFY_RESULTS) {
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    auto run =
                        RunIterative(*pdx_index, query + l * d, KNN, ivf_nprobe, passing_row_ids);
                    BenchmarkUtils::VerifyResult<true>(
                        recalls, run.result, KNN, int_ground_truth, l
                    );
                    total_iterations += run.iterations;
                    total_remaining += run.clusters_remaining;
                    exhausted += run.clusters_remaining == 0;
                }
            }

            std::vector<PhasesRuntime> runtimes;
            runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
            TicToc clock;
            for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    clock.Reset();
                    clock.Tic();
                    RunIterative(*pdx_index, query + l * d, KNN, ivf_nprobe, passing_row_ids);
                    clock.Toc();
                    runtimes[j + l * NUM_MEASURE_RUNS] = {clock.accum_time};
                }
            }
            PDX::Profiler::Get().PrintHierarchical();
            PDX::Profiler::Get().Reset();

            std::cout << "nprobe " << ivf_nprobe << ": avg iterations "
                      << static_cast<double>(total_iterations) / NUM_QUERIES
                      << ", avg clusters left "
                      << static_cast<double>(total_remaining) / NUM_QUERIES
                      << ", exhausted queries " << exhausted << "/" << NUM_QUERIES << "\n";

            BenchmarkMetadata results_metadata = {
                dataset,
                ALGORITHM,
                NUM_MEASURE_RUNS,
                NUM_QUERIES,
                ivf_nprobe,
                KNN,
                recalls,
                selectivity_value
            };
            BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
        }
    }
    return 0;
}

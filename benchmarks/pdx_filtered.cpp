#include "benchmark_utils.hpp"
#include "pdx/index.hpp"
#include "pdx/utils.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset> [index_type] [nprobe] [selectivity]\n";
        std::cerr << "Index types: pdx_f32 (default), pdx_u8, pdx_tree_f32, pdx_tree_u8\n";
        std::cerr << "Available datasets:";
        for (const auto& [name, _] : RAW_DATASET_PARAMS) {
            std::cerr << " " << name;
        }
        std::cerr << "\n";
        return 1;
    }

    std::string arg_dataset = argv[1];
    std::string index_type = "pdx_f32";
    size_t arg_ivf_nprobe = 0;
    std::string arg_selectivity = "0_99";

    if (argc > 2) {
        index_type = argv[2];
    }
    if (argc > 3) {
        arg_ivf_nprobe = atoi(argv[3]);
    }
    if (argc > 4) {
        arg_selectivity = argv[4];
    }

    std::cout << "==> PDX IVF ADSampling Filtered (" << index_type << ")\n";
    std::cout << "==> Selectivity: " << arg_selectivity << "\n";

    std::string ALGORITHM = "adsampling_filtered";
    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;
    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    // Build results file name from index type
    std::string index_type_upper = index_type;
    for (auto& c : index_type_upper)
        c = toupper(c);
    std::string RESULTS_PATH =
        BENCHMARK_UTILS.RESULTS_DIR_PATH + index_type_upper + "_ADSAMPLING_FILTERED.csv";

    // Parse selectivity string to float for metadata
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
        std::cout << "Index in-memory size: " << std::fixed << std::setprecision(2)
                  << static_cast<double>(pdx_index->GetInMemorySizeInBytes()) / (1024.0 * 1024.0)
                  << " MB\n";

        // Load queries
        std::unique_ptr<char[]> query_ptr =
            MmapFile(BenchmarkUtils::QUERIES_DATA + info.pdx_dataset_name);
        auto* query = reinterpret_cast<float*>(query_ptr.get());
        NUM_QUERIES = info.num_queries;
        query += 1; // skip number of embeddings header

        // Load filtered ground truth
        std::unique_ptr<char[]> ground_truth = MmapFile(
            BenchmarkUtils::FILTERED_GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm_" +
            arg_selectivity
        );
        auto* int_ground_truth = reinterpret_cast<uint32_t*>(ground_truth.get());

        // Load passing row IDs (binary format: [uint32 count][uint32[] ids])
        std::vector<size_t> passing_row_ids = LoadPassingRowIds(
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
            if (arg_ivf_nprobe > 0 && ivf_nprobe != arg_ivf_nprobe) {
                continue;
            }
            std::vector<PhasesRuntime> runtimes;
            runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
            pdx_index->SetNProbe(ivf_nprobe);

            float recalls = 0;
            if (VERIFY_RESULTS) {
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    auto result = pdx_index->FilteredSearch(
                        query + l * pdx_index->GetNumDimensions(), KNN, passing_row_ids
                    );
                    BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
                }
            }
            TicToc clock;
            for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    clock.Reset();
                    clock.Tic();
                    pdx_index->FilteredSearch(
                        query + l * pdx_index->GetNumDimensions(), KNN, passing_row_ids
                    );
                    clock.Toc();
                    runtimes[j + l * NUM_MEASURE_RUNS] = {clock.accum_time};
                }
            }
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

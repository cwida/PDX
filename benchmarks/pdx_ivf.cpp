#include <memory>
#include <iostream>
#include "pdx/utils.hpp"
#include "pdx/index.hpp"
#include "benchmark_utils.hpp"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <index_type> [dataset] [nprobe]\n";
        std::cerr << "Index types: pdx_f32, pdx_u8, pdx_tree_f32, pdx_tree_u8\n";
        std::cerr << "Available datasets:";
        for (const auto &[name, _] : RAW_DATASET_PARAMS) {
            std::cerr << " " << name;
        }
        std::cerr << "\n";
        return 1;
    }

    std::string index_type = argv[1];
    std::string arg_dataset;
    size_t arg_ivf_nprobe = 0;
    if (argc > 2) {
        arg_dataset = argv[2];
    }
    if (argc > 3) {
        arg_ivf_nprobe = atoi(argv[3]);
    }

    std::cout << "==> PDX IVF ADSampling (" << index_type << ")\n";

    std::string ALGORITHM = "adsampling";
    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    // Build results file name from index type (e.g., "pdx_f32" -> "PDX_F32_ADSAMPLING.csv")
    std::string index_type_upper = index_type;
    for (auto &c : index_type_upper) c = toupper(c);
    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + index_type_upper + "_ADSAMPLING.csv";

    for (const auto &[dataset, info] : RAW_DATASET_PARAMS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset) {
            continue;
        }

        std::string index_path = BenchmarkUtils::PDX_DATA + dataset + "-" + index_type;
        std::cout << "Loading " << index_path << "...\n";
        auto pdx_index = PDX::LoadPDXIndex(index_path);

        std::unique_ptr<char[]> query_ptr = MmapFile(BenchmarkUtils::QUERIES_DATA + info.pdx_dataset_name);
        auto *query = reinterpret_cast<float *>(query_ptr.get());

        NUM_QUERIES = info.num_queries;
        std::unique_ptr<char[]> ground_truth = MmapFile(
            BenchmarkUtils::GROUND_TRUTH_DATA + info.pdx_dataset_name + "_100_norm");
        auto *int_ground_truth = reinterpret_cast<uint32_t *>(ground_truth.get());
        query += 1; // skip number of embeddings

        std::vector<size_t> nprobes_to_use;
        if (arg_ivf_nprobe > 0) {
            nprobes_to_use = {arg_ivf_nprobe};
        } else {
            nprobes_to_use.assign(std::begin(BenchmarkUtils::IVF_PROBES), std::end(BenchmarkUtils::IVF_PROBES));
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
                    auto result = pdx_index->Search(query + l * pdx_index->GetNumDimensions(), KNN);
                    BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
                }
            }
            TicToc clock;
            for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    clock.Reset();
                    clock.Tic();
                    pdx_index->Search(query + l * pdx_index->GetNumDimensions(), KNN);
                    clock.Toc();
                    runtimes[j + l * NUM_MEASURE_RUNS] = {
                        clock.accum_time
                    };
                }
            }
            float real_selectivity = 1 - BenchmarkUtils::SELECTIVITY_THRESHOLD;
            BenchmarkMetadata results_metadata = {
                dataset,
                ALGORITHM,
                NUM_MEASURE_RUNS,
                NUM_QUERIES,
                ivf_nprobe,
                KNN,
                recalls,
                real_selectivity
            };
            BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
        }
    }
    return 0;
}

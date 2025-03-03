#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#ifndef PDX_USE_EXPLICIT_SIMD
#define PDX_USE_EXPLICIT_SIMD = true
#endif

#include <iostream>
#include <chrono>
#include "utils/file_reader.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "pdx/bond.hpp"
#include "utils/benchmark_utils.hpp"

int main(int argc, char *argv[]) {
    std::string arg_dataset;
    size_t arg_ivf_nprobe = 0;
    if (argc > 1){
        arg_dataset = argv[1];
    }
    if (argc > 2){
        arg_ivf_nprobe = atoi(argv[2]);
    }
    std::cout << "Doing something...\n";

    const bool USE_IVF = false;

    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    size_t IVF_NPROBE = 128;
    if (arg_ivf_nprobe > 0){
        IVF_NPROBE = arg_ivf_nprobe;
    }
    std::string ALGORITHM = "pdx-bond";

    std::string RESULTS_PATH;
    if (USE_IVF) {
        RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "IVF_PDX_BOND_TUNING.csv";
    } else {
        RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "PDX_BOND_TUNING.csv";
    }

    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        PDX::IndexPDXIVFFlat pdx_data = PDX::IndexPDXIVFFlat();
        if (USE_IVF) {
            pdx_data.Restore(BenchmarkUtils::PDX_DATA + dataset + "-ivf");
        } else {
            pdx_data.Restore(BenchmarkUtils::PDX_DATA + dataset + "-flat");
        }
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = ((uint32_t *)query)[0];
        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *)ground_truth;
        query += 1; // skip number of embeddings

        PDX::PDXBondSearcher searcher = PDX::PDXBondSearcher(pdx_data, 0.0, IVF_NPROBE, 0, PDX::DISTANCE_TO_MEANS_IMPROVED);

        for (float selectivity_thres: BenchmarkUtils::SELECTIVITY_THRESHOLDS){
                if (USE_IVF && pdx_data.num_vectorgroups < IVF_NPROBE){
                    continue;
                }
                std::vector<PhasesRuntime> runtimes;
                runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
                searcher.SetSelectivityThreshold(selectivity_thres);

                float recalls = 0;
                for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
                    for (size_t l = 0; l < NUM_QUERIES; ++l) {
                        searcher.Search(query + l * pdx_data.num_dimensions, KNN);
                        runtimes[j + l * NUM_MEASURE_RUNS] = {
                            searcher.end_to_end_clock.accum_time,
                            searcher.find_nearest_buckets_clock.accum_time,
                            searcher.query_preprocessing_clock.accum_time,
                            searcher.bounds_evaluation_clock.accum_time,
                    };
                    }
                }
                float real_selectivity = 1 - selectivity_thres;
                BenchmarkMetadata results_metadata = {
                        dataset,
                        ALGORITHM,
                        NUM_MEASURE_RUNS,
                        NUM_QUERIES,
                        IVF_NPROBE,
                        KNN,
                        recalls,
                        real_selectivity,
                };
                BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
        }
    }
}
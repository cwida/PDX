#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#ifndef PDX_USE_EXPLICIT_SIMD
#define PDX_USE_EXPLICIT_SIMD = true
#endif

#include <iostream>
#include "utils/file_reader.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "pdx/bond.hpp"
#include "utils/benchmark_utils.hpp"

int main(int argc, char *argv[]) {
    std::string arg_dataset;
    size_t arg_ivf_nprobe = 0;
    std::string ALGORITHM = "pdx-bond";
    PDX::PDXearchDimensionsOrder DIMENSION_ORDER = PDX::DISTANCE_TO_MEANS_IMPROVED;
    if (argc > 1){
        arg_dataset = argv[1];
    }
    if (argc > 2){
        arg_ivf_nprobe = atoi(argv[2]);
    }
    if (argc > 3){
        // enum PDXearchDimensionsOrder {
        //     SEQUENTIAL,
        //     DISTANCE_TO_MEANS,
        //     DECREASING,
        //     DISTANCE_TO_MEANS_IMPROVED,
        //     DECREASING_IMPROVED,
        //     DIMENSION_ZONES 
        // };
        DIMENSION_ORDER = static_cast<PDX::PDXearchDimensionsOrder>(atoi(argv[3]));
        if (DIMENSION_ORDER == PDX::DISTANCE_TO_MEANS_IMPROVED){
            ALGORITHM = "pdx-bond";
        }
        else if (DIMENSION_ORDER == PDX::DISTANCE_TO_MEANS){
            ALGORITHM = "pdx-bond-dtm";
        }
        else if (DIMENSION_ORDER == PDX::DECREASING_IMPROVED){
            ALGORITHM = "pdx-bond-dec";
        }
        else if (DIMENSION_ORDER == PDX::SEQUENTIAL){
            ALGORITHM = "pdx-bond-sec";
        }
        else if (DIMENSION_ORDER == PDX::DIMENSION_ZONES){
            ALGORITHM = "pdx-bond-dz";
        }

    }
    std::cout << "==> PDX IVF BOND\n";

    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    float SELECTIVITY_THRESHOLD = BenchmarkUtils::SELECTIVITY_THRESHOLD;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "IVF_DSM_BOND.csv";


    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        PDX::IndexPDXIVFFlat pdx_data = PDX::IndexPDXIVFFlat();

        pdx_data.RestoreDSM(BenchmarkUtils::DSM_BOND_DATA + dataset + "-ivf");
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = ((uint32_t *)query)[0];
        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *)ground_truth;
        query += 1; // skip number of embeddings

        PDX::PDXBondSearcher searcher = PDX::PDXBondSearcher(pdx_data, SELECTIVITY_THRESHOLD, 1, 0, DIMENSION_ORDER);
        searcher.SetDSM(true);

        for (size_t ivf_nprobe : BenchmarkUtils::IVF_PROBES) {
            if (pdx_data.num_vectorgroups < ivf_nprobe) {
                continue;
            }
            if (arg_ivf_nprobe > 0 && ivf_nprobe != arg_ivf_nprobe){
                continue;
            }
            std::vector<PhasesRuntime> runtimes;
            runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
            searcher.SetNProbe(ivf_nprobe);

            float recalls = 0;
            if (VERIFY_RESULTS){
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    auto result = searcher.Search(query + l * pdx_data.num_dimensions, KNN);
                    BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
                }
            }
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
            float real_selectivity = 1 - SELECTIVITY_THRESHOLD;
            BenchmarkMetadata results_metadata = {
                    dataset,
                    ALGORITHM,
                    NUM_MEASURE_RUNS,
                    NUM_QUERIES,
                    ivf_nprobe,
                    KNN,
                    recalls,
                    real_selectivity,
            };
            BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
        }
    }
}

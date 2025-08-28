#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#ifndef PDX_USE_EXPLICIT_SIMD
#define PDX_USE_EXPLICIT_SIMD = true
#endif

#include <memory>
#include <iostream>
#include "utils/file_reader.hpp"
#include "index_base/pdx_ivf.hpp"
#include "pruners/bond.hpp"
#include "pdxearch.hpp"
#include "utils/benchmark_utils.hpp"

int main(int argc, char *argv[]) {
    std::string arg_dataset;
    size_t arg_ivf_nprobe = 0;
    std::string ALGORITHM = "pdx-bond";
    PDX::DimensionsOrder DIMENSION_ORDER = PDX::SEQUENTIAL;
    DIMENSION_ORDER = PDX::DIMENSION_ZONES;
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
        DIMENSION_ORDER = static_cast<PDX::DimensionsOrder>(atoi(argv[3]));
        ALGORITHM = "pdx-bond";
        if (DIMENSION_ORDER == PDX::DISTANCE_TO_MEANS){
            ALGORITHM = "pdx-bond-dtm";
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

    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "IVF_PDX_BOND.csv";


    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        PDX::IndexPDXIVF pdx_data = PDX::IndexPDXIVF<PDX::F32>();

        pdx_data.Restore(BenchmarkUtils::PDX_DATA + dataset + "-ivf");

        std::unique_ptr<char[]> query_ptr = MmapFile(BenchmarkUtils::QUERIES_DATA + dataset);
        auto *query = reinterpret_cast<float*>(query_ptr.get());

        NUM_QUERIES = 1000;
        std::unique_ptr<char[]> ground_truth = MmapFile(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_100_norm");
        auto *int_ground_truth = reinterpret_cast<uint32_t*>(ground_truth.get());
        query += 1; // skip number of embeddings

        auto pruner = PDX::BondPruner(pdx_data.num_dimensions);
        PDX::PDXearch searcher = PDX::PDXearch(pdx_data, pruner, 0, DIMENSION_ORDER);

        std::vector<size_t> nprobes_to_use;
        if (arg_ivf_nprobe > 0) {
            nprobes_to_use = {arg_ivf_nprobe};
        } else {
            nprobes_to_use.assign(std::begin(BenchmarkUtils::IVF_PROBES), std::end(BenchmarkUtils::IVF_PROBES));
        }

        for (size_t ivf_nprobe : nprobes_to_use) {
            if (pdx_data.num_clusters < ivf_nprobe) {
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
                            searcher.end_to_end_clock.accum_time
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

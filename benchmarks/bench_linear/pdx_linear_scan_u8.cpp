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
    PDX::DimensionsOrder DIMENSION_ORDER = PDX::SEQUENTIAL;
    std::string ALGORITHM = "vertical-scan";
    if (argc > 1){
        arg_dataset = argv[1];
    }
    std::cout << "==> U8 PDX EXACT LINEAR SCAN\n";


    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    float SELECTIVITY_THRESHOLD = BenchmarkUtils::SELECTIVITY_THRESHOLD;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "EXACT_PDX_u8.csv";

    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        PDX::IndexPDXIVF pdx_data = PDX::IndexPDXIVF<PDX::U8>();
        pdx_data.Restore(BenchmarkUtils::PDX_DATA + dataset + "-u8x4-flat-blocks");
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = 100; // ((uint32_t *)query)[0];
        std::cout << NUM_QUERIES << "\n";
        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *)ground_truth;
        query += 1; // skip number of embeddings

        std::vector<PhasesRuntime> runtimes;
        runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
        PDX::PDXBondSearcher searcher = PDX::PDXBondSearcher<PDX::U8>(pdx_data, 0, 0, PDX::SEQUENTIAL);

        float recalls = 0;
        if (VERIFY_RESULTS){
            for (size_t l = 0; l < NUM_QUERIES; ++l) {
                auto result = searcher.LinearScan(query + l * pdx_data.num_dimensions, KNN);
                BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
            }
        }
        for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
            for (size_t l = 0; l < NUM_QUERIES; ++l) {
                searcher.LinearScan(query + l * pdx_data.num_dimensions, KNN);
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
                0,
                KNN,
                recalls,
                real_selectivity,
        };
        BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
    }
}

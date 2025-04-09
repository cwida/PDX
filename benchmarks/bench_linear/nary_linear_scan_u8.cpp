#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <iostream>
#include "utils/file_reader.hpp"
#include "nary/linear.hpp"
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
    std::cout << "==> N-ary U8 IVF LINEAR SCAN\n";
    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "U8_IVF_BRUTEFORCE.csv";

    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    std::string ALGORITHM = "brute-force";

    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        float *data = MmapFile32(BenchmarkUtils::NARY_DATA + dataset + "-u8");
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = 100; // ((uint32_t *)query)[0];
        query += 1; // skip number of embeddings
        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *)ground_truth;

        uint32_t num_dimensions = ((uint32_t *)data)[0];
        data += 2; // Skip num dimensions and number of partitions (which is always 1 in this case)
        uint32_t num_embeddings = ((uint32_t *)data)[0];
        data += 1;
        uint8_t * vectors = (uint8_t*) data;
        std::vector<PhasesRuntime> runtimes;
        runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
        LinearSearcher<PDX::U8> searcher {num_dimensions, num_embeddings};

        float recalls = 0.0;
        if (VERIFY_RESULTS) {
            for (size_t l = 0; l < NUM_QUERIES; ++l) {
                // check result
                auto result = searcher.Search(query + l * num_dimensions, vectors, KNN);
                BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
            }
        }
        for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
            for (size_t l = 0; l < NUM_QUERIES; ++l) {
                searcher.Search(query + l * num_dimensions, vectors, KNN);
                runtimes[j + l * NUM_MEASURE_RUNS] = {
                        searcher.end_to_end_clock.accum_time
                };
            }
        }
        BenchmarkMetadata results_metadata = {
                dataset,
                ALGORITHM,
                NUM_MEASURE_RUNS,
                NUM_QUERIES,
                0,
                KNN,
                recalls
        };
        BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
    }
}
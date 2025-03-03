#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <vector>
#include <iostream>
#include "utils/file_reader.hpp"
#include "nary/adsampling.h"
#include "utils/benchmark_utils.hpp"

// TODO: Needs to utilize the dual-block layout
int main() {
    std::cout << "==> N-ary ADSampling Bruteforce\n";
    
    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "NARY_ADSAMPLING.csv";

    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    float EPSILON0 = BenchmarkUtils::EPSILON0;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    std::vector<PhasesRuntime> runtimes;
    runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
    std::string ALGORITHM = "adsampling";

    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        int DELTA_D = 32;
        float *data = MmapFile32(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset);
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = ((uint32_t *)query)[0];
        float *_matrix = MmapFile32(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset + "-matrix");
        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *)ground_truth;
        query += 1; // skip number of embeddings
        uint32_t num_dimensions = ((uint32_t *)data)[0];
        data += 1;
        uint32_t num_embeddings = ((uint32_t *)data)[0];
        data += 1;

        if (num_dimensions <= 128) {
            DELTA_D = num_dimensions / 4;
        }

        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, num_dimensions, num_dimensions);
        matrix = matrix.inverse();
        NaryADSamplingSearcher searcher = NaryADSamplingSearcher(matrix, num_dimensions, num_embeddings, EPSILON0, DELTA_D, 0);

        float recalls = 0.0;
        if (VERIFY_RESULTS){
            for (size_t l = 0; l < NUM_QUERIES; ++l) {
                auto result = searcher.Search(query + l * num_dimensions, data, KNN);
                BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
            }
        }
        for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
            for (size_t l = 0; l < NUM_QUERIES; ++l) {
                searcher.Search(query + l * num_dimensions, data, KNN);
                runtimes[j + l * NUM_MEASURE_RUNS] = {
                            searcher.end_to_end_clock.accum_time,
                            searcher.find_nearest_buckets_clock.accum_time,
                            searcher.query_preprocessing_clock.accum_time,
                            searcher.bounds_evaluation_clock.accum_time,
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
    return 0;
}
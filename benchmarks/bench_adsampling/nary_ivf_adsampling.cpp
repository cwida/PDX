#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <vector>
#include <iostream>
#include "utils/file_reader.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "nary/adsampling.h"
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
    std::cout << "==> N-ary ADSampling IVF\n";

    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "IVF_NARY_ADSAMPLING.csv";

    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    float EPSILON0 = BenchmarkUtils::EPSILON0;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    std::string ALGORITHM = "adsampling";

    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        int DELTA_D = 32;
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = ((uint32_t *)query)[0];
        float *_matrix = MmapFile32(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset + "-matrix");
        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *) ground_truth;
        query += 1; // skip number of embeddings

        // Using the dual-block layout
        PDX::IndexPDXIVF nary_dual_block = PDX::IndexPDXIVF<PDX::F32>();
        nary_dual_block.Restore(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset + "-ivf-dual-block");

        uint32_t num_dimensions = nary_dual_block.num_dimensions;
        uint32_t num_embeddings = 1;

        if (num_dimensions <= 128) {
            DELTA_D = num_dimensions / 4;
        }

        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, num_dimensions, num_dimensions);
        matrix = matrix.inverse();

        NaryADSamplingSearcher searcher = NaryADSamplingSearcher(matrix, num_dimensions,
                                                                 num_embeddings, EPSILON0, DELTA_D,
                                                                 1);

        for (size_t ivf_nprobe : BenchmarkUtils::IVF_PROBES) {
            if (nary_dual_block.num_vectorgroups < ivf_nprobe){
                continue;
            }
            if (arg_ivf_nprobe > 0 && ivf_nprobe != arg_ivf_nprobe){
                continue;
            }
            std::vector<PhasesRuntime> runtimes;
            runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);

            searcher.SetNProbe(ivf_nprobe);

            float recalls = 0.0;
            if (VERIFY_RESULTS){
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    auto result = searcher.SearchIVF(query + l * num_dimensions, KNN, nary_dual_block);
                    BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
                }
            }
            for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    searcher.SearchIVF(query + l * num_dimensions, KNN, nary_dual_block);
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
                    ivf_nprobe,
                    KNN,
                    recalls,
                    0,
                    EPSILON0
            };
            BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
        }
    }
    return 0;
}
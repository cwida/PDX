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
#include "pdx/bsa.hpp"
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
    std::cout << "==> PDX IVF BSA\n";

    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    float SELECTIVITY_THRESHOLD = BenchmarkUtils::SELECTIVITY_THRESHOLD;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    PDX::DimensionsOrder DIMENSION_ORDER = PDX::SEQUENTIAL;
    std::string ALGORITHM = "bsa-pdx";

    std::string RESULTS_PATH;
    RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "IVF_PDX_BSA.csv";

    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        float MULTIPLIER_M = BenchmarkUtils::BSA_MULTIPLIERS_M[dataset];
        PDX::IndexPDXIVF pdx_data = PDX::IndexPDXIVF<PDX::F32>();
        pdx_data.Restore(BenchmarkUtils::PDX_BSA_DATA + dataset + "-ivf");
        float *_matrix = MmapFile32(BenchmarkUtils::NARY_BSA_DATA + dataset + "-matrix");
        float *base_square = MmapFile32(BenchmarkUtils::NARY_BSA_DATA + dataset + "-base-square");
        float *dimension_means = MmapFile32(BenchmarkUtils::NARY_BSA_DATA + dataset + "-dimension-means");
        float *dimension_variances = MmapFile32(BenchmarkUtils::NARY_BSA_DATA + dataset + "-dimension-variances");
        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, pdx_data.num_dimensions, pdx_data.num_dimensions);
        matrix = matrix.inverse();

        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = ((uint32_t *)query)[0];
        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *)ground_truth;
        query += 1; // skip number of embeddings

        PDX::BSASearcher searcher = PDX::BSASearcher<PDX::F32>(
            pdx_data,
            1, MULTIPLIER_M, matrix,dimension_variances,
            dimension_means, base_square, DIMENSION_ORDER);

        for (size_t ivf_nprobe : BenchmarkUtils::IVF_PROBES) {
            if (pdx_data.num_vectorgroups < ivf_nprobe){
                continue;
            }
            if (arg_ivf_nprobe > 0 && ivf_nprobe != arg_ivf_nprobe){
                continue;
            }
            std::vector<PhasesRuntime> runtimes;
            runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
            searcher.SetNProbe(ivf_nprobe);

            float recalls = 0;
            if (VERIFY_RESULTS) {
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    auto result = searcher.Search(query + l * pdx_data.num_dimensions, KNN);
                    BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
                }
            }
            // measure time
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
                    MULTIPLIER_M
            };
            BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
        }
    }
    return 0;
}
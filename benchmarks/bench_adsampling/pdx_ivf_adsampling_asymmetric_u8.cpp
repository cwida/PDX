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
#include "pdx/adsampling.hpp"
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
    std::cout << "==> PDX IVF ADSampling\n";

    std::string ALGORITHM = "adsampling";
    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    float EPSILON0 = BenchmarkUtils::EPSILON0;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    PDX::DimensionsOrder DIMENSION_ORDER = PDX::SEQUENTIAL;

    std::string RESULTS_PATH;
    RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "U8_IVF_PDX_ADSAMPLING.csv";


    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        PDX::IndexPDXIVF pdx_data = PDX::IndexPDXIVF<PDX::Quantization::ASYMMETRIC_U8>();
        pdx_data.Restore(BenchmarkUtils::PDX_ADSAMPLING_DATA + dataset + "-u8-v4-h64-ivf");
        float * _matrix = MmapFile32(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset + "-u8-v4-h64-matrix");
//         pdx_data.Restore(BenchmarkUtils::PDX_ADSAMPLING_DATA + dataset + "-u7x4-ivf");
//         float * _matrix = MmapFile32(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset + "-u7-matrix");
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(_matrix, pdx_data.num_dimensions, pdx_data.num_dimensions);
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = 1000; // ((uint32_t *)query)[0];
        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN) + "_norm");
        auto *int_ground_truth = (uint32_t *)ground_truth;
        query += 1; // skip number of embeddings

        uint8_t lep_exponent_idx = BenchmarkUtils::PDX_EXPONENTS[dataset];
        int lep_exponent = BenchmarkUtils::POW_10[lep_exponent_idx];

        // Doing the transformation of many queries at once is much better than one by one
        // This is probably due to better SIMDizing
//        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(query, 1000, pdx_data.num_dimensions);
//        TicToc tt = TicToc();
//        tt.Tic();
//        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> t_queries = q_matrix * matrix;
//        tt.Toc();
//        std::cout << 1.0 * tt.accum_time / 1e6 << "miliseconds\n";
//        std::cout << (1.0 * tt.accum_time / 1e6) / 1000 << "miliseconds per query\n";


        PDX::ADSamplingSearcher searcher = PDX::ADSamplingSearcher<PDX::ASYMMETRIC_U8>(pdx_data, 1, EPSILON0, matrix, DIMENSION_ORDER);
        searcher.SetExponent(lep_exponent);

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
                    BenchmarkUtils::VerifyResult<true, PDX::ASYMMETRIC_U8>(recalls, result, KNN, int_ground_truth, l);
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
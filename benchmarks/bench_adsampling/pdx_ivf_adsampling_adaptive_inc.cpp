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
    std::cout << "==> ADAPTIVENESS EFECTIVENESS PDX IVF ADSampling\n";

    uint8_t KNN = BenchmarkUtils::KNN;
    float SELECTIVITY_THRESHOLD = BenchmarkUtils::SELECTIVITY_THRESHOLD;
    float EPSILON0 = BenchmarkUtils::EPSILON0;
    size_t NUM_QUERIES;

    PDX::PDXearchDimensionsOrder DIMENSION_ORDER = PDX::SEQUENTIAL;

    std::string RESULTS_PATH;
    RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "ADAPTIVENESS_INC_IVF_PDX_ADSAMPLING.csv";
    //RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "ADAPTIVENESS_32_IVF_PDX_ADSAMPLING.csv";

    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        // Searcher 1
        PDX::IndexPDXIVFFlat pdx_data = PDX::IndexPDXIVFFlat();
        pdx_data.Restore(BenchmarkUtils::PDX_ADSAMPLING_DATA + dataset + "-ivf");
        float * _matrix = MmapFile32(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset + "-matrix");
        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, pdx_data.num_dimensions, pdx_data.num_dimensions);
        matrix = matrix.inverse();
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = ((uint32_t *)query)[0];
        query += 1; // skip number of embeddings

        float *ground_truth = MmapFile32(BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *)ground_truth;

        PDX::ADSamplingSearcher searcher = PDX::ADSamplingSearcher(pdx_data, SELECTIVITY_THRESHOLD, 1, EPSILON0, matrix, DIMENSION_ORDER);

        std::vector<size_t> runtimes;
        runtimes.resize(NUM_QUERIES);
        searcher.SetNProbe(arg_ivf_nprobe);

        bool write_header = true;
        if (std::filesystem::exists(RESULTS_PATH)){
            write_header = false;
        }
        std::ofstream file{RESULTS_PATH, std::ios::app};
        if (write_header){
            file << "dataset,query,avg,knn,ivf_nprobe" << "\n";
        }
        size_t REPETITION_NUMBER = 100;
        NUM_QUERIES = 1000;
        for (size_t l = 0; l < NUM_QUERIES; ++l) {
            for (size_t i = 0; i < 10; i ++){
                searcher.Search(query + l * pdx_data.num_dimensions, KNN);
            }
            runtimes[l] = 0;
            for (size_t i = 0; i < REPETITION_NUMBER; i ++){
                TicToc query_clock = TicToc();
                query_clock.Tic();
                auto result = searcher.Search(query + l * pdx_data.num_dimensions, KNN);
                query_clock.Toc();
                float recalls = 0;
                BenchmarkUtils::VerifyResult<true>(recalls, result, KNN, int_ground_truth, l);
                runtimes[l] += query_clock.accum_time;
            }
        }
        for (size_t i = 0; i < NUM_QUERIES; ++i){
            float runtime = ((1.0 * runtimes[i]) / (1.0 * REPETITION_NUMBER)) / 1000000;
            file << dataset << "," << i << "," << std::setprecision(6) << runtime << "," << +KNN << "," << arg_ivf_nprobe << "\n";
        }
        file.close();
    }
    return 0;
}

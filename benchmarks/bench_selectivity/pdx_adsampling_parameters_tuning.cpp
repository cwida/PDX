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
    std::cout << "Doing something...\n";

    const bool USE_IVF = true;
    const bool VERIFY_RESULTS = BenchmarkUtils::VERIFY_RESULTS;

    uint8_t KNN = BenchmarkUtils::KNN;
    float EPSILON0 = BenchmarkUtils::EPSILON0;
    size_t NUM_QUERIES;
    size_t NUM_MEASURE_RUNS = BenchmarkUtils::NUM_MEASURE_RUNS;

    PDX::PDXearchDimensionsOrder DIMENSION_ORDER = PDX::SEQUENTIAL;
    std::string ALGORITHM = "adsampling";

    std::string RESULTS_PATH;
    if (USE_IVF) {
        RESULTS_PATH =  BENCHMARK_UTILS.RESULTS_DIR_PATH + "IVF_PDX_ADSAMPLING_TUNING.csv";
    }

    size_t IVF_NPROBE = 128;
    if (arg_ivf_nprobe > 0){
        IVF_NPROBE = arg_ivf_nprobe;
    }

    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        PDX::IndexPDXIVFFlat pdx_data = PDX::IndexPDXIVFFlat();
        if (USE_IVF) {
            pdx_data.Restore(BenchmarkUtils::PDX_ADSAMPLING_DATA + dataset + "-ivf");
        } else {
            pdx_data.Restore(BenchmarkUtils::PDX_ADSAMPLING_DATA + dataset + "-flat");
        }

        float * _matrix = MmapFile32(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset + "-matrix");
        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, pdx_data.num_dimensions, pdx_data.num_dimensions);
        matrix = matrix.inverse();
        float *query = MmapFile32(BenchmarkUtils::QUERIES_DATA + dataset);
        NUM_QUERIES = ((uint32_t *)query)[0];
        float *ground_truth = MmapFile32( BenchmarkUtils::GROUND_TRUTH_DATA + dataset + "_" + std::to_string(KNN));
        auto *int_ground_truth = (uint32_t *)ground_truth;
        query += 1; // skip number of embeddings

        PDX::ADSamplingSearcher searcher = PDX::ADSamplingSearcher(pdx_data, 0.0, IVF_NPROBE, EPSILON0, matrix, DIMENSION_ORDER);

        for (float selectivity_thres: BenchmarkUtils::SELECTIVITY_THRESHOLDS){
                if (USE_IVF && pdx_data.num_vectorgroups < IVF_NPROBE){
                    continue;
                }
                std::vector<PhasesRuntime> runtimes;
                runtimes.resize(NUM_MEASURE_RUNS * NUM_QUERIES);
                searcher.SetSelectivityThreshold(selectivity_thres);

                float recalls = 0;
                if (VERIFY_RESULTS) {
                    for (size_t l = 0; l < NUM_QUERIES; ++l) {
                        auto result = searcher.Search(query + l * pdx_data.num_dimensions, KNN);
                        BenchmarkUtils::VerifyResult<USE_IVF>(recalls, result, KNN, int_ground_truth, l);
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
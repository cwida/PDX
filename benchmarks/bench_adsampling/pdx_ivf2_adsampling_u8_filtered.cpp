#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#ifndef PDX_USE_EXPLICIT_SIMD
#define PDX_USE_EXPLICIT_SIMD = true
#endif

#include <memory>
#include <iostream>
#include "utils/file_reader.hpp"
#include "index_base/pdx_ivf2.hpp"
#include "pruners/adsampling.hpp"
#include "pdxearch.hpp"
#include "db_mock/predicate_evaluator.hpp"
#include "utils/benchmark_utils.hpp"

int main(int argc, char *argv[]) {
    std::string arg_dataset;
    std::string arg_selectivity;
    size_t arg_ivf_nprobe = 0;
    if (argc > 1){
        arg_dataset = argv[1];
    }
    if (argc > 2){
        arg_ivf_nprobe = atoi(argv[2]);
    }
    if (argc > 3){
        arg_selectivity = argv[3];
    } else {
        arg_selectivity = "0_99";
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
    RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "U8_IMI_PDX_ADSAMPLING_FILTERED.csv";

    std::cout << "==> SELECTIVITY: " << arg_selectivity << std::endl;
    for (const auto & dataset : BenchmarkUtils::DATASETS) {
        if (arg_dataset.size() > 0 && arg_dataset != dataset){
            continue;
        }
        PDX::IndexPDXIVF2 pdx_data = PDX::IndexPDXIVF2<PDX::Quantization::U8>();
        pdx_data.Restore(BenchmarkUtils::PDX_ADSAMPLING_DATA + dataset + "-ivf2-u8");
        std::unique_ptr<char[]> _matrix_ptr = MmapFile(BenchmarkUtils::NARY_ADSAMPLING_DATA + dataset + "-ivf2-u8-matrix");
        auto *_matrix = reinterpret_cast<float*>(_matrix_ptr.get());

        std::unique_ptr<char[]> query_ptr = MmapFile(BenchmarkUtils::QUERIES_DATA + dataset);
        auto *query = reinterpret_cast<float*>(query_ptr.get());

        NUM_QUERIES = 1000;
        std::unique_ptr<char[]> ground_truth = MmapFile(BenchmarkUtils::FILTERED_GROUND_TRUTH_DATA + dataset + "_100_norm_" + arg_selectivity);
        auto *int_ground_truth = reinterpret_cast<uint32_t*>(ground_truth.get());
        query += 1; // skip number of embeddings

        PDX::PredicateEvaluator predicate_evaluator = PDX::PredicateEvaluator(pdx_data.num_clusters);
        predicate_evaluator.LoadSelectionVectorFromFile(BenchmarkUtils::SELECTION_VECTOR_DATA + dataset + "_" + arg_selectivity + ".bin");
        PDX::ADSamplingPruner pruner = PDX::ADSamplingPruner<PDX::U8>(pdx_data.num_dimensions, EPSILON0, _matrix);
        PDX::PDXearch searcher = PDX::PDXearch<PDX::U8, PDX::IndexPDXIVF2<PDX::U8>>(pdx_data, pruner, 1, DIMENSION_ORDER);

        std::vector<size_t> nprobes_to_use;
        if (arg_ivf_nprobe > 0) {
            nprobes_to_use = {arg_ivf_nprobe};
        } else {
            nprobes_to_use.assign(std::begin(BenchmarkUtils::IVF_PROBES), std::end(BenchmarkUtils::IVF_PROBES));
        }

        for (size_t ivf_nprobe : nprobes_to_use) {
            if (pdx_data.num_clusters < ivf_nprobe){
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
                    auto result = searcher.FilteredSearch(query + l * pdx_data.num_dimensions, KNN, predicate_evaluator);
                    BenchmarkUtils::VerifyResult<true, PDX::U8>(recalls, result, KNN, int_ground_truth, l);
                }
            }
            for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
                for (size_t l = 0; l < NUM_QUERIES; ++l) {
                    searcher.FilteredSearch(query + l * pdx_data.num_dimensions, KNN, predicate_evaluator);
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
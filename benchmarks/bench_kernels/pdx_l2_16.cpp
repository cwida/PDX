#include <iostream>
#include <string>
#include "distance_computers.hpp"


int main(int argc, char *argv[]) {
    std::string ALGORITHM = "pdx";
    size_t DIMENSION;
    size_t N_VECTORS;
    if (argc > 1){
        N_VECTORS = atoi(argv[1]);
    }
    if (argc > 2){
        DIMENSION = atoi(argv[2]);
    }
    std::cout << "==> PURE SCAN PDX\n";

    size_t NUM_WARMUP_RUNS = 30;
    size_t NUM_MEASURE_RUNS = 300;
    if (N_VECTORS <= 8192){
        NUM_WARMUP_RUNS = 300;
        NUM_MEASURE_RUNS = 3000;
    }

    std::cout << "RUNS: " << NUM_MEASURE_RUNS << "\n";

    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "PURESCAN_PDX_L2_16.csv";

    std::string filename = "16x" + std::to_string(N_VECTORS) + "x"+ std::to_string(DIMENSION) + "-pdx";
    std::string dataset = std::to_string(N_VECTORS) + "x" + std::to_string(DIMENSION);

    float *raw_data = MmapFile32( BenchmarkUtils::PURESCAN_DATA + filename);
    float query[DIMENSION];
    memcpy(query, raw_data, sizeof(float) * DIMENSION);
    raw_data += DIMENSION;

    std::vector<PhasesRuntime> runtimes;
    runtimes.resize(NUM_MEASURE_RUNS);

    float * data;
    size_t skipping_size = DIMENSION * 16;
    size_t vector_chunks = N_VECTORS / 16;
    for (size_t j = 0; j < NUM_WARMUP_RUNS; ++j) {
        data = raw_data;
        for (size_t vector_chunk = 0; vector_chunk < vector_chunks; ++vector_chunk){
            PDXScanner<PDX::L2, 16>::ResetDistances();
            PDXScanner<PDX::L2, 16>::CalculateVerticalDistancesVectorized(query, data, DIMENSION);
            data += skipping_size;
        }
    }
    std::cout << std::setprecision(16) << PDXScanner<PDX::L2, 16>::distances[16 - 1] << "\n";
    std::cout << std::setprecision(16) << PDXScanner<PDX::L2, 16>::distances[0] << "\n";

    for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
        data = raw_data;
        TicToc clock = TicToc();
        for (size_t vector_chunk = 0; vector_chunk < vector_chunks; ++vector_chunk){
            PDXScanner<PDX::L2, 16>::ResetDistances();
            PDXScanner<PDX::L2, 16>::CalculateVerticalDistancesVectorized(query, data, DIMENSION);
            data += skipping_size;
        }
        clock.Toc();
        size_t benchmark_time = clock.accum_time;
        runtimes[j] = {benchmark_time};
    }
    std::cout << std::setprecision(16) << PDXScanner<PDX::L2, 16>::distances[16 - 1] << "\n";

    BenchmarkMetadata results_metadata = {
            dataset,
            ALGORITHM,
            NUM_MEASURE_RUNS,
            NUM_WARMUP_RUNS,
            1,
            0,
            0
    };
    BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
}

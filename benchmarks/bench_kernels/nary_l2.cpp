#include <iostream>
#include <string>
#include "standalone_distance_computers.hpp"


void __attribute__((noinline)) work(const float *__restrict query, const float *__restrict data, size_t DIMENSION, float *__restrict distances, size_t vector_idx){
	distances[vector_idx] = SIMDScanner<PDX::L2>::CalculateDistance(query, data, DIMENSION);
}

int main(int argc, char *argv[]) {
    std::string ALGORITHM = "simd";
    size_t DIMENSION;
    size_t N_VECTORS;
    if (argc > 1){
        N_VECTORS = atoi(argv[1]);
    }
    if (argc > 2){
        DIMENSION = atoi(argv[2]);
    }
    std::cout << "==> PURE SCAN SIMD\n";

    size_t NUM_WARMUP_RUNS = 300;
    size_t NUM_MEASURE_RUNS = 3000;
    if (N_VECTORS <= 8192){
        NUM_WARMUP_RUNS = 300;
        NUM_MEASURE_RUNS = 3000;
    }

    std::cout << "RUNS: " << NUM_MEASURE_RUNS << "\n";

    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "PURESCAN_SIMD_L2.csv";

    std::string filename = std::to_string(N_VECTORS) + "x"+ std::to_string(DIMENSION);
    std::string dataset = std::to_string(N_VECTORS) + "x" + std::to_string(DIMENSION);

    float *raw_data = MmapFile32( BenchmarkUtils::PURESCAN_DATA + filename);
    float query[DIMENSION];
    memcpy(query, raw_data, sizeof(float) * DIMENSION);
    raw_data += DIMENSION;

    std::vector<PhasesRuntime> runtimes;
    runtimes.resize(NUM_MEASURE_RUNS);

    float * data;
    alignas(64) float distances[N_VECTORS];
    for (size_t j = 0; j < NUM_WARMUP_RUNS; ++j) {
        data = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx){
            work(query, data, DIMENSION, distances, vector_idx);
            data += DIMENSION;
        }
    }
    std::cout << std::setprecision(16) << distances[N_VECTORS-1] << "\n"; // Dummy print so the compiler does the job

    for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
        data = raw_data;
        TicToc clock = TicToc();
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx){
            work(query, data, DIMENSION, distances, vector_idx);
            data += DIMENSION;
        }
    	clock.Toc();
        size_t benchmark_time = clock.accum_time;
        runtimes[j] = {benchmark_time};
    }
    std::cout << std::setprecision(16) << distances[N_VECTORS-1] << "\n"; // Dummy print so the compiler does the job

    BenchmarkMetadata results_metadata = {
            dataset,
            ALGORITHM,
            NUM_MEASURE_RUNS,
            1,
            0,
            0
    };
    BenchmarkUtils::SaveResults(runtimes, RESULTS_PATH, results_metadata);
}

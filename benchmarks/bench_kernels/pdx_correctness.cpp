#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <iostream>
#include <string>
#include "distance_computers.hpp"


int main(int argc, char *argv[]) {
    std::string ALGORITHM = "simd";
    size_t DIMENSION;
    size_t N_VECTORS;
    size_t N_RUNS = 1;
    size_t WARMUP_RUNS = 0;
    if (argc > 1) {
        N_VECTORS = atoi(argv[1]);
    }
    if (argc > 2) {
        DIMENSION = atoi(argv[2]);
    }
    if (argc > 3){
        WARMUP_RUNS = atoi(argv[3]);
    }
    if (argc > 4){
        N_RUNS = atoi(argv[4]);
    }
    std::cout << "==> CORRECTNESS ANALYSIS\n";

    std::string filename = std::to_string(N_VECTORS) + "x" + std::to_string(DIMENSION);
    float *raw_data = MmapFile32(BenchmarkUtils::PURESCAN_DATA + filename);
    float query[DIMENSION];
    memcpy(query, raw_data, sizeof(float) * DIMENSION);
    raw_data += DIMENSION;
    float *data;

    // SIMD SCANNER
    float distances[N_VECTORS];
    for (size_t n_runs = 0; n_runs < WARMUP_RUNS; ++n_runs) {
        data = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx) {
            distances[vector_idx] = SIMDScanner<PDX::IP>::CalculateDistance(query, data, DIMENSION);
            data += DIMENSION;
        }
    }
    TicToc clock_simd = TicToc();
    for (size_t n_runs = 0; n_runs < N_RUNS; ++n_runs){
        data = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx) {
            distances[vector_idx] = SIMDScanner<PDX::IP>::CalculateDistance(query, data, DIMENSION);
            data += DIMENSION;
        }
    }
    clock_simd.Toc();
    size_t benchmark_time_simd = clock_simd.accum_time;


    // SCALAR SCANNER
    float * data_scalar;
    float distances_scalar[N_VECTORS];
    for (size_t n_runs = 0; n_runs < WARMUP_RUNS; ++n_runs) {
        data_scalar = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx) {
            distances_scalar[vector_idx] = ScalarScanner<PDX::IP>::CalculateDistance(query, data_scalar, DIMENSION);
            data_scalar += DIMENSION;
        }
    }
    TicToc clock_scalar = TicToc();
    for (size_t n_runs = 0; n_runs < N_RUNS; ++n_runs) {
        data_scalar = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx) {
            distances_scalar[vector_idx] = ScalarScanner<PDX::IP>::CalculateDistance(query, data_scalar, DIMENSION);
            data_scalar += DIMENSION;
        }
    }
    clock_scalar.Toc();
    size_t benchmark_time_scalar = clock_scalar.accum_time;


    // PRECISE SCANNER
    float * data_precise;
    double distances_precise[N_VECTORS];
    for (size_t n_runs = 0; n_runs < WARMUP_RUNS; ++n_runs) {
        data_precise = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx) {
            distances_precise[vector_idx] = ScalarScanner<PDX::IP>::CalculatePreciseDistance(query, data_precise, DIMENSION);
            data_precise += DIMENSION;
        }
    }
    TicToc clock_precise = TicToc();
    for (size_t n_runs = 0; n_runs < N_RUNS; ++n_runs) {
        data_precise = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx) {
            distances_precise[vector_idx] = ScalarScanner<PDX::IP>::CalculatePreciseDistance(query, data_precise, DIMENSION);
            data_precise += DIMENSION;
        }
    }
    clock_scalar.Toc();
    size_t benchmark_time_precise = clock_scalar.accum_time;


    // IMPRECISE SCANNER
    float * data_imprecise;
    float distances_imprecise[N_VECTORS];
    for (size_t n_runs = 0; n_runs < WARMUP_RUNS; ++n_runs) {
        data_imprecise = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx) {
            distances_imprecise[vector_idx] = ImpreciseScanner<PDX::IP>::CalculateDistance(query, data_imprecise, DIMENSION);
            data_imprecise += DIMENSION;
        }
    }
    TicToc clock_imprecise = TicToc();
    for (size_t n_runs = 0; n_runs < N_RUNS; ++n_runs) {
        data_imprecise = raw_data;
        for (size_t vector_idx = 0; vector_idx < N_VECTORS; ++vector_idx) {
            distances_imprecise[vector_idx] = ImpreciseScanner<PDX::IP>::CalculateDistance(query, data_imprecise, DIMENSION);
            data_imprecise += DIMENSION;
        }
    }
    clock_imprecise.Toc();
    size_t benchmark_time_imprecise = clock_imprecise.accum_time;

    // PDX SCANNER
    std::string filename_pdx = "64x" + std::to_string(N_VECTORS) + "x"+ std::to_string(DIMENSION) + "-pdx";
    float *raw_data_pdx = MmapFile32( BenchmarkUtils::PURESCAN_DATA + filename_pdx);
    float query_pdx[DIMENSION];
    memcpy(query_pdx, raw_data_pdx, sizeof(float) * DIMENSION);
    raw_data_pdx += DIMENSION;
    float * data_pdx;
    size_t skipping_size = DIMENSION * PDXScanner<PDX::IP>::PDX_VECTOR_SIZE;
    size_t vector_chunks = N_VECTORS / PDXScanner<PDX::IP>::PDX_VECTOR_SIZE;
    for (size_t n_runs = 0; n_runs < WARMUP_RUNS; ++n_runs) {
        data_pdx = raw_data_pdx;
        for (size_t vector_chunk = 0; vector_chunk < vector_chunks; ++vector_chunk) {
            PDXScanner<PDX::IP>::ResetDistances();
            PDXScanner<PDX::IP>::CalculateVerticalDistancesVectorized(query_pdx, data_pdx, DIMENSION);
            data_pdx += skipping_size;
        }
    }
    TicToc clock_pdx = TicToc();
    for (size_t n_runs = 0; n_runs < N_RUNS; ++n_runs) {
        data_pdx = raw_data_pdx;
        for (size_t vector_chunk = 0; vector_chunk < vector_chunks; ++vector_chunk) {
            PDXScanner<PDX::IP>::ResetDistances();
            PDXScanner<PDX::IP>::CalculateVerticalDistancesVectorized(query_pdx, data_pdx, DIMENSION);
            data_pdx += skipping_size;
        }
    }
    clock_pdx.Toc();
    size_t benchmark_time_pdx = clock_pdx.accum_time;

    std::cout << "Scalar:\t\t" << std::setprecision(16) << benchmark_time_scalar << "\n";
    std::cout << "Precise:\t" << std::setprecision(16) << benchmark_time_precise << "\n";
    std::cout << "SIMD:\t\t" << std::setprecision(16) << benchmark_time_simd << "\n";
    std::cout << "PDX:\t\t" << std::setprecision(16) << benchmark_time_pdx << "\n";
    std::cout << "IMP:\t\t" << std::setprecision(16) << benchmark_time_imprecise << "\n";

    std::cout << "PDX Distance:\t\t" <<  std::setprecision(16) << PDXScanner<PDX::IP>::distances[PDXScanner<>::PDX_VECTOR_SIZE - 1] << "\n";
    std::cout << "Scalar Distance:\t" <<  std::setprecision(16) << distances_scalar[N_VECTORS-1] << "\n";
    std::cout << "Precise Distance:\t" <<  std::setprecision(16) << distances_precise[N_VECTORS-1] << "\n";
    std::cout << "SIMD Distance:\t\t" <<  std::setprecision(16) << distances[N_VECTORS-1] << "\n";
    std::cout << "IMP Distance:\t\t" <<  std::setprecision(16) << distances_imprecise[N_VECTORS-1] << "\n";

}

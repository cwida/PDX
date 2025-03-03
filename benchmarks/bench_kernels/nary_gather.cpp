#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif

#ifdef __AVX2__ 
#include <immintrin.h>
#endif


#include <iostream>
#include <string>
#include <limits>
#include "utils/file_reader.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "pdx/bond.hpp"
#include "utils/benchmark_utils.hpp"


template<PDX::DistanceFunction ALPHA=PDX::L2>
class GatherPDXScanner {

public:
    static constexpr uint16_t PDX_VECTOR_SIZE = 64;
    alignas(64) inline static float distances[PDX_VECTOR_SIZE];

    static void ResetDistances(){
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(float));
    }

#if defined(__ARM_NEON)
    alignas(32) static inline size_t indixes_to_gather[64];
    static void FillIndexesToGather(size_t end_dimension){
        uint32_t increment = end_dimension;
        uint32_t start_0 = ((0 * 16) * increment);
        uint32_t start_1 = ((1 * 16) * increment);
        uint32_t start_2 = ((2 * 16) * increment);
        uint32_t start_3 = ((3 * 16) * increment);
        indixes_to_gather[0] = start_0 + (0x00 * increment);
        indixes_to_gather[1] = start_0 + (0x01 * increment);
        indixes_to_gather[2] = start_0 + (0x02 * increment);
        indixes_to_gather[3] = start_0 + (0x03 * increment);
        indixes_to_gather[4] = start_0 + (0x04 * increment);
        indixes_to_gather[5] = start_0 + (0x05 * increment);
        indixes_to_gather[6] = start_0 + (0x06 * increment);
        indixes_to_gather[7] = start_0 + (0x07 * increment);
        indixes_to_gather[8] = start_0 + (0x08 * increment);
        indixes_to_gather[9] = start_0 + (0x09 * increment);
        indixes_to_gather[10] = start_0 + (0x0A * increment);
        indixes_to_gather[11] = start_0 + (0x0B * increment);
        indixes_to_gather[12] = start_0 + (0x0C * increment);
        indixes_to_gather[13] = start_0 + (0x0D * increment);
        indixes_to_gather[14] = start_0 + (0x0E * increment);
        indixes_to_gather[15] = start_0 + (0x0F * increment);

        indixes_to_gather[16] = start_1 + (0x00 * increment);
        indixes_to_gather[17] = start_1 + (0x01 * increment);
        indixes_to_gather[18] = start_1 + (0x02 * increment);
        indixes_to_gather[19] = start_1 + (0x03 * increment);
        indixes_to_gather[20] = start_1 + (0x04 * increment);
        indixes_to_gather[21] = start_1 + (0x05 * increment);
        indixes_to_gather[22] = start_1 + (0x06 * increment);
        indixes_to_gather[23] = start_1 + (0x07 * increment);
        indixes_to_gather[24] = start_1 + (0x08 * increment);
        indixes_to_gather[25] = start_1 + (0x09 * increment);
        indixes_to_gather[26] = start_1 + (0x0A * increment);
        indixes_to_gather[27] = start_1 + (0x0B * increment);
        indixes_to_gather[28] = start_1 + (0x0C * increment);
        indixes_to_gather[29] = start_1 + (0x0D * increment);
        indixes_to_gather[30] = start_1 + (0x0E * increment);
        indixes_to_gather[31] = start_1 + (0x0F * increment);

        indixes_to_gather[32] = start_2 + (0x00 * increment);
        indixes_to_gather[33] = start_2 + (0x01 * increment);
        indixes_to_gather[34] = start_2 + (0x02 * increment);
        indixes_to_gather[35] = start_2 + (0x03 * increment);
        indixes_to_gather[36] = start_2 + (0x04 * increment);
        indixes_to_gather[37] = start_2 + (0x05 * increment);
        indixes_to_gather[38] = start_2 + (0x06 * increment);
        indixes_to_gather[39] = start_2 + (0x07 * increment);
        indixes_to_gather[40] = start_2 + (0x08 * increment);
        indixes_to_gather[41] = start_2 + (0x09 * increment);
        indixes_to_gather[42] = start_2 + (0x0A * increment);
        indixes_to_gather[43] = start_2 + (0x0B * increment);
        indixes_to_gather[44] = start_2 + (0x0C * increment);
        indixes_to_gather[45] = start_2 + (0x0D * increment);
        indixes_to_gather[46] = start_2 + (0x0E * increment);
        indixes_to_gather[47] = start_2 + (0x0F * increment);

        indixes_to_gather[48] = start_3 + (0x00 * increment);
        indixes_to_gather[49] = start_3 + (0x01 * increment);
        indixes_to_gather[50] = start_3 + (0x02 * increment);
        indixes_to_gather[51] = start_3 + (0x03 * increment);
        indixes_to_gather[52] = start_3 + (0x04 * increment);
        indixes_to_gather[53] = start_3 + (0x05 * increment);
        indixes_to_gather[54] = start_3 + (0x06 * increment);
        indixes_to_gather[55] = start_3 + (0x07 * increment);
        indixes_to_gather[56] = start_3 + (0x08 * increment);
        indixes_to_gather[57] = start_3 + (0x09 * increment);
        indixes_to_gather[58] = start_3 + (0x0A * increment);
        indixes_to_gather[59] = start_3 + (0x0B * increment);
        indixes_to_gather[60] = start_3 + (0x0C * increment);
        indixes_to_gather[61] = start_3 + (0x0D * increment);
        indixes_to_gather[62] = start_3 + (0x0E * increment);
        indixes_to_gather[63] = start_3 + (0x0F * increment);
    }
#elif defined(__AVX512F__)
    static inline __m512i indexes_to_gather[4];

    static void FillIndexesToGather(size_t end_dimension){
        uint32_t increment = end_dimension;
        uint32_t start_0 = ((0 * 16) * increment);
        uint32_t start_1 = ((1 * 16) * increment);
        uint32_t start_2 = ((2 * 16) * increment);
        uint32_t start_3 = ((3 * 16) * increment);
        indexes_to_gather[0] = _mm512_set_epi32(
                start_0 +  (0x0F * increment),
                start_0 +  (0x0E * increment),
                start_0 +  (0x0D * increment),
                start_0 +  (0x0C * increment),
                start_0 +  (0x0B * increment),
                start_0 +  (0x0A * increment),
                start_0 +  (0x09 * increment),
                start_0 +  (0x08 * increment),
                start_0 +  (0x07 * increment),
                start_0 +  (0x06 * increment),
                start_0 +  (0x05 * increment),
                start_0 +  (0x04 * increment),
                start_0 +  (0x03 * increment),
                start_0 +  (0x02 * increment),
                start_0 +  (0x01 * increment),
                start_0 +  (0x00 * increment)
        );
        indexes_to_gather[1] = _mm512_set_epi32(
                start_1 + (0x0F * increment),
                start_1 + (0x0E * increment),
                start_1 + (0x0D * increment),
                start_1 + (0x0C * increment),
                start_1 + (0x0B * increment),
                start_1 + (0x0A * increment),
                start_1 + (0x09 * increment),
                start_1 + (0x08 * increment),
                start_1 + (0x07 * increment),
                start_1 + (0x06 * increment),
                start_1 + (0x05 * increment),
                start_1 + (0x04 * increment),
                start_1 + (0x03 * increment),
                start_1 + (0x02 * increment),
                start_1 + (0x01 * increment),
                start_1 + (0x00 * increment)
        );
        indexes_to_gather[2] = _mm512_set_epi32(
                start_2 + (0x0F * increment),
                start_2 + (0x0E * increment),
                start_2 + (0x0D * increment),
                start_2 + (0x0C * increment),
                start_2 + (0x0B * increment),
                start_2 + (0x0A * increment),
                start_2 + (0x09 * increment),
                start_2 + (0x08 * increment),
                start_2 + (0x07 * increment),
                start_2 + (0x06 * increment),
                start_2 + (0x05 * increment),
                start_2 + (0x04 * increment),
                start_2 + (0x03 * increment),
                start_2 + (0x02 * increment),
                start_2 + (0x01 * increment),
                start_2 + (0x00 * increment)
        );
        indexes_to_gather[3] = _mm512_set_epi32(
                start_3 + (0x0F * increment),
                start_3 + (0x0E * increment),
                start_3 + (0x0D * increment),
                start_3 + (0x0C * increment),
                start_3 + (0x0B * increment),
                start_3 + (0x0A * increment),
                start_3 + (0x09 * increment),
                start_3 + (0x08 * increment),
                start_3 + (0x07 * increment),
                start_3 + (0x06 * increment),
                start_3 + (0x05 * increment),
                start_3 + (0x04 * increment),
                start_3 + (0x03 * increment),
                start_3 + (0x02 * increment),
                start_3 + (0x01 * increment),
                start_3 + (0x00 * increment)
        );
    }
#endif

    static void CalculateVerticalDistancesVectorized(
        const float *__restrict query, float *__restrict data, size_t end_dimension, size_t n_vectors, size_t current_vector,
        const uint32_t *__restrict indexes_array, size_t &index_offset){
#if defined(__ARM_NEON)
        float32x4_t vec2;
        float32x4_t d_vec;
        float32x4_t res[16];
        for (size_t i = 0; i < 16; ++i){
            res[i] = vld1q_f32(&distances[i * 4]);
        }
        for (size_t dim_idx = 0; dim_idx < end_dimension; dim_idx++) {
            float32x4_t vec1 = vdupq_n_f32(query[dim_idx]);
            for (int i = 0; i < 16; ++i) {
                vec2 = {
                        data[indixes_to_gather[i * 4]],
                        data[indixes_to_gather[i * 4 + 1]],
                        data[indixes_to_gather[i * 4 + 2]],
                        data[indixes_to_gather[i * 4 + 3]],
                };
                d_vec = vsubq_f32(vec1, vec2);
                res[i] = vfmaq_f32(res[i], d_vec, d_vec);
            }
            data +=1;
        }
        for (int i = 0; i < 16; ++i) {
            vst1q_f32(&distances[i * 4], res[i]);
        }
#elif defined(__AVX512F__)
        __m512 vec2;
        __m512 d_vec;
        __m512 res[4];
        for (size_t i = 0; i < 4; ++i){
            res[i] = _mm512_load_ps(&distances[i * 16]);
        }
        for (size_t dim_idx = 0; dim_idx < end_dimension; dim_idx++) {
            __m512 vec1 = _mm512_set1_ps(query[dim_idx]);
            for (size_t i = 0; i < 4; i++){
                vec2 = _mm512_i32gather_ps(indexes_to_gather[i], data, 4);
                d_vec = _mm512_sub_ps(vec1, vec2);
                res[i] = _mm512_fmadd_ps(d_vec, d_vec, res[i]);
            }
            data += 1;
        }
        for (int i = 0; i < 4; ++i) {
            _mm512_store_ps(&distances[i * 16], res[i]); 
        }
#endif
    }
};

void __attribute__((noinline)) work(float *__restrict data, const float *__restrict query, size_t DIMENSION, size_t N_VECTORS, size_t cur_vector, const uint32_t *__restrict indexes_array, size_t &index_offset){
            GatherPDXScanner<PDX::L2>::ResetDistances();
            GatherPDXScanner<PDX::L2>::CalculateVerticalDistancesVectorized(query, data, DIMENSION, N_VECTORS, cur_vector, indexes_array, index_offset);
}


int main(int argc, char *argv[]) {
    std::string ALGORITHM = "gather";
    size_t DIMENSION;
    size_t N_VECTORS;
    if (argc > 1){
        N_VECTORS = atoi(argv[1]);
    }
    if (argc > 2){
        DIMENSION = atoi(argv[2]);
    }
    std::cout << "==> PURE SCAN GATHER\n";

    size_t NUM_WARMUP_RUNS = 10;
    size_t NUM_MEASURE_RUNS = 100;
    if (N_VECTORS <= 8192){
        NUM_WARMUP_RUNS = 300;
        NUM_MEASURE_RUNS = 3000;
    }

    std::cout << "RUNS: " << NUM_MEASURE_RUNS << "\n";

    std::string RESULTS_PATH = BENCHMARK_UTILS.RESULTS_DIR_PATH + "PURESCAN_GATHER_L2.csv";

    std::string filename = std::to_string(N_VECTORS) + "x"+ std::to_string(DIMENSION);
    std::string dataset = std::to_string(N_VECTORS) + "x" + std::to_string(DIMENSION);

    float *raw_data = MmapFile32( BenchmarkUtils::PURESCAN_DATA + filename);
    float query[DIMENSION];
    memcpy(query, raw_data, sizeof(float) * DIMENSION);
    raw_data += DIMENSION;

    std::vector<PhasesRuntime> runtimes;
    runtimes.resize(NUM_MEASURE_RUNS);

    float * data;
    size_t skipping_size = DIMENSION * GatherPDXScanner<PDX::L2>::PDX_VECTOR_SIZE;
    alignas(64) uint32_t indexes_array[64];

    GatherPDXScanner<PDX::L2>::FillIndexesToGather(DIMENSION);
    size_t vector_chunks = N_VECTORS / GatherPDXScanner<PDX::L2>::PDX_VECTOR_SIZE;
    for (size_t j = 0; j < NUM_WARMUP_RUNS; ++j) {
        size_t cur_vector = 0;
        size_t index_offset = 0;
        data = raw_data;
        for (size_t vector_chunk = 0; vector_chunk < vector_chunks; ++vector_chunk){
            work(data, query, DIMENSION, N_VECTORS, cur_vector, indexes_array, index_offset);
            cur_vector += GatherPDXScanner<PDX::L2>::PDX_VECTOR_SIZE;
            data += skipping_size;
        }
    }
    std::cout << std::setprecision(16) << GatherPDXScanner<PDX::L2>::distances[GatherPDXScanner<>::PDX_VECTOR_SIZE - 1] << "\n";


    for (size_t j = 0; j < NUM_MEASURE_RUNS; ++j) {
        size_t cur_vector = 0;
        size_t index_offset = 0;
        data = raw_data;
        TicToc clock = TicToc();
        for (size_t vector_chunk = 0; vector_chunk < vector_chunks; ++vector_chunk){
            work(data, query, DIMENSION, N_VECTORS, cur_vector, indexes_array, index_offset);
            cur_vector += GatherPDXScanner<PDX::L2>::PDX_VECTOR_SIZE;
            data += skipping_size;
        }
        clock.Toc();
        size_t benchmark_time = clock.accum_time;
        runtimes[j] = {benchmark_time};
    }
    std::cout << std::setprecision(16) << GatherPDXScanner<PDX::L2>::distances[GatherPDXScanner<>::PDX_VECTOR_SIZE - 1] << "\n";

    
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

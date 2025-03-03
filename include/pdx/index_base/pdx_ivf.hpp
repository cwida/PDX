#ifndef EMBEDDINGSEARCH_DATA_LOADER_HPP
#define EMBEDDINGSEARCH_DATA_LOADER_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include "utils/file_reader.hpp"

namespace PDX {

struct Vectorgroup {
    uint32_t num_embeddings{};
    uint32_t *indices = nullptr;
    float *data = nullptr;
};

/******************************************************************
 * Very rudimentary memory to IVF index reader
 ******************************************************************/
class IndexPDXIVFFlat {
public:

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    std::vector<Vectorgroup> vectorgroups;
    float *means{};
    bool is_ivf{};
    float *centroids{};
    float *centroids_pdx{};

    void Restore(const std::string &filename) {
        size_t num_tuples;
        char *input = (char *) MmapFile32(num_tuples, filename);
        Load(input);
    }

    void Load(char *input) {
        char *next_value = input;
        num_dimensions = ((uint32_t *) input)[0];
        next_value += sizeof(uint32_t);
        num_vectorgroups = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);
        auto *nums_embeddings = (uint32_t *) next_value;
        next_value += num_vectorgroups * sizeof(uint32_t);
        vectorgroups.resize(num_vectorgroups);
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            Vectorgroup &vectorgroup = vectorgroups[i];
            vectorgroup.num_embeddings = nums_embeddings[i];
            vectorgroup.data = (float *) next_value;
            next_value += sizeof(float) * vectorgroup.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            Vectorgroup &vectorgroup = vectorgroups[i];
            vectorgroup.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup.num_embeddings;
        }
        means = (float *) next_value;
        next_value += sizeof(float) * num_dimensions;
        is_ivf = ((char *) next_value)[0];
        next_value += sizeof(char);
        if (is_ivf) {
            centroids = (float *) next_value;
            next_value += sizeof(float) * num_vectorgroups * num_dimensions;
            centroids_pdx = (float *) next_value;
        }
    }

    void RestoreDSM(const std::string &filename) {
        size_t num_tuples;
        char *input = (char *) MmapFile32(num_tuples, filename);
        LoadDSM(input);
    }

    // Full DSM readers
    void LoadDSM(char *input) {
        char *next_value = input;
        num_dimensions = ((uint32_t *) input)[0];
        next_value += sizeof(uint32_t);
        num_vectorgroups = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);
        auto *nums_embeddings = (uint32_t *) next_value;
        next_value += num_vectorgroups * sizeof(uint32_t);
        vectorgroups.resize(num_vectorgroups);
        size_t all_embeddings = 0;
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            Vectorgroup &vectorgroup = vectorgroups[i];
            vectorgroup.num_embeddings = nums_embeddings[i];
            vectorgroup.data = (float *) next_value;
            // Pointers in DSM are to the start (D0) of the first vector in the logical block
            next_value += sizeof(float) * vectorgroup.num_embeddings;
            all_embeddings += vectorgroup.num_embeddings;
        }
        // Then we advance the pointer to the end of the data
        next_value += sizeof(float) * all_embeddings * (num_dimensions - 1);
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            Vectorgroup &vectorgroup = vectorgroups[i];
            vectorgroup.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup.num_embeddings;
        }
        means = (float *) next_value;
        next_value += sizeof(float) * num_dimensions;
        is_ivf = ((char *) next_value)[0];
        next_value += sizeof(char);
        if (is_ivf) {
            centroids = (float *) next_value;
            next_value += sizeof(float) * num_vectorgroups * num_dimensions;
            centroids_pdx = (float *) next_value;
        }
    }

};

} // namespace PDX

#endif //EMBEDDINGSEARCH_DATA_LOADER_HPP

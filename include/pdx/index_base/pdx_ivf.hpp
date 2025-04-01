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

struct VectorgroupU8 {
    uint32_t num_embeddings{};
    uint32_t *indices = nullptr;
    uint8_t *data = nullptr;
    int32_t *for_bases{};
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
};

class IndexPDXIVFFlatU8 {
public:

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    std::vector<VectorgroupU8> vectorgroups;
    float *means{};
    bool is_ivf{};
    float *centroids{};
    float *centroids_pdx{};

    void Restore(const std::string &filename) {
        size_t num_tuples;
        char *input = (char *) MmapFile8(num_tuples, filename);
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
            VectorgroupU8 &vectorgroup = vectorgroups[i];
            vectorgroup.num_embeddings = nums_embeddings[i];
            vectorgroup.data = (uint8_t *) next_value;
            next_value += sizeof(uint8_t) * vectorgroup.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VectorgroupU8 &vectorgroup = vectorgroups[i];
            vectorgroup.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup.num_embeddings;
        }
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VectorgroupU8 &vectorgroup = vectorgroups[i];
            vectorgroup.for_bases = (int32_t *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
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

#ifndef PDX_IVF_HPP
#define PDX_IVF_HPP

#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include "utils/file_reader.hpp"
#include "common.hpp"

namespace PDX {


/******************************************************************
 * Very rudimentary memory to IVF index reader
 ******************************************************************/
template <Quantization q>
class IndexPDXIVF{};

template <>
class IndexPDXIVF<F32> {
public:

    using VECTORGROUP_TYPE = Vectorgroup<F32>;

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<VECTORGROUP_TYPE> vectorgroups;
    float *means{};
    bool is_ivf{};
    bool is_normalized{};
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
        num_vertical_dimensions = ((uint32_t *) input)[1];
        num_horizontal_dimensions = ((uint32_t *) input)[2];

        next_value += sizeof(uint32_t) * 3;
        num_vectorgroups = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);
        auto *nums_embeddings = (uint32_t *) next_value;
        next_value += num_vectorgroups * sizeof(uint32_t);
        vectorgroups.resize(num_vectorgroups);
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.num_embeddings = nums_embeddings[i];
            vectorgroup.data = (float *) next_value;
            next_value += sizeof(float) * vectorgroup.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup.num_embeddings;
        }
        means = (float *) next_value;
        next_value += sizeof(float) * num_dimensions;
        is_normalized = ((char *) next_value)[0];
        next_value += sizeof(char);
        is_ivf = ((char *) next_value)[0];
        next_value += sizeof(char);
        if (is_ivf) {
            centroids = (float *) next_value;
            next_value += sizeof(float) * num_vectorgroups * num_dimensions;
            centroids_pdx = (float *) next_value;
        }
    }
};

template <>
class IndexPDXIVF<U8> {
public:

    using VECTORGROUP_TYPE = Vectorgroup<U8>;

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<Vectorgroup<U8>> vectorgroups;
    float *means{};
    bool is_ivf{};
    bool is_normalized{};
    float *centroids{};
    float *centroids_pdx{};

    float for_base {};
    float scale_factor {};

    void Restore(const std::string &filename) {
        size_t num_tuples;
        char *input = (char *) MmapFile8(num_tuples, filename);
        Load(input);
    }

    void Load(char *input) {
        char *next_value = input;
        num_dimensions = ((uint32_t *) input)[0];
        num_vertical_dimensions = ((uint32_t *) input)[1];
        num_horizontal_dimensions = ((uint32_t *) input)[2];

        next_value += sizeof(uint32_t) * 3;
        num_vectorgroups = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);
        auto *nums_embeddings = (uint32_t *) next_value;
        next_value += num_vectorgroups * sizeof(uint32_t);
        vectorgroups.resize(num_vectorgroups);
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.num_embeddings = nums_embeddings[i];
            vectorgroup.data = (uint8_t *) next_value;
            next_value += sizeof(uint8_t) * vectorgroup.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup.num_embeddings;
        }
        // means = (float *) next_value;
        // next_value += sizeof(float) * num_dimensions;
        is_normalized = ((char *) next_value)[0];
        next_value += sizeof(char);
        is_ivf = ((char *) next_value)[0];
        next_value += sizeof(char);
        if (is_ivf) {
            centroids = (float *) next_value;
            next_value += sizeof(float) * num_vectorgroups * num_dimensions;
            centroids_pdx = (float *) next_value;
        }

        for_base = ((float *) next_value)[0];
        next_value += sizeof(float);
        scale_factor = ((float *) next_value)[0];
        next_value += sizeof(float);
    }
};

} // namespace PDX

#endif //PDX_IVF_HPP

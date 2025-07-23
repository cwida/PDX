#ifndef PDX_IMI_HPP
#define PDX_IMI_HPP

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
 * Very rudimentary memory to IMI index reader
 ******************************************************************/
template <Quantization q>
class IndexPDXIMI{};

template <>
class IndexPDXIMI<F32> {
public:

    using VECTORGROUP_TYPE = Vectorgroup<F32>;
    using VECTORGROUP_TYPE_L0 = Vectorgroup<F32>;

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};

    std::vector<VECTORGROUP_TYPE> vectorgroups;
    uint32_t num_vectorgroups_l0 {};
    std::vector<Vectorgroup<F32>> vectorgroups_l0;

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
        num_horizontal_dimensions = (uint32_t)(num_dimensions * PROPORTION_VERTICAL_DIM);
        num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        if (num_horizontal_dimensions % 64 != 0) {
            num_horizontal_dimensions = static_cast<int>(std::round(num_horizontal_dimensions / 64.0)) * 64;
            num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        }
        if (num_vertical_dimensions == 0) {
            num_horizontal_dimensions = 64;
            num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        }

        // std::cout << "Vertical dims: " << num_vertical_dimensions << "\n";
        // std::cout << "Horizontal dims: " << num_horizontal_dimensions << "\n";

        next_value += sizeof(uint32_t);
        num_vectorgroups = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);

        num_vectorgroups_l0 = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);

        // L0 load
        auto *nums_embeddings_l0 = (uint32_t *) next_value;
        next_value += num_vectorgroups_l0 * sizeof(uint32_t);
        // std::cout << "N buckets L0: " << num_vectorgroups_l0 << "\n";

        vectorgroups_l0.resize(num_vectorgroups_l0);

        for (size_t i = 0; i < num_vectorgroups_l0; ++i) {
            VECTORGROUP_TYPE_L0 &vectorgroup_l0 = vectorgroups_l0[i];
            vectorgroup_l0.num_embeddings = nums_embeddings_l0[i];
            vectorgroup_l0.data = (float *) next_value;
            next_value += sizeof(float) * vectorgroup_l0.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_vectorgroups_l0; ++i) {
            VECTORGROUP_TYPE_L0 &vectorgroup_l0 = vectorgroups_l0[i];
            vectorgroup_l0.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup_l0.num_embeddings;
        }
        // std::cout << "Finished loading L0\n";

        auto *nums_embeddings = (uint32_t *) next_value;
        next_value += num_vectorgroups * sizeof(uint32_t);
        vectorgroups.resize(num_vectorgroups);
        // std::cout << "N buckets L1: " << num_vectorgroups << "\n";
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
        is_normalized = ((char *) next_value)[0];
        next_value += sizeof(char);

        centroids_pdx = (float *) next_value;
    }
};

template <>
class IndexPDXIMI<U8> {
public:

    using VECTORGROUP_TYPE = Vectorgroup<U8>;
    using VECTORGROUP_TYPE_L0 = Vectorgroup<F32>;

    uint32_t num_dimensions {};
    uint32_t num_vectorgroups {};
    uint32_t num_horizontal_dimensions {};
    uint32_t num_vertical_dimensions {};

    std::vector<Vectorgroup<U8>> vectorgroups;

    uint32_t num_vectorgroups_l0 {};
    std::vector<Vectorgroup<F32>> vectorgroups_l0;

    bool is_normalized{};
    float *centroids{};
    float *centroids_pdx{};

    float for_base{};
    float scale_factor{};

    void Restore(const std::string &filename) {
        size_t num_tuples;
        char *input = (char *) MmapFile8(num_tuples, filename);
        Load(input);
    }

    void Load(char *input) {
        char *next_value = input;
        num_dimensions = ((uint32_t *) input)[0];
        num_horizontal_dimensions = (uint32_t)(num_dimensions * PROPORTION_VERTICAL_DIM);
        num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        if (num_horizontal_dimensions % 64 != 0) {
            num_horizontal_dimensions = static_cast<int>(std::round(num_horizontal_dimensions / 64.0)) * 64;
            num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        }
        if (num_vertical_dimensions == 0) {
            num_horizontal_dimensions = 64;
            num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        }
        // std::cout << "Dims: " << num_dimensions << "\n";
        // std::cout << "Vertical dims: " << num_vertical_dimensions << "\n";
        // std::cout << "Horizontal dims: " << num_horizontal_dimensions << "\n";
        next_value += sizeof(uint32_t);
        num_vectorgroups = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);
        num_vectorgroups_l0 = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);

        // L0 load
        auto *nums_embeddings_l0 = (uint32_t *) next_value;
        next_value += num_vectorgroups_l0 * sizeof(uint32_t);
        // std::cout << "N buckets L0: " << num_vectorgroups_l0 << "\n";

        vectorgroups_l0.resize(num_vectorgroups_l0);

        for (size_t i = 0; i < num_vectorgroups_l0; ++i) {
            VECTORGROUP_TYPE_L0 &vectorgroup_l0 = vectorgroups_l0[i];
            vectorgroup_l0.num_embeddings = nums_embeddings_l0[i];
            vectorgroup_l0.data = (float *) next_value;
            next_value += sizeof(float) * vectorgroup_l0.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_vectorgroups_l0; ++i) {
            VECTORGROUP_TYPE_L0 &vectorgroup_l0 = vectorgroups_l0[i];
            vectorgroup_l0.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup_l0.num_embeddings;
        }
        // std::cout << "Finished loading L0\n";

        // L1 load
        auto *nums_embeddings = (uint32_t *) next_value;
        next_value += num_vectorgroups * sizeof(uint32_t);
        vectorgroups.resize(num_vectorgroups);
        // std::cout << "N buckets L1: " << num_vectorgroups << "\n";
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
        is_normalized = ((char *) next_value)[0];
        next_value += sizeof(char);

        centroids_pdx = (float *) next_value;
        next_value += sizeof(float) * num_vectorgroups_l0 * num_dimensions;

        for_base = ((float *) next_value)[0];
        next_value += sizeof(float);
        scale_factor = ((float *) next_value)[0];
    }
};

} // namespace PDX

#endif //PDX_IMI_HPP

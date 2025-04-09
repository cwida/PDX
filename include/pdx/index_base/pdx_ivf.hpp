#ifndef PDX_DATA_LOADER_HPP
#define PDX_DATA_LOADER_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include "utils/file_reader.hpp"
#include "pdx/common.hpp"

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
    std::vector<VECTORGROUP_TYPE> vectorgroups;
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
    std::vector<Vectorgroup<U8>> vectorgroups;
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
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
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

template <>
class IndexPDXIVF<U6> {
public:

    using VECTORGROUP_TYPE = Vectorgroup<U6>;

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    std::vector<Vectorgroup<U6>> vectorgroups;
    float *means{};
    bool is_ivf{};
    float *centroids{};
    float *centroids_pdx{};
    size_t BW=6; // in bits
    size_t EXCEPTION_SIZE=1; // byte

    template<class T, T val=8>
    static constexpr std::uint32_t AlignValue(T n) {
        return ((n + (val - 1)) / val) * val;
    }

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
        // Vectors Data
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.num_embeddings = nums_embeddings[i];
            vectorgroup.data = (uint8_t *) next_value;
            // Values are byte aligned every 4 dimensions
            // So we have to manually count when reading the index
//            for (size_t d = 0; d < num_dimensions; d+=4){
//                next_value += AlignValue<uint32_t, 8>(BW * vectorgroup.num_embeddings * 4) / 8;
//            }
            next_value += (int)(AlignValue<uint32_t, 1024>(vectorgroup.num_embeddings * num_dimensions) * 0.75); // 6 / 8
        }
        // Indices
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup.num_embeddings;
        }
        // For Bases
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.for_bases = (int32_t *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
        }
        // Exceptions
//        for (size_t i = 0; i < num_vectorgroups; ++i) {
//            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
//            vectorgroup.exceptions_n = ((uint32_t *) next_value)[0];
//            next_value += sizeof(uint32_t);
//            vectorgroup.exceptions_per_dimension = (uint16_t *) next_value;
//            next_value += sizeof(uint16_t) * num_dimensions;
//            vectorgroup.exceptions = (uint8_t*) next_value;
//            size_t total_bytes_exceptions = 0;
//            for (size_t n = 0; n < num_dimensions; ++n){
//                // For now no alignment for SIMD, just scalar
//                total_bytes_exceptions += vectorgroup.exceptions_per_dimension[n] * sizeof(uint8_t);
//            }
//            next_value += total_bytes_exceptions;
//        }
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


template <>
class IndexPDXIVF<U4> {
public:
    
    using VECTORGROUP_TYPE = Vectorgroup<U4>;

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    std::vector<VECTORGROUP_TYPE> vectorgroups;
    float *means{};
    bool is_ivf{};
    float *centroids{};
    float *centroids_pdx{};
    size_t BW=4; // in bits
    size_t EXCEPTION_SIZE=1; // byte

    template<class T, T val=8>
    static constexpr std::uint32_t AlignValue(T n) {
        return ((n + (val - 1)) / val) * val;
    }

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
        // Vectors Data
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.num_embeddings = nums_embeddings[i];
            vectorgroup.data = (uint8_t *) next_value;
            // Values are byte aligned every 4 dimensions
            // So we have to manually count when reading the index
//            for (size_t d = 0; d < num_dimensions; d+=4){
//                next_value += AlignValue<uint32_t, 8>(BW * vectorgroup.num_embeddings * 4) / 8;
//            }
            next_value += (int)(AlignValue<uint32_t, 1024>(vectorgroup.num_embeddings * num_dimensions) * 0.50); // 4 / 8
        }
        // Indices
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup.num_embeddings;
        }
        // For Bases
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.for_bases = (int32_t *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
        }
        // Exceptions
//        for (size_t i = 0; i < num_vectorgroups; ++i) {
//            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
//            vectorgroup.exceptions_n = ((uint32_t *) next_value)[0];
//            next_value += sizeof(uint32_t);
//            vectorgroup.exceptions_per_dimension = (uint16_t *) next_value;
//            next_value += sizeof(uint16_t) * num_dimensions;
//            vectorgroup.exceptions = (uint8_t*) next_value;
//            size_t total_bytes_exceptions = 0;
//            for (size_t n = 0; n < num_dimensions; ++n){
//                // For now no alignment for SIMD, just scalar
//                total_bytes_exceptions += vectorgroup.exceptions_per_dimension[n] * sizeof(uint8_t);
//            }
//            next_value += total_bytes_exceptions;
//        }
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

#endif //PDX_DATA_LOADER_HPP

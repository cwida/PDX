#ifndef PDX_DATA_LOADER_HPP
#define PDX_DATA_LOADER_HPP

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
        num_vertical_dimensions = num_dimensions;
        num_horizontal_dimensions = 0;
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

    void Restore(const std::string &filename) {
        size_t num_tuples;
        char *input = (char *) MmapFile8(num_tuples, filename);
        Load(input);
    }

    void Load(char *input) {
        char *next_value = input;
        num_dimensions = ((uint32_t *) input)[0];
        num_horizontal_dimensions = (uint32_t)(num_dimensions * 0.75);
        num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        if (num_horizontal_dimensions % 64 != 0) {
            num_horizontal_dimensions = static_cast<int>(std::round(num_horizontal_dimensions / 64.0)) * 64;
            num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        }
        // TODO: UNCOMMENT WHEN GOING BACK TO FULL VERTICAL
//        num_vertical_dimensions = num_dimensions;
//        num_horizontal_dimensions = 0;
        std::cout << "Vertical dims: " << num_vertical_dimensions << "\n";
        std::cout << "Horizontal dims: " << num_horizontal_dimensions << "\n";
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
        // TODO: Should not always load!
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.for_bases = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
            vectorgroup.scale_factors = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
            //vectorgroup.norms = (float *) next_value;
            //next_value += sizeof(float) * vectorgroup.num_embeddings;
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
class IndexPDXIVF<ASYMMETRIC_U8> {
public:

    using VECTORGROUP_TYPE = Vectorgroup<ASYMMETRIC_U8>;

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<Vectorgroup<ASYMMETRIC_U8>> vectorgroups;
    float *means{};
    bool is_ivf{};
    bool is_normalized{};
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
        num_horizontal_dimensions = (uint32_t)(num_dimensions * 0.75);
        num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        if (num_horizontal_dimensions % 64 != 0) {
            num_horizontal_dimensions = static_cast<int>(std::round(num_horizontal_dimensions / 64.0)) * 64;
            num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        }
        // TODO: UNCOMMENT WHEN GOING BACK TO FULL VERTICAL
//        num_vertical_dimensions = num_dimensions;
//        num_horizontal_dimensions = 0;
        std::cout << "Vertical dims: " << num_vertical_dimensions << "\n";
        std::cout << "Horizontal dims: " << num_horizontal_dimensions << "\n";
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
        // TODO: Should not always load!
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.for_bases = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
            vectorgroup.scale_factors = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
            //vectorgroup.norms = (float *) next_value;
            //next_value += sizeof(float) * vectorgroup.num_embeddings;
        }
        //means = (float *) next_value;
        //next_value += sizeof(float) * num_dimensions;
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

// LEP 8 always admit 4-bit data
template <>
class IndexPDXIVF<ASYMMETRIC_LEP_U8> {
public:

    using VECTORGROUP_TYPE = Vectorgroup<ASYMMETRIC_LEP_U8>;
    const size_t BW = 4;

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<Vectorgroup<ASYMMETRIC_LEP_U8>> vectorgroups;
    float *means{};
    bool is_ivf{};
    bool is_normalized{};
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
        num_horizontal_dimensions = (uint32_t)(num_dimensions * 0.75);
        num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        if (num_horizontal_dimensions % 64 != 0) {
            num_horizontal_dimensions = static_cast<int>(std::round(num_horizontal_dimensions / 64.0)) * 64;
            num_vertical_dimensions = num_dimensions - num_horizontal_dimensions;
        }
        // TODO: UNCOMMENT WHEN GOING BACK TO FULL VERTICAL
//        num_vertical_dimensions = num_dimensions;
//        num_horizontal_dimensions = 0;
        std::cout << "Vertical dims: " << num_vertical_dimensions << "\n";
        std::cout << "Horizontal dims: " << num_horizontal_dimensions << "\n";
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
            // Each group of 4 dimensions in the vectorgroup is aligned at the byte level
            for (size_t d = 0; d < num_dimensions; d+=4){
                // They should be the same, as I am grouping every 4 dimensions an 4*4 = 16 --always byte aligned
                assert((BW * vectorgroup.num_embeddings * 4 / 8) == (AlignValue<uint32_t, 8>(BW * vectorgroup.num_embeddings * 4) / 8));
                next_value += AlignValue<uint32_t, 8>(BW * vectorgroup.num_embeddings * 4) / 8;
            }
        }
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * vectorgroup.num_embeddings;
        }
        // TODO: Should not always load!
        for (size_t i = 0; i < num_vectorgroups; ++i) {
            VECTORGROUP_TYPE &vectorgroup = vectorgroups[i];
            vectorgroup.for_bases = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
            vectorgroup.scale_factors = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
            //vectorgroup.norms = (float *) next_value;
            //next_value += sizeof(float) * vectorgroup.num_embeddings;

            vectorgroup.num_exceptions = ((uint32_t *) next_value)[0];
            next_value += sizeof(uint32_t);

            vectorgroup.for_bases_exceptions = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;

            vectorgroup.scale_factors_exceptions = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;

            if (vectorgroup.num_exceptions == 0) continue;

            vectorgroup.exceptions_positions = (uint16_t *) next_value;
            next_value += sizeof(uint16_t) * (vectorgroup.num_exceptions * num_dimensions);

            vectorgroup.data_exceptions = (uint8_t *) next_value;
            next_value += sizeof(uint8_t) * (vectorgroup.num_exceptions * num_dimensions);

        }
        //means = (float *) next_value;
        //next_value += sizeof(float) * num_dimensions;
        is_normalized = ((char *) next_value)[0];
        next_value += sizeof(char);
        is_ivf = ((char *) next_value)[0];
        next_value += sizeof(char);
        std::cout << "LOADED" << "\n";
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
    bool is_normalized{};
    float *centroids{};
    float *centroids_pdx{};
    size_t BW=6; // in bits
    size_t EXCEPTION_SIZE=1; // byte

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
            vectorgroup.for_bases = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
            vectorgroup.scale_factors = (float *) next_value;
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
class IndexPDXIVF<U4> {
public:
    
    using VECTORGROUP_TYPE = Vectorgroup<U4>;

    uint32_t num_dimensions{};
    uint32_t num_vectorgroups{};
    std::vector<VECTORGROUP_TYPE> vectorgroups;
    float *means{};
    bool is_ivf{};
    bool is_normalized{};
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
            vectorgroup.for_bases = (float *) next_value;
            next_value += sizeof(uint32_t) * num_dimensions;
            vectorgroup.scale_factors = (float *) next_value;
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

} // namespace PDX

#endif //PDX_DATA_LOADER_HPP

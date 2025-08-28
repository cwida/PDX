#ifndef PDX_IVF2_HPP
#define PDX_IVF2_HPP

#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include "utils/file_reader.hpp"
#include "common.hpp"

namespace PDX {


/******************************************************************
 * Very rudimentary memory to IVF2 index reader
 ******************************************************************/
template <Quantization q>
class IndexPDXIVF2{};

template <>
class IndexPDXIVF2<F32> {
public:

    using CLUSTER_TYPE = Cluster<F32>;
    using CLUSTER_TYPE_L0 = Cluster<F32>;

    std::unique_ptr<char[]> file_buffer;

    uint32_t num_dimensions{};
    uint32_t num_clusters{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};

    std::vector<CLUSTER_TYPE> clusters;
    uint32_t num_clusters_l0 {};
    std::vector<Cluster<F32>> clusters_l0;

    float *means{};
    bool is_ivf{};
    bool is_normalized{};
    float *centroids{};
    float *centroids_pdx{};

    void Restore(const std::string &filename) {
        file_buffer = MmapFile(filename);
        Load(file_buffer.get());
    }

    void Load(char *input) {
        char *next_value = input;
        num_dimensions = ((uint32_t *) input)[0];
        num_vertical_dimensions = ((uint32_t *) input)[1];
        num_horizontal_dimensions = ((uint32_t *) input)[2];


        next_value += sizeof(uint32_t) * 3;
        num_clusters = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);

        num_clusters_l0 = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);

        // L0 load
        auto *nums_embeddings_l0 = (uint32_t *) next_value;
        next_value += num_clusters_l0 * sizeof(uint32_t);

        clusters_l0.resize(num_clusters_l0);

        for (size_t i = 0; i < num_clusters_l0; ++i) {
            CLUSTER_TYPE_L0 &cluster_l0 = clusters_l0[i];
            cluster_l0.num_embeddings = nums_embeddings_l0[i];
            cluster_l0.data = (float *) next_value;
            next_value += sizeof(float) * cluster_l0.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_clusters_l0; ++i) {
            CLUSTER_TYPE_L0 &cluster_l0 = clusters_l0[i];
            cluster_l0.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * cluster_l0.num_embeddings;
        }

        auto *nums_embeddings = (uint32_t *) next_value;
        next_value += num_clusters * sizeof(uint32_t);
        clusters.resize(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            CLUSTER_TYPE &cluster = clusters[i];
            cluster.num_embeddings = nums_embeddings[i];
            cluster.data = (float *) next_value;
            next_value += sizeof(float) * cluster.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            CLUSTER_TYPE &cluster = clusters[i];
            cluster.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * cluster.num_embeddings;
        }
        is_normalized = ((char *) next_value)[0];
        next_value += sizeof(char);

        centroids_pdx = (float *) next_value;
    }
};

template <>
class IndexPDXIVF2<U8> {
public:

    using CLUSTER_TYPE = Cluster<U8>;
    using CLUSTER_TYPE_L0 = Cluster<F32>;

    std::unique_ptr<char[]> file_buffer;

    uint32_t num_dimensions {};
    uint32_t num_clusters {};
    uint32_t num_horizontal_dimensions {};
    uint32_t num_vertical_dimensions {};

    std::vector<Cluster<U8>> clusters;

    uint32_t num_clusters_l0 {};
    std::vector<Cluster<F32>> clusters_l0;

    float *means{};
    bool is_normalized{};
    float *centroids{};
    float *centroids_pdx{};

    float for_base{};
    float scale_factor{};

    void Restore(const std::string &filename) {
        file_buffer = MmapFile(filename);
        Load(file_buffer.get());
    }

    void Load(char *input) {
        char *next_value = input;
        num_dimensions = ((uint32_t *) input)[0];
        num_vertical_dimensions = ((uint32_t *) input)[1];
        num_horizontal_dimensions = ((uint32_t *) input)[2];

        next_value += sizeof(uint32_t) * 3;
        num_clusters = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);
        num_clusters_l0 = ((uint32_t *) next_value)[0];
        next_value += sizeof(uint32_t);

        // L0 load
        auto *nums_embeddings_l0 = (uint32_t *) next_value;
        next_value += num_clusters_l0 * sizeof(uint32_t);

        clusters_l0.resize(num_clusters_l0);

        for (size_t i = 0; i < num_clusters_l0; ++i) {
            CLUSTER_TYPE_L0 &cluster_l0 = clusters_l0[i];
            cluster_l0.num_embeddings = nums_embeddings_l0[i];
            cluster_l0.data = (float *) next_value;
            next_value += sizeof(float) * cluster_l0.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_clusters_l0; ++i) {
            CLUSTER_TYPE_L0 &cluster_l0 = clusters_l0[i];
            cluster_l0.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * cluster_l0.num_embeddings;
        }

        // L1 load
        auto *nums_embeddings = (uint32_t *) next_value;
        next_value += num_clusters * sizeof(uint32_t);
        clusters.resize(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            CLUSTER_TYPE &cluster = clusters[i];
            cluster.num_embeddings = nums_embeddings[i];
            cluster.data = (uint8_t *) next_value;
            next_value += sizeof(uint8_t) * cluster.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            CLUSTER_TYPE &cluster = clusters[i];
            cluster.indices = (uint32_t *) next_value;
            next_value += sizeof(uint32_t) * cluster.num_embeddings;
        }
        is_normalized = ((char *) next_value)[0];
        next_value += sizeof(char);

        centroids_pdx = (float *) next_value;
        next_value += sizeof(float) * num_clusters_l0 * num_dimensions;

        for_base = ((float *) next_value)[0];
        next_value += sizeof(float);
        scale_factor = ((float *) next_value)[0];
    }
};

} // namespace PDX

#endif //PDX_IVF2_HPP

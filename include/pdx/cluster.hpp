#pragma once

#include "pdx/common.hpp"
#include "pdx/utils.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <vector>

namespace PDX {

template <Quantization Q>
struct Cluster {
    using data_t = pdx_data_t<Q>;

    Cluster(uint32_t num_embeddings, uint32_t num_dimensions)
        : num_embeddings(num_embeddings), num_dimensions(num_dimensions),
            indices(new uint32_t[num_embeddings]),
            data(new data_t[static_cast<uint64_t>(num_embeddings) * num_dimensions]) {}

    ~Cluster() {
        delete[] data;
        delete[] indices;
    }

    uint32_t num_embeddings{};
    const uint32_t num_dimensions{};
    uint32_t* indices = nullptr;
    data_t* data = nullptr;

    size_t GetInMemorySizeInBytes() const {
        return sizeof(*this) + num_embeddings * sizeof(*indices) +
                num_embeddings * static_cast<uint64_t>(num_dimensions) * sizeof(*data);
    }
};

} // namespace PDX

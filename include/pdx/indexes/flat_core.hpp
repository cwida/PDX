#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace PDX {

// Row-major block of transformed float32 embeddings; the Flat counterpart of IVF<Q>
struct Flat {
    uint32_t num_dimensions = 0;
    bool is_normalized = false;
    size_t num_embeddings = 0;
    std::vector<float> data;
    std::vector<uint32_t> indices;
    std::vector<uint8_t> tombstones;

    Flat() = default;

    Flat(uint32_t num_dimensions, bool is_normalized)
        : num_dimensions(num_dimensions), is_normalized(is_normalized) {}

    [[nodiscard]] size_t UsedCapacity() const { return indices.size(); }

    [[nodiscard]] bool HasTombstone(size_t position) const { return tombstones[position] != 0; }

    [[nodiscard]] const float* GetEmbedding(size_t position) const {
        return data.data() + position * num_dimensions;
    }

    uint32_t AppendEmbedding(uint32_t row_id, const float* embedding) {
        const auto position = static_cast<uint32_t>(indices.size());
        data.insert(data.end(), embedding, embedding + num_dimensions);
        indices.push_back(row_id);
        tombstones.push_back(0);
        num_embeddings++;
        return position;
    }

    void DeleteEmbedding(uint32_t position) {
        tombstones[position] = 1;
        num_embeddings--;
    }

    void Clear() {
        data.clear();
        indices.clear();
        tombstones.clear();
        num_embeddings = 0;
    }

    [[nodiscard]] size_t GetInMemorySizeInBytes() const {
        return sizeof(*this) + data.capacity() * sizeof(float) +
               indices.capacity() * sizeof(uint32_t) + tombstones.capacity() * sizeof(uint8_t);
    }
};

} // namespace PDX

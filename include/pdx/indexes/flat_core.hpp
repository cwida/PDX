#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

#include "pdx/utils.hpp"

namespace PDX {

// Row-major block of transformed float32 embeddings; the Flat counterpart of IVF<Q>
struct Flat {
    uint32_t num_dimensions = 0;
    bool is_normalized = false;
    size_t num_embeddings = 0;
    std::vector<float> data;
    std::vector<uint32_t> indices;
    std::vector<uint8_t> tombstones;
    std::vector<double> embeddings_sum;

    Flat() = default;

    Flat(uint32_t num_dimensions, bool is_normalized)
        : num_dimensions(num_dimensions), is_normalized(is_normalized),
          embeddings_sum(num_dimensions, 0.0) {}

    [[nodiscard]] size_t UsedCapacity() const { return indices.size(); }

    [[nodiscard]] bool HasTombstone(size_t position) const { return tombstones[position] != 0; }

    [[nodiscard]] const float* GetEmbeddingPtrAtPosition(size_t position) const {
        return data.data() + position * num_dimensions;
    }

    uint32_t AppendEmbedding(uint32_t row_id, const float* embedding) {
        const auto position = static_cast<uint32_t>(indices.size());
        data.insert(data.end(), embedding, embedding + num_dimensions);
        indices.push_back(row_id);
        tombstones.push_back(0);
        num_embeddings++;
        for (uint32_t d = 0; d < num_dimensions; d++) {
            embeddings_sum[d] += embedding[d];
        }
        return position;
    }

    void DeleteEmbedding(uint32_t position) {
        tombstones[position] = 1;
        num_embeddings--;
        const float* embedding = GetEmbeddingPtrAtPosition(position);
        for (uint32_t d = 0; d < num_dimensions; d++) {
            embeddings_sum[d] -= embedding[d];
        }
    }

    void Clear() {
        data.clear();
        indices.clear();
        tombstones.clear();
        num_embeddings = 0;
        embeddings_sum.assign(num_dimensions, 0.0);
    }

    // Only the live rows, so the saved index is compacted.
    void Save(std::ostream& out) const {
        WriteValue(out, num_dimensions);
        WriteValue(out, static_cast<uint8_t>(is_normalized));
        WriteValue(out, static_cast<uint64_t>(num_embeddings));
        for (size_t position = 0; position < UsedCapacity(); position++) {
            if (!HasTombstone(position)) {
                WriteValue(out, indices[position]);
            }
        }
        for (size_t position = 0; position < UsedCapacity(); position++) {
            if (!HasTombstone(position)) {
                out.write(
                    reinterpret_cast<const char*>(GetEmbeddingPtrAtPosition(position)),
                    static_cast<std::streamsize>(sizeof(float) * num_dimensions)
                );
            }
        }
        out.write(
            reinterpret_cast<const char*>(embeddings_sum.data()),
            static_cast<std::streamsize>(sizeof(double) * num_dimensions)
        );
    }

    template <class Reader>
    void Load(Reader& reader) {
        num_dimensions = ReadValue<uint32_t>(reader);
        is_normalized = ReadValue<uint8_t>(reader) != 0;
        num_embeddings = static_cast<size_t>(ReadValue<uint64_t>(reader));
        indices.resize(num_embeddings);
        reader.Read(indices.data(), sizeof(uint32_t) * num_embeddings);
        data.resize(num_embeddings * num_dimensions);
        reader.Read(data.data(), sizeof(float) * data.size());
        tombstones.assign(num_embeddings, 0);
        embeddings_sum.resize(num_dimensions);
        reader.Read(embeddings_sum.data(), sizeof(double) * num_dimensions);
    }

    [[nodiscard]] size_t GetInMemorySizeInBytes() const {
        return sizeof(*this) + data.capacity() * sizeof(float) +
               indices.capacity() * sizeof(uint32_t) + tombstones.capacity() * sizeof(uint8_t) +
               embeddings_sum.capacity() * sizeof(double);
    }
};

} // namespace PDX

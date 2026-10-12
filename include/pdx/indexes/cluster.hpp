#pragma once

#include "pdx/common.hpp"
#include "pdx/profiler.hpp"
#include "pdx/utils.hpp"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace PDX {

template <Quantization Q>
struct Cluster : skmeans::AllocatorHolder {
    using data_t = pdx_data_t<Q>;
    using tombstones_t = std::unordered_set<uint32_t>;

    constexpr static float CAPACITY_THRESHOLD = 1.3f; // 30% more than the current capacity
    constexpr static float MIN_CAPACITY_THRESHOLD = 0.5f;
    constexpr static uint32_t DEFAULT_MAX_CAPACITY_LOWER_BOUND = 256;

    static uint32_t CapacityFor(uint32_t num_embeddings, uint32_t max_cluster_size) {
        if (max_cluster_size == 0) {
            return std::max(
                static_cast<uint32_t>(num_embeddings * CAPACITY_THRESHOLD),
                DEFAULT_MAX_CAPACITY_LOWER_BOUND
            );
        }
        // Guaranteeing a free slot for the next append
        return std::max(max_cluster_size, num_embeddings + 1);
    }

    Cluster(uint32_t num_embeddings, uint32_t num_dimensions, MemoryAllocator* allocator = nullptr)
        : Cluster(num_embeddings, CapacityFor(num_embeddings, 0), num_dimensions, allocator) {}

    Cluster(
        uint32_t num_embeddings,
        uint32_t max_capacity,
        uint32_t num_dimensions,
        MemoryAllocator* allocator = nullptr,
        bool allocate_data = true
    )
        : num_embeddings(num_embeddings), used_capacity(num_embeddings), max_capacity(max_capacity),
          min_capacity(static_cast<uint32_t>(num_embeddings * MIN_CAPACITY_THRESHOLD)),
          num_dimensions(num_dimensions) {
        SetAllocator(allocator);
        if (allocate_data) {
            indices = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), max_capacity);
            data = skmeans::AllocateBuffer<data_t>(
                GetAllocator(), static_cast<uint64_t>(max_capacity) * num_dimensions
            );
        }
    }

    Cluster(Cluster&& other) noexcept
        : skmeans::AllocatorHolder(other), num_embeddings(other.num_embeddings),
          used_capacity(other.used_capacity), max_capacity(other.max_capacity),
          min_capacity(other.min_capacity), num_dimensions(other.num_dimensions),
          n_accessed(other.n_accessed.load()), n_inserted(other.n_inserted),
          n_deleted(other.n_deleted), id(other.id), mesocluster_id(other.mesocluster_id),
          indices(std::move(other.indices)), data(std::move(other.data)),
          tombstones(std::move(other.tombstones)) {}

    Cluster(const Cluster&) = delete;
    Cluster& operator=(const Cluster&) = delete;

    // Move-assignment: transfers data ownership, keeps destination's mutex and const
    // num_dimensions. Caller must ensure no concurrent access to *this during assignment.
    Cluster& operator=(Cluster&& other) noexcept {
        if (this != &other) {
            assert(num_dimensions == other.num_dimensions);
            skmeans::AllocatorHolder::operator=(other);
            num_embeddings = other.num_embeddings;
            used_capacity = other.used_capacity;
            max_capacity = other.max_capacity;
            min_capacity = other.min_capacity;
            // num_dimensions: const and guaranteed same — skip
            // cluster_mutex: keep our own — skip
            n_accessed = other.n_accessed.load();
            n_inserted = other.n_inserted;
            n_deleted = other.n_deleted;
            id = other.id;
            mesocluster_id = other.mesocluster_id;
            indices = std::move(other.indices);
            data = std::move(other.data);
            tombstones = std::move(other.tombstones);
        }
        return *this;
    }

    uint32_t num_embeddings{
    }; // Number of valid embeddings in the cluster, i.e., excluding tombstones
    uint32_t used_capacity{}; // Total capacity of the cluster, i.e., including tombstones but
                              // excluding empty slots that are not yet used
    uint32_t max_capacity{};
    uint32_t min_capacity{};
    const uint32_t num_dimensions{};
    std::mutex cluster_mutex;
    std::atomic<size_t> n_accessed{0};
    size_t n_inserted = 0;
    size_t n_deleted = 0;
    uint32_t id{};             // Position in IVF::clusters vector
    uint32_t mesocluster_id{}; // Which L0 meso-cluster contains this L1 cluster (L1 only)

    skmeans::AllocatedBuffer<uint32_t> indices; // !These are row_ids
    skmeans::AllocatedBuffer<data_t> data;
    tombstones_t tombstones; // ! Need to have indexes, not row_ids

    void AddTombstone(uint32_t index) { tombstones.insert(index); }

    bool HasTombstone(uint32_t index) const { return tombstones.count(index); }

    uint32_t PopTombstone() {
        auto it = tombstones.begin();
        uint32_t val = *it;
        tombstones.erase(it);
        return val;
    }

    void RemoveTombstone(uint32_t index) { tombstones.erase(index); }

    // Returns the index in cluster of the newly appended embedding
    uint32_t AppendEmbedding(uint32_t row_id, const data_t* PDX_RESTRICT embedding) {
        PDX_PROFILE_SCOPE("LeafAppend");
        std::lock_guard<std::mutex> lock(cluster_mutex);
        uint32_t next_free_idx = used_capacity;
        bool replaced_tombstone = false;
        if (!tombstones.empty()) {
            next_free_idx = PopTombstone();
            replaced_tombstone = true;
        }
        if (next_free_idx >= max_capacity) {
            throw std::runtime_error(
                "AppendEmbedding: cluster buffer overflow (used_capacity=" +
                std::to_string(used_capacity) + ", max_capacity=" + std::to_string(max_capacity) +
                ")"
            );
        }
        InsertEmbedding(next_free_idx, row_id, embedding);
        num_embeddings++;
        if (!replaced_tombstone) {
            used_capacity++;
        }
        assert(num_embeddings <= used_capacity);

        n_inserted++;
        return next_free_idx;
    }

    void DeleteEmbedding(uint32_t index_in_cluster) {
        PDX_PROFILE_SCOPE("LeafDelete");
        std::lock_guard<std::mutex> lock(cluster_mutex);
        AddTombstone(index_in_cluster);
        num_embeddings--;
        n_deleted++;
    }

    size_t GetInMemorySizeInBytes() const {
        const size_t data_size =
            data ? max_capacity * static_cast<uint64_t>(num_dimensions) * sizeof(data_t) : 0;
        const size_t indices_size = indices ? max_capacity * sizeof(uint32_t) : 0;
        return sizeof(*this) + data_size + indices_size +
               tombstones.size() * (sizeof(uint32_t) + 2 * sizeof(void*)) +
               tombstones.bucket_count() * sizeof(void*);
    }

    // Gather all embeddings from the PDX layout into a contiguous row-major buffer.
    // Assumes no tombstones (call CompactCluster first).
    // Uses blocked transpose for the vertical block and group-first iteration for
    // the horizontal block to maximise cache locality on the source side.
    skmeans::AllocatedBuffer<data_t> GetHorizontalEmbeddingsFromPDXBuffer() const {
        auto out = skmeans::AllocateBuffer<data_t>(
            GetAllocator(), static_cast<size_t>(num_embeddings) * num_dimensions
        );
        for (uint32_t i = 0; i < num_embeddings; i++) {
            ReadEmbeddingFromPDXBuffer(i, out.get() + static_cast<size_t>(i) * num_dimensions);
        }
        return out;
    }

    // Gather a single embedding from the PDX layout into a row-major buffer.
    std::unique_ptr<data_t[]> GetHorizontalEmbeddingFromPDXBuffer(uint32_t idx_in_cluster) const {
        PDX_PROFILE_SCOPE("DePDXify-One");
        std::unique_ptr<data_t[]> out(new data_t[num_dimensions]);
        ReadEmbeddingFromPDXBuffer(idx_in_cluster, out.get());
        return out;
    }

    // Writes the valid PDX data row-by-row, stripping stride gaps.
    // Assumes no tombstones (call CompactCluster first).
    void SavePDXData(std::ostream& out) const {
        const auto split = GetPDXDimensionSplit(num_dimensions);
        const uint32_t vertical_d = split.vertical_dimensions;
        const uint32_t horizontal_d = split.horizontal_dimensions;
        const size_t stride = max_capacity;

        if constexpr (Q == Quantization::F32) {
            for (uint32_t d = 0; d < vertical_d; d++) {
                out.write(
                    reinterpret_cast<const char*>(data.get() + d * stride),
                    sizeof(data_t) * num_embeddings
                );
            }
        } else {
            uint32_t d = 0;
            for (; d + U8_INTERLEAVE_SIZE <= vertical_d; d += U8_INTERLEAVE_SIZE) {
                out.write(
                    reinterpret_cast<const char*>(data.get() + d * stride),
                    num_embeddings * U8_INTERLEAVE_SIZE
                );
            }
            if (d < vertical_d) {
                uint32_t remaining = vertical_d - d;
                out.write(
                    reinterpret_cast<const char*>(data.get() + d * stride),
                    static_cast<std::streamsize>(num_embeddings) * remaining
                );
            }
        }

        const data_t* h_base = data.get() + stride * vertical_d;
        for (uint32_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
            out.write(
                reinterpret_cast<const char*>(h_base), sizeof(data_t) * num_embeddings * H_DIM_SIZE
            );
            h_base += stride * H_DIM_SIZE;
        }
    }

    // Reads compact PDX data (as SavePDXData writes it) and places it into the strided buffer.
    template <class Reader>
    void LoadPDXData(Reader& reader) {
        const auto split = GetPDXDimensionSplit(num_dimensions);
        const uint32_t vertical_d = split.vertical_dimensions;
        const uint32_t horizontal_d = split.horizontal_dimensions;
        const size_t stride = max_capacity;

        if constexpr (Q == Quantization::F32) {
            for (uint32_t d = 0; d < vertical_d; d++) {
                reader.Read(data.get() + d * stride, sizeof(data_t) * num_embeddings);
            }
        } else {
            uint32_t d = 0;
            for (; d + U8_INTERLEAVE_SIZE <= vertical_d; d += U8_INTERLEAVE_SIZE) {
                reader.Read(
                    data.get() + d * stride,
                    static_cast<size_t>(num_embeddings) * U8_INTERLEAVE_SIZE
                );
            }
            if (d < vertical_d) {
                uint32_t remaining = vertical_d - d;
                reader.Read(
                    data.get() + d * stride, static_cast<size_t>(num_embeddings) * remaining
                );
            }
        }

        data_t* h_base = data.get() + stride * vertical_d;
        for (uint32_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
            reader.Read(h_base, sizeof(data_t) * num_embeddings * H_DIM_SIZE);
            h_base += stride * H_DIM_SIZE;
        }
    }

    // Store row-major embeddings into this cluster's preallocated buffers in the transposed PDX
    // layout, from slot 0. See the README of the following for a description of the PDX layout:
    // https://github.com/cwida/pdx
    void StoreEmbeddings(const data_t* embeddings, const size_t num_rows) {
        const auto split = GetPDXDimensionSplit(num_dimensions);
        const uint32_t vertical_d = split.vertical_dimensions;
        const uint32_t horizontal_d = split.horizontal_dimensions;

        if constexpr (Q == Quantization::F32) {
            using h_matrix_t = Eigen::Matrix<float, Eigen::Dynamic, H_DIM_SIZE, Eigen::RowMajor>;
            const auto stride = static_cast<Eigen::Index>(max_capacity);

            Eigen::Map<const eigen_matrix_t> in(embeddings, num_rows, num_dimensions);

            // Vertical block: (vertical_d x num_rows) with row stride = max_capacity
            Eigen::Map<eigen_matrix_t, 0, Eigen::OuterStride<Eigen::Dynamic>> out(
                data.get(), vertical_d, num_rows, Eigen::OuterStride<Eigen::Dynamic>(stride)
            );
            out.noalias() = in.leftCols(vertical_d).transpose();

            float* horizontal_out = data.get() + stride * vertical_d;
            for (size_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
                Eigen::Map<h_matrix_t> out_h(horizontal_out, num_rows, H_DIM_SIZE);
                out_h.noalias() = in.block(0, vertical_d + j, num_rows, H_DIM_SIZE);
                horizontal_out += stride * H_DIM_SIZE;
            }
        } else {
            using u8_matrix_t =
                Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            using u8_v_matrix_t =
                Eigen::Matrix<uint8_t, Eigen::Dynamic, U8_INTERLEAVE_SIZE, Eigen::RowMajor>;
            using u8_h_matrix_t =
                Eigen::Matrix<uint8_t, Eigen::Dynamic, H_DIM_SIZE, Eigen::RowMajor>;
            const auto stride = static_cast<size_t>(max_capacity);

            Eigen::Map<const u8_matrix_t> in(embeddings, num_rows, num_dimensions);

            size_t dim = 0;
            for (; dim + U8_INTERLEAVE_SIZE <= vertical_d; dim += U8_INTERLEAVE_SIZE) {
                Eigen::Map<u8_v_matrix_t> out_v(
                    data.get() + dim * stride, num_rows, U8_INTERLEAVE_SIZE
                );
                out_v.noalias() = in.block(0, dim, num_rows, U8_INTERLEAVE_SIZE);
            }
            if (dim < vertical_d) {
                auto remaining = static_cast<Eigen::Index>(vertical_d - dim);
                Eigen::Map<u8_matrix_t> out_v(data.get() + dim * stride, num_rows, remaining);
                out_v.noalias() = in.block(0, dim, num_rows, remaining);
            }

            uint8_t* horizontal_out = data.get() + stride * vertical_d;
            for (size_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
                Eigen::Map<u8_h_matrix_t> out_h(horizontal_out, num_rows, H_DIM_SIZE);
                out_h.noalias() = in.block(0, vertical_d + j, num_rows, H_DIM_SIZE);
                horizontal_out += stride * H_DIM_SIZE;
            }
        }
    }

    // Caller must hold cluster_mutex
    // Returns: vector of (row_id, new_index_in_cluster) for each moved embedding
    // TODO(@lkuffo, med): I dont like this while loops too much. Its confusing (but it works)
    std::vector<std::pair<uint32_t, uint32_t>> CompactCluster() {
        PDX_PROFILE_SCOPE("CompactCluster");
        std::vector<std::pair<uint32_t, uint32_t>> moves;
        if (tombstones.empty()) {
            return moves;
        }
        moves.reserve(tombstones.size());

        // shrink past any tombstoned tail positions (no data movement needed)
        while (used_capacity > 0 && HasTombstone(used_capacity - 1)) {
            RemoveTombstone(used_capacity - 1);
            indices[used_capacity - 1] = 0;
            used_capacity--;
        }

        // fill remaining interior tombstones by moving from the tail
        while (!tombstones.empty()) {
            uint32_t tombstone_idx = PopTombstone();
            uint32_t last_idx = used_capacity - 1;
            CopyEmbeddingInPDXLayout(last_idx, tombstone_idx);
            indices[tombstone_idx] = indices[last_idx];
            moves.emplace_back(indices[tombstone_idx], tombstone_idx);
            indices[last_idx] = 0;
            used_capacity--;
            // The new tail might also be a tombstone, drain it
            while (used_capacity > 0 && HasTombstone(used_capacity - 1)) {
                RemoveTombstone(used_capacity - 1);
                indices[used_capacity - 1] = 0;
                used_capacity--;
            }
        }

        assert(num_embeddings == used_capacity);
        return moves;
    }

    // Re-lays out the rows in CAPACITY_THRESHOLD times as many slots, keeping their slots (so row
    // ids, centroid and the index's row id mapping stay valid). For a full cluster that a split
    // cannot separate. Assumes no tombstones; the index must recompute its cluster offsets.
    void Grow() {
        PDX_PROFILE_SCOPE("GrowCluster");
        auto rows = GetHorizontalEmbeddingsFromPDXBuffer();
        const uint32_t new_max_capacity = CapacityFor(num_embeddings, 0);
        auto new_indices = skmeans::AllocateBuffer<uint32_t>(GetAllocator(), new_max_capacity);
        auto new_data = skmeans::AllocateBuffer<data_t>(
            GetAllocator(), static_cast<uint64_t>(new_max_capacity) * num_dimensions
        );
        std::memcpy(new_indices.get(), indices.get(), num_embeddings * sizeof(uint32_t));
        indices = std::move(new_indices);
        data = std::move(new_data);
        max_capacity = new_max_capacity;
        min_capacity = static_cast<uint32_t>(num_embeddings * MIN_CAPACITY_THRESHOLD);
        StoreEmbeddings(rows.get(), num_embeddings);
    }

    // Gather-reads one embedding from a transposed PDX buffer of the given stride into a horizontal
    // (row-major) output. Reverse of InsertEmbedding.
    static void ReadEmbeddingFromPDXBuffer(
        const data_t* data,
        size_t stride,
        uint32_t num_dimensions,
        uint32_t idx_in_cluster,
        data_t* out
    ) {
        const auto split = GetPDXDimensionSplit(num_dimensions);
        const uint32_t vertical_d = split.vertical_dimensions;
        const uint32_t horizontal_d = split.horizontal_dimensions;

        if constexpr (Q == Quantization::F32) {
            for (uint32_t d = 0; d < vertical_d; d++) {
                out[d] = data[d * stride + idx_in_cluster];
            }
        } else {
            uint32_t d = 0;
            for (; d + U8_INTERLEAVE_SIZE <= vertical_d; d += U8_INTERLEAVE_SIZE) {
                memcpy(
                    out + d,
                    data + d * stride + static_cast<size_t>(idx_in_cluster) * U8_INTERLEAVE_SIZE,
                    U8_INTERLEAVE_SIZE
                );
            }
            if (d < vertical_d) {
                uint32_t remaining = vertical_d - d;
                memcpy(
                    out + d,
                    data + d * stride + static_cast<size_t>(idx_in_cluster) * remaining,
                    remaining
                );
            }
        }

        const data_t* h_base = data + stride * vertical_d;
        for (uint32_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
            memcpy(
                out + vertical_d + j,
                h_base + static_cast<size_t>(idx_in_cluster) * H_DIM_SIZE,
                H_DIM_SIZE * sizeof(data_t)
            );
            h_base += stride * H_DIM_SIZE;
        }
    }

  private:
    void ReadEmbeddingFromPDXBuffer(uint32_t idx_in_cluster, data_t* out) const {
        ReadEmbeddingFromPDXBuffer(data.get(), max_capacity, num_dimensions, idx_in_cluster, out);
    }

    // Scatter-writes a horizontal (row-major) embedding into the transposed PDX buffer layout.
    // This function assumes thread safety (caller must hold cluster_mutex).
    void InsertEmbedding(uint32_t idx_in_cluster, uint32_t row_id, const data_t* embedding) {
        const auto split = GetPDXDimensionSplit(num_dimensions);
        const uint32_t vertical_d = split.vertical_dimensions;
        const uint32_t horizontal_d = split.horizontal_dimensions;
        const size_t stride = max_capacity;

        if constexpr (Q == Quantization::F32) {
            // Vertical: column-major, one float per dimension row
            for (uint32_t d = 0; d < vertical_d; d++) {
                data[d * stride + idx_in_cluster] = embedding[d];
            }
        } else {
            // U8 Vertical: interleaved in groups of U8_INTERLEAVE_SIZE
            uint32_t d = 0;
            for (; d + U8_INTERLEAVE_SIZE <= vertical_d; d += U8_INTERLEAVE_SIZE) {
                memcpy(
                    data.get() + d * stride +
                        static_cast<size_t>(idx_in_cluster) * U8_INTERLEAVE_SIZE,
                    embedding + d,
                    U8_INTERLEAVE_SIZE
                );
            }
            if (d < vertical_d) {
                uint32_t remaining = vertical_d - d;
                memcpy(
                    data.get() + d * stride + static_cast<size_t>(idx_in_cluster) * remaining,
                    embedding + d,
                    remaining
                );
            }
        }

        // Horizontal: groups of H_DIM_SIZE, row-major within each group
        data_t* h_base = data.get() + stride * vertical_d;
        for (uint32_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
            memcpy(
                h_base + static_cast<size_t>(idx_in_cluster) * H_DIM_SIZE,
                embedding + vertical_d + j,
                H_DIM_SIZE * sizeof(data_t)
            );
            h_base += stride * H_DIM_SIZE;
        }

        indices[idx_in_cluster] = row_id;
    }

    // Copies an embedding within the PDX buffer from one position to another.
    // This function assumes thread safety (caller must hold cluster_mutex).
    void CopyEmbeddingInPDXLayout(uint32_t src_idx, uint32_t dst_idx) {
        const auto split = GetPDXDimensionSplit(num_dimensions);
        const uint32_t vertical_d = split.vertical_dimensions;
        const uint32_t horizontal_d = split.horizontal_dimensions;
        const size_t stride = max_capacity;

        if constexpr (Q == Quantization::F32) {
            for (uint32_t d = 0; d < vertical_d; d++) {
                data[d * stride + dst_idx] = data[d * stride + src_idx];
            }
        } else {
            uint32_t d = 0;
            for (; d + U8_INTERLEAVE_SIZE <= vertical_d; d += U8_INTERLEAVE_SIZE) {
                memcpy(
                    data.get() + d * stride + static_cast<size_t>(dst_idx) * U8_INTERLEAVE_SIZE,
                    data.get() + d * stride + static_cast<size_t>(src_idx) * U8_INTERLEAVE_SIZE,
                    U8_INTERLEAVE_SIZE
                );
            }
            if (d < vertical_d) {
                uint32_t remaining = vertical_d - d;
                memcpy(
                    data.get() + d * stride + static_cast<size_t>(dst_idx) * remaining,
                    data.get() + d * stride + static_cast<size_t>(src_idx) * remaining,
                    remaining
                );
            }
        }

        data_t* h_base = data.get() + stride * vertical_d;
        for (uint32_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
            memcpy(
                h_base + static_cast<size_t>(dst_idx) * H_DIM_SIZE,
                h_base + static_cast<size_t>(src_idx) * H_DIM_SIZE,
                H_DIM_SIZE * sizeof(data_t)
            );
            h_base += stride * H_DIM_SIZE;
        }
    }
};

} // namespace PDX

#pragma once

#include "pdx/common.hpp"
#include "pdx/distance_computers/base_computers.hpp"
#include "pdx/quantizers/lvq.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>

namespace PDX {

class Centroids : public skmeans::AllocatorHolder {
  public:
    Centroids() = default;

    Centroids(
        CentroidQuantization quantization,
        uint32_t num_dimensions,
        MemoryAllocator* allocator = nullptr
    )
        : quantization(quantization), num_dimensions(num_dimensions),
          encoded_centroid_size(GetEncodedCentroidSize(quantization, num_dimensions)) {
        SetAllocator(allocator);
    }

    CentroidQuantization GetQuantization() const { return quantization; }

    void StoreCentroids(const float* centroids, size_t num_new_centroids) {
        num_centroids = 0;
        Reserve(num_new_centroids);
        num_centroids = num_new_centroids;
        for (size_t centroid_id = 0; centroid_id < num_centroids; ++centroid_id) {
            EncodeCentroid(centroid_id, centroids + centroid_id * num_dimensions);
        }
        squared_norms_stale = true;
    }

    void StoreCentroid(uint32_t centroid_id, const float* centroid) {
        EncodeCentroid(centroid_id, centroid);
        squared_norms_stale = true;
    }

    void AppendCentroid(const float* centroid) {
        if (num_centroids == GetCapacity()) {
            Reserve(std::max<size_t>(1, 2 * num_centroids));
        }
        EncodeCentroid(num_centroids++, centroid);
        squared_norms_stale = true;
    }

    void RemoveCentroidBySwapWithLast(uint32_t centroid_id) {
        const size_t last_id = num_centroids - 1;
        if (centroid_id != last_id) {
            std::memcpy(
                GetEncodedCentroidPtr(centroid_id),
                GetEncodedCentroidPtr(last_id),
                encoded_centroid_size
            );
        }
        num_centroids--;
        squared_norms_stale = true;
    }

    void ShrinkToFit() {
        if (num_centroids == GetCapacity()) {
            return;
        }
        if (num_centroids == 0) {
            encoded_centroids.reset();
            return;
        }
        auto shrunk = skmeans::AllocateBuffer<uint8_t>(
            GetAllocator(), num_centroids * encoded_centroid_size
        );
        std::memcpy(shrunk.get(), encoded_centroids.get(), num_centroids * encoded_centroid_size);
        encoded_centroids = std::move(shrunk);
    }

    void ComputeDistancesToCentroids(const float* PDX_RESTRICT query, float* PDX_RESTRICT out)
        const {
        switch (quantization) {
        case CentroidQuantization::F32:
            for (size_t centroid_id = 0; centroid_id < num_centroids; ++centroid_id) {
                out[centroid_id] = DistanceComputer<DistanceMetric::L2SQ, F32>::Horizontal(
                    query, GetStoredF32CentroidPtr(centroid_id), num_dimensions
                );
            }
            break;
        case CentroidQuantization::LVQ8: {
            const LVQQuantizer<8> quantizer(num_dimensions);
            float query_sum = 0.0f;
            float query_squared_norm = 0.0f;
            PDX_VECTORIZE_LOOP
            for (size_t i = 0; i < num_dimensions; ++i) {
                query_sum += query[i];
                query_squared_norm += query[i] * query[i];
            }
            for (size_t centroid_id = 0; centroid_id < num_centroids; ++centroid_id) {
                out[centroid_id] = quantizer.ComputeL2SQ(
                    query,
                    query_sum,
                    query_squared_norm,
                    GetLVQCodesPtr(centroid_id),
                    ReadLVQParams(centroid_id)
                );
            }
            break;
        }
        }
    }

    void DecodeCentroid(uint32_t centroid_id, float* out) const {
        if (quantization == CentroidQuantization::F32) {
            std::memcpy(out, GetStoredF32CentroidPtr(centroid_id), num_dimensions * sizeof(float));
            return;
        }
        DecodeLVQCentroid(centroid_id, out);
    }

    const float* GetF32CentroidPtr(uint32_t centroid_id, float* scratch) const {
        if (quantization == CentroidQuantization::F32) {
            return GetStoredF32CentroidPtr(centroid_id);
        }
        DecodeLVQCentroid(centroid_id, scratch);
        return scratch;
    }

    const float* GetF32CentroidsPtr(skmeans::AllocatedBuffer<float>& scratch) const {
        if (quantization == CentroidQuantization::F32) {
            return GetStoredF32CentroidPtr(0);
        }
        skmeans::EnsureBufferSize(scratch, GetAllocator(), num_centroids * num_dimensions);
        for (size_t centroid_id = 0; centroid_id < num_centroids; ++centroid_id) {
            DecodeLVQCentroid(centroid_id, scratch.get() + centroid_id * num_dimensions);
        }
        return scratch.get();
    }

    const float* GetF32CentroidsSquaredNormsPtr() {
        if (squared_norms_stale) {
            skmeans::EnsureBufferSize(squared_norms, GetAllocator(), num_centroids);
            switch (quantization) {
            case CentroidQuantization::F32:
                Eigen::Map<Eigen::VectorXf>(squared_norms.get(), num_centroids).noalias() =
                    Eigen::Map<const eigen_matrix_t>(
                        GetStoredF32CentroidPtr(0), num_centroids, num_dimensions
                    )
                        .rowwise()
                        .squaredNorm();
                break;
            case CentroidQuantization::LVQ8:
                for (size_t centroid_id = 0; centroid_id < num_centroids; ++centroid_id) {
                    squared_norms[centroid_id] = ReadLVQParams(centroid_id).dequantized_squared_norm;
                }
                break;
            }
            squared_norms_stale = false;
        }
        return squared_norms.get();
    }

    void Save(std::ostream& out) const {
        out.write(
            reinterpret_cast<const char*>(encoded_centroids.get()),
            static_cast<std::streamsize>(num_centroids * encoded_centroid_size)
        );
    }

    template <class Reader>
    void Load(Reader& reader, size_t num_saved_centroids) {
        num_centroids = 0;
        Reserve(num_saved_centroids);
        if (num_saved_centroids > 0) {
            reader.Read(encoded_centroids.get(), num_saved_centroids * encoded_centroid_size);
        }
        num_centroids = num_saved_centroids;
        squared_norms_stale = true;
    }

    size_t GetAllocatedSizeInBytes() const {
        const size_t encoded_centroids_size =
            encoded_centroids ? encoded_centroids.get_deleter().Size() : 0;
        const size_t squared_norms_size =
            squared_norms ? squared_norms.get_deleter().Size() * sizeof(float) : 0;
        return encoded_centroids_size + squared_norms_size;
    }

  private:
    static size_t GetEncodedCentroidSize(
        CentroidQuantization quantization,
        uint32_t num_dimensions
    ) {
        switch (quantization) {
        case CentroidQuantization::F32:
            return num_dimensions * sizeof(float);
        case CentroidQuantization::LVQ8:
            return sizeof(LVQParams) + num_dimensions;
        }
        throw std::runtime_error(
            "Unknown PDX centroid quantization: " +
            std::to_string(static_cast<int>(quantization))
        );
    }

    size_t GetCapacity() const {
        return encoded_centroids ? encoded_centroids.get_deleter().Size() / encoded_centroid_size
                                 : 0;
    }

    void Reserve(size_t capacity) {
        if (capacity <= GetCapacity()) {
            return;
        }
        auto grown =
            skmeans::AllocateBuffer<uint8_t>(GetAllocator(), capacity * encoded_centroid_size);
        if (num_centroids > 0) {
            std::memcpy(
                grown.get(), encoded_centroids.get(), num_centroids * encoded_centroid_size
            );
        }
        encoded_centroids = std::move(grown);
    }

    uint8_t* GetEncodedCentroidPtr(size_t centroid_id) const {
        return encoded_centroids.get() + centroid_id * encoded_centroid_size;
    }

    const float* GetStoredF32CentroidPtr(size_t centroid_id) const {
        return reinterpret_cast<const float*>(encoded_centroids.get()) +
               centroid_id * num_dimensions;
    }

    const uint8_t* GetLVQCodesPtr(size_t centroid_id) const {
        return GetEncodedCentroidPtr(centroid_id) + sizeof(LVQParams);
    }

    LVQParams ReadLVQParams(size_t centroid_id) const {
        LVQParams params{};
        std::memcpy(&params, GetEncodedCentroidPtr(centroid_id), sizeof(LVQParams));
        return params;
    }

    void EncodeCentroid(size_t centroid_id, const float* centroid) {
        switch (quantization) {
        case CentroidQuantization::F32:
            std::memcpy(GetEncodedCentroidPtr(centroid_id), centroid, encoded_centroid_size);
            break;
        case CentroidQuantization::LVQ8: {
            const LVQParams params =
                LVQQuantizer<8>(num_dimensions)
                    .QuantizeEmbedding(
                        centroid, GetEncodedCentroidPtr(centroid_id) + sizeof(LVQParams)
                    );
            std::memcpy(GetEncodedCentroidPtr(centroid_id), &params, sizeof(LVQParams));
            break;
        }
        }
    }

    void DecodeLVQCentroid(size_t centroid_id, float* out) const {
        LVQQuantizer<8>(num_dimensions)
            .DequantizeEmbedding(GetLVQCodesPtr(centroid_id), ReadLVQParams(centroid_id), out);
    }

    CentroidQuantization quantization = CentroidQuantization::F32;
    uint32_t num_dimensions = 0;
    size_t encoded_centroid_size = 0;
    size_t num_centroids = 0;
    skmeans::AllocatedBuffer<uint8_t> encoded_centroids;
    skmeans::AllocatedBuffer<float> squared_norms;
    bool squared_norms_stale = true;
};

} // namespace PDX

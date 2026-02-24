#pragma once

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#include "pdx/index.hpp"

namespace py = pybind11;

namespace PDX {

inline DistanceMetric ToDistanceMetric(uint8_t metric) {
    switch (metric) {
    case 0:
        return DistanceMetric::L2SQ;
    case 1:
        return DistanceMetric::COSINE;
    case 2:
        return DistanceMetric::IP;
    default:
        throw std::runtime_error("Unknown distance metric: " + std::to_string(metric));
    }
}

class PyPDXIndex {
    std::unique_ptr<IPDXIndex> index;

    PyPDXIndex() = default;

  public:
    PyPDXIndex(
        const std::string& index_type,
        uint32_t num_dimensions,
        uint8_t distance_metric,
        uint32_t seed,
        uint32_t num_clusters,
        uint32_t num_meso_clusters,
        bool normalize,
        float sampling_fraction,
        uint32_t kmeans_iters
    ) {
        PDXIndexConfig config{
            .num_dimensions = num_dimensions,
            .distance_metric = ToDistanceMetric(distance_metric),
            .seed = seed,
            .num_clusters = num_clusters,
            .num_meso_clusters = num_meso_clusters,
            .normalize = normalize,
            .sampling_fraction = sampling_fraction,
            .kmeans_iters = kmeans_iters,
        };
        if (index_type == "pdx_f32") {
            index = std::make_unique<PDXIndexF32>(config);
        } else if (index_type == "pdx_u8") {
            index = std::make_unique<PDXIndexU8>(config);
        } else if (index_type == "pdx_tree_f32") {
            index = std::make_unique<PDXTreeIndexF32>(config);
        } else if (index_type == "pdx_tree_u8") {
            index = std::make_unique<PDXTreeIndexU8>(config);
        } else {
            throw std::runtime_error(
                "Unknown index type: " + index_type +
                ". Valid types: pdx_f32, pdx_u8, pdx_tree_f32, pdx_tree_u8"
            );
        }
    }

    static PyPDXIndex LoadFromFile(const std::string& path) {
        PyPDXIndex self;
        self.index = PDX::LoadPDXIndex(path);
        return self;
    }

    void BuildIndex(const py::array_t<float>& data) {
        auto buf = data.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("data must be a 2D numpy array (n_embeddings x dimensions)");
        }
        auto* ptr = static_cast<const float*>(buf.ptr);
        size_t n = static_cast<size_t>(buf.shape[0]);
        index->BuildIndex(ptr, n);
    }

    std::pair<py::array_t<uint32_t>, py::array_t<float>> Search(
        const py::array_t<float>& query,
        uint32_t knn
    ) const {
        auto buf = query.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("query must be a 1D numpy array");
        }
        auto* ptr = static_cast<const float*>(buf.ptr);
        auto results = index->Search(ptr, knn);
        size_t n = results.size();
        py::array_t<uint32_t> ids(n);
        py::array_t<float> distances(n);
        auto ids_ptr = ids.mutable_unchecked<1>();
        auto distances_ptr = distances.mutable_unchecked<1>();
        for (size_t i = 0; i < n; ++i) {
            ids_ptr(i) = results[i].index;
            distances_ptr(i) = results[i].distance;
        }
        return {ids, distances};
    }

    void SetNProbe(uint32_t n) const { index->SetNProbe(n); }

    void Save(const std::string& path) const { index->Save(path); }

    uint32_t GetNumDimensions() const { return index->GetNumDimensions(); }

    uint32_t GetNumClusters() const { return index->GetNumClusters(); }

    size_t GetInMemorySizeInBytes() const { return index->GetInMemorySizeInBytes(); }
};

} // namespace PDX

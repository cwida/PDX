#ifndef PDX_LIB
#define PDX_LIB

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
#include "utils/file_reader.hpp"
#include "index_base/pdx_ivf.hpp"
#include "index_base/pdx_ivf2.hpp"
#include "pdxearch.hpp"
#include "pruners/bond.hpp"
#include "pruners/adsampling.hpp"

// TODO: the python wrapper and the core API should not be interleaved
namespace py = pybind11;

/******************************************************************
 * Very rudimentary wrappers for python bindings
 * Probably a lot of room for improvement (TODO)
 ******************************************************************/
namespace PDX {

class IndexADSamplingIVF2SQ8 {

    using KNNCandidate = KNNCandidate<U8>;
    using Index = IndexPDXIVF2<U8>;
    using Pruner = ADSamplingPruner<U8>;
    using Searcher = PDXearch<U8, Index>;

public:
    Index index = Index();
    std::unique_ptr<Searcher> searcher = nullptr;
    std::unique_ptr<char[]> transformation_matrix = nullptr;
    constexpr static float epsilon0 = 1.5;

    void Load(const py::bytes& data, const py::array_t<float>& _matrix){
        py::buffer_info info(py::buffer(data).request());
        auto data_ = static_cast<char*>(info.ptr);
        index.Load(data_);

        auto matrix_buf = _matrix.request();
        auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
        Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix_ptr);
        searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
    }

    void Restore(const std::string &path, const std::string &matrix_path){
        index.Restore(path);
        transformation_matrix = MmapFile(matrix_path);
        auto *_matrix = reinterpret_cast<float*>(transformation_matrix.get());
        Pruner pruner = Pruner(index.num_dimensions, epsilon0, _matrix);
        searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
    }

    void SetPruningConfidence(float confidence) const {
        searcher->pruner.SetEpsilon0(confidence);
    }

    std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
        return searcher->Search(q, k);
    }

    // Serialize return value
    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_Search(const py::array_t<float>& q, uint32_t k, uint32_t n_probe) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        searcher->SetNProbe(n_probe);
        std::vector<KNNCandidate> results = searcher->Search(query, k);
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

    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_FilteredSearch(
        const py::array_t<float>& q, uint32_t k, uint32_t n_probe,
        const py::array_t<uint32_t>& n_passing_tuples, const py::array_t<uint8_t>& selection_vector
    ) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        auto n_passing_tuples_p = static_cast<uint32_t*>(n_passing_tuples.request().ptr);
        auto selection_vector_p = static_cast<uint8_t*>(selection_vector.request().ptr);

        PredicateEvaluator pe = PredicateEvaluator(searcher->pdx_data.num_clusters);
        pe.LoadSelectionVector(n_passing_tuples_p, selection_vector_p);

        searcher->SetNProbe(n_probe);
        std::vector<KNNCandidate> results = searcher->FilteredSearch(query, k, pe);
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

};

class IndexADSamplingIVF2Flat {

    using KNNCandidate = KNNCandidate<F32>;
    using Index = IndexPDXIVF2<F32>;
    using Pruner = ADSamplingPruner<F32>;
    using Searcher = PDXearch<F32, Index>;

public:
    Index index = Index();
    std::unique_ptr<Searcher> searcher = nullptr;
    std::unique_ptr<char[]> transformation_matrix = nullptr;
    constexpr static float epsilon0 = 1.5;

    void Load(const py::bytes& data, const py::array_t<float>& _matrix){
        py::buffer_info info(py::buffer(data).request());
        auto data_ = static_cast<char*>(info.ptr);
        index.Load(data_);

        auto matrix_buf = _matrix.request();
        auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
        Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix_ptr);
        searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
    }

    void Restore(const std::string &path, const std::string &matrix_path){
        index.Restore(path);
        transformation_matrix = MmapFile(matrix_path);
        auto *_matrix = reinterpret_cast<float*>(transformation_matrix.get());
        Pruner pruner = Pruner(index.num_dimensions, epsilon0, _matrix);
        searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
    }

    void SetPruningConfidence(float confidence) const {
        searcher->pruner.SetEpsilon0(confidence);
    }

    std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
        return searcher->Search(q, k);
    }

    // Serialize return value
    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_Search(const py::array_t<float>& q, uint32_t k, uint32_t n_probe) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        searcher->SetNProbe(n_probe);
        std::vector<KNNCandidate> results = searcher->Search(query, k);
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

    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_FilteredSearch(
        const py::array_t<float>& q, uint32_t k, uint32_t n_probe,
        const py::array_t<uint32_t>& n_passing_tuples, const py::array_t<uint8_t>& selection_vector
    ) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        auto n_passing_tuples_p = static_cast<uint32_t*>(n_passing_tuples.request().ptr);
        auto selection_vector_p = static_cast<uint8_t*>(selection_vector.request().ptr);

        PredicateEvaluator pe = PredicateEvaluator(searcher->pdx_data.num_clusters);
        pe.LoadSelectionVector(n_passing_tuples_p, selection_vector_p);

        searcher->SetNProbe(n_probe);
        std::vector<KNNCandidate> results = searcher->FilteredSearch(query, k, pe);
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

};

class IndexADSamplingIVFFlat {
    
    using KNNCandidate = KNNCandidate<F32>;
    using Index = IndexPDXIVF<F32>;
    using Pruner = ADSamplingPruner<F32>;
    using Searcher = PDXearch<F32>;
    
public:
    Index index = Index();
    std::unique_ptr<Searcher> searcher = nullptr;
    std::unique_ptr<char[]> transformation_matrix = nullptr;
    constexpr static float epsilon0 = 1.5;

    void Load(const py::bytes& data, const py::array_t<float>& _matrix){
        py::buffer_info info(py::buffer(data).request());
        auto data_ = static_cast<char*>(info.ptr);
        index.Load(data_);

        auto matrix_buf = _matrix.request();
        auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
        Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix_ptr);
        searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
    }

    void Restore(const std::string &path, const std::string &matrix_path){
        index.Restore(path);
        transformation_matrix = MmapFile(matrix_path);
        auto *_matrix = reinterpret_cast<float*>(transformation_matrix.get());
        Pruner pruner = Pruner(index.num_dimensions, epsilon0, _matrix);
        searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
    }

    void SetPruningConfidence(float confidence) const {
        searcher->pruner.SetEpsilon0(confidence);
    }

    std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
        return searcher->Search(q, k);
    }

    // Serialize return value
    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_Search(const py::array_t<float>& q, uint32_t k, uint32_t n_probe) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        searcher->SetNProbe(n_probe);
        std::vector<KNNCandidate> results = searcher->Search(query, k);
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

    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_FilteredSearch(
        const py::array_t<float>& q, uint32_t k, uint32_t n_probe,
        const py::array_t<uint32_t>& n_passing_tuples, const py::array_t<uint8_t>& selection_vector
    ) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        auto n_passing_tuples_p = static_cast<uint32_t*>(n_passing_tuples.request().ptr);
        auto selection_vector_p = static_cast<uint8_t*>(selection_vector.request().ptr);

        PredicateEvaluator pe = PredicateEvaluator(searcher->pdx_data.num_clusters);
        pe.LoadSelectionVector(n_passing_tuples_p, selection_vector_p);

        searcher->SetNProbe(n_probe);
        std::vector<KNNCandidate> results = searcher->FilteredSearch(query, k, pe);
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

};

class IndexBONDIVFFlat {
    using KNNCandidate = KNNCandidate<F32>;
    using Index = IndexPDXIVF<F32>;
    using Pruner = BondPruner<F32>;
    using Searcher = PDXearch<F32, Index, Global8Quantizer<F32>, L2, BondPruner<F32>>;

public:
    Index index = Index();
    std::unique_ptr<Searcher> searcher = nullptr;

    void Load(const py::bytes& data){
        py::buffer_info info(py::buffer(data).request());
        auto data_ = static_cast<char*>(info.ptr);
        index.Load(data_);
#if false && defined(__AVX512FP16__)
        // In Intel architectures with low bandwidth at L3/DRAM, the DISTANCE_TO_MEANS criteria performs better
        pruner = Pruner(index.num_dimensions);
        searcher = std::make_unique<Searcher>(index, pruner, 0, DISTANCE_TO_MEANS);
#else
        Pruner pruner = Pruner(index.num_dimensions);
        searcher = std::make_unique<Searcher>(index, pruner, 0, DIMENSION_ZONES);
#endif
    }

    void Restore(const std::string &path){
        index.Restore(path);
#if false && defined(__AVX512FP16__)
        // In Intel architectures with low bandwidth at L3/DRAM, the DISTANCE_TO_MEANS criteria performs better
        pruner = Pruner(index.num_dimensions);
        searcher = std::make_unique<Searcher>(index, pruner, 0, DISTANCE_TO_MEANS);
#else
        Pruner pruner = Pruner(index.num_dimensions);
        searcher = std::make_unique<Searcher>(index, pruner, 0, DIMENSION_ZONES);
#endif
    }

    std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
        return searcher->Search(q, k);
    }

    // Serialize return value
    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_Search(const py::array_t<float>& q, uint32_t k, uint32_t n_probe) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        searcher->SetNProbe(n_probe);
        std::vector<KNNCandidate> results = searcher->Search(query, k);
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

    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_FilteredSearch(
        const py::array_t<float>& q, uint32_t k, uint32_t n_probe,
        const py::array_t<uint32_t>& n_passing_tuples, const py::array_t<uint8_t>& selection_vector
    ) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        auto n_passing_tuples_p = static_cast<uint32_t*>(n_passing_tuples.request().ptr);
        auto selection_vector_p = static_cast<uint8_t*>(selection_vector.request().ptr);

        PredicateEvaluator pe = PredicateEvaluator(searcher->pdx_data.num_clusters);
        pe.LoadSelectionVector(n_passing_tuples_p, selection_vector_p);

        searcher->SetNProbe(n_probe);
        std::vector<KNNCandidate> results = searcher->FilteredSearch(query, k, pe);
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

};

class IndexBONDFlat {
    using KNNCandidate = KNNCandidate<F32>;
    using Index = IndexPDXIVF<F32>;
    using Pruner = BondPruner<F32>;
    using Searcher = PDXearch<F32, Index, Global8Quantizer<F32>, L2, BondPruner<F32>>;
public:
    Index index = Index();
    std::unique_ptr<Searcher> searcher = nullptr;

    void Load(const py::bytes& data){
        py::buffer_info info(py::buffer(data).request());
        auto data_ = static_cast<char*>(info.ptr);
        index.Load(data_);
        Pruner pruner = Pruner(index.num_dimensions);
        searcher = std::make_unique<Searcher>(index, pruner, 0, DISTANCE_TO_MEANS);
    }

    void Restore(const std::string &path){
        index.Restore(path);
        Pruner pruner = Pruner(index.num_dimensions);
        searcher = std::make_unique<Searcher>(index, pruner, 0, DISTANCE_TO_MEANS);
    }

    std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
        return searcher->Search(q, k);
    }

    // Serialize return value
    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_Search(const py::array_t<float>& q, uint32_t k) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        std::vector<KNNCandidate> results = searcher->Search(query, k);
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

    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_FilteredSearch(
        const py::array_t<float>& q, uint32_t k,
        const py::array_t<uint32_t>& n_passing_tuples, const py::array_t<uint8_t>& selection_vector
    ) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        auto n_passing_tuples_p = static_cast<uint32_t*>(n_passing_tuples.request().ptr);
        auto selection_vector_p = static_cast<uint8_t*>(selection_vector.request().ptr);

        PredicateEvaluator pe = PredicateEvaluator(searcher->pdx_data.num_clusters);
        pe.LoadSelectionVector(n_passing_tuples_p, selection_vector_p);

        std::vector<KNNCandidate> results = searcher->FilteredSearch(query, k, pe);
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

};

class IndexADSamplingFlat {

    using KNNCandidate = KNNCandidate<F32>;
    using Index = IndexPDXIVF<F32>;
    using Pruner = ADSamplingPruner<F32>;
    using Searcher = PDXearch<F32>;

public:
    Index index = Index();
    std::unique_ptr<Searcher> searcher = nullptr;
    std::unique_ptr<char[]> transformation_matrix = nullptr;
    constexpr static float epsilon0 = 1.5;

    void Load(const py::bytes& data, const py::array_t<float>& _matrix){
        py::buffer_info info(py::buffer(data).request());
        auto data_ = static_cast<char*>(info.ptr);
        index.Load(data_);

        auto matrix_buf = _matrix.request();
        auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
        Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix_ptr);
        searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
    }

    void Restore(const std::string &path, const std::string &matrix_path){
        index.Restore(path);
        transformation_matrix = MmapFile(matrix_path);
        auto *_matrix = reinterpret_cast<float*>(transformation_matrix.get());
        Pruner pruner = Pruner(index.num_dimensions, epsilon0, _matrix);
        searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
    }

    void SetPruningConfidence(float confidence) const {
        searcher->pruner.SetEpsilon0(confidence);
    }

    std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
        return searcher->Search(q, k);
    }

        // Serialize return value
    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_Search(const py::array_t<float>& q, uint32_t k) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        std::vector<KNNCandidate> results = searcher->Search(query, k);
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

    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_FilteredSearch(
        const py::array_t<float>& q, uint32_t k,
        const py::array_t<uint32_t>& n_passing_tuples, const py::array_t<uint8_t>& selection_vector
    ) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        auto n_passing_tuples_p = static_cast<uint32_t*>(n_passing_tuples.request().ptr);
        auto selection_vector_p = static_cast<uint8_t*>(selection_vector.request().ptr);

        PredicateEvaluator pe = PredicateEvaluator(searcher->pdx_data.num_clusters);
        pe.LoadSelectionVector(n_passing_tuples_p, selection_vector_p);

        std::vector<KNNCandidate> results = searcher->FilteredSearch(query, k, pe);
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

};

class IndexPDXFlat {
    using KNNCandidate = KNNCandidate<F32>;
    using Index = IndexPDXIVF<F32>;
    using Pruner = BondPruner<F32>;
    using Searcher = PDXearch<F32, Index, Global8Quantizer<F32>, L2, BondPruner<F32>>;

public:
    Index index = Index();
    std::unique_ptr<Searcher> searcher = nullptr;

    void Load(const py::bytes& data){
        py::buffer_info info(py::buffer(data).request());
        auto data_ = static_cast<char*>(info.ptr);
        index.Load(data_);
        Pruner pruner = Pruner(index.num_dimensions);
        searcher = std::make_unique<Searcher>(index, pruner, 0, SEQUENTIAL);
    }

    void Restore(const std::string &path){
        index.Restore(path);
        Pruner pruner = Pruner(index.num_dimensions);
        searcher = std::make_unique<Searcher>(index, pruner, 0, SEQUENTIAL);
    }

    std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
        return searcher->LinearScan(q, k);
    }

        // Serialize return value
    std::pair<py::array_t<uint32_t>, py::array_t<float>>
    _py_Search(const py::array_t<float>& q, uint32_t k) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        std::vector<KNNCandidate> results = searcher->Search(query, k);
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
};

} // namespace PDX

#endif // PDX_LIB
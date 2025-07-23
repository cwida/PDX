#ifndef PDX_LIB
#define PDX_LIB

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "utils/file_reader.hpp"
#include "index_base/pdx_ivf.hpp"
#include "index_base/pdx_imi.hpp"
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

class IndexADSamplingIMISQ8 {

    using KNNCandidate = KNNCandidate<U8>;
    using Index = IndexPDXIMI<U8>;
    using Pruner = ADSamplingPruner<U8>;
    using Searcher = PDXearch<U8, Index>;

    public:
        Index index = Index();
        std::unique_ptr<Searcher> searcher = nullptr;
        constexpr static float epsilon0 = 1.5;

        void Load(const py::bytes& data, const py::array_t<float>& _matrix){
            py::buffer_info info(py::buffer(data).request());
            auto data_ = static_cast<char*>(info.ptr);
            index.Load(data_);

            auto matrix_buf = _matrix.request();
            auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(matrix_ptr, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix);
            searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
        }

        void Restore(const std::string &path, const std::string &matrix_path){
            index.Restore(path);
            float * _matrix = MmapFile32(matrix_path); // TODO: Fix and put in same index file
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix);
            searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
        }

        std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
            return searcher->Search(q, k);
        }

        // Serialize return value
        std::vector<KNNCandidate> _py_Search(const py::array_t<float>& q, uint32_t k, uint32_t n_probe) const {
            auto buf = q.request();  // Get buffer info
            if (buf.ndim != 1) {
                throw std::runtime_error("Input should be a 1-D NumPy array");
            }
            auto query = static_cast<float*>(buf.ptr);
            searcher->SetNProbe(n_probe);
            return searcher->Search(query, k);
        }
};

class IndexADSamplingIMIFlat {

    using KNNCandidate = KNNCandidate<F32>;
    using Index = IndexPDXIMI<F32>;
    using Pruner = ADSamplingPruner<F32>;
    using Searcher = PDXearch<F32, Index>;

    public:
        Index index = Index();
        std::unique_ptr<Searcher> searcher = nullptr;
        constexpr static float epsilon0 = 1.5;

        void Load(const py::bytes& data, const py::array_t<float>& _matrix){
            py::buffer_info info(py::buffer(data).request());
            auto data_ = static_cast<char*>(info.ptr);
            index.Load(data_);

            auto matrix_buf = _matrix.request();
            auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(matrix_ptr, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix);
            searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
        }

        void Restore(const std::string &path, const std::string &matrix_path){
            index.Restore(path);
            float * _matrix = MmapFile32(matrix_path); // TODO: Fix and put in same index file
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix);
            searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
        }

        std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
            return searcher->Search(q, k);
        }

        // Serialize return value
        std::vector<KNNCandidate> _py_Search(const py::array_t<float>& q, uint32_t k, uint32_t n_probe) const {
            auto buf = q.request();  // Get buffer info
            if (buf.ndim != 1) {
                throw std::runtime_error("Input should be a 1-D NumPy array");
            }
            auto query = static_cast<float*>(buf.ptr);
            searcher->SetNProbe(n_probe);
            return searcher->Search(query, k);
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
        constexpr static float epsilon0 = 1.5;

        void Load(const py::bytes& data, const py::array_t<float>& _matrix){
            py::buffer_info info(py::buffer(data).request());
            auto data_ = static_cast<char*>(info.ptr);
            index.Load(data_);

            auto matrix_buf = _matrix.request();
            auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(matrix_ptr, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix);
            searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
        }

        void Restore(const std::string &path, const std::string &matrix_path){
            index.Restore(path);
            float * _matrix = MmapFile32(matrix_path); // TODO: Fix and put in same index file
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix);
            searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
        }

        std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
            return searcher->Search(q, k);
        }

        // Serialize return value
        std::vector<KNNCandidate> _py_Search(const py::array_t<float>& q, uint32_t k, uint32_t n_probe) const {
            auto buf = q.request();  // Get buffer info
            if (buf.ndim != 1) {
                throw std::runtime_error("Input should be a 1-D NumPy array");
            }
            auto query = static_cast<float*>(buf.ptr);
            searcher->SetNProbe(n_probe);
            return searcher->Search(query, k);
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
        std::vector<KNNCandidate> _py_Search(const py::array_t<float>& q, uint32_t k, uint32_t n_probe) const {
            auto buf = q.request();  // Get buffer info
            if (buf.ndim != 1) {
                throw std::runtime_error("Input should be a 1-D NumPy array");
            }
            auto query = static_cast<float*>(buf.ptr);
            searcher->SetNProbe(n_probe);
            return searcher->Search(query, k);
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
    std::vector<KNNCandidate> _py_Search(const py::array_t<float>& q, uint32_t k) const {
        auto buf = q.request();  // Get buffer info
        if (buf.ndim != 1) {
            throw std::runtime_error("Input should be a 1-D NumPy array");
        }
        auto query = static_cast<float*>(buf.ptr);
        return searcher->Search(query, k);
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
        constexpr static float epsilon0 = 1.5;

        void Load(const py::bytes& data, const py::array_t<float>& _matrix){
            py::buffer_info info(py::buffer(data).request());
            auto data_ = static_cast<char*>(info.ptr);
            index.Load(data_);

            auto matrix_buf = _matrix.request();
            auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(matrix_ptr, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();

            Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix);
            searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
        }

        void Restore(const std::string &path, const std::string &matrix_path){
            index.Restore(path);
            float * _matrix = MmapFile32(matrix_path); // TODO: Should be inside same index file?
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            Pruner pruner = Pruner(index.num_dimensions, epsilon0, matrix);
            searcher = std::make_unique<Searcher>(index, pruner, 1, SEQUENTIAL);
        }

        std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
            return searcher->Search(q, k);
        }

        // Serialize return value
        std::vector<KNNCandidate> _py_Search(const py::array_t<float>& q, uint32_t k) const {
            auto buf = q.request();  // Get buffer info
            if (buf.ndim != 1) {
                throw std::runtime_error("Input should be a 1-D NumPy array");
            }
            auto query = static_cast<float*>(buf.ptr);
            return searcher->Search(query, k);
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
        std::vector<KNNCandidate> _py_Search(const py::array_t<float>& q, uint32_t k) const {
            auto buf = q.request();  // Get buffer info
            if (buf.ndim != 1) {
                throw std::runtime_error("Input should be a 1-D NumPy array");
            }
            auto query = static_cast<float*>(buf.ptr);
            return searcher->LinearScan(query, k);
        }
    };

} // namespace PDX

#endif // PDX_LIB
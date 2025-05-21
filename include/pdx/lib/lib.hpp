#ifndef PDX_LIB
#define PDX_LIB

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "utils/file_reader.hpp"
#include "pdx/index_base/pdx_ivf.hpp"
#include "pdx/bond.hpp"
#include "pdx/adsampling.hpp"
#include "pdx/bsa.hpp"

// TODO: the python wrapper and the core API should not be interleaved
namespace py = pybind11;

/******************************************************************
 * Very rudimentary wrappers for python bindings
 * Probably a lot of room for improvement (TODO)
 ******************************************************************/
namespace PDX {

class IndexADSamplingIVFFlat {
    
    using KNNCandidate = KNNCandidate<PDX::F32>;
    using IndexPDXIVF = IndexPDXIVF<PDX::F32>;
    using ADSamplingSearcher = ADSamplingSearcher<PDX::F32>;
    
    public:
        IndexPDXIVF index = IndexPDXIVF();
        std::unique_ptr<ADSamplingSearcher> searcher = nullptr;
        constexpr static const float epsilon0 = 2.1;

        // TODO: A preprocess and create function, right now it will reside on Python

        void Load(const py::bytes& data, const py::array_t<float>& _matrix){
            py::buffer_info info(py::buffer(data).request());
            auto data_ = static_cast<char*>(info.ptr);
            index.Load(data_);

            auto matrix_buf = _matrix.request();
            auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(matrix_ptr, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();

            searcher = std::make_unique<ADSamplingSearcher>(index, 64, epsilon0, matrix, SEQUENTIAL);
        }

        void Restore(const std::string &path, const std::string &matrix_path){
            index.Restore(path);
            float * _matrix = MmapFile32(matrix_path); // TODO: Fix and put in same index file
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            searcher = std::make_unique<ADSamplingSearcher>(index, 64, epsilon0, matrix, SEQUENTIAL);
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
            if (n_probe >= 0){
                searcher->SetNProbe(n_probe);
            }
            return searcher->Search(query, k);
        }
    };

class IndexBONDIVFFlat {
    using KNNCandidate = KNNCandidate<PDX::F32>;
    using IndexPDXIVF = IndexPDXIVF<PDX::F32>;
    using PDXBondSearcher = PDXBondSearcher<PDX::F32>;
    public:
        IndexPDXIVF index = IndexPDXIVF();
        std::unique_ptr<PDXBondSearcher> searcher = nullptr;

        // TODO: A preprocess and create function, right now it will reside on Python

        void Load(const py::bytes& data){
            py::buffer_info info(py::buffer(data).request());
            auto data_ = static_cast<char*>(info.ptr);
            index.Load(data_);
#if defined(__AVX512FP16__)
            // In Intel architectures with low bandwidth at L3/DRAM, the DISTANCE_TO_MEANS criteria performs better
            searcher = std::make_unique<PDXBondSearcher>(index, 64, 0, DISTANCE_TO_MEANS);
#else
            searcher = std::make_unique<PDXBondSearcher>(index, 64, 0, DIMENSION_ZONES);
#endif
        }

        void Restore(const std::string &path){
            index.Restore(path);
#if defined(__AVX512FP16__)
            // In Intel architectures with low bandwidth at L3/DRAM, the DISTANCE_TO_MEANS criteria performs better
            searcher = std::make_unique<PDXBondSearcher>(index, 64, 0, DISTANCE_TO_MEANS);
#else
            searcher = std::make_unique<PDXBondSearcher>(index, 64, 0, DIMENSION_ZONES);
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
            if (n_probe >= 0){
                searcher->SetNProbe(n_probe);
            }
            return searcher->Search(query, k);
        }
    };

class IndexBONDFlat {
    using KNNCandidate = KNNCandidate<PDX::F32>;
    using IndexPDXIVF = IndexPDXIVF<PDX::F32>;
    using PDXBondSearcher = PDXBondSearcher<PDX::F32>;
public:
    IndexPDXIVF index = IndexPDXIVF();
    std::unique_ptr<PDXBondSearcher> searcher = nullptr;

    // TODO: A preprocess and create function, right now it will reside on Python

    void Load(const py::bytes& data){
        py::buffer_info info(py::buffer(data).request());
        auto data_ = static_cast<char*>(info.ptr);
        index.Load(data_);
        searcher = std::make_unique<PDXBondSearcher>(index, 0, 0, DISTANCE_TO_MEANS);
    }

    void Restore(const std::string &path){
        index.Restore(path);
        searcher = std::make_unique<PDXBondSearcher>(index, 0, 0, DISTANCE_TO_MEANS);
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
    using KNNCandidate = KNNCandidate<PDX::F32>;
    using IndexPDXIVF = IndexPDXIVF<PDX::F32>;
    using ADSamplingSearcher = ADSamplingSearcher<PDX::F32>;
    public:
        IndexPDXIVF index = IndexPDXIVF();
        std::unique_ptr<ADSamplingSearcher> searcher = nullptr;
        constexpr static const float epsilon0 = 2.1;

        // TODO: A preprocess and create function, right now it will reside on Python

        void Load(const py::bytes& data, const py::array_t<float>& _matrix){
            py::buffer_info info(py::buffer(data).request());
            auto data_ = static_cast<char*>(info.ptr);
            index.Load(data_);

            auto matrix_buf = _matrix.request();
            auto matrix_ptr = static_cast<float*>(matrix_buf.ptr);
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(matrix_ptr, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();

            searcher = std::make_unique<ADSamplingSearcher>(index, 0, epsilon0, matrix, SEQUENTIAL);
        }

        void Restore(const std::string &path, const std::string &matrix_path){
            index.Restore(path);
            float * _matrix = MmapFile32(matrix_path); // TODO: Should be inside same index file?
            Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(_matrix, index.num_dimensions, index.num_dimensions);
            matrix = matrix.inverse();
            searcher = std::make_unique<ADSamplingSearcher>(index, 0, epsilon0, matrix, SEQUENTIAL);
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
    using KNNCandidate = KNNCandidate<PDX::F32>;
    using IndexPDXIVF = IndexPDXIVF<PDX::F32>;
    using PDXBondSearcher = PDXBondSearcher<PDX::F32>;
    public:
        IndexPDXIVF index = IndexPDXIVF();
        std::unique_ptr<PDXBondSearcher> searcher = nullptr;

        // TODO: A preprocess and create function, right now it will reside on Python

        void Load(const py::bytes& data){
            py::buffer_info info(py::buffer(data).request());
            auto data_ = static_cast<char*>(info.ptr);
            index.Load(data_);
            searcher = std::make_unique<PDXBondSearcher>(index, 0, 0, SEQUENTIAL);
        }

        void Restore(const std::string &path){
            index.Restore(path);
            searcher = std::make_unique<PDXBondSearcher>(index, 0, 0, SEQUENTIAL);
        }

        // std::vector<KNNCandidate> Search(float *q, uint32_t k) const {
        //     return searcher->LinearScan(q, k);
        // }

        // Serialize return value
        // std::vector<KNNCandidate> _py_Search(const py::array_t<float>& q, uint32_t k) const {
        //     auto buf = q.request();  // Get buffer info
        //     if (buf.ndim != 1) {
        //         throw std::runtime_error("Input should be a 1-D NumPy array");
        //     }
        //     auto query = static_cast<float*>(buf.ptr);
        //     return searcher->LinearScan(query, k);
        // }
    };

} // namespace PDX

#endif // PDX_LIB
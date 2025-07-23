#include <pybind11/pybind11.h>
#include "lib/lib.hpp"

namespace py = pybind11;

/******************************************************************
 * Wrapper for Python bindings
 * TODO: Implement quantized classes
 ******************************************************************/
PYBIND11_MODULE(compiled, m) {

    m.doc() = "A library to do vertical pruned vector similarity search";

    py::class_<PDX::Vectorgroup<PDX::F32>>(m, "Vectorgroup")
            .def(py::init<>())
            .def_readwrite("num_embeddings", &PDX::Vectorgroup<PDX::F32>::num_embeddings)
            .def_readwrite("indices", &PDX::Vectorgroup<PDX::F32>::indices)
            .def_readwrite("data", &PDX::Vectorgroup<PDX::F32>::data);

    py::class_<PDX::IndexPDXIVF<PDX::F32>>(m, "IndexPDXIVFFlat")
            .def_readwrite("num_dimensions", &PDX::IndexPDXIVF<PDX::F32>::num_dimensions)
            .def_readwrite("num_vectorgroups", &PDX::IndexPDXIVF<PDX::F32>::num_vectorgroups)
            .def_readwrite("vectorgroups", &PDX::IndexPDXIVF<PDX::F32>::vectorgroups)
            .def_readwrite("means", &PDX::IndexPDXIVF<PDX::F32>::means)
            .def_readwrite("is_ivf", &PDX::IndexPDXIVF<PDX::F32>::is_ivf)
            .def_readwrite("centroids", &PDX::IndexPDXIVF<PDX::F32>::centroids)
            .def_readwrite("centroids_pdx", &PDX::IndexPDXIVF<PDX::F32>::centroids_pdx)
            .def("restore", &PDX::IndexPDXIVF<PDX::F32>::Restore);

    py::class_<PDX::KNNCandidate<PDX::F32>>(m, "KNNCandidate")
            .def(py::init<>())
            .def_readwrite("index", &PDX::KNNCandidate<PDX::F32>::index)
            .def_readwrite("distance", &PDX::KNNCandidate<PDX::F32>::distance);

        py::class_<PDX::KNNCandidate<PDX::U8>>(m, "KNNCandidateSQ8")
        .def(py::init<>())
        .def_readwrite("index", &PDX::KNNCandidate<PDX::U8>::index)
        .def_readwrite("distance", &PDX::KNNCandidate<PDX::U8>::distance);

        py::class_<PDX::IndexADSamplingIMISQ8>(m, "IndexADSamplingIMISQ8")
                .def(py::init<>())
                .def("restore", &PDX::IndexADSamplingIMISQ8::Restore, py::arg("path"), py::arg("matrix_path"))
                .def("load", &PDX::IndexADSamplingIMISQ8::Load, py::arg("data"), py::arg("matrix"))
                .def("search", &PDX::IndexADSamplingIMISQ8::_py_Search, py::arg("q"), py::arg("k"), py::arg("n_probe"));

        py::class_<PDX::IndexADSamplingIMIFlat>(m, "IndexADSamplingIMIFlat")
        .def(py::init<>())
        .def("restore", &PDX::IndexADSamplingIMIFlat::Restore, py::arg("path"), py::arg("matrix_path"))
        .def("load", &PDX::IndexADSamplingIMIFlat::Load, py::arg("data"), py::arg("matrix"))
        .def("search", &PDX::IndexADSamplingIMIFlat::_py_Search, py::arg("q"), py::arg("k"), py::arg("n_probe"));

    py::class_<PDX::IndexADSamplingIVFFlat>(m, "IndexADSamplingIVFFlat")
            .def(py::init<>())
            .def("restore", &PDX::IndexADSamplingIVFFlat::Restore, py::arg("path"), py::arg("matrix_path"))
            .def("load", &PDX::IndexADSamplingIVFFlat::Load, py::arg("data"), py::arg("matrix"))
            .def("search", &PDX::IndexADSamplingIVFFlat::_py_Search, py::arg("q"), py::arg("k"), py::arg("n_probe"));

    py::class_<PDX::IndexBONDIVFFlat>(m, "IndexBONDIVFFlat")
            .def(py::init<>())
            .def("restore", &PDX::IndexBONDIVFFlat::Restore, py::arg("path"))
            .def("load", &PDX::IndexBONDIVFFlat::Load, py::arg("data"))
            .def("search", &PDX::IndexBONDIVFFlat::_py_Search, py::arg("q"), py::arg("k"), py::arg("n_probe"));

    py::class_<PDX::IndexBONDFlat>(m, "IndexBONDFlat")
            .def(py::init<>())
            .def("restore", &PDX::IndexBONDFlat::Restore, py::arg("path"))
            .def("load", &PDX::IndexBONDFlat::Load, py::arg("data"))
            .def("search", &PDX::IndexBONDFlat::_py_Search, py::arg("q"), py::arg("k"));

    py::class_<PDX::IndexADSamplingFlat>(m, "IndexADSamplingFlat")
            .def(py::init<>())
            .def("restore", &PDX::IndexADSamplingFlat::Restore, py::arg("path"), py::arg("matrix_path"))
            .def("load", &PDX::IndexADSamplingFlat::Load, py::arg("data"), py::arg("matrix"))
            .def("search", &PDX::IndexADSamplingFlat::_py_Search, py::arg("q"), py::arg("k"));

    py::class_<PDX::IndexPDXFlat>(m, "IndexPDXFlat")
            .def(py::init<>())
            .def("restore", &PDX::IndexPDXFlat::Restore, py::arg("path"))
            .def("load", &PDX::IndexPDXFlat::Load, py::arg("data"))
            .def("search", &PDX::IndexPDXFlat::_py_Search, py::arg("q"), py::arg("k"));



}
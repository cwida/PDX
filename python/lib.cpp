#include <pybind11/pybind11.h>
#include <pdx/lib/lib.hpp>

namespace py = pybind11;

/******************************************************************
 * Wrapper for Python bindings
 ******************************************************************/
PYBIND11_MODULE(compiled, m) {

    m.doc() = "A library to do vertical pruned vector similarity search";

    py::class_<PDX::Vectorgroup>(m, "Vectorgroup")
            .def(py::init<>())
            .def_readwrite("num_embeddings", &PDX::Vectorgroup::num_embeddings)
            .def_readwrite("indices", &PDX::Vectorgroup::indices)
            .def_readwrite("data", &PDX::Vectorgroup::data);

    py::class_<PDX::IndexPDXIVFFlat>(m, "IndexPDXIVFFlat")
            .def_readwrite("num_dimensions", &PDX::IndexPDXIVFFlat::num_dimensions)
            .def_readwrite("num_vectorgroups", &PDX::IndexPDXIVFFlat::num_vectorgroups)
            .def_readwrite("vectorgroups", &PDX::IndexPDXIVFFlat::vectorgroups)
            .def_readwrite("means", &PDX::IndexPDXIVFFlat::means)
            .def_readwrite("is_ivf", &PDX::IndexPDXIVFFlat::is_ivf)
            .def_readwrite("centroids", &PDX::IndexPDXIVFFlat::centroids)
            .def_readwrite("centroids_pdx", &PDX::IndexPDXIVFFlat::centroids_pdx)
            .def("restore", &PDX::IndexPDXIVFFlat::Restore);

    py::class_<KNNCandidate>(m, "KNNCandidate")
            .def(py::init<>())
            .def_readwrite("index", &KNNCandidate::index)
            .def_readwrite("distance", &KNNCandidate::distance);

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
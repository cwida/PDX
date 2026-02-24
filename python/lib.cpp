#include "pdx/lib/lib.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(compiled, m) {
    m.doc() = "PDXearch: Faster similarity search with a transposed data layout for vectors";

    py::class_<PDX::PyPDXIndex>(m, "PDXIndex")
        .def(
            py::init<
                const std::string&,
                uint32_t,
                uint8_t,
                uint32_t,
                uint32_t,
                uint32_t,
                bool,
                float,
                uint32_t>(),
            py::arg("index_type"),
            py::arg("num_dimensions"),
            py::arg("distance_metric") = 0,
            py::arg("seed") = 42,
            py::arg("num_clusters") = 0,
            py::arg("num_meso_clusters") = 0,
            py::arg("normalize") = false,
            py::arg("sampling_fraction") = 0.0f,
            py::arg("kmeans_iters") = 10
        )
        .def("build_index", &PDX::PyPDXIndex::BuildIndex, py::arg("data"))
        .def("search", &PDX::PyPDXIndex::Search, py::arg("query"), py::arg("knn"))
        .def(
            "filtered_search",
            &PDX::PyPDXIndex::FilteredSearch,
            py::arg("query"),
            py::arg("knn"),
            py::arg("row_ids")
        )
        .def("set_nprobe", &PDX::PyPDXIndex::SetNProbe, py::arg("nprobe"))
        .def("save", &PDX::PyPDXIndex::Save, py::arg("path"))
        .def("get_num_dimensions", &PDX::PyPDXIndex::GetNumDimensions)
        .def("get_num_clusters", &PDX::PyPDXIndex::GetNumClusters)
        .def("get_cluster_size", &PDX::PyPDXIndex::GetClusterSize, py::arg("cluster_id"))
        .def("get_cluster_row_ids", &PDX::PyPDXIndex::GetClusterRowIds, py::arg("cluster_id"))
        .def("get_in_memory_size_in_bytes", &PDX::PyPDXIndex::GetInMemorySizeInBytes);

    m.def(
        "load_index",
        &PDX::PyPDXIndex::LoadFromFile,
        py::arg("path"),
        "Load a PDX index from a single file (auto-detects type)."
    );
}

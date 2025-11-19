/**
 * @file python_bindings.cpp
 * @brief Python bindings for parallel IVF using pybind11
 * @date 11-18-2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "IVFBase.h"

namespace py = pybind11;

PYBIND11_MODULE(parallel_ivf, m) {
    m.doc() = "Python bindings for parallel IVF (Inverted File Index)";

    // Expose IVF base class
    py::class_<IVF>(m, "IVF", "Base class for Inverted File Index")
        .def_readwrite("d", &IVF::d, "Dimensionality of vectors")
        .def_readwrite("nlist", &IVF::nlist, "Number of inverted lists")
        .def_readwrite("nprobe", &IVF::nprobe, "Number of lists to probe during search")
        .def_readwrite("inv_lists", &IVF::inv_lists, "Inverted lists storing vectors")
        .def_readwrite("labels", &IVF::labels, "Labels for vectors in inverted lists")
        .def_readwrite("centroids", &IVF::centroids, "Cluster centroids");

    // Expose IVFBase implementation
    py::class_<IVFBase, IVF>(m, "IVFBase", "Base implementation of IVF")
        .def(py::init<size_t, size_t>(),
             py::arg("d"), 
             py::arg("nlist"),
             "Construct an IVFBase index\n\n"
             "Parameters:\n"
             "  d: dimensionality of vectors\n"
             "  nlist: number of inverted lists (clusters)")
        
        .def("train", 
             [](IVFBase &self, py::array_t<float> train_data) {
                 py::buffer_info buf = train_data.request();
                 
                 // Validate input
                 if (buf.ndim != 2) {
                     throw std::runtime_error("train_data must be a 2D array");
                 }
                 
                 size_t n_train = buf.shape[0];
                 size_t d = buf.shape[1];
                 
                 if (d != self.d) {
                     throw std::runtime_error(
                         "Dimension mismatch: expected " + std::to_string(self.d) + 
                         " but got " + std::to_string(d));
                 }
                 
                 if (n_train < self.nlist) {
                     throw std::runtime_error(
                         "Number of training vectors (" + std::to_string(n_train) + 
                         ") must be >= nlist (" + std::to_string(self.nlist) + ")");
                 }
                 
                 self.train(n_train, static_cast<float*>(buf.ptr));
             },
             py::arg("train_data"),
             "Train the index with training vectors\n\n"
             "Parameters:\n"
             "  train_data: numpy array of shape (n_train, d)")
        
        .def("add", 
             [](IVFBase &self, py::array_t<float> add_data) {
                 py::buffer_info buf = add_data.request();
                 
                 if (buf.ndim != 2) {
                     throw std::runtime_error("add_data must be a 2D array");
                 }
                 
                 size_t n_add = buf.shape[0];
                 size_t d = buf.shape[1];
                 
                 if (d != self.d) {
                     throw std::runtime_error(
                         "Dimension mismatch: expected " + std::to_string(self.d) + 
                         " but got " + std::to_string(d));
                 }
                 
                 self.add(n_add, static_cast<float*>(buf.ptr));
             },
             py::arg("add_data"),
             "Add vectors to the index\n\n"
             "Parameters:\n"
             "  add_data: numpy array of shape (n_add, d)")
        
        .def("search", 
             [](const IVFBase &self, py::array_t<float> queries, 
                size_t k, size_t nprobe) {
                 py::buffer_info buf = queries.request();
                 
                 if (buf.ndim != 2) {
                     throw std::runtime_error("queries must be a 2D array");
                 }
                 
                 size_t n_queries = buf.shape[0];
                 size_t d = buf.shape[1];
                 
                 if (d != self.d) {
                     throw std::runtime_error(
                         "Dimension mismatch: expected " + std::to_string(self.d) + 
                         " but got " + std::to_string(d));
                 }
                 
                 if (nprobe > self.nlist) {
                     throw std::runtime_error(
                         "nprobe (" + std::to_string(nprobe) + 
                         ") cannot be greater than nlist (" + std::to_string(self.nlist) + ")");
                 }
                 
                 self.search(n_queries, static_cast<float*>(buf.ptr), k, nprobe);
             },
             py::arg("queries"),
             py::arg("k"),
             py::arg("nprobe"),
             "Search for nearest neighbors\n\n"
             "Parameters:\n"
             "  queries: numpy array of shape (n_queries, d)\n"
             "  k: number of nearest neighbors to return\n"
             "  nprobe: number of lists to probe")
        
        .def("__repr__",
             [](const IVFBase &self) {
                 return "<IVFBase d=" + std::to_string(self.d) + 
                        ", nlist=" + std::to_string(self.nlist) + 
                        ", nprobe=" + std::to_string(self.nprobe) + ">";
             });

    // Version info
    m.attr("__version__") = "0.1.0";
}


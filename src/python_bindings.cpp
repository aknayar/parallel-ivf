/**
 * @file python_bindings.cpp
 * @brief Python bindings for parallel IVF using pybind11
 * @date 11-18-2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "IVF.h"

namespace py = pybind11;

// Helper template to bind any IVF instantiation with minimal code
template <DistanceKernel Kernel>
void bind_ivf(py::module &m, const char *class_name, const char *doc) {
    using IVFType = IVF<Kernel>;
    
    py::class_<IVFType>(m, class_name, doc)
        .def(py::init<size_t, size_t>(),
             py::arg("d"), 
             py::arg("nlist"),
             "Construct an IVF index\n\n"
             "Parameters:\n"
             "  d: dimensionality of vectors\n"
             "  nlist: number of inverted lists (clusters)")
        
        // Attributes
        .def_readwrite("d", &IVFType::d, "Dimensionality of vectors")
        .def_readwrite("nlist", &IVFType::nlist, "Number of inverted lists")
        .def_readwrite("nprobe", &IVFType::nprobe, "Number of lists to probe during search")
        .def_readwrite("maxlabel", &IVFType::maxlabel, "Maximum label assigned")
        .def_readwrite("inv_lists", &IVFType::inv_lists, "Inverted lists storing vectors")
        .def_readwrite("labels", &IVFType::labels, "Labels for vectors in inverted lists")
        .def_readwrite("centroids", &IVFType::centroids, "Cluster centroids")
        
        // train()
        .def("train", 
             [](IVFType &self, py::array_t<float> train_data) {
                 py::buffer_info buf = train_data.request();
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
             "Train the index with training vectors")
        
        // build()
        .def("build", 
             [](IVFType &self, py::array_t<float> train_data) {
                 py::buffer_info buf = train_data.request();
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
                 if (self.centroids.empty()) {
                     throw std::runtime_error(
                         "Cannot build IVF: centroids are empty. Call train() first.");
                 }
                 self.build(n_train, static_cast<float*>(buf.ptr));
             },
             py::arg("train_data"),
             "Build inverted lists using the training data")
        
        // add()
        .def("add", 
             [](IVFType &self, py::array_t<float> add_data) {
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
             "Add vectors to the index")
        
        // search()
        .def("search", 
             [](const IVFType &self, py::array_t<float> queries, 
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
                 return self.search(n_queries, static_cast<float*>(buf.ptr), k, nprobe);
             },
             py::arg("queries"),
             py::arg("k"),
             py::arg("nprobe"),
             "Search for nearest neighbors")
        
        // __repr__()
        .def("__repr__",
             [class_name](const IVFType &self) {
                 return "<" + std::string(class_name) + " d=" + std::to_string(self.d) + 
                        ", nlist=" + std::to_string(self.nlist) + 
                        ", nprobe=" + std::to_string(self.nprobe) + ">";
             });
}

PYBIND11_MODULE(parallel_ivf, m) {
    m.doc() = "Python bindings for parallel IVF (Inverted File Index)";

    // Expose DistanceKernel enum
    py::enum_<DistanceKernel>(m, "DistanceKernel")
        .value("SCALAR", DistanceKernel::SCALAR, "Scalar distance computation")
        .value("SIMD", DistanceKernel::SIMD, "SIMD distance computation")
        .export_values();

    // Bind all IVF instantiations - just one line each!
    bind_ivf<DistanceKernel::SCALAR>(m, "IVFBase", "IVF with scalar distance kernel");
    bind_ivf<DistanceKernel::SIMD>(m, "IVFSIMD", "IVF with SIMD distance kernel");
    
    // To add more in the future, just add another line:
    // bind_ivf<DistanceKernel::AVX512>(m, "IVFAVX512", "IVF with AVX512 distance kernel");

    // Version info
    m.attr("__version__") = "0.1.0";
}

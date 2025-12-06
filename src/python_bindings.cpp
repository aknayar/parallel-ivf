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
template <DistanceKernel DKernel, ParallelType PType>
void bind_ivf(py::module &m, const char *class_name, const char *doc) {
    using IVFType = IVF<DKernel, PType>;
    
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
        .value("CACHE", DistanceKernel::CACHE, "Cache friendly distance computation")
        .value("CACHESIMD", DistanceKernel::CACHESIMD, "Cache friendly distance computation with SIMD")
        .export_values();

    // Expose ParallelType enum
    py::enum_<ParallelType>(m, "ParallelType")
        .value("SERIAL", ParallelType::SERIAL, "Serial implementation")
        .value("QUERY_PARALLEL", ParallelType::QUERY_PARALLEL, "Query-level parallelism")
        .value("CANDIDATE_PARALLEL", ParallelType::CANDIDATE_PARALLEL, "Candidate-level parallelism")
        .value("QUERYCANDIDATE_PARALLEL", ParallelType::QUERYCANDIDATE_PARALLEL, "Both")
        .export_values();

    // Bind all IVF instantiations - one line per combination!
    // SCALAR variants
    bind_ivf<DistanceKernel::SCALAR, ParallelType::SERIAL>(
        m, "IVFScalarSerial", "IVF with scalar distance, serial");
    bind_ivf<DistanceKernel::SCALAR, ParallelType::QUERY_PARALLEL>(
        m, "IVFScalarQueryParallel", "IVF with scalar distance, query-parallel");
    bind_ivf<DistanceKernel::SCALAR, ParallelType::CANDIDATE_PARALLEL>(
        m, "IVFScalarCandidateParallel", "IVF with scalar distance, candidate-parallel");
    bind_ivf<DistanceKernel::SCALAR, ParallelType::QUERYCANDIDATE_PARALLEL>(
        m, "IVFScalarQueryCandidateParallel", "IVF with scalar distance, candidate-parallel");
    
    // SIMD variants
    bind_ivf<DistanceKernel::SIMD, ParallelType::SERIAL>(
        m, "IVFSIMDSerial", "IVF with SIMD distance, serial");
    bind_ivf<DistanceKernel::SIMD, ParallelType::QUERY_PARALLEL>(
        m, "IVFSIMDQueryParallel", "IVF with SIMD distance, query-parallel");
    bind_ivf<DistanceKernel::SIMD, ParallelType::CANDIDATE_PARALLEL>(
        m, "IVFSIMDCandidateParallel", "IVF with SIMD distance, candidate-parallel");
    bind_ivf<DistanceKernel::SIMD, ParallelType::QUERYCANDIDATE_PARALLEL>(
        m, "IVFSIMDQueryCandidateParallel", "IVF with SIMD distance, candidate-parallel");

    bind_ivf<DistanceKernel::CACHE, ParallelType::SERIAL>(
        m, "IVFCacheSerial", "IVF with cache friendly distance, serial");
    bind_ivf<DistanceKernel::CACHE, ParallelType::QUERY_PARALLEL>(
        m, "IVFCacheQueryParallel", "IVF with cache friendly distance, query-parallel");
    bind_ivf<DistanceKernel::CACHE, ParallelType::CANDIDATE_PARALLEL>(
        m, "IVFCacheCandidateParallel", "IVF with cache friendly distance, candidate-parallel");
    bind_ivf<DistanceKernel::CACHE, ParallelType::QUERYCANDIDATE_PARALLEL>(
        m, "IVFCacheQueryCandidateParallel", "IVF with cache friendly distance, candidate-parallel");

    bind_ivf<DistanceKernel::CACHESIMD, ParallelType::SERIAL>(
        m, "IVFCacheSIMDSerial", "IVF with cache friendly distance, serial");
    bind_ivf<DistanceKernel::CACHESIMD, ParallelType::QUERY_PARALLEL>(
        m, "IVFCacheSIMDQueryParallel", "IVF with cache friendly distance, query-parallel");
    bind_ivf<DistanceKernel::CACHESIMD, ParallelType::CANDIDATE_PARALLEL>(
        m, "IVFCacheSIMDCandidateParallel", "IVF with cache friendly distance, candidate-parallel");
    bind_ivf<DistanceKernel::CACHESIMD, ParallelType::QUERYCANDIDATE_PARALLEL>(
        m, "IVFCacheSIMDQueryCandidateParallel", "IVF with cache friendly distance, candidate-parallel");
    
    // Aliases for convenience (default = SCALAR + SERIAL)
    m.attr("IVFBase") = m.attr("IVFScalarSerial");
    m.attr("IVFSIMD") = m.attr("IVFSIMDSerial");
    m.attr("IVFCache") = m.attr("IVFCacheSerial");
    m.attr("IVFCacheSIMD") = m.attr("IVFCacheSIMDSerial");
    m.attr("IVFSIMDQueryParallel") = m.attr("IVFSIMDQueryParallel");
    m.attr("IVFSIMDCandidateParallel") = m.attr("IVFSIMDCandidateParallel");
    m.attr("IVFScalarQueryParallel") = m.attr("IVFScalarQueryParallel");
    m.attr("IVFScalarCandidateParallel") = m.attr("IVFScalarCandidateParallel");
    m.attr("IVFCacheQueryParallel") = m.attr("IVFCacheQueryParallel");
    m.attr("IVFCacheCandidateParallel") = m.attr("IVFCacheCandidateParallel");
    m.attr("IVFCacheSIMDQueryParallel") = m.attr("IVFCacheQueryParallel");
    m.attr("IVFCacheSIMDCandidateParallel") = m.attr("IVFCacheCandidateParallel");

    // Version info
    m.attr("__version__") = "0.1.0";
}

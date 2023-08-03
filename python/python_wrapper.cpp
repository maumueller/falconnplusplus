#include <NNDataStructure.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace python {

namespace py = pybind11;

PYBIND11_MODULE(falconnpp, m) {
    py::class_<FalconnPP>(m, "FalconnPP")
        .def(py::init<const int&, const int&, const int&>())
        .def("build", &FalconnPP::buildIndex)
        .def("search", &FalconnPP::query);
}
} // namespace python

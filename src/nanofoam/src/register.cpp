#include "functions.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(MODULE_NAME, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", &add, "a"_a, "b"_a);
    m.def("subtract", &subtract, "a"_a, "b"_a);
    m.def("multiply", &multiply, "a"_a, "b"_a);
    m.def("divide", &divide, "a"_a, "b"_a);
}

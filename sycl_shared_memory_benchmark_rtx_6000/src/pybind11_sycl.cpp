#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>
#include "sycl_shared_core.h"

namespace py = pybind11;

static long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double ns_ms(long long ns) {
    return static_cast<double>(ns) / 1.0e6;
}

static py::tuple init(size_t n) {
    int rc = sycl_shared_init(n);
    if (rc != 0) throw std::runtime_error("sycl_shared_init failed");
    py::array_t<float> a({n, n}, {sizeof(float) * n, sizeof(float)}, sycl_shared_a(), py::none());
    py::array_t<float> b({n, n}, {sizeof(float) * n, sizeof(float)}, sycl_shared_b(), py::none());
    py::array_t<float> c({n, n}, {sizeof(float) * n, sizeof(float)}, sycl_shared_c(), py::none());
    return py::make_tuple(a, b, c);
}

static py::dict matmul(py::array_t<float, py::array::c_style | py::array::forcecast> a,
                       py::array_t<float, py::array::c_style | py::array::forcecast> b,
                       py::array_t<float, py::array::c_style | py::array::forcecast> c,
                       long long py_call_ns) {
    long long entry_ns = now_ns();
    long long validate0 = now_ns();
    auto abuf = a.request();
    auto bbuf = b.request();
    auto cbuf = c.request();
    if (abuf.ndim != 2 || abuf.shape[0] != abuf.shape[1]) throw std::runtime_error("expected square arrays");
    size_t n = static_cast<size_t>(abuf.shape[0]);
    long long validate1 = now_ns();
    SyclSharedTimings t{};
    int rc = sycl_shared_matmul(static_cast<float*>(abuf.ptr),
                                static_cast<float*>(bbuf.ptr),
                                static_cast<float*>(cbuf.ptr),
                                n, &t);
    if (rc != 0) throw std::runtime_error("sycl_shared_matmul failed");
    py::dict out;
    out["marshal_in_ms"] = ns_ms(entry_ns - py_call_ns);
    out["pybind_buffer_ms"] = ns_ms(validate1 - validate0);
    out["kernel_ms"] = t.kernel_ms;
    out["sycl_total_ms"] = t.total_ms;
    return out;
}

PYBIND11_MODULE(pybind11_sycl, m) {
    m.def("has_gpu", []() { return sycl_shared_has_gpu() != 0; });
    m.def("init", &init);
    m.def("finalize", []() { sycl_shared_finalize(); });
    m.def("matmul", &matmul);
}

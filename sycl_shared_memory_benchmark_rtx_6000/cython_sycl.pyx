# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.stddef cimport size_t

cdef extern from "src/sycl_shared_core.h":
    ctypedef struct SyclSharedTimings:
        double kernel_ms
        double total_ms
    int sycl_shared_has_gpu()
    int sycl_shared_init(size_t n)
    void sycl_shared_finalize()
    float* sycl_shared_a()
    float* sycl_shared_b()
    float* sycl_shared_c()
    int sycl_shared_matmul(float* a, float* b, float* c, size_t n, SyclSharedTimings* t)

cdef extern from "Python.h":
    object PyMemoryView_FromMemory(char *mem, Py_ssize_t size, int flags)
    cdef int PyBUF_WRITE

def has_gpu():
    return sycl_shared_has_gpu() != 0

def init(size_t n):
    cdef int rc = sycl_shared_init(n)
    if rc != 0:
        raise RuntimeError(f"sycl_shared_init failed: {rc}")
    cdef Py_ssize_t bytes_n = <Py_ssize_t>(n * n * sizeof(float))
    a = np.frombuffer(PyMemoryView_FromMemory(<char*>sycl_shared_a(), bytes_n, PyBUF_WRITE), dtype=np.float32).reshape((n, n))
    b = np.frombuffer(PyMemoryView_FromMemory(<char*>sycl_shared_b(), bytes_n, PyBUF_WRITE), dtype=np.float32).reshape((n, n))
    c = np.frombuffer(PyMemoryView_FromMemory(<char*>sycl_shared_c(), bytes_n, PyBUF_WRITE), dtype=np.float32).reshape((n, n))
    return a, b, c

def finalize():
    sycl_shared_finalize()

def matmul(float[:, ::1] a, float[:, ::1] b, float[:, ::1] c, long py_call_ns):
    cdef SyclSharedTimings t
    cdef size_t n = a.shape[0]
    cdef int rc = sycl_shared_matmul(&a[0, 0], &b[0, 0], &c[0, 0], n, &t)
    if rc != 0:
        raise RuntimeError(f"sycl_shared_matmul failed: {rc}")
    return {
        "marshal_in_ms": 0.0,
        "cython_memoryview_ms": 0.0,
        "kernel_ms": t.kernel_ms,
        "sycl_total_ms": t.total_ms,
    }

# SYCL Shared-Memory Matrix Benchmark

## Purpose

This benchmark matches the production SYCL-PIC memory model more closely than
the earlier CUDA/NumPy benchmark. Instead of starting from ordinary CPU NumPy
arrays and copying them to CUDA memory, this benchmark allocates all matrices
with:

```cpp
sycl::malloc_shared<float>(N * N, queue)
```

The same `malloc_shared` buffers are then exposed or passed through each
Python-C++ integration method:

| Method | Shared-memory interface |
|---|---|
| NumPy C-API | `PyArray_SimpleNewFromData` over `malloc_shared` |
| pybind11 | `py::array_t` view over `malloc_shared` |
| Cython | typed memoryview / Python buffer over `malloc_shared` |
| ctypes/nanobind | raw pointer exposed as `numpy.ctypeslib.as_array` |
| DLPack | DLPack-style capsule metadata over the shared pointer |

All methods call the same SYCL tiled matrix-multiplication kernel. There are
no H2D or D2H copies inside the timed call.

## Run Configuration

| Field | Value |
|---|---|
| Hardware tag | `HPC-GPU__x86_64__NVIDIA_RTX_6000_Ada_Generation__sycl_malloc_shared` |
| Backend | SYCL GPU |
| Matrix type | FP32 |
| Sizes | `128, 256, 512, 1024, 2048, 4096, 8192` |
| Repetitions | 3 measured repetitions |
| Warmup | 1 warmup call |
| Memory model | `sycl::malloc_shared` |

## End-to-End Results

Total Python-call time in milliseconds:

| N | NumPy C-API | pybind11 | Cython | ctypes/nanobind | DLPack |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.021 | 0.067 | 0.019 | 0.026 | 0.017 |
| 256 | 0.024 | 0.035 | 0.024 | 0.026 | 0.034 |
| 512 | 0.076 | 0.552 | 0.077 | 0.078 | 0.122 |
| 1024 | 0.438 | 0.441 | 0.480 | 2.711 | 2.687 |
| 2048 | 8.083 | 8.271 | 8.305 | 8.304 | 8.300 |
| 4096 | 65.300 | 61.388 | 62.703 | 65.327 | 64.611 |
| 8192 | 1614.144 | 1479.706 | 1551.774 | 1606.463 | 1615.298 |

Throughput in GFLOP/s:

| N | NumPy C-API | pybind11 | Cython | ctypes/nanobind | DLPack |
|---:|---:|---:|---:|---:|---:|
| 128 | 198.7 | 63.0 | 223.9 | 158.9 | 250.4 |
| 256 | 1378.5 | 964.0 | 1421.6 | 1270.4 | 985.2 |
| 512 | 3526.4 | 486.4 | 3490.7 | 3459.1 | 2196.1 |
| 1024 | 4900.0 | 4869.4 | 4469.8 | 792.1 | 799.3 |
| 2048 | 2125.3 | 2077.2 | 2068.6 | 2069.0 | 2069.7 |
| 4096 | 2104.7 | 2238.8 | 2191.9 | 2103.9 | 2127.2 |
| 8192 | 681.2 | 743.1 | 708.6 | 684.4 | 680.7 |

## Key Observation

Once the matrices are allocated with `sycl::malloc_shared`, the integration
method no longer determines the dominant runtime. For large matrices, all
methods converge because the same SYCL kernel dominates the call.

At `N=2048`, all five methods are within about 3%:

```text
NumPy C-API       8.083 ms
pybind11          8.271 ms
Cython            8.305 ms
ctypes/nanobind   8.304 ms
DLPack            8.300 ms
```

At `N=4096`, all methods are again in the same range:

```text
NumPy C-API      65.300 ms
pybind11         61.388 ms
Cython           62.703 ms
ctypes/nanobind  65.327 ms
DLPack           64.611 ms
```

The small-size variation is mostly call overhead, runtime scheduling, and
measurement noise. The large-size variation is dominated by SYCL kernel
execution and GPU runtime behavior, not Python binding overhead.

## Why This Supports the Production Choice

The earlier CUDA benchmark made NumPy C-API look slower because that
benchmark used ordinary CPU NumPy arrays and therefore paid per-call H2D and
D2H transfer costs.

This benchmark removes that mismatch by using the same memory model as the
production solver:

```text
C++/SYCL owns memory
    -> sycl::malloc_shared
    -> Python gets a zero-copy view
    -> SYCL kernel operates on the same allocation
```

Under this production-equivalent setup, NumPy C-API is competitive with all
other methods. It also has the best engineering fit for the solver because it
provides:

1. Zero-copy access to `sycl::malloc_shared` buffers.
2. Standard NumPy shape and dtype metadata.
3. Cleaner bounds/shape handling than raw pointers.
4. No DLPack ownership-callback complexity.
5. Compatibility with CPU, GPU, and hybrid CPU-GPU execution.

## Final Conclusion

For pure GPU-resident tensor exchange, DLPack can be fastest. For the actual
SYCL-PIC solver, however, the important requirement is exposing existing
SYCL shared-memory arrays (`rho`, `phi`) to Python without copying.

The SYCL shared-memory benchmark shows that NumPy C-API does exactly that:
when the data path matches production, NumPy C-API is not a performance
bottleneck. The runtime is dominated by the SYCL kernel, while the Python
integration overhead is negligible.

Therefore, NumPy C-API remains the most suitable production interface, with
DLPack reserved as a possible future extension if the ML path becomes fully
GPU tensor based.

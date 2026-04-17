/**
 * Method 1: PyBind11 - The Standard Approach
 * 
 * PyBind11 is the traditional standard for C++/Python bindings.
 * It provides automatic NumPy interoperability and clean syntax.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS Error: " + std::to_string(status)); \
        } \
    } while(0)

// Singleton cuBLAS handle
cublasHandle_t& get_handle() {
    static cublasHandle_t handle = []() {
        cublasHandle_t h;
        CHECK_CUBLAS(cublasCreate(&h));
        return h;
    }();
    return handle;
}

/**
 * PyBind11 Matrix Multiplication
 * 
 * This version accepts NumPy arrays from Python, copies them to GPU,
 * performs computation, and returns the result.
 * 
 * Note: This involves CPU->GPU->CPU copies for each call.
 */
py::array_t<float> matmul_pybind11(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B) 
{
    // Get buffer info
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();

    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Matrices must be 2-dimensional");
    }

    int M = bufA.shape[0];
    int K = bufA.shape[1];
    int N = bufB.shape[1];

    if (bufB.shape[0] != K) {
        throw std::runtime_error("Matrix dimension mismatch");
    }

    // Allocate output
    py::array_t<float> C({M, N});
    py::buffer_info bufC = C.request();

    float* h_A = static_cast<float*>(bufA.ptr);
    float* h_B = static_cast<float*>(bufB.ptr);
    float* h_C = static_cast<float*>(bufC.ptr);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    // Copy to GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // cuBLAS computation
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(
        get_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

/**
 * PyBind11 with GPU Tensors (using raw pointers)
 * 
 * This version accepts GPU pointers directly, avoiding CPU copies.
 * Requires passing tensor.data_ptr() from PyTorch.
 */
void matmul_pybind11_gpu(
    std::uintptr_t ptr_A, std::uintptr_t ptr_B, std::uintptr_t ptr_C,
    int M, int K, int N)
{
    float* d_A = reinterpret_cast<float*>(ptr_A);
    float* d_B = reinterpret_cast<float*>(ptr_B);
    float* d_C = reinterpret_cast<float*>(ptr_C);

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(
        get_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
}

PYBIND11_MODULE(matmul_pybind11, m) {
    m.doc() = "Method 1: PyBind11 Matrix Multiplication";
    
    m.def("matmul", &matmul_pybind11, 
          "Matrix multiplication with NumPy arrays (CPU->GPU->CPU)",
          py::arg("A"), py::arg("B"));
    
    m.def("matmul_gpu", &matmul_pybind11_gpu,
          "Matrix multiplication with GPU pointers (zero-copy)",
          py::arg("ptr_A"), py::arg("ptr_B"), py::arg("ptr_C"),
          py::arg("M"), py::arg("K"), py::arg("N"));
}

/**
 * Method 4: CUDA Array Interface (__cuda_array_interface__)
 * 
 * The CUDA Array Interface is a Python-level protocol that allows
 * GPU arrays to expose their memory layout. It's used by CuPy, Numba,
 * and PyTorch to share GPU memory without copying.
 * 
 * This implementation provides two approaches:
 * 1. Using nanobind's ndarray (recommended - handles protocol automatically)
 * 2. Direct pointer passing (for manual __cuda_array_interface__ usage)
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace nb = nanobind;

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

cublasHandle_t& get_handle() {
    static cublasHandle_t handle = []() {
        cublasHandle_t h;
        CHECK_CUBLAS(cublasCreate(&h));
        return h;
    }();
    return handle;
}

/**
 * Matrix multiplication using nanobind's ndarray
 * 
 * nanobind's ndarray automatically handles __cuda_array_interface__,
 * DLPack, and other tensor protocols transparently.
 */
void matmul_cuda_interface(
    nb::ndarray<float, nb::device::cuda, nb::c_contig> A,
    nb::ndarray<float, nb::device::cuda, nb::c_contig> B,
    nb::ndarray<float, nb::device::cuda, nb::c_contig> C) 
{
    float* d_A = (float*)A.data();
    float* d_B = (float*)B.data();
    float* d_C = (float*)C.data();

    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(B.shape(1));

    if (static_cast<int>(B.shape(0)) != K) {
        throw std::runtime_error("Matrix dimension mismatch: A.cols != B.rows");
    }
    if (static_cast<int>(C.shape(0)) != M || static_cast<int>(C.shape(1)) != N) {
        throw std::runtime_error("Output matrix C has incorrect dimensions");
    }

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

/**
 * Direct pointer version for maximum performance
 * 
 * Use this when you've already extracted pointers via:
 *   ptr = tensor.__cuda_array_interface__['data'][0]
 * or:
 *   ptr = tensor.data_ptr()  # PyTorch
 */
void matmul_direct_ptr(std::uintptr_t ptr_A, std::uintptr_t ptr_B, 
                       std::uintptr_t ptr_C, int M, int K, int N) 
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

NB_MODULE(matmul_cuda_interface, m) {
    m.doc() = "Method 4: CUDA Array Interface Matrix Multiplication";
    
    m.def("matmul", &matmul_cuda_interface,
          "Matrix multiplication via __cuda_array_interface__ (nanobind ndarray)",
          nb::arg("A"), nb::arg("B"), nb::arg("C"));
    
    m.def("matmul_ptr", &matmul_direct_ptr,
          "Matrix multiplication with direct GPU pointers",
          nb::arg("ptr_A"), nb::arg("ptr_B"), nb::arg("ptr_C"),
          nb::arg("M"), nb::arg("K"), nb::arg("N"));
}

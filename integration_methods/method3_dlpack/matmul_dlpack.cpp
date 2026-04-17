/**
 * Method 3: DLPack - Zero-Copy Tensor Exchange Standard
 * 
 * DLPack is a cross-framework tensor memory sharing protocol.
 * It allows PyTorch, TensorFlow, JAX, and custom C++ code to share
 * GPU tensors without copying data.
 * 
 * This is the recommended approach for high-performance ML pipelines.
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
 * DLPack Managed Tensor Wrapper
 * 
 * This function receives DLPack tensors from Python frameworks
 * and performs zero-copy matrix multiplication.
 */
void matmul_dlpack(nb::ndarray<float, nb::device::cuda, nb::c_contig> A,
                   nb::ndarray<float, nb::device::cuda, nb::c_contig> B,
                   nb::ndarray<float, nb::device::cuda, nb::c_contig> C) 
{
    // Extract GPU pointers directly - ZERO COPY!
    float* d_A = (float*)A.data();
    float* d_B = (float*)B.data();
    float* d_C = (float*)C.data();

    // Get dimensions
    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(B.shape(1));

    // Validate dimensions
    if (static_cast<int>(B.shape(0)) != K) {
        throw std::runtime_error("Matrix dimension mismatch: A.cols != B.rows");
    }
    if (static_cast<int>(C.shape(0)) != M || static_cast<int>(C.shape(1)) != N) {
        throw std::runtime_error("Output matrix C has incorrect dimensions");
    }

    float alpha = 1.0f, beta = 0.0f;

    // Row-major to column-major trick
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
 * DLPack with explicit capsule handling
 * 
 * This version shows how to manually handle DLPack capsules
 * for frameworks that don't support nanobind's ndarray directly.
 */
void matmul_dlpack_capsule(nb::object A_capsule, nb::object B_capsule, 
                           nb::object C_capsule, int M, int K, int N)
{
    // Convert from __dlpack__() protocol
    // PyTorch: tensor.__dlpack__() returns a PyCapsule containing DLManagedTensor
    
    // For simplicity, we extract data pointers via Python's data_ptr()
    // In production, you'd parse the DLManagedTensor structure
    
    auto A_ptr = nb::cast<std::uintptr_t>(A_capsule);
    auto B_ptr = nb::cast<std::uintptr_t>(B_capsule);
    auto C_ptr = nb::cast<std::uintptr_t>(C_capsule);
    
    float* d_A = reinterpret_cast<float*>(A_ptr);
    float* d_B = reinterpret_cast<float*>(B_ptr);
    float* d_C = reinterpret_cast<float*>(C_ptr);
    
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

NB_MODULE(matmul_dlpack, m) {
    m.doc() = "Method 3: DLPack Zero-Copy Matrix Multiplication";
    
    m.def("matmul", &matmul_dlpack,
          "Zero-copy matrix multiplication using DLPack/nanobind ndarray",
          nb::arg("A"), nb::arg("B"), nb::arg("C"));
    
    m.def("matmul_ptr", &matmul_dlpack_capsule,
          "Matrix multiplication with raw pointers (for DLPack capsules)",
          nb::arg("A_ptr"), nb::arg("B_ptr"), nb::arg("C_ptr"),
          nb::arg("M"), nb::arg("K"), nb::arg("N"));
}

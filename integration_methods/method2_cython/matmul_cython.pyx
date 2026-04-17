# Method 2: Cython Implementation
# 
# Cython allows writing C extensions for Python using a Python-like syntax.
# This file contains the Cython wrapper for cuBLAS matrix multiplication.
# 
# Cython requires a .pyx file (this file) and a setup.py to build.

# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as np
from libc.stdint cimport uintptr_t
from libcpp cimport bool

# Declare external CUDA/cuBLAS functions
cdef extern from "cuda_runtime.h":
    ctypedef int cudaError_t
    ctypedef void* cudaStream_t
    
    cudaError_t cudaMalloc(void** devPtr, size_t size)
    cudaError_t cudaFree(void* devPtr)
    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind)
    cudaError_t cudaDeviceSynchronize()
    
    int cudaMemcpyHostToDevice
    int cudaMemcpyDeviceToHost

cdef extern from "cublas_v2.h":
    ctypedef void* cublasHandle_t
    ctypedef int cublasStatus_t
    ctypedef int cublasOperation_t
    
    int CUBLAS_OP_N
    int CUBLAS_STATUS_SUCCESS
    
    cublasStatus_t cublasCreate(cublasHandle_t* handle)
    cublasStatus_t cublasDestroy(cublasHandle_t handle)
    cublasStatus_t cublasSgemm(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float* alpha,
        const float* A, int lda,
        const float* B, int ldb,
        const float* beta,
        float* C, int ldc
    )

# Global cuBLAS handle
cdef cublasHandle_t _handle = NULL

cdef cublasHandle_t get_handle():
    global _handle
    if _handle == NULL:
        cublasCreate(&_handle)
    return _handle

def matmul_cython(np.ndarray[np.float32_t, ndim=2, mode='c'] A,
                  np.ndarray[np.float32_t, ndim=2, mode='c'] B):
    """
    Matrix multiplication using Cython + cuBLAS.
    
    This version copies data from CPU to GPU and back.
    
    Args:
        A: numpy array of shape (M, K), float32, C-contiguous
        B: numpy array of shape (K, N), float32, C-contiguous
    
    Returns:
        C: numpy array of shape (M, N), float32
    """
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]
    
    if B.shape[0] != K:
        raise ValueError("Matrix dimension mismatch")
    
    # Allocate output
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] C = np.zeros((M, N), dtype=np.float32)
    
    # Get pointers to numpy data
    cdef float* h_A = <float*>A.data
    cdef float* h_B = <float*>B.data
    cdef float* h_C = <float*>C.data
    
    # Allocate GPU memory
    cdef float* d_A
    cdef float* d_B
    cdef float* d_C
    cdef size_t sizeA = M * K * sizeof(float)
    cdef size_t sizeB = K * N * sizeof(float)
    cdef size_t sizeC = M * N * sizeof(float)
    
    cudaMalloc(<void**>&d_A, sizeA)
    cudaMalloc(<void**>&d_B, sizeB)
    cudaMalloc(<void**>&d_C, sizeC)
    
    # Copy to GPU
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice)
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice)
    
    # cuBLAS computation
    cdef float alpha = 1.0
    cdef float beta = 0.0
    
    cublasSgemm(
        get_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    )
    cudaDeviceSynchronize()
    
    # Copy result back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost)
    
    # Free GPU memory
    cudaFree(d_A)
    cudaFree(d_B)
    cudaFree(d_C)
    
    return C


def matmul_cython_gpu(uintptr_t ptr_A, uintptr_t ptr_B, uintptr_t ptr_C,
                      int M, int K, int N):
    """
    Matrix multiplication with GPU pointers (zero-copy).
    
    Pass tensor.data_ptr() from PyTorch CUDA tensors.
    
    Args:
        ptr_A: GPU pointer to matrix A
        ptr_B: GPU pointer to matrix B
        ptr_C: GPU pointer to matrix C (output)
        M, K, N: Matrix dimensions (A is M×K, B is K×N, C is M×N)
    """
    cdef float* d_A = <float*>ptr_A
    cdef float* d_B = <float*>ptr_B
    cdef float* d_C = <float*>ptr_C
    
    cdef float alpha = 1.0
    cdef float beta = 0.0
    
    cublasSgemm(
        get_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    )
    cudaDeviceSynchronize()

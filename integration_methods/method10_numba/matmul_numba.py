"""
Numba CUDA Matrix Multiplication

Numba allows writing CUDA kernels in pure Python syntax.
The @cuda.jit decorator compiles Python to GPU machine code.

Key Concepts:
- Write GPU kernels in Python (no C++ needed!)
- JIT compilation at first call
- Direct GPU memory access
- Good for custom kernels, not as optimized as cuBLAS for standard ops
"""

import numpy as np
import math
import time

try:
    from numba import cuda
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not installed. Install with: pip install numba")


if NUMBA_AVAILABLE:
    
    # =========================================================================
    # Method 1: Naive CUDA Kernel (Educational)
    # =========================================================================
    
    @cuda.jit
    def matmul_naive_kernel(A, B, C):
        """
        Naive matrix multiplication kernel.
        Each thread computes one element of C.
        
        This is simple but NOT optimal - no shared memory usage.
        """
        # Get thread position
        i, j = cuda.grid(2)
        
        # Bounds check
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp
    
    
    # =========================================================================
    # Method 2: Tiled/Shared Memory Kernel (Optimized)
    # =========================================================================
    
    TILE_SIZE = 16  # Threads per block dimension
    
    @cuda.jit
    def matmul_shared_kernel(A, B, C):
        """
        Optimized matrix multiplication using shared memory tiling.
        
        Shared memory is much faster than global memory.
        We load tiles into shared memory and reuse them.
        """
        # Shared memory for tiles
        sA = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=numba.float32)
        sB = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=numba.float32)
        
        # Thread indices
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        
        # Block indices
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        # Global row and column
        row = by * TILE_SIZE + ty
        col = bx * TILE_SIZE + tx
        
        # Accumulator
        tmp = 0.0
        
        # Number of tiles
        num_tiles = (A.shape[1] + TILE_SIZE - 1) // TILE_SIZE
        
        for tile in range(num_tiles):
            # Load tile into shared memory
            a_col = tile * TILE_SIZE + tx
            b_row = tile * TILE_SIZE + ty
            
            if row < A.shape[0] and a_col < A.shape[1]:
                sA[ty, tx] = A[row, a_col]
            else:
                sA[ty, tx] = 0.0
            
            if b_row < B.shape[0] and col < B.shape[1]:
                sB[ty, tx] = B[b_row, col]
            else:
                sB[ty, tx] = 0.0
            
            # Wait for all threads to load
            cuda.syncthreads()
            
            # Compute partial dot product
            for k in range(TILE_SIZE):
                tmp += sA[ty, k] * sB[k, tx]
            
            # Wait before loading next tile
            cuda.syncthreads()
        
        # Write result
        if row < C.shape[0] and col < C.shape[1]:
            C[row, col] = tmp
    
    
    def matmul_numba_naive(A, B, C):
        """
        Wrapper for naive Numba CUDA matmul.
        """
        M, K = A.shape
        K, N = B.shape
        
        # Configure grid
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(N / threads_per_block[0])
        blocks_per_grid_y = math.ceil(M / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Launch kernel
        matmul_naive_kernel[blocks_per_grid, threads_per_block](A, B, C)
    
    
    def matmul_numba_shared(A, B, C):
        """
        Wrapper for optimized Numba CUDA matmul with shared memory.
        """
        M, K = A.shape
        K, N = B.shape
        
        # Configure grid
        threads_per_block = (TILE_SIZE, TILE_SIZE)
        blocks_per_grid_x = math.ceil(N / TILE_SIZE)
        blocks_per_grid_y = math.ceil(M / TILE_SIZE)
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Launch kernel
        matmul_shared_kernel[blocks_per_grid, threads_per_block](A, B, C)
    
    
    def benchmark_numba_cuda(sizes=[512, 1024, 2048], iterations=10):
        """Benchmark Numba CUDA implementations."""
        print("=" * 60)
        print("Numba CUDA Matrix Multiplication Benchmark")
        print("=" * 60)
        
        results = {'naive': [], 'shared': []}
        
        for size in sizes:
            print(f"\nSize {size}×{size}:")
            
            # Create device arrays
            A = cuda.to_device(np.ones((size, size), dtype=np.float32))
            B = cuda.to_device(np.full((size, size), 2.0, dtype=np.float32))
            C = cuda.device_array((size, size), dtype=np.float32)
            expected = 2.0 * size
            
            # ---- Naive kernel ----
            # Warmup (includes JIT compilation)
            matmul_numba_naive(A, B, C)
            cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(iterations):
                matmul_numba_naive(A, B, C)
            cuda.synchronize()
            naive_time = (time.perf_counter() - start) / iterations
            
            C_host = C.copy_to_host()
            naive_accuracy = 100.0 if abs(C_host.max() - expected) < 0.01 else 0.0
            naive_tflops = 2 * size**3 / naive_time / 1e12
            
            print(f"  Naive:  {naive_time*1000:.3f} ms, {naive_tflops:.2f} TFLOPS, {naive_accuracy:.0f}% accuracy")
            results['naive'].append({
                'size': size, 'time_ms': naive_time*1000, 
                'tflops': naive_tflops, 'accuracy': naive_accuracy
            })
            
            # ---- Shared memory kernel ----
            C = cuda.device_array((size, size), dtype=np.float32)
            
            # Warmup
            matmul_numba_shared(A, B, C)
            cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(iterations):
                matmul_numba_shared(A, B, C)
            cuda.synchronize()
            shared_time = (time.perf_counter() - start) / iterations
            
            C_host = C.copy_to_host()
            shared_accuracy = 100.0 if abs(C_host.max() - expected) < 0.01 else 0.0
            shared_tflops = 2 * size**3 / shared_time / 1e12
            
            print(f"  Shared: {shared_time*1000:.3f} ms, {shared_tflops:.2f} TFLOPS, {shared_accuracy:.0f}% accuracy")
            results['shared'].append({
                'size': size, 'time_ms': shared_time*1000,
                'tflops': shared_tflops, 'accuracy': shared_accuracy
            })
        
        return results


else:
    def benchmark_numba_cuda(*args, **kwargs):
        print("Numba not available. Install with: pip install numba")
        return None


if __name__ == "__main__":
    if NUMBA_AVAILABLE:
        benchmark_numba_cuda([512, 1024, 2048, 4096])
    else:
        print("Please install Numba: pip install numba")

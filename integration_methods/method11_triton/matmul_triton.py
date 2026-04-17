"""
Triton Matrix Multiplication

Triton is OpenAI's language for writing highly efficient GPU kernels.
It provides a higher-level abstraction than CUDA while achieving near-cuBLAS performance.

Key Concepts:
- Block-level programming (not thread-level like CUDA)
- Automatic memory coalescing and shared memory management
- JIT compilation with autotuning
- Often matches or exceeds hand-written CUDA
"""

import torch
import time

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not installed. Install with: pip install triton")


if TRITON_AVAILABLE:
    
    # =========================================================================
    # Triton Matrix Multiplication Kernel
    # =========================================================================
    
    @triton.jit
    def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides (how many elements to skip to get to next row/col)
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Block sizes (tile dimensions)
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        Triton matrix multiplication kernel.
        
        Computes C = A @ B where:
        - A is (M, K)
        - B is (K, N)
        - C is (M, N)
        
        This kernel uses tiling for better memory access patterns.
        """
        # Program ID - which block are we?
        pid = tl.program_id(axis=0)
        
        # Number of blocks in each dimension
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        
        # Convert 1D program ID to 2D block indices
        # Using a grouped ordering for better L2 cache utilization
        num_pid_in_group = 8  # Group size for swizzling
        group_id = pid // (num_pid_in_group * num_pid_n)
        first_pid_m = group_id * num_pid_in_group
        group_size_m = min(num_pid_m - first_pid_m, num_pid_in_group)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % (num_pid_in_group * num_pid_n)) // group_size_m
        
        # Calculate starting offsets for this block
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # Pointers to first block of A and B
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        # Initialize accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Iterate over K dimension in tiles
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load tiles from A and B
            # Masking handles edge cases where tile extends beyond matrix
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            
            # Compute tile multiplication and accumulate
            accumulator += tl.dot(a, b)
            
            # Advance pointers to next K tile
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        # Convert to output dtype
        c = accumulator.to(tl.float32)
        
        # Calculate output pointers
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        
        # Store result
        tl.store(c_ptrs, c, mask=c_mask)
    
    
    def matmul_triton(A, B):
        """
        Triton matrix multiplication wrapper.
        
        Args:
            A: (M, K) tensor
            B: (K, N) tensor
        
        Returns:
            C: (M, N) tensor = A @ B
        """
        # Check constraints
        assert A.shape[1] == B.shape[0], "Incompatible dimensions"
        assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
        
        M, K = A.shape
        K, N = B.shape
        
        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        # Block sizes - these can be tuned for different GPUs
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        
        # Calculate grid size
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        # Launch kernel
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return C
    
    
    def matmul_triton_inplace(A, B, C):
        """
        Triton matrix multiplication with pre-allocated output.
        """
        assert A.shape[1] == B.shape[0], "Incompatible dimensions"
        assert A.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA"
        
        M, K = A.shape
        K, N = B.shape
        
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    
    
    # =========================================================================
    # Autotuned Version (finds best block sizes)
    # =========================================================================
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def matmul_kernel_autotuned(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """Same kernel but with autotuning to find best block sizes."""
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = 8
        group_id = pid // (num_pid_in_group * num_pid_n)
        first_pid_m = group_id * num_pid_in_group
        group_size_m = min(num_pid_m - first_pid_m, num_pid_in_group)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % (num_pid_in_group * num_pid_n)) // group_size_m
        
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        c = accumulator.to(tl.float32)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    
    
    def matmul_triton_autotuned(A, B):
        """Autotuned Triton matmul - finds best block sizes automatically."""
        assert A.shape[1] == B.shape[0], "Incompatible dimensions"
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        matmul_kernel_autotuned[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
        return C
    
    
    def benchmark_triton(sizes=[512, 1024, 2048, 4096], iterations=10):
        """Benchmark Triton implementations."""
        print("=" * 60)
        print("Triton Matrix Multiplication Benchmark")
        print("=" * 60)
        
        results = {'basic': [], 'autotuned': []}
        
        for size in sizes:
            print(f"\nSize {size}×{size}:")
            
            A = torch.ones(size, size, device='cuda', dtype=torch.float32)
            B = torch.full((size, size), 2.0, device='cuda', dtype=torch.float32)
            expected = 2.0 * size
            
            # ---- Basic Triton ----
            # Warmup (includes JIT compilation)
            C = matmul_triton(A, B)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(iterations):
                C = matmul_triton(A, B)
            torch.cuda.synchronize()
            basic_time = (time.perf_counter() - start) / iterations
            
            basic_accuracy = 100.0 if abs(C.max().item() - expected) < 0.01 else 0.0
            basic_tflops = 2 * size**3 / basic_time / 1e12
            
            print(f"  Basic:     {basic_time*1000:.3f} ms, {basic_tflops:.2f} TFLOPS, {basic_accuracy:.0f}% accuracy")
            results['basic'].append({
                'size': size, 'time_ms': basic_time*1000,
                'tflops': basic_tflops, 'accuracy': basic_accuracy
            })
            
            # ---- Autotuned Triton ----
            # Warmup (includes autotuning on first call)
            C = matmul_triton_autotuned(A, B)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(iterations):
                C = matmul_triton_autotuned(A, B)
            torch.cuda.synchronize()
            auto_time = (time.perf_counter() - start) / iterations
            
            auto_accuracy = 100.0 if abs(C.max().item() - expected) < 0.01 else 0.0
            auto_tflops = 2 * size**3 / auto_time / 1e12
            
            print(f"  Autotuned: {auto_time*1000:.3f} ms, {auto_tflops:.2f} TFLOPS, {auto_accuracy:.0f}% accuracy")
            results['autotuned'].append({
                'size': size, 'time_ms': auto_time*1000,
                'tflops': auto_tflops, 'accuracy': auto_accuracy
            })
        
        return results


else:
    def benchmark_triton(*args, **kwargs):
        print("Triton not available. Install with: pip install triton")
        return None
    
    def matmul_triton(*args, **kwargs):
        raise ImportError("Triton not available")
    
    def matmul_triton_autotuned(*args, **kwargs):
        raise ImportError("Triton not available")


if __name__ == "__main__":
    if TRITON_AVAILABLE:
        benchmark_triton([512, 1024, 2048, 4096])
    else:
        print("Please install Triton: pip install triton")

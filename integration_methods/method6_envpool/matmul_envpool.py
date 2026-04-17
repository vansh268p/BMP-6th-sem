"""
EnvPool-Style Batched Matrix Multiplication

EnvPool manages multiple environments in C++ and exposes them as batched tensors.
This demonstrates the same pattern for matrix multiplication:
- Batch multiple matrix operations in C++
- Return results as a single batched tensor
- 10-100x faster than Python loops

Key Concepts:
- Vectorized environments (batch of matrices)
- C++ manages the batch, Python sees one tensor
- Amortizes Python overhead across batch
"""

import torch
import numpy as np
import time


class BatchedMatMulPool:
    """
    EnvPool-style batched matrix multiplication.
    
    Instead of:
        for i in range(batch_size):
            C[i] = A[i] @ B[i]  # Python loop - slow!
    
    We do:
        C = batched_matmul(A, B)  # Single call - fast!
    
    This mirrors how EnvPool batches environment steps.
    """
    
    def __init__(self, batch_size, matrix_size, device="cuda"):
        self.batch_size = batch_size
        self.matrix_size = matrix_size
        self.device = device
        
        # Pre-allocate batched tensors
        self.A = torch.empty(
            batch_size, matrix_size, matrix_size,
            dtype=torch.float32, device=device
        )
        self.B = torch.empty(
            batch_size, matrix_size, matrix_size,
            dtype=torch.float32, device=device
        )
        self.C = torch.empty(
            batch_size, matrix_size, matrix_size,
            dtype=torch.float32, device=device
        )
        
    def reset(self):
        """Initialize all matrices in the batch."""
        self.A.normal_()
        self.B.normal_()
        self.C.zero_()
        return self.A.clone()
    
    def step(self, B_batch=None):
        """
        Perform batched matrix multiplication.
        
        Uses torch.bmm (batched matrix multiply) which:
        1. Keeps all data on GPU
        2. Processes entire batch in one kernel launch
        3. Much faster than Python loops
        """
        if B_batch is not None:
            self.B = B_batch
        
        # Batched matrix multiplication - single CUDA kernel!
        # This is equivalent to EnvPool's vectorized step()
        self.C = torch.bmm(self.A, self.B)
        
        # Update state for next step (like env.step updating obs)
        self.A = self.C.clone()
        
        return self.C
    
    def step_sequential(self, B_batch=None):
        """
        Sequential version for comparison (what EnvPool avoids).
        
        This is how naive Python would do it - SLOW!
        """
        if B_batch is not None:
            self.B = B_batch
            
        for i in range(self.batch_size):
            self.C[i] = torch.mm(self.A[i], self.B[i])
        
        self.A = self.C.clone()
        return self.C


def benchmark_envpool_style(batch_sizes=[8, 32, 128], matrix_size=512, iterations=10):
    """
    Benchmark EnvPool-style batching vs sequential.
    
    This shows why EnvPool is 10-100x faster:
    - Batched: One kernel launch for all matrices
    - Sequential: N kernel launches (Python loop overhead)
    """
    print("=" * 60)
    print("EnvPool-Style Batched Matrix Multiplication")
    print("=" * 60)
    print(f"Matrix size: {matrix_size}×{matrix_size}")
    print()
    
    results = []
    
    for batch_size in batch_sizes:
        pool = BatchedMatMulPool(batch_size, matrix_size, device="cuda")
        pool.reset()
        
        # Warmup
        for _ in range(3):
            pool.step()
        torch.cuda.synchronize()
        
        # Benchmark batched (EnvPool style)
        start = time.perf_counter()
        for _ in range(iterations):
            C = pool.step()
        torch.cuda.synchronize()
        batched_time = (time.perf_counter() - start) / iterations
        
        # Benchmark sequential (naive Python)
        pool.reset()
        for _ in range(3):
            pool.step_sequential()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            C = pool.step_sequential()
        torch.cuda.synchronize()
        seq_time = (time.perf_counter() - start) / iterations
        
        # Calculate metrics
        total_flops = batch_size * 2 * matrix_size ** 3
        batched_tflops = total_flops / batched_time / 1e12
        seq_tflops = total_flops / seq_time / 1e12
        speedup = seq_time / batched_time
        
        print(f"Batch size {batch_size:3d}:")
        print(f"  Batched (EnvPool):   {batched_time*1000:8.3f} ms, {batched_tflops:.2f} TFLOPS")
        print(f"  Sequential (naive):  {seq_time*1000:8.3f} ms, {seq_tflops:.2f} TFLOPS")
        print(f"  Speedup: {speedup:.1f}x")
        print()
        
        results.append({
            'batch_size': batch_size,
            'batched_time': batched_time,
            'sequential_time': seq_time,
            'speedup': speedup,
            'batched_tflops': batched_tflops
        })
    
    return results


if __name__ == "__main__":
    benchmark_envpool_style()

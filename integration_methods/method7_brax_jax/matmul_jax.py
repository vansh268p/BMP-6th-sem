"""
Brax/JAX-Style GPU Matrix Multiplication

Brax uses JAX for differentiable GPU physics. Key features:
- JIT compilation (compile once, run fast)
- Automatic differentiation (gradients through physics!)
- Vectorization with vmap (automatic batching)
- Pure functional style (no side effects)

This demonstrates JAX patterns for matrix multiplication.
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not installed. Install with: pip install jax jaxlib")

import numpy as np
import time


if JAX_AVAILABLE:
    # ===========================================
    # JAX Matrix Multiplication Functions
    # ===========================================
    
    @jit
    def matmul_jax(A, B):
        """
        Basic JAX matrix multiplication.
        
        @jit decorator compiles this to XLA (Accelerated Linear Algebra).
        First call is slow (compilation), subsequent calls are fast.
        """
        return jnp.matmul(A, B)
    
    
    @jit
    def matmul_chain(A, B, C):
        """
        Chained matrix multiplication: A @ B @ C
        
        JAX's XLA compiler optimizes the entire computation graph,
        potentially fusing operations for better performance.
        """
        return jnp.matmul(jnp.matmul(A, B), C)
    
    
    # Vectorized (batched) matmul using vmap
    # vmap = "vectorizing map" - automatically batches a function
    batched_matmul = jit(vmap(lambda a, b: jnp.matmul(a, b), in_axes=(0, 0)))
    
    
    @jit
    def matmul_with_grad(A, B):
        """
        Matrix multiplication that we can differentiate through.
        
        In Brax, this allows computing gradients through physics:
        - Forward: state_new = physics(state, action)
        - Backward: d(loss)/d(action) for policy gradient
        """
        C = jnp.matmul(A, B)
        # Return scalar for grad (sum of squared elements)
        return jnp.sum(C ** 2)
    
    # Gradient function - computes d(output)/d(A)
    grad_matmul = jit(grad(matmul_with_grad, argnums=0))
    
    
    def benchmark_jax_style(sizes=[512, 1024, 2048], iterations=10):
        """Benchmark JAX-style matrix multiplication."""
        print("=" * 60)
        print("Brax/JAX-Style Matrix Multiplication")
        print("=" * 60)
        
        results = []
        
        for size in sizes:
            # Create JAX arrays (automatically on GPU if available)
            key = jax.random.PRNGKey(0)
            A = jax.random.normal(key, (size, size), dtype=jnp.float32)
            B = jax.random.normal(key, (size, size), dtype=jnp.float32)
            
            # Warmup (includes JIT compilation on first call)
            print(f"\nSize {size}×{size}:")
            print("  Compiling (first call)...", end=" ")
            start = time.perf_counter()
            C = matmul_jax(A, B).block_until_ready()
            compile_time = time.perf_counter() - start
            print(f"{compile_time*1000:.1f} ms")
            
            # Benchmark after compilation
            start = time.perf_counter()
            for _ in range(iterations):
                C = matmul_jax(A, B).block_until_ready()
            elapsed = (time.perf_counter() - start) / iterations
            
            flops = 2 * size * size * size
            tflops = flops / elapsed / 1e12
            
            print(f"  After JIT:    {elapsed*1000:.3f} ms, {tflops:.2f} TFLOPS")
            
            # Benchmark gradient computation
            start = time.perf_counter()
            for _ in range(iterations):
                dA = grad_matmul(A, B).block_until_ready()
            grad_elapsed = (time.perf_counter() - start) / iterations
            
            print(f"  With gradient: {grad_elapsed*1000:.3f} ms")
            
            results.append({
                'size': size,
                'time': elapsed,
                'tflops': tflops,
                'compile_time': compile_time,
                'grad_time': grad_elapsed
            })
        
        return results
    
    
    def benchmark_vmap_batching(batch_sizes=[8, 32, 128], matrix_size=512, iterations=10):
        """
        Benchmark JAX's vmap for automatic batching.
        
        vmap transforms a single-matrix function into a batched version
        WITHOUT rewriting the function. This is Brax's secret weapon.
        """
        print("\n" + "=" * 60)
        print("JAX vmap Automatic Batching")
        print("=" * 60)
        print(f"Matrix size: {matrix_size}×{matrix_size}")
        
        key = jax.random.PRNGKey(0)
        
        for batch_size in batch_sizes:
            # Create batched inputs
            A_batch = jax.random.normal(key, (batch_size, matrix_size, matrix_size))
            B_batch = jax.random.normal(key, (batch_size, matrix_size, matrix_size))
            
            # Warmup
            C = batched_matmul(A_batch, B_batch).block_until_ready()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                C = batched_matmul(A_batch, B_batch).block_until_ready()
            elapsed = (time.perf_counter() - start) / iterations
            
            total_flops = batch_size * 2 * matrix_size ** 3
            tflops = total_flops / elapsed / 1e12
            
            print(f"Batch {batch_size:3d}: {elapsed*1000:.3f} ms, {tflops:.2f} TFLOPS")


else:
    def benchmark_jax_style(*args, **kwargs):
        print("JAX not available. Install with: pip install jax jaxlib")
        return []
    
    def benchmark_vmap_batching(*args, **kwargs):
        print("JAX not available. Install with: pip install jax jaxlib")


if __name__ == "__main__":
    if JAX_AVAILABLE:
        benchmark_jax_style()
        benchmark_vmap_batching()
    else:
        print("Please install JAX: pip install jax jaxlib")

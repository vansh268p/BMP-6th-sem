"""
Benchmark All RL Environment Integration Methods for Matrix Multiplication

This script compares:
1. Gymnasium - Standard RL environment API
2. EnvPool - Batched C++ environments
3. Brax/JAX - JIT-compiled GPU operations
4. DM Control - Richer timestep API
5. Isaac Lab - Full GPU-native pipeline

All methods perform matrix multiplication, demonstrating different
integration patterns used in reinforcement learning.
"""

import sys
import time
import torch
import numpy as np

# Add method directories to path
sys.path.insert(0, 'method5_gymnasium')
sys.path.insert(0, 'method6_envpool')
sys.path.insert(0, 'method7_brax_jax')
sys.path.insert(0, 'method8_dm_control')
sys.path.insert(0, 'method9_isaac_lab')


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def benchmark_pytorch_baseline(sizes, iterations=10):
    """Baseline: Pure PyTorch matrix multiplication."""
    print_header("BASELINE: PyTorch Native")
    
    results = []
    for size in sizes:
        A = torch.randn(size, size, device="cuda", dtype=torch.float32)
        B = torch.randn(size, size, device="cuda", dtype=torch.float32)
        
        # Warmup
        for _ in range(3):
            C = torch.mm(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            C = torch.mm(A, B)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iterations
        
        flops = 2 * size ** 3
        tflops = flops / elapsed / 1e12
        
        print(f"Size {size:5d}×{size}: {elapsed*1000:10.3f} ms, {tflops:6.2f} TFLOPS")
        results.append({'size': size, 'time': elapsed, 'tflops': tflops})
    
    return results


def benchmark_gymnasium_method(sizes, iterations=10):
    """Method 5: Gymnasium-style wrapper."""
    print_header("METHOD 5: Gymnasium Style")
    
    try:
        from matmul_gymnasium import MatMulEnv
    except ImportError as e:
        print(f"Error importing Gymnasium method: {e}")
        print("Install with: pip install gymnasium")
        return []
    
    results = []
    for size in sizes:
        try:
            env = MatMulEnv(matrix_size=size, device="cuda")
            obs, info = env.reset()
            action = np.random.randn(size, size).astype(np.float32)
            
            # Warmup
            for _ in range(3):
                env.step(action)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                obs, reward, term, trunc, info = env.step(action)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / iterations
            
            flops = 2 * size ** 3
            tflops = flops / elapsed / 1e12
            
            print(f"Size {size:5d}×{size}: {elapsed*1000:10.3f} ms, {tflops:6.2f} TFLOPS")
            results.append({'size': size, 'time': elapsed, 'tflops': tflops})
        except Exception as e:
            print(f"Size {size}: FAILED - {e}")
    
    return results


def benchmark_envpool_method(batch_sizes, matrix_size=512, iterations=10):
    """Method 6: EnvPool-style batching."""
    print_header("METHOD 6: EnvPool Style (Batched)")
    print(f"Matrix size: {matrix_size}×{matrix_size}")
    
    try:
        from matmul_envpool import BatchedMatMulPool
    except ImportError as e:
        print(f"Error importing EnvPool method: {e}")
        return []
    
    results = []
    for batch_size in batch_sizes:
        try:
            pool = BatchedMatMulPool(batch_size, matrix_size, device="cuda")
            pool.reset()
            
            # Warmup
            for _ in range(3):
                pool.step()
            torch.cuda.synchronize()
            
            # Benchmark batched
            start = time.perf_counter()
            for _ in range(iterations):
                C = pool.step()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / iterations
            
            total_flops = batch_size * 2 * matrix_size ** 3
            tflops = total_flops / elapsed / 1e12
            
            print(f"Batch {batch_size:4d}: {elapsed*1000:10.3f} ms, {tflops:6.2f} TFLOPS")
            results.append({'batch_size': batch_size, 'time': elapsed, 'tflops': tflops})
        except Exception as e:
            print(f"Batch {batch_size}: FAILED - {e}")
    
    return results


def benchmark_jax_method(sizes, iterations=10):
    """Method 7: Brax/JAX style."""
    print_header("METHOD 7: Brax/JAX Style")
    
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
        JAX_AVAILABLE = True
    except ImportError:
        print("JAX not available. Install with: pip install jax jaxlib")
        return []
    
    @jit
    def matmul_jax(A, B):
        return jnp.matmul(A, B)
    
    results = []
    key = jax.random.PRNGKey(0)
    
    for size in sizes:
        try:
            A = jax.random.normal(key, (size, size), dtype=jnp.float32)
            B = jax.random.normal(key, (size, size), dtype=jnp.float32)
            
            # Warmup (includes compilation)
            C = matmul_jax(A, B).block_until_ready()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                C = matmul_jax(A, B).block_until_ready()
            elapsed = (time.perf_counter() - start) / iterations
            
            flops = 2 * size ** 3
            tflops = flops / elapsed / 1e12
            
            print(f"Size {size:5d}×{size}: {elapsed*1000:10.3f} ms, {tflops:6.2f} TFLOPS")
            results.append({'size': size, 'time': elapsed, 'tflops': tflops})
        except Exception as e:
            print(f"Size {size}: FAILED - {e}")
    
    return results


def benchmark_dm_control_method(sizes, iterations=10):
    """Method 8: DM Control style."""
    print_header("METHOD 8: DM Control Style")
    
    try:
        from matmul_dm_control import MatMulEnvironment
    except ImportError as e:
        print(f"Error importing DM Control method: {e}")
        return []
    
    results = []
    for size in sizes:
        try:
            env = MatMulEnvironment(matrix_size=size, device="cuda")
            timestep = env.reset()
            action = np.random.randn(size, size).astype(np.float32) * 0.01
            
            # Warmup
            for _ in range(3):
                timestep = env.step(action)
                if timestep.last():
                    timestep = env.reset()
            torch.cuda.synchronize()
            
            timestep = env.reset()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                timestep = env.step(action)
                if timestep.last():
                    timestep = env.reset()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / iterations
            
            flops = 2 * size ** 3
            tflops = flops / elapsed / 1e12
            
            print(f"Size {size:5d}×{size}: {elapsed*1000:10.3f} ms, {tflops:6.2f} TFLOPS")
            results.append({'size': size, 'time': elapsed, 'tflops': tflops})
        except Exception as e:
            print(f"Size {size}: FAILED - {e}")
    
    return results


def benchmark_isaac_lab_method(num_envs_list, matrix_size=128, iterations=100):
    """Method 9: Isaac Lab style (GPU-native)."""
    print_header("METHOD 9: Isaac Lab Style (GPU-Native)")
    print(f"Matrix size: {matrix_size}×{matrix_size}")
    
    try:
        from matmul_isaac_lab import DirectRLMatMulEnv, IsaacLabConfig
    except ImportError as e:
        print(f"Error importing Isaac Lab method: {e}")
        return []
    
    results = []
    for num_envs in num_envs_list:
        try:
            cfg = IsaacLabConfig(
                num_envs=num_envs,
                matrix_size=matrix_size,
                device="cuda"
            )
            env = DirectRLMatMulEnv(cfg)
            env.reset()
            
            # Actions MUST be created on GPU directly
            actions = torch.randn(
                num_envs, matrix_size, matrix_size,
                device=torch.device("cuda"), dtype=torch.float32
            ) * 0.01
            
            # Warmup
            for _ in range(10):
                env.step(actions)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                obs, rew, done, info = env.step(actions)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            steps_per_second = num_envs * iterations / elapsed
            time_per_step = elapsed / iterations * 1000
            flops_per_step = num_envs * 2 * matrix_size ** 3
            tflops = flops_per_step * iterations / elapsed / 1e12
            
            print(f"Envs {num_envs:5d}: {time_per_step:8.3f} ms/step, "
                  f"{steps_per_second:>12,.0f} steps/sec, {tflops:6.2f} TFLOPS")
            results.append({
                'num_envs': num_envs,
                'time_per_step': time_per_step,
                'steps_per_second': steps_per_second,
                'tflops': tflops
            })
            
            env.close()
        except Exception as e:
            print(f"Envs {num_envs}: FAILED - {e}")
    
    return results


def main():
    print("=" * 70)
    print(" RL ENVIRONMENT INTEGRATION METHODS BENCHMARK")
    print(" Matrix Multiplication Performance Comparison")
    print("=" * 70)
    
    # Test sizes
    sizes = [512, 1024, 2048, 4096]
    batch_sizes = [8, 32, 128, 512]
    num_envs_list = [64, 256, 1024, 4096]
    
    # Run all benchmarks
    results = {}
    
    results['baseline'] = benchmark_pytorch_baseline(sizes)
    results['gymnasium'] = benchmark_gymnasium_method(sizes)
    results['envpool'] = benchmark_envpool_method(batch_sizes)
    results['jax'] = benchmark_jax_method(sizes)
    results['dm_control'] = benchmark_dm_control_method(sizes)
    results['isaac_lab'] = benchmark_isaac_lab_method(num_envs_list)
    
    # Summary
    print_header("SUMMARY")
    print("""
Method Comparison:

┌─────────────────┬──────────────────┬─────────────────────────────────────┐
│ Method          │ Pattern          │ Best Use Case                       │
├─────────────────┼──────────────────┼─────────────────────────────────────┤
│ PyTorch Native  │ Direct calls     │ Simple prototyping                  │
│ Gymnasium       │ step()/reset()   │ Standard RL, Stable-Baselines3      │
│ EnvPool         │ Batched C++      │ High-throughput CPU simulation      │
│ JAX/Brax        │ JIT + vmap       │ Differentiable physics, TPU/GPU     │
│ DM Control      │ TimeStep API     │ Research, richer observation types  │
│ Isaac Lab       │ GPU-native       │ Maximum throughput, robotics sim    │
└─────────────────┴──────────────────┴─────────────────────────────────────┘

Key Insights:
1. Isaac Lab achieves highest steps/second by keeping everything on GPU
2. EnvPool batching provides significant speedup over sequential
3. JAX JIT compilation eliminates Python overhead after first call
4. Gymnasium/DM Control have similar performance (both transfer to CPU)
5. For RL training, choose based on framework compatibility
""")
    
    return results


if __name__ == "__main__":
    main()

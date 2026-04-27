# SYCL Shared-Memory Integration Benchmark

This benchmark uses the same memory model as the production SYCL-PIC code:

```text
sycl::malloc_shared -> Python integration method -> same SYCL tiled kernel
```

Unlike the CUDA/NumPy benchmarks, there are no H2D or D2H copies inside the
timed matrix-multiplication call.

Build:

```bash
bash build_core.sh
bash build_wrappers.sh
```

Run:

```bash
python3 benchmark.py --sizes 128 256 512 1024 2048 4096 8192 --reps 3 --warmup 1
python3 plot_results.py
```

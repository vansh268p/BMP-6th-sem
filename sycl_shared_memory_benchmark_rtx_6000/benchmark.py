"""SYCL malloc_shared matrix multiplication integration benchmark.

This benchmark mirrors the production SYCL-PIC memory model:
matrices are allocated once with ``sycl::malloc_shared`` and then exposed or
passed through different Python/C++ integration methods without H2D/D2H
copies inside the timed call.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy_c_api_sycl  # noqa: E402
import pybind11_sycl  # noqa: E402
import dlpack_sycl  # noqa: E402
import cython_sycl  # noqa: E402


class SyclSharedTimings(ctypes.Structure):
    _fields_ = [
        ("kernel_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
    ]


def load_core():
    lib = ctypes.CDLL(str(HERE / "src" / "libsycl_shared_core.so"))
    lib.sycl_shared_has_gpu.argtypes = []
    lib.sycl_shared_has_gpu.restype = ctypes.c_int
    lib.sycl_shared_init.argtypes = [ctypes.c_size_t]
    lib.sycl_shared_init.restype = ctypes.c_int
    lib.sycl_shared_finalize.argtypes = []
    lib.sycl_shared_finalize.restype = None
    lib.sycl_shared_a.argtypes = []
    lib.sycl_shared_a.restype = ctypes.POINTER(ctypes.c_float)
    lib.sycl_shared_b.argtypes = []
    lib.sycl_shared_b.restype = ctypes.POINTER(ctypes.c_float)
    lib.sycl_shared_c.argtypes = []
    lib.sycl_shared_c.restype = ctypes.POINTER(ctypes.c_float)
    lib.sycl_shared_matmul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(SyclSharedTimings),
    ]
    lib.sycl_shared_matmul.restype = ctypes.c_int
    return lib


CORE = load_core()


def device_label():
    host = socket.gethostname()
    gpu = "sycl_gpu"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpu = r.stdout.strip().splitlines()[0].replace(" ", "_")
    except Exception:
        pass
    cpu = platform.processor() or "cpu"
    return f"{host}__{cpu.replace(' ', '_')}__{gpu}__sycl_malloc_shared"


def init_ctypes(n):
    rc = CORE.sycl_shared_init(n)
    if rc != 0:
        raise RuntimeError(f"sycl_shared_init failed: {rc}")
    elems = n * n
    a_ptr = CORE.sycl_shared_a()
    b_ptr = CORE.sycl_shared_b()
    c_ptr = CORE.sycl_shared_c()
    a = np.ctypeslib.as_array(a_ptr, shape=(elems,)).reshape(n, n)
    b = np.ctypeslib.as_array(b_ptr, shape=(elems,)).reshape(n, n)
    c = np.ctypeslib.as_array(c_ptr, shape=(elems,)).reshape(n, n)
    return a_ptr, b_ptr, c_ptr, a, b, c


def finalize_ctypes():
    CORE.sycl_shared_finalize()


METHODS = {
    "NumPy C-API": {
        "init": numpy_c_api_sycl.init,
        "matmul": numpy_c_api_sycl.matmul,
        "finalize": numpy_c_api_sycl.finalize,
    },
    "pybind11": {
        "init": pybind11_sycl.init,
        "matmul": pybind11_sycl.matmul,
        "finalize": pybind11_sycl.finalize,
    },
    "Cython": {
        "init": cython_sycl.init,
        "matmul": cython_sycl.matmul,
        "finalize": cython_sycl.finalize,
    },
    "DLPack": {
        "init": dlpack_sycl.init,
        "matmul": dlpack_sycl.matmul,
        "finalize": dlpack_sycl.finalize,
    },
}


def ctypes_matmul(a_ptr, b_ptr, c_ptr, n, py_call_ns):
    timings = SyclSharedTimings()
    rc = CORE.sycl_shared_matmul(a_ptr, b_ptr, c_ptr, n, ctypes.byref(timings))
    if rc != 0:
        raise RuntimeError(f"sycl_shared_matmul failed: {rc}")
    return {
        "marshal_in_ms": 0.0,
        "ctypes_pointer_ms": 0.0,
        "kernel_ms": timings.kernel_ms,
        "sycl_total_ms": timings.total_ms,
    }


def run_one(method_name, n, reps, warmup):
    if method_name == "ctypes/nanobind":
        init_result = init_ctypes(n)
        a_ptr, b_ptr, c_ptr = init_result[:3]
        call_args = (a_ptr, b_ptr, c_ptr, n)
        matmul = ctypes_matmul
        finalize = finalize_ctypes
    else:
        method = METHODS[method_name]
        init_result = method["init"](n)
        call_args = tuple(init_result)
        matmul = method["matmul"]
        finalize = method["finalize"]

    try:
        for _ in range(warmup):
            t0 = time.perf_counter_ns()
            matmul(*call_args, t0)

        samples = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            rec = matmul(*call_args, t0)
            t1 = time.perf_counter_ns()
            rec = dict(rec)
            rec["py_total_ms"] = (t1 - t0) / 1e6
            known = sum(
                float(rec.get(k, 0.0))
                for k in rec
                if k.endswith("_ms") and k not in {"py_total_ms", "marshal_out_ms"}
            )
            rec["marshal_out_ms"] = max(0.0, rec["py_total_ms"] - known)
            samples.append(rec)
    finally:
        finalize()

    numeric_keys = [k for k, v in samples[0].items() if isinstance(v, (int, float))]
    avg = {k: float(np.mean([s.get(k, 0.0) for s in samples])) for k in numeric_keys}
    avg["py_total_ms_std"] = float(np.std([s["py_total_ms"] for s in samples]))
    avg["method"] = method_name
    avg["size"] = n
    avg["backend"] = "sycl"
    avg["kind"] = "malloc_shared_tiled"
    avg["h2d_ms"] = 0.0
    avg["d2h_ms"] = 0.0
    avg["gflops"] = 2 * n**3 / (avg["py_total_ms"] * 1e-3) / 1e9
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--methods", nargs="+", default=["NumPy C-API", "pybind11", "Cython", "ctypes/nanobind", "DLPack"])
    ap.add_argument("--tag", default=device_label())
    args = ap.parse_args()

    if not CORE.sycl_shared_has_gpu():
        raise SystemExit("SYCL GPU device is not available")

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / f"{args.tag}.csv"

    print("[cfg] benchmark=SYCL malloc_shared tiled matmul")
    print("[cfg] H2D/D2H copies are not part of the timed call")
    print(f"[cfg] tag={args.tag}")
    print(f"[cfg] sizes={args.sizes} reps={args.reps} warmup={args.warmup}")

    rows = []

    def write_rows():
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    for n in args.sizes:
        for method in args.methods:
            print(f"  [{n:>5}] {method:16s} ...", end="", flush=True)
            rec = run_one(method, n, args.reps, args.warmup)
            rec["tag"] = args.tag
            rows.append(rec)
            write_rows()
            print(f" {rec['py_total_ms']:9.3f} ms ({rec['gflops']:8.1f} GFLOP/s)")

    print(f"\n[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()

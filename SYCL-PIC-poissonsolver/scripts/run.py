#!/usr/bin/env python3
"""
SYCL Particle Simulation Runner
==============================

Compiles launcher.cpp + sycl_pic_module.cpp (+ all supporting .cpp files)
into a single executable that embeds the Python interpreter and registers
`sycl_pic` as a built-in module.  MKL Pardiso works because all symbols
are resolved at link time — no dlopen.

Then runs the executable, which immediately launches main.py.  From the
user's perspective, main.py IS the simulation driver.
"""

import os
import sys
import subprocess
import argparse
import sysconfig
import glob
import json


def get_sycl_compiler():
    """Detect SYCL compiler."""
    compiler = "/opt/intel/oneapi/compiler/2025.1/bin/icx"
    if not os.path.exists(compiler):
        compiler = "icx"
    try:
        subprocess.run([compiler, '--version'], capture_output=True, check=True)
        return compiler
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ICX SYCL compiler not found.")


def get_embed_python_prefix():
    """Pick the Python environment used by the embedded launcher."""
    prefix = os.environ.get("PHYSICSNEMO_ENV_PREFIX", "/home/student/.conda/envs/physicsnemo")
    if os.path.isdir(prefix):
        return prefix
    return "/home/anaconda3"


def get_embed_python_info(prefix):
    """Query Python/sysconfig details from the target environment."""
    py_exe = os.path.join(prefix, "bin", "python")
    if not os.path.exists(py_exe):
        raise RuntimeError(f"Embedded Python executable not found: {py_exe}")

    query = (
        "import json, sys, sysconfig, numpy; "
        "print(json.dumps({"
        "'include': sysconfig.get_path('include'), "
        "'libdir': sysconfig.get_config_var('LIBDIR'), "
        "'ldlibrary': sysconfig.get_config_var('LDLIBRARY'), "
        "'version': f'{sys.version_info.major}.{sys.version_info.minor}', "
        "'stdlib': sysconfig.get_path('stdlib'), "
        "'purelib': sysconfig.get_path('purelib'), "
        "'platlib': sysconfig.get_path('platlib'), "
        "'numpy_include': numpy.get_include()"
        "}))"
    )

    result = subprocess.run([py_exe, "-c", query], capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip())


def compile_extension(compiler, build_dir="build", devices=['gpu']):
    """Compile sycl_pic_launcher executable (embeds Python + sycl_pic module)."""
    print(f"Compiling sycl_pic_launcher with {compiler} for devices: {devices}")
    os.makedirs(build_dir, exist_ok=True)

    # Source files: launcher + sycl_pic_module + all simulation sources
    src_files = [
        "src/launcher.cpp",
        "src/sycl_pic_module.cpp",
        "src/utils.cpp",
        "src/particles.cpp",
        "src/mover.cpp",
        "src/interpolation.cpp",
        "src/types.cpp",
        "src/poissonSolver.cpp",
    ]

    # SYCL device targets
    target_map = {'gpu': 'nvptx64-nvidia-cuda', 'cpu': 'spir64_x86_64'}
    targets_flag = ",".join(target_map[d] for d in devices if d in target_map)

    # Python / NumPy paths from target embedded environment
    embed_prefix = get_embed_python_prefix()
    py_info = get_embed_python_info(embed_prefix)
    python_include = py_info["include"]
    numpy_include = py_info["numpy_include"]
    python_lib = os.path.join(py_info["libdir"], py_info["ldlibrary"])
    if not os.path.exists(python_lib):
        version = py_info["version"]
        fallback_libs = [
            f"libpython{version}.so",
            f"libpython{version}.so.1.0",
            f"libpython{version}m.so",
        ]
        for libname in fallback_libs:
            candidate = os.path.join(py_info["libdir"], libname)
            if os.path.exists(candidate):
                python_lib = candidate
                break
    if not os.path.exists(python_lib):
        print(f"Error: {python_lib} not found"); return False

    output = os.path.join(build_dir, "sycl_pic_launcher")

    cmd = [
        compiler,
        "-fsycl",
        f"-fsycl-targets={targets_flag}",
        "-I./include",
        f"-I{python_include}",
        f"-I{numpy_include}",
        "-I/opt/intel/oneapi/dpl/2022.8/include",
        "-O3", "-std=c++17",
        "-DMKL_ILP64",
        "-I/opt/intel/oneapi/mkl/2025.1/include",
        "-L/opt/intel/oneapi/mkl/2025.1/lib",
        "-L/opt/intel/oneapi/tbb/2022.3/lib",
        "-lmkl_sycl_blas", "-lmkl_sycl_lapack", "-lmkl_sycl_dft",
        "-lmkl_intel_ilp64", "-lmkl_tbb_thread", "-lmkl_core",
        "-lsycl", "-lOpenCL", "-ltbb", "-lpthread", "-ldl", "-lm",
        "-w",
        *src_files,
        python_lib,
        "-o", output,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Compilation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        print(f"Compilation successful!  → {output}")
        return True
    except Exception as e:
        print(f"Compilation error: {e}")
        return False


def run_simulation(input_file1, input_file2, device, wg_size, prob_size,
                   build_dir="build"):
    """Run the sycl_pic_launcher executable, which runs main.py internally."""
    exe = os.path.join(build_dir, "sycl_pic_launcher")
    if not os.path.exists(exe):
        print(f"Launcher executable not found: {exe}")
        return False

    cmd = [
        exe,
        input_file1, input_file2, device, str(wg_size), str(prob_size),
    ]

    print(f"Running: {' '.join(cmd)}")
    env = os.environ.copy()
    
    # Runtime library paths
    cuda_adapter = "/opt/intel/oneapi/compiler/2025.1/lib"
    mkl_lib      = "/opt/intel/oneapi/mkl/2025.1/lib"
    tbb_lib      = "/opt/intel/oneapi/tbb/2022.3/lib"
    embed_prefix = get_embed_python_prefix()
    py_info = get_embed_python_info(embed_prefix)
    embed_lib = os.path.join(embed_prefix, "lib")
    extra = ":".join([cuda_adapter, mkl_lib, tbb_lib, embed_lib])
    env["LD_LIBRARY_PATH"] = extra + ":" + env.get("LD_LIBRARY_PATH", "")

    # Python environment for embedded interpreter
    py_stdlib = py_info["stdlib"]
    py_site_packages = py_info["purelib"]
    py_platlib = py_info["platlib"]
    py_dynload = os.path.join(py_stdlib, "lib-dynload")

    env["PYTHONHOME"] = embed_prefix
    physics_root = "/home/student/Dhruvil_Om_BMP_2026"
    physics_solver_root = "/home/student/Dhruvil_Om_BMP_2026/PoissonSolver"
    existing_pythonpath = env.get("PYTHONPATH", "")

    python_paths = [
        physics_root,
        physics_solver_root,
        py_stdlib,
        py_site_packages,
        py_platlib,
        py_dynload,
    ]

    # Keep only entries from the selected environment to avoid mixing
    # incompatible stdlib/site-packages (e.g., 3.11 vs 3.12).
    if existing_pythonpath:
        safe_existing = [
            p for p in existing_pythonpath.split(":")
            if p.startswith(embed_prefix)
        ]
        python_paths.extend(safe_existing)

    env["PYTHONPATH"] = ":".join(path for path in python_paths if path)
    env["PHYSICSNEMO_ROOT"] = physics_root

    # Ensure user-site packages do not override the selected embedded env.
    env["PYTHONNOUSERSITE"] = "1"

    # Avoid torch inductor spawning external compile worker subprocesses that
    # were triggering SRE mismatch under embedded Python.
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed with return code {e.returncode}")
        return False


def validate_inputs(input_file1, input_file2, device, wg_size, prob_size):
    for f in [input_file1, input_file2]:
        if not os.path.exists(f):
            raise ValueError(f"Input file not found: {f}")
    if device not in ('cpu', 'gpu'):
        raise ValueError("Device must be 'cpu' or 'gpu'")
    if wg_size <= 0 or prob_size <= 0:
        raise ValueError("wg_size and prob_size must be positive")


def main():
    parser = argparse.ArgumentParser(
        description="Compile and run SYCL-PIC simulation (Python-driven)")
    parser.add_argument("input_file1", help="First particle input file (.bin)")
    parser.add_argument("input_file2", help="Second particle input file (.bin)")
    parser.add_argument("--device", choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument("--wg-size", type=int, default=256)
    parser.add_argument("--prob-size", type=int, default=24)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()

    try:
        if not args.run_only:
            validate_inputs(args.input_file1, args.input_file2,
                            args.device, args.wg_size, args.prob_size)

        if not args.run_only:
            compiler = get_sycl_compiler()
            devices = [args.device]
            if not compile_extension(compiler, args.build_dir, devices):
                return 1

        if not args.compile_only:
            if not run_simulation(args.input_file1, args.input_file2,
                                  args.device, args.wg_size, args.prob_size,
                                  args.build_dir):
                return 1

        print("\nSimulation completed successfully!")
        print("Output files: Mesh_0.out, Mesh_1.out")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

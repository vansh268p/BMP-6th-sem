# SYCL-PIC: GPU-Accelerated Particle-in-Cell Simulation with Zero-Copy Python Interop

**BTech Mini Project**

## Abstract

This project implements a high-performance 2D **Particle-in-Cell (PIC)** plasma simulation accelerated on NVIDIA GPUs using **Intel SYCL** (DPC++). The core contribution is a **zero-copy Python–C++/SYCL interoperability layer** that allows a Python driver (`main.py`) to fully orchestrate the GPU simulation — including modifying GPU-resident arrays from Python — without any data copies between languages. The simulation models electron and ion dynamics in a Penning-type discharge geometry with electrostatic field solving via MKL Pardiso.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Zero-Copy Python Interop](#zero-copy-python-interop)
- [Simulation Physics](#simulation-physics)
- [Project Structure](#project-structure)
- [Key Source Files](#key-source-files)
- [Technical Challenges Solved](#technical-challenges-solved)
- [Hardware and Software Environment](#hardware-and-software-environment)
- [Build and Run Instructions](#build-and-run-instructions)
- [Simulation Parameters](#simulation-parameters)
- [Output Files](#output-files)
- [Results](#results)
- [Work Done Summary](#work-done-summary)

---

## Project Overview

The original codebase was a monolithic C++ PIC simulation (`main.cpp`) that ran entirely in C++. Our work **replaced the C++ driver with a Python driver** (`main.py`) while keeping all GPU-accelerated computation in SYCL C++. This enables:

1. **Rapid prototyping** — modify simulation logic in Python without recompilation
2. **Zero-copy data access** — Python/NumPy arrays point directly at SYCL `malloc_shared` GPU memory
3. **Python ecosystem integration** — use NumPy, SciPy, matplotlib etc. on live simulation data
4. **Full simulation control from Python** — the simulation loop, physics modifications (e.g., `phi = rand_int * rho`), and diagnostics are all driven from Python

## Architecture

The system consists of three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  src/main.py  (Python Driver)                                   │
│  ─ Simulation loop, timing, phi = rand_int * rho via NumPy     │
│  ─ import sycl_pic  → calls C++ functions directly              │
├─────────────────────────────────────────────────────────────────┤
│  src/sycl_pic_module.cpp  (Python C Extension Module)           │
│  ─ 18 wrapped C++ functions exposed to Python                   │
│  ─ PyArray_SimpleNewFromData for zero-copy rho/phi arrays       │
│  ─ Compiled INTO the executable (not a .so)                     │
├─────────────────────────────────────────────────────────────────┤
│  src/launcher.cpp  (Executable Entry Point)                     │
│  ─ PyImport_AppendInittab("sycl_pic", PyInit_sycl_pic)         │
│  ─ Embeds Python interpreter, runs src/main.py                  │
│  ─ All MKL symbols resolved at link time (no dlopen)            │
├─────────────────────────────────────────────────────────────────┤
│  C++ SYCL Simulation Kernels                                    │
│  ─ interpolation.cpp (charge deposition on GPU)                 │
│  ─ mover.cpp (particle movement on GPU)                         │
│  ─ poissonSolver.cpp (MKL Pardiso electrostatic solver)         │
│  ─ particles.cpp (species registration, I/O)                    │
│  ─ utils.cpp (initialization, diagnostics, mesh output)         │
│  ─ types.cpp (global variable definitions)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Flow

```
User runs:  python scripts/run.py input3.bin input3.bin --device gpu --wg-size 24 --prob-size 24
                │
                ▼
        scripts/run.py compiles all C++ into build/sycl_pic_launcher
                │
                ▼
        build/sycl_pic_launcher starts
                │
                ├── PyImport_AppendInittab("sycl_pic", ...)  ← register built-in module
                ├── Py_Initialize()                           ← start Python interpreter
                └── PyRun_SimpleFile("src/main.py")           ← hand control to Python
                        │
                        ▼
                src/main.py executes:
                    import sycl_pic
                    sycl_pic.init(...)        ← SYCL queue + Pardiso init
                    rho = sycl_pic.get_rho()  ← zero-copy NumPy view
                    phi = sycl_pic.get_phi()  ← zero-copy NumPy view
                    for iteration in range(timesteps):
                        sycl_pic.charge_deposition(sp)
                        phi[:] = rand_int * rho   ← Python writes to GPU memory
                        sycl_pic.poisson_solve(iteration)
                        sycl_pic.move_particles(sp, iteration)
                        ...
```

## Zero-Copy Python Interop

The key technical achievement is **zero-copy bidirectional data sharing** between Python/NumPy and SYCL C++:

### How It Works

1. **SYCL shared memory allocation** (in C++):
   ```cpp
   rho = sycl::malloc_shared<double>(GRID_X * GRID_Y, queue);
   phi = sycl::malloc_shared<double>(GRID_X * GRID_Y, queue);
   ```

2. **NumPy array wrapping** (in `sycl_pic_module.cpp`):
   ```cpp
   // Wrap raw pointer as NumPy array — NO data copied
   PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)rho);
   // Ensure Python doesn't free SYCL-owned memory
   PyArray_CLEARFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
   ```

3. **Python-side modification** (in `main.py`):
   ```python
   rho = sycl_pic.get_rho()  # NumPy view into SYCL shared memory
   phi = sycl_pic.get_phi()  # NumPy view into SYCL shared memory
   phi[:] = rand_int * rho   # Writes directly to GPU-accessible memory
   ```

4. **C++ reads the same memory** — the Poisson solver sees the values Python wrote:
   ```cpp
   poissonSolver(rho, phi, iter, queue);  // phi already contains Python's values
   ```

### Memory Layout

```
SYCL malloc_shared memory (accessible by both CPU and GPU):

    ┌──────────────────────────────────┐
    │  rho[0] rho[1] ... rho[128756]  │  ← 128,757 doubles (501×257 grid)
    └──────────────────────────────────┘
         ▲                    ▲
         │                    │
    C++ writes here      Python/NumPy reads here
    (charge deposition)  (phi[:] = rand_int * rho)

    ┌──────────────────────────────────┐
    │  phi[0] phi[1] ... phi[128756]  │  ← 128,757 doubles (501×257 grid)
    └──────────────────────────────────┘
         ▲                    ▲
         │                    │
    Python writes here   C++ reads here
    (phi[:] = rand*rho)  (Poisson solver)
```

**No memcpy, no serialization, no IPC** — both languages access the same physical memory.

## Simulation Physics

This is a 2D electrostatic Particle-in-Cell simulation modeling:

| Parameter | Value |
|-----------|-------|
| Simulation domain | 0.025 m x 0.0128 m |
| Grid | 501 x 257 = 128,757 cells |
| Timestep | 5 x 10^-12 s |
| Species | 2 (electrons + Xe+ ions) |
| Particles per species | 9.6 x 10^6 simulation particles |
| Actual particles represented | 5 x 10^16 per species |
| Gas | Xenon (Xe) at 0.6 Pa |
| Electron temperature | 10 eV |
| Ion temperature | 0.5 eV |
| Boundary conditions | Periodic |
| Boundary voltages | 200 V (left), 0 V (right, top, bottom) |

### PIC Simulation Loop (per timestep)

1. **Zero charge density** — `memset(rho, 0, ...)`
2. **Charge deposition** — interpolate particle charges onto grid (GPU kernel)
3. **Phi modification** — `phi = rand_int * rho` (Python, zero-copy)
4. **Poisson solve** — solve nabla^2(phi) = -rho/epsilon_0 using MKL Pardiso (CPU, sparse direct solver)
5. **Particle mover** — advance positions and velocities using electric field (GPU kernel)
6. **Particle compaction** — defragment particle arrays (GPU kernel)
7. **Hashmap cleanup** — reset particle-to-grid mapping (GPU kernel)
8. **New particle generation** — create electron-ion pairs to maintain quasi-neutrality
9. **Diagnostics** — output density, energy, electric field, potential

## Project Structure

```
SYCL-PIC/
├── src/
│   ├── main.py                 # Python simulation driver (replaces main.cpp)
│   ├── launcher.cpp            # C++ executable that embeds Python
│   ├── sycl_pic_module.cpp     # Python C extension (18 wrapped functions)
│   ├── main.cpp                # Original C++ driver (kept for reference)
│   ├── sycl_pic_ctypes.cpp     # Alternative ctypes wrapper (experimental)
│   ├── interpolation.cpp       # Charge deposition GPU kernels
│   ├── mover.cpp               # Particle movement GPU kernels
│   ├── poissonSolver.cpp       # MKL Pardiso electrostatic solver
│   ├── particles.cpp           # Species registration, binary I/O
│   ├── utils.cpp               # Initialization, diagnostics, output
│   └── types.cpp               # Global variable definitions
├── include/
│   ├── types.hpp               # Data structures, global declarations
│   ├── particles.hpp           # Particle registration interface
│   ├── species_ds.hpp          # GridParams, Particle struct definitions
│   ├── point.hpp               # Point (particle) data structure
│   ├── mover.hpp               # Particle mover interface
│   ├── interpolation.hpp       # Charge deposition interface
│   ├── utils.hpp               # Utility function interface
│   ├── poissonSolver.hpp       # Poisson solver interface
│   └── sycl_hashmap.hpp        # Custom SYCL GPU hashmap (linear probing)
├── scripts/
│   └── run.py                  # Build and run automation script
├── M_PICDAT.DAT                # Simulation configuration file
├── geometry.conf               # Boundary geometry definition
├── Makefile                    # Legacy build system
├── CMakeLists.txt              # CMake build system
└── README.md                   # This file
```

## Key Source Files

### `src/main.py` — Python Simulation Driver

The main simulation loop, fully replacing `main.cpp`. Calls all C++ SYCL functions through the `sycl_pic` module. Performs `phi = rand_int * rho` in Python using zero-copy NumPy views. Accumulates timing for each phase and prints a detailed report.

### `src/sycl_pic_module.cpp` — Python C Extension

Wraps 18 C++ simulation functions as callable Python methods:

| Python Function | C++ Equivalent | Description |
|----------------|----------------|-------------|
| `sycl_pic.init()` | `initVariables()` + `initPoissonSolver()` | Full initialization |
| `sycl_pic.get_rho()` | `PyArray_SimpleNewFromData(rho)` | Zero-copy NumPy view of charge density |
| `sycl_pic.get_phi()` | `PyArray_SimpleNewFromData(phi)` | Zero-copy NumPy view of potential |
| `sycl_pic.charge_deposition(sp)` | `interpolate()` | GPU charge deposition |
| `sycl_pic.poisson_solve(iter)` | `poissonSolver()` | MKL Pardiso solve |
| `sycl_pic.move_particles(sp, iter)` | `mover()` | GPU particle advance |
| `sycl_pic.compact_particles(sp)` | `compact_particles()` | GPU array defragmentation |
| `sycl_pic.cleanup_map(sp)` | `cleanup_map()` | GPU hashmap reset |
| `sycl_pic.generate_new_particles()` | `generate_new_electrons()` + `generate_pairs()` | Pair generation |
| `sycl_pic.print_diagnostics(dir, iter)` | `printNumberDensity()` etc. | Output diagnostics |
| `sycl_pic.save_meshes()` | File output loop | Write Mesh_*.out |
| `sycl_pic.finalize()` | `freeVariables()` | Free SYCL memory |

### `src/launcher.cpp` — Embedded Python Launcher

A thin C++ executable (~90 lines) that:
1. Registers `sycl_pic` as a built-in Python module via `PyImport_AppendInittab()`
2. Initializes the embedded Python interpreter
3. Passes command-line arguments through to `sys.argv`
4. Runs `src/main.py` via `PyRun_SimpleFile()`

This design is critical — it avoids `dlopen` entirely, which solves the MKL Pardiso crash (see Technical Challenges).

### `scripts/run.py` — Build and Run Script

Automates compilation and execution:
- Detects the Intel DPC++ compiler
- Compiles all C++ sources + launcher into a single executable
- Links against SYCL, MKL (ILP64), TBB, Python, and NumPy
- Sets up `LD_LIBRARY_PATH`, `PYTHONHOME`, `PYTHONPATH` for the launcher

## Technical Challenges Solved

### 1. MKL Pardiso Segfault with dlopen

**Problem:** Intel MKL's Pardiso sparse direct solver crashes with a segmentation fault when called from any shared library loaded via `dlopen()`. This affects:
- Python C extensions loaded via `import` (Python uses `dlopen` internally)
- `ctypes.CDLL()` (explicit `dlopen`)
- PyBind11, Cython, and all other binding approaches

**Root Cause:** MKL Pardiso's internal thread initialization requires symbols resolved at static link time. When loaded via `dlopen`, the lazy symbol resolution causes Pardiso's internal state to be corrupted.

**Approaches Attempted and Failed:**
- `LD_PRELOAD` with MKL libraries
- `ctypes.CDLL` with `RTLD_GLOBAL` flag
- `sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)`
- `memset` of Pardiso arrays before initialization
- Linker flags: `-Wl,-z,now -Wl,--no-as-needed`

**Solution:** The `launcher.cpp` approach — compile the C extension directly into the executable and register it as a built-in module via `PyImport_AppendInittab()`. This resolves all MKL symbols at link time, completely bypassing `dlopen`.

### 2. Zero-Copy Memory Safety

**Problem:** When wrapping SYCL `malloc_shared` pointers as NumPy arrays, Python's garbage collector could attempt to `free()` the SYCL-owned memory, causing a double-free crash.

**Solution:** After creating the NumPy array with `PyArray_SimpleNewFromData()`, immediately clear the ownership flag:
```cpp
PyArray_CLEARFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
```
This tells NumPy the array does not own its data buffer, preventing any deallocation.

### 3. In-Place Array Writes

**Problem:** The Python assignment `phi = rand_int * rho` creates a NEW array and rebinds the variable — it doesn't write into the SYCL shared memory buffer.

**Solution:** Use slice assignment: `phi[:] = rand_int * rho`. The `[:]` forces NumPy to write element-by-element into the existing buffer rather than creating a new allocation.

### 4. SYCL on NVIDIA GPUs with Intel DPC++

**Problem:** Intel DPC++ 2025.3 dropped the CUDA adapter (`libur_adapter_cuda.so`), failing to target NVIDIA GPUs.

**Solution:** Pin to Intel DPC++ 2025.1 (`/opt/intel/oneapi/compiler/2025.1/bin/icx`) which includes the CUDA backend. Compile with:
```
-fsycl -fsycl-targets=nvptx64-nvidia-cuda
```

## Hardware and Software Environment

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA RTX 6000 Ada Generation |
| **CUDA** | 12.4 |
| **Compiler** | Intel DPC++ 2025.1 (icx) with CUDA adapter |
| **MKL** | Intel oneAPI MKL 2025.1 (ILP64 interface) |
| **TBB** | Intel oneAPI TBB 2022.3 |
| **DPL** | Intel oneAPI DPL 2022.8 |
| **Python** | Anaconda Python 3.12 |
| **NumPy** | (from Anaconda distribution) |
| **OS** | Linux |

## Build and Run Instructions

### Prerequisites

- Intel oneAPI DPC++ Compiler 2025.1 with CUDA adapter
- Intel MKL 2025.1
- Anaconda Python 3.12 with NumPy
- NVIDIA GPU with CUDA 12.x drivers

### Quick Start

```bash
cd SYCL-PIC

# Compile and run (one command)
python scripts/run.py input3.bin input3.bin --device gpu --wg-size 24 --prob-size 24
```

### Separate Compile and Run

```bash
# Compile only
python scripts/run.py input3.bin input3.bin --device gpu --wg-size 24 --prob-size 24 --compile-only

# Run only (after compilation)
python scripts/run.py input3.bin input3.bin --device gpu --wg-size 24 --prob-size 24 --run-only
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `input_file1` | Electron particle binary input file (e.g., `input3.bin`) |
| `input_file2` | Ion particle binary input file (e.g., `input3.bin`) |
| `--device` | Target device: `cpu` or `gpu` (default: `gpu`) |
| `--wg-size` | SYCL work-group size (default: 24) |
| `--prob-size` | Hashmap probing size (default: 24) |
| `--compile-only` | Only compile, do not run |
| `--run-only` | Only run (assumes already compiled) |

## Simulation Parameters

Simulation parameters are defined in `M_PICDAT.DAT`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| TIMESTEPS | 1000 | Number of simulation iterations |
| XMAX | 0.025 m | Domain width |
| YMAX | 0.0128 m | Domain height |
| NX | 500 | Cells in X |
| NY | 256 | Cells in Y |
| dt | 5e-12 s | Time step |
| Te | 10 eV | Electron temperature |
| Ti | 0.5 eV | Ion temperature |
| Pressure | 0.6 Pa | Gas pressure |
| Gas | Xe (Xenon) | Simulation gas |
| printInterval | 10 | Diagnostic output interval |

Boundary geometry is defined in `geometry.conf` (Penning trap with 200 V on left electrode).

## Output Files

| File | Description |
|------|-------------|
| `Mesh_0.out` | Final electron charge deposition mesh (501 x 257) |
| `Mesh_1.out` | Final ion charge deposition mesh (501 x 257) |
| `particle_trajectory.out` | Particle trajectory tracking |
| `out/electronDensity/` | Electron number density snapshots |
| `out/ionDensity/` | Ion number density snapshots |
| `out/electronenergy/` | Electron energy snapshots |
| `out/ionenergy/` | Ion energy snapshots |
| `out/potential/` | Electrostatic potential snapshots |
| `out/Ex/`, `out/Ey/` | Electric field component snapshots |
| `out/electronNumber/` | Electron count per cell |
| `out/ionNumber/` | Ion count per cell |

## Results

### Performance (10 iterations, 501x257 grid, 9.6M particles/species)

| Phase | Time (s) |
|-------|----------|
| Charge Deposition | 0.544 |
| Particle Mover | 0.342 |
| Poisson Solver (MKL Pardiso) | 0.781 |
| Map Cleanup | ~0.1 |
| Particle Compaction | ~0.05 |
| New Particle Generation | ~0.05 |
| **Total** | **~2.13** |

### Zero-Copy Verification

The zero-copy interop was verified by computing `phi = rand_int * rho` in Python and checking every element in C++:
- **128,757 grid elements** checked per iteration
- **10 iterations** verified
- **0 mismatches** — all elements satisfy `|phi[i] - rand_int * rho[i]| < 1e-12`

This confirms that Python's NumPy array and C++'s raw pointer access the identical memory with no data corruption.

## Work Done Summary

### Phase 1: Baseline C++ Simulation
- Studied the existing SYCL-PIC codebase (`main.cpp`, 7 C++ source files, 9 headers)
- Understood the PIC simulation loop: charge deposition → Poisson solve → particle move → compaction
- Identified all global state, GPU kernels, and MKL Pardiso dependencies

### Phase 2: Zero-Copy Python Interop in C++ (Proof of Concept)
- Embedded Python inside `main.cpp` using `Py_Initialize()` / `PyRun_File()`
- Wrapped `rho` and `phi` SYCL shared memory pointers as NumPy arrays via `PyArray_SimpleNewFromData()`
- Created `scripts/modify_phi.py` to compute `phi = rand_int * rho` from Python
- Added C++ verification loop to confirm zero-copy correctness
- **Result**: All 128,757 grid elements matched across 10 iterations

### Phase 3: Created Python C Extension Module
- Wrote `src/sycl_pic_module.cpp` — a Python C extension wrapping all 18 simulation functions
- Each function uses Python C API (`PyArg_ParseTuple`, `Py_BuildValue`, `PyFloat_FromDouble`) to bridge Python and C++
- Zero-copy array access via `PyArray_SimpleNewFromData()` with `PyArray_CLEARFLAGS(NPY_ARRAY_OWNDATA)`

### Phase 4: Explored Multiple C++/Python Integration Methods
- Investigated 11+ integration approaches (PyBind11, Cython, ctypes, DLPack, CUDA Array Interface, Gymnasium, EnvPool, Brax/JAX, DM Control, Isaac Lab, Numba, Triton)
- Identified that **all dlopen-based approaches fail** due to MKL Pardiso's requirement for static symbol resolution

### Phase 5: Solved MKL Pardiso Crash — Launcher Executable Approach
- Created `src/launcher.cpp` — a thin C++ executable that registers `sycl_pic` as a built-in Python module via `PyImport_AppendInittab()`
- This compiles the C extension **into the executable**, bypassing `dlopen` entirely
- MKL Pardiso initializes and runs successfully
- **This was the critical breakthrough** that enabled the full Python driver

### Phase 6: Python Driver (main.py) Replaces main.cpp
- Wrote `src/main.py` — complete Python simulation driver with 1:1 functional parity to `main.cpp`
- All simulation logic (loop, timing, diagnostics) in Python
- `phi = rand_int * rho` computed natively in NumPy (3 lines vs 80+ lines in the C++ embedded Python approach)
- Rewrote `scripts/run.py` to compile the launcher executable and run simulations

### Phase 7: Verification and Successful Execution
- Full simulation runs end-to-end with Python as the driver
- All 10 iterations complete successfully
- Output files (`Mesh_0.out`, `Mesh_1.out`, diagnostics) generated correctly
- Total execution time: ~2.13 seconds (comparable to pure C++ driver)
- Zero-copy verified: no data mismatches between Python and C++

---

**Authors**: Manya & Vansh
**Course**: BTech Mini Project
**Hardware**: NVIDIA RTX 6000 Ada GPU, Intel oneAPI DPC++ 2025.1

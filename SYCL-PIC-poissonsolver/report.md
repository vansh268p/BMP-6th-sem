# Integration Report: ML Poisson Solver in SYCL-PIC

## 1. Objective

This report documents the integration of a machine-learning Poisson solver into the SYCL-PIC runtime so that:

- charge density rho is produced by the SYCL kernels,
- rho is passed to the ML solver using zero-copy NumPy views,
- predicted phi is written back in-place to SYCL shared memory,
- particle mover continues using updated electric fields derived from the predicted phi,
- the original external predictor file remains unchanged,
- integration logic stays outside main.py as a separate module.

## 2. Final Integration Design

The runtime now uses this flow:

1. SYCL kernels compute rho in C++ and expose it to Python as a zero-copy array.
2. main.py calls a separate bridge module, not inline model code.
3. Bridge module loads src/predict_poisson_ml.py (copied and adapted from the external project).
4. ML predictor reads rho and writes phi in-place to the same shared memory buffer.
5. C-extension recomputes electric field from phi using update_fields_from_phi().
6. Mover and remaining simulation steps proceed as before.

## 3. Files Added and Updated

### 3.1 New files

- src/poisson_ml_bridge.py
  - New integration bridge that dynamically loads the predictor script and runs in-memory inference.
  - Uses absolute paths for external config and checkpoint locations by default:
    - /home/student/Dhruvil_Om_BMP_2026/PoissonSolver/conf/config_deeponet.yaml
    - /home/student/Dhruvil_Om_BMP_2026/PoissonSolver/outputs_poisson/checkpoints

- src/predict_poisson_ml.py
  - Local copy of external predictor script.
  - Extended with reusable in-memory APIs for repeated simulation calls:
    - PoissonPredictor
    - predict_from_array
    - predict_into_arrays
  - Keeps model loading cached for better runtime performance.

### 3.2 Updated files

- src/main.py
  - Added robust import path handling for embedded Python:
    - inserts its own src directory into sys.path.
  - Imports ExternalPoissonMLSolver from the separate bridge module.
  - Gets zero-copy rho and phi arrays via sycl_pic.get_rho() and sycl_pic.get_phi().
  - Replaces direct C++ Poisson solve call with ML solve call:
    - ml_poisson_solver.solve_inplace(rho, phi, grid_x, grid_y, iteration)
  - Calls sycl_pic.update_fields_from_phi() after ML prediction.

- src/sycl_pic_module.cpp
  - Added new methods exposed to Python:
    - get_grid_x()
    - get_grid_y()
    - update_fields_from_phi()
  - update_fields_from_phi() applies boundary overwrite and recomputes electric field from current phi.

- src/launcher.cpp
  - Removed hardcoded Python home assumption.
  - Now reads PYTHONHOME from environment and uses it for embedded interpreter setup.
  - This enables selecting a non-base conda environment at runtime.

- scripts/run.py
  - Major environment alignment updates so compile and runtime both use the same embedded Python environment.
  - Added helper methods:
    - get_embed_python_prefix()
    - get_embed_python_info(prefix)
  - Compile stage now queries include/lib/numpy paths from selected environment Python.
  - Runtime stage now sets:
    - PYTHONHOME
    - PYTHONPATH
    - LD_LIBRARY_PATH
    consistently from that environment.
  - Added fallback for libpython shared library discovery when sysconfig points to static archive.

## 4. Original External File Handling

Per requirement, this file was reverted and kept untouched for integration:

- /home/student/Dhruvil_Om_BMP_2026/PoissonSolver/predict_poisson.py

All custom integration changes were moved into:

- src/predict_poisson_ml.py

## 5. Zero-Copy Data Path

### 5.1 rho and phi ownership

- rho and phi are SYCL-owned shared-memory buffers.
- sycl_pic_module.cpp wraps them as NumPy arrays without copy.
- Python receives views to the same memory addresses.

### 5.2 Inference write-back

- Bridge reshapes flat views to (grid_y, grid_x) using NumPy reshape views.
- predict_into_arrays writes predicted phi directly into caller-provided phi_out.
- No intermediate memcpy back into C++ is required.

## 6. Why update_fields_from_phi() was added

Previously poisson_solve() in C++ did two things:

1. solved for phi,
2. recalculated electric field from phi.

After switching to ML phi prediction, step 2 still must happen before mover. The new update_fields_from_phi() API preserves that physical dependency by reusing existing electric-field update logic in C++.

## 7. Environment Activation Fix

The failure ModuleNotFoundError: No module named physicsnemo.sym happened because embedded Python was effectively running with base stdlib/site-packages while the model package lived in another environment.

Fix implemented:

- scripts/run.py now resolves Python include/lib/site-packages from selected conda env.
- launcher respects PYTHONHOME from run.py.
- compile and run phases now target the same Python ABI/runtime.

Default selected environment prefix:

- /home/student/.conda/envs/physicsnemo

Override supported via environment variable:

- PHYSICSNEMO_ENV_PREFIX

## 8. Run Instructions

From project root:

- Compile:
  - python scripts/run.py input3.bin input3.bin --device gpu --wg-size 24 --prob-size 24 --compile-only

- Run:
  - python scripts/run.py input3.bin input3.bin --device gpu --wg-size 24 --prob-size 24 --run-only

Optional environment override example:

- PHYSICSNEMO_ENV_PREFIX=/home/student/.conda/envs/physicsnemo python scripts/run.py input3.bin input3.bin --device gpu --wg-size 24 --prob-size 24 --run-only

## 9. Validation Status

Validated outcomes:

- The prior bridge import error for poisson_ml_bridge was fixed.
- The prior physicsnemo.sym import error was fixed by environment alignment.
- Runtime progressed through many iterations in smoke run after fixes.

Note:

- Full long-run numerical validation against C++ Poisson baseline was not executed in this integration pass.

## 10. Important Assumptions and Constraints

- Predictor currently expects 257x257 input/output shape.
- This integration assumes simulation grid matches model grid.
- If grid resolution changes, predictor preprocessing/model must be adapted.

## 11. Summary

The SYCL-PIC pipeline now uses an external ML Poisson solver in a clean, modular, zero-copy manner:

- separate bridge module,
- separate predictor file in src,
- unchanged original external predictor,
- C-extension support for post-phi field update,
- robust embedded Python environment activation for PhysicsNeMo.

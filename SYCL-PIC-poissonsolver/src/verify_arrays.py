#!/usr/bin/env python3
"""
verify_arrays.py — Verify rho/phi arrays are passed correctly between C++ and Python.

This script runs inside the simulation (called from main.py) to check that
the numpy arrays returned by sycl_pic.get_rho() / get_phi() point to the
exact same memory as the C++ rho/phi pointers.

How it works:
  1. From Python: dump rho/phi numpy arrays to binary files
  2. From C++: call sycl_pic.dump_rho_phi_cpp() to dump the same arrays
  3. Compare the two binary files byte-for-byte
  If they match → zero-copy is working. If not → something is wrong.
"""

import os
import numpy as np


def verify_zero_copy(rho, phi, grid_x, grid_y, sycl_pic_module, prefix="./verify", iteration=0):
    """
    Dump rho/phi from both Python and C++ sides, compare byte-for-byte.
    Returns True if arrays match exactly on both sides.
    """
    os.makedirs(prefix, exist_ok=True)
    tag = f"{prefix}/iter{iteration:04d}"

    # ── Python side: dump raw bytes ─────────────────────────────
    rho_py_file = f"{tag}_rho_python.bin"
    phi_py_file = f"{tag}_phi_python.bin"

    rho_arr = np.asarray(rho)
    phi_arr = np.asarray(phi)

    rho_arr.tofile(rho_py_file)
    phi_arr.tofile(phi_py_file)

    # ── C++ side: dump raw bytes ────────────────────────────────
    sycl_pic_module.dump_rho_phi_cpp(tag)
    # This creates: {tag}_rho_cpp.bin and {tag}_phi_cpp.bin

    rho_cpp_file = f"{tag}_rho_cpp.bin"
    phi_cpp_file = f"{tag}_phi_cpp.bin"

    # ── Compare byte-for-byte ───────────────────────────────────
    rho_match = _compare_files(rho_py_file, rho_cpp_file, "rho", iteration)
    phi_match = _compare_files(phi_py_file, phi_cpp_file, "phi", iteration)

    return rho_match and phi_match


def _compare_files(py_file, cpp_file, name, iteration):
    """Compare two binary files byte-for-byte."""
    with open(py_file, "rb") as f:
        py_bytes = f.read()
    with open(cpp_file, "rb") as f:
        cpp_bytes = f.read()

    if len(py_bytes) != len(cpp_bytes):
        print(f"[VERIFY] ✗ {name} iter={iteration}: SIZE MISMATCH "
              f"(Python={len(py_bytes)} bytes, C++={len(cpp_bytes)} bytes)")
        return False

    if py_bytes == cpp_bytes:
        n_doubles = len(py_bytes) // 8
        print(f"[VERIFY] ✓ {name} iter={iteration}: EXACT MATCH "
              f"({n_doubles} doubles, {len(py_bytes)} bytes)")
        return True
    else:
        # Find where they differ
        py_arr = np.frombuffer(py_bytes, dtype=np.float64)
        cpp_arr = np.frombuffer(cpp_bytes, dtype=np.float64)
        diff = np.abs(py_arr - cpp_arr)
        n_diff = np.count_nonzero(diff)
        print(f"[VERIFY] ✗ {name} iter={iteration}: MISMATCH "
              f"({n_diff}/{len(py_arr)} elements differ, max_diff={diff.max():.6e})")
        return False

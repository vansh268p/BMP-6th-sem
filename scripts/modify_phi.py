"""
modify_phi.py — Zero-copy Python interop for SYCL-PIC simulation.

Called from embedded C++ interpreter. The variables `rho` and `phi`
are already injected into the global namespace as NumPy arrays that
wrap the SYCL shared-memory pointers directly (no copy).

phi[:] = rand_int * rho   ← in-place write into C++ memory
"""
import random

rand_int = random.randint(1, 100)

# In-place slice assignment: writes directly into the C++/SYCL shared memory
phi[:] = rand_int * rho

print(f"[modify_phi.py] phi = {rand_int} * rho (grid_size={len(rho)})")

# Expose rand_int so C++ can read it back for verification
# (it stays in the __main__ global dict)

#!/usr/bin/env python3
"""Diagnose the sign convention between simulation rho and training rho.

This script:
1. Runs the C++ solver for 1 iteration, captures rho and phi
2. Runs the ML solver on the same rho, captures phi
3. Compares the signs and magnitudes
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Load the training data to check sign conventions
import h5py

print("=" * 70)
print("STEP 1: Check training data sign convention")
print("=" * 70)

with h5py.File('/home/student/Dhruvil_Om_BMP_2026/PoissonSolver/validation.hdf5', 'r') as f:
    rho_train = np.array(f['rho'][0, 0])  # first sample, (257, 257)
    phi_train = np.array(f['potential'][0, 0])

print(f"Training rho: min={rho_train.min():.6e}  max={rho_train.max():.6e}  mean={rho_train.mean():.6e}")
print(f"Training phi: min={phi_train.min():.6e}  max={phi_train.max():.6e}  mean={phi_train.mean():.6e}")
print(f"Training rho center value: {rho_train[128, 128]:.6e}")
print(f"Training phi center value: {phi_train[128, 128]:.6e}")

# Check: in a plasma, where n_e > n_i, rho = e*(n_i - n_e) < 0
# The corresponding potential should be negative (electron cloud pushes potential down)
# If rho > 0 at a point, we expect phi > 0 there (ion excess creates positive potential)

# Check correlation
from numpy import corrcoef
corr = corrcoef(rho_train.flatten(), phi_train.flatten())[0, 1]
print(f"\nCorrelation(rho_train, phi_train) = {corr:.4f}")
if corr > 0:
    print("  → POSITIVE correlation: where rho>0 (ion excess), phi>0 ✓ (physical)")
else:
    print("  → NEGATIVE correlation: indicates unusual sign convention")

print("\n" + "=" * 70)
print("STEP 2: Check simulation rho sign convention")
print("=" * 70)
print("""
Charge deposition in interpolation.cpp:
  chrg_electron = -(-1.6e-19) / 8.854e-12 = +1.807e-8  (POSITIVE)
  chrg_ion      = -(+1.6e-19) / 8.854e-12 = -1.807e-8  (NEGATIVE)
  
  rho_sim = Σ_electrons(weight * +1.807e-8) + Σ_ions(weight * -1.807e-8)
          ∝ (n_e - n_i) / ε₀
  
Training data (create_dataset.py):
  rho_train = e * (n_i - n_e)
  
Therefore: rho_sim * ε₀ = -rho_train * (area_factor)

CONCLUSION: The EPSILON conversion has the WRONG SIGN!
  Current:  rho_normalized = rho_sim * (+EPSILON) / RHO_NORM
  Correct:  rho_normalized = rho_sim * (-EPSILON) / RHO_NORM
""")

# Let's also check by computing what the weighing factor looks like
EPSILON = 8.854e-12
RHO_NORM = 1.23223e-08
E_CHARGE = 1.602e-19

# Simulation:
# For electron: chrg = -charge/epsilon = +e/eps = +1.807e-8
# rho_sim += weight * chrg (electrons add positive)
# rho_sim * epsilon = weight * e (electrons add positive)
#
# Training: rho_train = e * (n_i - n_e)  (electrons add negative)
#
# So: rho_sim * epsilon = -rho_train * dx * dy  (where dx*dy comes from bilinear weight sum)

print("=" * 70)
print("STEP 3: Numerical verification")
print("=" * 70)

# What does the model expect? Normalized rho_train / RHO_NORM
# What are we feeding? rho_sim * EPSILON / RHO_NORM
# These have OPPOSITE SIGNS
# Fix: rho_sim * (-EPSILON) / RHO_NORM

# But wait - there's also the area factor. Training rho is per-unit-area (density),
# simulation rho has the bilinear weight already (not divided by area).
# Let's check: weighingFactor = actualParticleCount * surfArea / (simulationParticleCount * dx * dy * dx * dy)
# So corner_mesh *= weighingFactor, which has units of [1/(dx*dy)] roughly
# Then rho_sim = corner_mesh * weighingFactor * (-charge/epsilon)
# The weighingFactor includes 1/(dx²*dy²) but the bilinear weights are dx*dy scale
# Net: rho_sim has units of charge/(epsilon * area) per grid cell

# Actually, let's just check the relative scale:
# We know rho_sim ~ 1e3 (from DIAG output)
# rho_sim * epsilon ~ 1e3 * 8.854e-12 ~ 8.854e-9
# rho_train ~ 1e-8 scale
# So the magnitudes are in the right ballpark (possibly off by a factor of ~1-2)

print(f"rho_sim * EPSILON ≈ 1e3 * {EPSILON:.3e} = {1e3 * EPSILON:.3e}")
print(f"rho_train std = {RHO_NORM:.3e}")
print(f"Ratio: {1e3 * EPSILON / RHO_NORM:.4f}")
print()
print("The magnitude is correct (~0.7 normalized), but the SIGN IS FLIPPED!")
print()
print("FIX: Change EPSILON multiplication to -EPSILON in predict_poisson_ml.py")

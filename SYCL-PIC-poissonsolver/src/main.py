#!/usr/bin/env python3
"""
main.py — Python driver for the SYCL-PIC particle simulation.

All heavy SYCL/GPU work is done inside the `sycl_pic` C extension module,
which is registered as a built-in module by `launcher.cpp`.  Arrays
(rho, phi) are accessible via zero-copy NumPy views into SYCL shared
memory — no data is ever copied between C++ and Python.

Usage (called by launcher executable):
    ./build/sycl_pic_launcher <input1> <input2> <cpu|gpu> <wg_size> <prob_size>
"""

import sys
import time
from pathlib import Path
import numpy as np
import sycl_pic

# Keep ML Poisson logic in a separate file/module as requested.
# Embedded Python does not always include this script directory in sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poisson_ml_bridge import ExternalPoissonMLSolver
from verify_arrays import verify_zero_copy


def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <input1> <input2> <cpu|gpu> <wg_size> <prob_size>",
              file=sys.stderr)
        sys.exit(1)

    input1    = sys.argv[1]
    input2    = sys.argv[2]
    device    = sys.argv[3]
    wg_size   = int(sys.argv[4])
    prob_size = int(sys.argv[5])

    # ── Initialisation ──────────────────────────────────────────
    print(f"Work-Group Size: {wg_size}")
    print(f"Probing Size: {prob_size}")

    sycl_pic.init(device, wg_size, prob_size, input1, input2)

    timesteps      = sycl_pic.get_timesteps()
    print_interval = sycl_pic.get_print_interval()
    num_species    = sycl_pic.get_num_species()
    grid_size      = sycl_pic.get_grid_size()
    grid_x         = sycl_pic.get_grid_x()
    grid_y         = sycl_pic.get_grid_y()

    # Zero-copy NumPy views into SYCL shared memory.
    rho = sycl_pic.get_rho()
    phi = sycl_pic.get_phi()

    ml_poisson_solver = ExternalPoissonMLSolver()

    print(
        f"Grid size: {grid_size} ({grid_x}x{grid_y})  "
        f"Timesteps: {timesteps}  Species: {num_species}"
    )

    # ── Timing accumulators ─────────────────────────────────────
    t_charge   = 0.0
    t_mover    = 0.0
    t_compact  = 0.0
    t_alloc    = 0.0
    t_scan     = 0.0
    t_cleanup  = 0.0
    t_poisson  = 0.0
    t_print    = 0.0
    t_newpart  = 0.0

    # ── Main simulation loop ────────────────────────────────────
    for iteration in range(timesteps):
        ne, ni = sycl_pic.get_particle_counts()
        print(f"Iteration: {iteration}  Particles(e/i): {ne} {ni}")

        # 1. Zero rho
        sycl_pic.zero_rho()

        # 2. Charge deposition for each species
        for sp in range(num_species):
            t_charge += sycl_pic.charge_deposition(sp)

        # ── Verify rho BEFORE ML solver (first 3 iterations) ───
        # if iteration < 3:
        #     print(f"\n[VERIFY] === Checking rho/phi BEFORE ML solver (iter {iteration}) ===")
        #     verify_zero_copy(rho, phi, grid_x, grid_y, sycl_pic,
        #                      prefix="./verify_dumps/before_ml", iteration=iteration)

        # ── Diagnostic: check rho/phi scales (first 3 iters) ────
        if iteration < 3:
            rho_2d = rho.reshape(grid_y, grid_x)
            print(f"[DIAG iter={iteration}] rho  min={rho_2d.min():.6e}  max={rho_2d.max():.6e}  mean={rho_2d.mean():.6e}")

        # 3. Poisson solver — ML surrogate with periodic C++ correction
        #    iteration 0 MUST use C++ to initialize Pardiso factorization (phases 11+22)
        t_poisson += ml_poisson_solver.solve_inplace(rho, phi, grid_x, grid_y, iteration)
        sycl_pic.update_fields_from_phi()

        # Hybrid: C++ Pardiso reset every 10 iterations to correct accumulated ML error (comment to disable)
        # if iteration > 0 and iteration % 10 == 0: t_poisson += sycl_pic.poisson_solve(iteration)

        # 3. Poisson solver CPP (standalone, disabled when ML is active)
        # t_poisson += sycl_pic.poisson_solve(iteration)

        # ── Verify phi AFTER ML solver (first 3 iterations) ────
        # if iteration < 3:
        #     print(f"\n[VERIFY] === Checking rho/phi AFTER ML solver (iter {iteration}) ===")
        #     verify_zero_copy(rho, phi, grid_x, grid_y, sycl_pic,
        #                      prefix="./verify_dumps/after_ml", iteration=iteration)

        # 4. Mover + compaction + cleanup for each species
        for sp in range(num_species):
            t_mover += sycl_pic.move_particles(sp, iteration)

            t0 = time.perf_counter()
            # (energy printing placeholder — currently commented out in C++)
            t_print += time.perf_counter() - t0

            # Compaction (at print intervals)
            if iteration % print_interval == 0:
                elapsed, at, st = sycl_pic.compact_particles(sp)
                t_compact += elapsed
                t_alloc   += at
                t_scan    += st

            # Cleanup map (every iteration)
            t_cleanup += sycl_pic.cleanup_map(sp)

        # 5. Generate new particles
        # t_newpart += sycl_pic.generate_new_particles()

        # 6. Diagnostics
        if iteration % print_interval == 0:
            sycl_pic.print_diagnostics("./out", iteration)

    # ── Post-loop ───────────────────────────────────────────────
    sycl_pic.save_meshes()
    sycl_pic.finalize()

    # ── Timing report ───────────────────────────────────────────
    print(f"Charge Deposition time: {t_charge} seconds")
    print(f"Mover time: {t_mover} seconds")
    print(f"Particle Compaction time: {t_compact} seconds")
    print(f"Allocation time in Compaction: {t_alloc} seconds")
    print(f"Scan time in Compaction: {t_scan} seconds")
    print(f"Map Cleanup time: {t_cleanup} seconds")
    print(f"Poisson Solver time: {t_poisson} seconds")
    print(f"Print time: {t_print} seconds")
    print(f"New Particle Generation time: {t_newpart} seconds")
    total = t_charge + t_mover + t_compact + t_cleanup + t_poisson + t_print + t_newpart
    print(f"Total execution time: {total} seconds")


if __name__ == "__main__":
    main()

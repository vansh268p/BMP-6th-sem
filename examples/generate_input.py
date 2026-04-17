#!/usr/bin/env python3
"""
Example Input File Generator for SYCL Particle Simulation
========================================================

This script generates sample binary input files for testing the simulation.
"""

import struct
import random
import argparse

def generate_input_file(filename, nx=64, ny=64, num_points=1000, maxiter=100):
    """Generate a binary input file with random particle positions."""

    print(f"Generating {filename}...")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Particles: {num_points}")
    print(f"  Max iterations: {maxiter}")

    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('i', nx))        # NX
        f.write(struct.pack('i', ny))        # NY
        f.write(struct.pack('i', num_points)) # NUM_Points
        f.write(struct.pack('i', maxiter))   # Maxiter

        # Write particle positions (PointOld structures: x, y)
        for _ in range(num_points):
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)
            f.write(struct.pack('d', x))     # x coordinate
            f.write(struct.pack('d', y))     # y coordinate

    print(f"  Created: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate example input files")
    parser.add_argument("--nx", type=int, default=64, help="Grid X dimension")
    parser.add_argument("--ny", type=int, default=64, help="Grid Y dimension") 
    parser.add_argument("--particles", type=int, default=1000, help="Number of particles")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args()

    # Generate two example files
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    electrons_file = os.path.join(args.output_dir, "electrons.bin")
    ions_file = os.path.join(args.output_dir, "ions.bin")

    generate_input_file(electrons_file, args.nx, args.ny, args.particles, args.maxiter)
    generate_input_file(ions_file, args.nx, args.ny, args.particles, args.maxiter)

    print("\nExample usage:")
    print(f"python scripts/run.py {electrons_file} {ions_file} --device gpu --wg-size 256 --prob-size 24")

if __name__ == "__main__":
    main()
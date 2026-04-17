# SYCL Particle Simulation

A high-performance particle simulation code using SYCL for charge deposition and particle movement on heterogeneous computing platforms.

## Overview

This simulation performs particle-in-cell (PIC) computations with the following key features:
- **Charge Deposition**: Interpolation of particle charges onto a computational mesh
- **Particle Movement**: Stochastic particle mover with random relocation
- **Hash Map Management**: Efficient particle-to-grid mapping using custom SYCL hash maps
- **Multi-Species Support**: Handles different particle species (electrons, ions) with different charge signs

## Project Structure

```
sycl-particle-simulation/
├── src/                    # Source files (.cpp)
│   ├── main.cpp           # Main execution logic
│   ├── particles.cpp      # Particle registration and file I/O
│   ├── mover.cpp         # Particle movement functions
│   ├── interpolation.cpp # Charge deposition algorithms
│   ├── utils.cpp         # Utility functions
│   └── types.cpp         # Global variable definitions
├── include/               # Header files (.hpp)
│   ├── types.hpp         # Data structures and type definitions
│   ├── particles.hpp     # Particle registration interface
│   ├── mover.hpp         # Particle movement interface
│   ├── interpolation.hpp # Charge deposition interface
│   └── utils.hpp         # Utility function interface
├── scripts/              # Python automation scripts
│   └── run.py           # Compilation and execution script
├── examples/            # Example input files and documentation
├── Makefile            # Build system
└── README.md           # This file
```

## Dependencies

### Required
- **SYCL Compiler**: One of the following:
  - Intel oneAPI DPC++ Compiler (`icpx`)
  - Intel DPC++ Compiler (`dpcpp`)
  - Clang++ with SYCL support
- **Intel oneAPI Math Kernel Library (MKL)**: For random number generation
- **Custom Hash Map**: `sycl_hashmap_2.hpp` (must be provided separately)

### Optional
- **Python 3.6+**: For automated build and execution scripts
- **Make**: For traditional build system

## Installation

### 1. Install Intel oneAPI (Recommended)

Download and install Intel oneAPI toolkit from:
https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkit.html

Source the environment:
```bash
source /opt/intel/oneapi/setvars.sh
```

### 2. Clone and Setup

```bash
# Clone or extract the project
cd sycl-particle-simulation

# Ensure sycl_hashmap_2.hpp is available
# (This file must be provided separately)
```

## Building the Code

### Method 1: Using Python Script (Recommended)

```bash
# Compile only
python scripts/run.py input1.bin input2.bin --compile-only

# Compile and run
python scripts/run.py input1.bin input2.bin --device gpu --wg-size 256 --prob-size 24
```

### Method 2: Using Makefile

```bash
# Build the simulation
make

# Clean build files
make clean

# Show help
make help
```

### Method 3: Manual Compilation

```bash
icpx -fsycl -I./include -O3 -std=c++17 -DMKL_ILP64 -qmkl \
    src/*.cpp -o build/particle_simulation
```

## Running the Simulation

### Input Files

The simulation requires two binary input files containing particle data:
- **Format**: Binary files with header and particle positions
- **Header**: NX (int), NY (int), NUM_Points (int), Maxiter (int)
- **Data**: Array of PointOld structures (x, y coordinates)

### Command Line Usage

```bash
./build/particle_simulation <input_file1> <input_file2> <device> <wg_size> <prob_size>
```

**Parameters:**
- `input_file1`: First particle species input file (e.g., electrons)
- `input_file2`: Second particle species input file (e.g., ions)
- `device`: Target device (`cpu` or `gpu`)
- `wg_size`: Work group size (typically 128, 256, or 512)
- `prob_size`: Probe size parameter (typically 16, 24, or 32)

### Using Python Script

```bash
# Basic usage
python scripts/run.py electrons.bin ions.bin

# With specific parameters
python scripts/run.py electrons.bin ions.bin \
    --device gpu --wg-size 256 --prob-size 24

# CPU execution
python scripts/run.py electrons.bin ions.bin \
    --device cpu --wg-size 128 --prob-size 16
```

**Python Script Options:**
- `--device {cpu,gpu}`: Target device (default: gpu)
- `--wg-size INT`: Work group size (default: 256)
- `--prob-size INT`: Probe size (default: 24)
- `--build-dir DIR`: Build directory (default: build)
- `--compile-only`: Only compile, don't run
- `--run-only`: Only run (assume already compiled)

## Output Files

The simulation generates:
- `Mesh_0.out`: Charge density mesh for first particle species
- `Mesh_1.out`: Charge density mesh for second particle species

Each file contains a 2D grid of charge density values.

## Algorithm Details

### 1. Particle Registration
- Reads binary input files containing particle positions
- Initializes SYCL hash maps for efficient particle-to-grid mapping
- Allocates device memory for particle data and mesh structures

### 2. Charge Deposition (Interpolation)
- Uses bilinear interpolation to deposit particle charges onto grid points
- Employs parallel reduction to accumulate charges from multiple particles
- Implements efficient memory access patterns for optimal performance

### 3. Particle Movement
- Stochastic mover with probability-based particle relocation
- Uses Intel MKL random number generators for reproducible results
- Updates hash map entries when particles move between grid cells

### 4. Hash Map Management
- Custom SYCL hash map implementation for particle-to-grid mapping
- Cleanup operations to maintain data structure consistency
- Optimized for parallel access patterns

## Performance Tuning

### Work Group Size (`wg_size`)
- **GPU**: Typically 256 or 512 for modern GPUs
- **CPU**: Typically 64 or 128 for multi-core CPUs
- Should be a multiple of the hardware's preferred work group size

### Probe Size (`prob_size`)
- Controls memory access granularity in hash map operations
- Larger values may improve memory bandwidth utilization
- Typical range: 16-32

### Device Selection
- **GPU**: Better for large problem sizes with high parallelism
- **CPU**: Better for smaller problems or when GPU memory is limited

## Troubleshooting

### Compilation Issues

1. **SYCL compiler not found**:
   ```bash
   # Install Intel oneAPI and source environment
   source /opt/intel/oneapi/setvars.sh
   ```

2. **Missing MKL**:
   - Ensure Intel oneAPI is properly installed and sourced
   - Check that `-qmkl` flag is supported by your compiler

3. **Hash map header missing**:
   - Ensure `sycl_hashmap_2.hpp` is available in include path
   - This file must be provided separately

### Runtime Issues

1. **Invalid device type**:
   - Use only `cpu` or `gpu` as device parameter
   - Check that target device is available on your system

2. **Input file errors**:
   - Verify input files exist and are in correct binary format
   - Check file permissions

3. **Memory allocation failures**:
   - Reduce problem size or use CPU device for large problems
   - Check available GPU memory

### Performance Issues

1. **Poor GPU performance**:
   - Try different work group sizes (256, 512, 1024)
   - Ensure problem size is large enough to saturate GPU
   - Check for memory bandwidth bottlenecks

2. **Slow compilation**:
   - Use `-O2` instead of `-O3` for faster compilation during development
   - Consider parallel compilation with `make -j`

## Example Usage

```bash
# Compile the code
python scripts/run.py electrons.bin ions.bin --compile-only

# Run on GPU with default parameters
python scripts/run.py electrons.bin ions.bin --device gpu

# Run on CPU with custom parameters
python scripts/run.py electrons.bin ions.bin \
    --device cpu --wg-size 128 --prob-size 16

# Traditional makefile approach
make
./build/particle_simulation electrons.bin ions.bin gpu 256 24
```

## Code Organization

### Header Files (include/)
- **types.hpp**: Core data structures (Point, Particle, GridParams)
- **particles.hpp**: Particle I/O and registration functions
- **mover.hpp**: Particle movement algorithms
- **interpolation.hpp**: Charge deposition algorithms
- **utils.hpp**: Utility and cleanup functions

### Source Files (src/)
- **main.cpp**: Main simulation loop and timing
- **particles.cpp**: File I/O and particle initialization
- **mover.cpp**: Stochastic particle movement
- **interpolation.cpp**: Bilinear interpolation and charge deposition
- **utils.cpp**: Hash map cleanup and utilities
- **types.cpp**: Global variable definitions

## Contributing

When modifying the code:

1. **Maintain Structure**: Keep the separation between headers and implementation
2. **Follow Naming**: Use consistent naming conventions
3. **Document Changes**: Update this README for significant changes
4. **Test Thoroughly**: Verify both CPU and GPU execution paths

## License

[Specify your license here]

## Contact

[Specify contact information here]

# Quick Start Guide

## 1. Prerequisites

Install Intel oneAPI toolkit and source the environment:
```bash
source /opt/intel/oneapi/setvars.sh
```

## 2. Generate Example Input Files

```bash
cd sycl-particle-simulation
python examples/generate_input.py --particles 1000
```

This creates `electrons.bin` and `ions.bin` with 1000 particles each.

## 3. Build and Run

### Option A: Python Script (Recommended)
```bash
python scripts/run.py electrons.bin ions.bin --device gpu --wg-size 256 --prob-size 24
```

### Option B: Makefile
```bash
make
./build/particle_simulation electrons.bin ions.bin gpu 256 24
```

### Option C: CMake
```bash
mkdir build && cd build
cmake ..
make
./bin/particle_simulation ../electrons.bin ../ions.bin gpu 256 24
```

## 4. Check Output

The simulation will create:
- `Mesh_0.out`: Charge density for first species
- `Mesh_1.out`: Charge density for second species

## Common Parameters

| Parameter | GPU Default | CPU Default | Description |
|-----------|-------------|-------------|-------------|
| wg_size   | 256         | 128         | Work group size |
| prob_size | 24          | 16          | Probe size |
| device    | gpu         | cpu         | Target device |

## Troubleshooting

1. **No SYCL compiler**: Install Intel oneAPI
2. **Compilation fails**: Source oneAPI environment
3. **Runtime error**: Check input file format and device availability

For detailed documentation, see [README.md](README.md).
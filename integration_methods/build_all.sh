#!/bin/bash
# Build all integration method implementations
# Run from: ~/manya-vansh-bmp/integration_methods

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOBIND_DIR=$(python -c "import nanobind; print(nanobind.cmake_dir())")
PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || echo "")

echo "============================================"
echo "Building C++/Python Integration Methods"
echo "============================================"
echo ""

# Method 1: PyBind11
echo "[1/4] Building PyBind11 implementation..."
cd "$SCRIPT_DIR/method1_pybind11"
rm -rf build && mkdir -p build && cd build
if [ -n "$PYBIND11_DIR" ]; then
    cmake .. -Dpybind11_DIR="$PYBIND11_DIR" 2>&1 | tail -3
    cmake --build . 2>&1 | tail -3
    cp matmul_pybind11*.so ../
    echo "      ✓ Built matmul_pybind11"
else
    echo "      ⚠ pybind11 not found, skipping..."
fi

# Method 2: Cython
echo "[2/4] Building Cython implementation..."
cd "$SCRIPT_DIR/method2_cython"
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace 2>&1 | tail -5
if [ -f matmul_cython*.so ]; then
    echo "      ✓ Built matmul_cython"
else
    echo "      ⚠ Cython build may have failed"
fi

# Method 3: DLPack (nanobind)
echo "[3/4] Building DLPack implementation..."
cd "$SCRIPT_DIR/method3_dlpack"
rm -rf build && mkdir -p build && cd build
# Try to find dlpack headers
DLPACK_PATH=$(python -c "import torch; print(torch.__path__[0] + '/include')" 2>/dev/null || echo "/usr/include")
cmake .. -Dnanobind_DIR="$NANOBIND_DIR" -DDLPACK_INCLUDE_DIR="$DLPACK_PATH" 2>&1 | tail -3
cmake --build . 2>&1 | tail -3
cp matmul_dlpack*.so ../
echo "      ✓ Built matmul_dlpack"

# Method 4: CUDA Array Interface (nanobind)
echo "[4/4] Building CUDA Array Interface implementation..."
cd "$SCRIPT_DIR/method4_cuda_interface"
rm -rf build && mkdir -p build && cd build
cmake .. -Dnanobind_DIR="$NANOBIND_DIR" 2>&1 | tail -3
cmake --build . 2>&1 | tail -3
cp matmul_cuda_interface*.so ../
echo "      ✓ Built matmul_cuda_interface"

echo ""
echo "============================================"
echo "Build Complete!"
echo "============================================"
echo ""
echo "Built modules:"
ls -la "$SCRIPT_DIR"/method*/*.so 2>/dev/null || echo "No .so files found"

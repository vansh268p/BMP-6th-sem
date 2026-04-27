#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

EXT_SUFFIX="$(python3-config --extension-suffix)"
PY_INCLUDES="$(python3-config --includes)"
NP_INCLUDES="-I$(python3 -c 'import numpy as np; print(np.get_include())')"
PB_INCLUDES="-I$(python3 -c 'import pybind11; print(pybind11.get_include())')"

c++ -O3 -std=c++17 -fPIC -shared $PY_INCLUDES $NP_INCLUDES \
    -I. src/numpy_c_api_sycl.cpp -Lsrc -lsycl_shared_core \
    -Wl,-rpath,'$ORIGIN/src' \
    -o "numpy_c_api_sycl${EXT_SUFFIX}"

c++ -O3 -std=c++17 -fPIC -shared $PY_INCLUDES $PB_INCLUDES \
    -I. src/pybind11_sycl.cpp -Lsrc -lsycl_shared_core \
    -Wl,-rpath,'$ORIGIN/src' \
    -o "pybind11_sycl${EXT_SUFFIX}"

c++ -O3 -std=c++17 -fPIC -shared $PY_INCLUDES \
    -I. src/dlpack_sycl.cpp -Lsrc -lsycl_shared_core \
    -Wl,-rpath,'$ORIGIN/src' \
    -o "dlpack_sycl${EXT_SUFFIX}"

python3 setup_cython.py build_ext --inplace

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ICX="${ICX:-/opt/intel/oneapi/compiler/2025.1/bin/icx}"
"$ICX" -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
    -O3 -std=c++17 -fPIC -shared \
    src/sycl_shared_core.cpp \
    -o src/libsycl_shared_core.so

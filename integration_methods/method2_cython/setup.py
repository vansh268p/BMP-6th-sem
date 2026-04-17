"""
Setup script for Cython matrix multiplication extension.

Build with:
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# CUDA paths
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')

ext_modules = [
    Extension(
        "matmul_cython",
        sources=["matmul_cython.pyx"],
        include_dirs=[
            np.get_include(),
            os.path.join(CUDA_HOME, 'include'),
        ],
        library_dirs=[
            os.path.join(CUDA_HOME, 'lib64'),
        ],
        libraries=['cublas', 'cudart'],
        language='c++',
        extra_compile_args=['-O3'],
    )
]

setup(
    name="matmul_cython",
    ext_modules=cythonize(ext_modules, language_level=3),
)

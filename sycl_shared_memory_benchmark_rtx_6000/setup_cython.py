from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "cython_sycl",
    sources=["cython_sycl.pyx"],
    include_dirs=[np.get_include(), "."],
    libraries=["sycl_shared_core"],
    library_dirs=["src"],
    runtime_library_dirs=["$ORIGIN/src"],
    language="c++",
)

setup(
    name="cython_sycl",
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
)

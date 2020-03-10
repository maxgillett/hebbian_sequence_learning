from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(name='routines',
      ext_modules=cythonize(
        Extension(
            "routines",
            sources=["routines.pyx", "integration.cpp"],
            extra_compile_args=["-std=c++11"],
            extra_link_args=["-lgsl", "-lgslcblas", "-lm"],
            language="c++")
        )
)

"""Build the Cython Z80 extension in-place.

Usage:
    python setup_cy.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    "pygear.cpu.z80_cy",
    sources=["pygear/cpu/z80_cy.pyx"],
    extra_compile_args=["-O3", "-march=native"],
)

setup(
    name="pygear_cy",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)

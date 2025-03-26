# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="piece_move",
    ext_modules=cythonize("piece_move.pyx", language_level=3),
    include_dirs=[numpy.get_include()]
)

# setup.py

from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "machine_learning",  # The module name that will be imported from python
        sources=[
            "machine_learning.pyx",
            "dtw.c",
            "mann_kendall.c",
            "pelt.c",
            "theil_sen.c",
        ],
        include_dirs=[
            numpy.get_include()  # Necessary for NumPy 'C-API'
        ],
        language="c",
        extra_compile_args=["-mavx", "-ffast-math", "-std=c99"],
    )
]

setup(
    name="machine_learning_lib", # Name for meta
    ext_modules=cythonize(ext_modules),
)
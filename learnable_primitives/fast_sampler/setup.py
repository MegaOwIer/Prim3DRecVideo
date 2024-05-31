import numpy
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = []

extensions.append(
    Extension("learnable_primitives.fast_sampler._sampler", 
              sources=["learnable_primitives/fast_sampler/_sampler.pyx",
                       "learnable_primitives/fast_sampler/sampling.cpp"],
              extra_compile_args=["-O2", "-ffast-math", "-march=native", "-std=c++11"]))
setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include(), "learnable_primitives/fast_sampler"]
)

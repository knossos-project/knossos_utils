#!/usr/bin/env python
have_cython = False
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext as build_ext
    have_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext as build_ext
from setuptools import setup
from setuptools import Extension
import numpy
import os

extensions = []
if have_cython:
    extensions.append(Extension("mergelist_tools",
                                ['knossos_utils/mergelist_tools.pyx'],
                                include_dirs = [numpy.get_include()],
                                language="c++",
                                extra_compile_args=["-std=c++0x", "-include", "cmath"]))
else:
    extensions.append(Extension('mergelist_tools',
                                ['knossos_utils/mergelist_tools.cpp'],
                                include_dirs = [numpy.get_include()],
                                language="c++",
                                extra_compile_args=["-std=c++0x", "-include", "cmath"]))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup (
    name = "knossos_python_tools",
    version = "1.0",
    description = "Tools for generating or manipulating knossos datasets and annotation files",
    author = "Sven Dorkenwald, KNOSSOS team",
    author_email = "knossos-team@mpimf-heidelberg.mpg.de",
    url = "https://github.com/knossos-project/knossos_python_tools",
    license = "GPL",
    long_description = read("README.md"),
    packages = ["knossos_utils"],
    data_files = [("", ["LICENSE"])],
    ext_modules = extensions,
    cmdclass = {"build_ext": build_ext},
    install_requires=[
        "h5py",
        "numpy",
        "python-snappy",
        "scipy",
    ]
)

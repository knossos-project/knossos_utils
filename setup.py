#!/usr/bin/env python2

from setuptools import setup, Extension
import os
try:
    import numpy
except ImportError:
    print("Numpy not found. Please install Numpy manually: http://www.scipy.org/install.html")
    raise SystemExit

extensions = [Extension(
    "mergelist_tools",
    ["knossos_utils/mergelist_tools.pyx"],
    include_dirs=[numpy.get_include()],
    language="c++",
    extra_compile_args=["-std=c++0x", "-include", "cmath"])
]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="knossos_python_tools",
    version="1.0",
    description="Tools for generating or manipulating knossos datasets and annotation files",
    author="Sven Dorkenwald, KNOSSOS team",
    author_email="knossos-team@mpimf-heidelberg.mpg.de",
    url="https://github.com/knossos-project/knossos_python_tools",
    license="GPL",
    long_description=read("README.md"),
    packages=["knossos_utils"],
    data_files=[("", ["LICENSE"])],
    ext_modules=extensions,
    setup_requires=[
        "cython>=0.23",
        "setuptools>=18.0",
    ],
    install_requires=[
        "cython",
        "h5py",
        "numpy",
        "scipy",
    ],
    extras_require={
        "snappy": ["python-snappy"],
    },
)


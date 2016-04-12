#!/usr/bin/env python2

from setuptools import setup, Extension
import os
import sys

if sys.version_info[:2] != (2, 7):
    print('\nSorry, only Python 2.7 is currently supported.')
    print('\nYour current Python version is {}'.format(sys.version))
    sys.exit(1)

# Setuptools >=18.0 is needed for Cython to work correctly.
if parse_version(setuptools.__version__) < parse_version('18.0'):
    print('\nYour installed Setuptools version is too old.')
    print('Please upgrade it to at least 18.0, e.g. by running')
    print('$ python2 -m pip install --upgrade setuptools')
    print('If this fails, try additionally passing the "--user" switch to the install command, or use Anaconda2.')
    sys.exit(1)

try:
    import numpy
except ImportError:
    print("Numpy not found. Please install Numpy manually: http://www.scipy.org/install.html")
    sys.exit(1)

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
    ],
    install_requires=[  # TODO: Determine minimum requirement versions
        "cython",
        "h5py",
        "numpy",
        "scipy",
    ],
    extras_require={
        "snappy": ["python-snappy"],
    },
)


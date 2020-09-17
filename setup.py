#!/usr/bin/env python

import os
import sys
import setuptools
from setuptools import find_packages, setup, Extension
from pkg_resources import parse_version


# Setuptools >=18.0 is needed for Cython to work correctly.
if parse_version(setuptools.__version__) < parse_version('18.0'):
    print('\nYour installed Setuptools version is too old.')
    print('Please upgrade it to at least 18.0, e.g. by running')
    print('$ python{} -m pip install --upgrade setuptools'.format(sys.version_info[0]))
    print('If this fails, try additionally passing the "--user" switch to the install command, or use Anaconda.')
    sys.stdout.flush()
    sys.exit(1)

try:
    import numpy
except ImportError:
    print("Numpy not found. Please install Numpy manually: http://www.scipy.org/install.html")
    sys.stdout.flush()
    sys.exit(1)

extensions = [Extension(
    "knossos_utils.mergelist_tools",
    ["knossos_utils/mergelist_tools.pyx"],
    include_dirs=[numpy.get_include()],
    language="c++",
    extra_compile_args=["-std=c++0x", "-include", "cmath"])
]

install_requires = [
    "cython>=0.23",
    "h5py>=2.5",
    "imageio",
    "numpy>=1.10",
    "scipy>=0.16",
    "toml",
    "networkx>=1.11",
    "requests>=2.12",
    "matplotlib",
    "Pillow"
]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="knossos_utils",
    version="0.1",
    description="Tools for generating or manipulating knossos datasets and annotation files",
    author="Sven Dorkenwald, KNOSSOS team",
    author_email="knossosteam@gmail.com",
    url="https://github.com/knossos-project/knossos_utils",
    license="GPL",
    long_description=read("README.md"),
    packages=find_packages(),
    data_files=[("", ["LICENSE"])],
    ext_modules=extensions,
    setup_requires=[
        "cython>=0.23",
    ],
    install_requires=install_requires,
    extras_require={
        "snappy": ["python-snappy>=0.5"],
        # "skeletopyze": only needed for importing skeletopyze skeletons. See https://github.com/funkey/skeletopyze
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)

from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('knossos_utils/mergelist_tools.pyx'))

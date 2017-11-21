from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension

#
# Run this as python mergelist_tools_setup.py build_ext --inplace 
# to generate mergelist_tools.so which can be imported by python.
#

extensions = [
    Extension('knossos_utils.mergelist_tools',
        sources=['knossos_utils/mergelist_tools.pyx'],
        include_dirs = [np.get_include()],
        language='c++',
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"])
]

setup(
    ext_modules = cythonize(extensions),
    language='c++',
)

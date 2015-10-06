from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
import numpy

extensions = [
    Extension(
    	"mergelist_tools",
    	['mergelist_tools.pyx'],
        include_dirs = [numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++0x", "-include", "cmath"]
    ),
]

setup(
    ext_modules = cythonize(extensions),
)

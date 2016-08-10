# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function
# builtins is either provided by Python 3 or by the "future" module for Python 2 (http://python-future.org/)
from builtins import range, map, zip, filter, round, next, input, bytes, hex, oct, chr, int  # TODO: Import all other necessary builtins after testing
from functools import reduce

import sys
from setuptools import setup

install_requires = [
    'numpy>=1.10',
    'scipy>=0.16',
    'Pillow',
    'PyQt5',
]


entry_points = {
    'console_scripts': [
        'knossos_cuber = knossos_cuber.knossos_cuber:main',
    ],
    'gui_scripts': [
        'knossos_cuber_gui = knossos_cuber.knossos_cuber_gui:main'
    ]
}

if sys.version_info < (3, 0):
    # Python 2 needs builtins provided by future
    install_requires.append('future>=0.15')

# Decide if/how the GUI should be installed:
try:
    import PyQt5
    # If the line above works, there is no need to install PyQt5, so we should remove it from install_requires:
    install_requires.remove('PyQt5')
except ImportError:  # PyQt5 not available
    if sys.version_info >= (3, 5):
        # PyQt5 will be installed by setuptools via install_requires.
        pass
    else:
        # PyQt5 currently can't be pip installed on python<3.5. Either it is there (system package) or you can't use it.
        print('PyQt5 not found. knossos_cuber_gui will not be available.')
        print('(This problem occurs only in old Python versions.')
        print(' If you use Python 3.5 or later, PyQt5 will be automatically provided.)')
        print('You can also try installing PyQt5 via your system package manager (apt, yum etc.) and then re-install.')
        print(flush=True)
        del entry_points['gui_scripts']

setup(
    name='knossos_cuber',
    packages=['knossos_cuber'],
    entry_points=entry_points,
    include_package_data=True,
    version='1.0',
    description='A script that converts images into a KNOSSOS-readable format.',
    author='Jörgen Kornfeld, Fabian Svara',
    author_email='Jörgen Kornfeld <joergen.kornfeld@mpimf-heidelberg.mpg.de>,'
                 'Fabian Svara <fabian.svara@mpimf-heidelberg.mpg.de>',
    url='https://github.com/knossos-project/knossos_cuber',  # TODO: Actually push it there
    keywords=['converter', 'skeletonization', 'segmentation'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2 :: Only',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    install_requires=install_requires,
)

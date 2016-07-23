# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

from setuptools import setup

setup(
    name='knossos_cuber',
    packages=['knossos_cuber'],
    entry_points={
        'console_scripts': [
            'knossos_cuber = knossos_cuber.knossos_cuber:main',
        ],
        'gui_scripts': [
            'knossos_cuber_gui = knossos_cuber.knossos_cuber_gui:main',  # TODO: Fix config loading
        ]
    },
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
    install_requires=[
        'numpy>=1.10',
        'scipy>=0.16',
        'future>=0.15',  # only required for Python 2
        'Pillow',
        # 'PyQt4',  # (no packages on PyPi)
    ],
)

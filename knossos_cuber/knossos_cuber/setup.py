from __future__ import absolute_import, print_function

from distutils.core import setup
import py2exe

setup(
    windows=['knossos_cuber_gui.py'],
    options={"py2exe" : {"includes" : ["sip"]}}
)

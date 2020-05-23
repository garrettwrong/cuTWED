#!/usr/bin/env python
"""
Copyright 2020 Garrett Wright, Gestalt Group LLC

This file is part of cuTWED.

cuTWED is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
cuTWED is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with cuTWED.  If not, see <https://www.gnu.org/licenses/>.

"""

import os

from cffi import FFI

ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
with open(os.path.join("reference_implementation", "twed.h")) as fh:
    ffibuilder.cdef(fh.read())

# set_source() gives the name of the python extension module to
# produce, and some C source code as a string.  This C code needs
# to make the declarated functions, types and globals available,
# so it is often just the "#include".
ffibuilder.set_source(
    "_CTWED",
    """
    #include "twed.h"   // the C header of the library
    """,
    sources=[os.path.join('reference_implementation', 'twed.c')],
    include_dirs=['reference_implementation'],
    libraries=['m'],   # shared library names, for the linker
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

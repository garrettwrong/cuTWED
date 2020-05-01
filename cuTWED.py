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

import numpy as np
import ctypes

try:
    _libcuTWED = ctypes.CDLL('libcuTWED.so')
except OSError as e:
    print("Ensure you have added 'libcuTWED.so' somewhere in your LD_LIBRARY_PATH")
    raise e

_twed = _libcuTWED.twed
_twed.restype = ctypes.c_double
_twed.argtypes = (np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  ctypes.c_int,
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  ctypes.c_int,
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  ctypes.c_double,
                  ctypes.c_double,
                  ctypes.c_int,
                  ctypes.c_void_p)


def twed(A, TA, B, TB, nu, lamb, degree):
    """
    Invokes CUDA based twed using ctypes wrapper.

    A, B  : Arrays of time series values.
    TA, TB: Arrays of corresponding time series timestamps.
    nu, lamb, degree: algo parameters.
    """
    
    nA = len(A)
    nB = len(B)
    assert nA == len(TA)
    assert nB == len(TB)
    assert degree>0

    return _twed(A, nA, TA, B, nB, TB, nu, lamb, degree, None)

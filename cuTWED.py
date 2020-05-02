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

try:
    _libcuTWED_32 = ctypes.CDLL('libcuTWED_32.so')
except OSError as e:
    print("Ensure you have added 'libcuTWED_32.so' somewhere in your LD_LIBRARY_PATH")
    raise e

_twed = _libcuTWED.twed
_twed.restype = ctypes.c_double
_twed.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  ctypes.c_int,
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  ctypes.c_int,
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  ctypes.c_double,
                  ctypes.c_double,
                  ctypes.c_int,
                  ctypes.c_void_p]

_twed_32 = _libcuTWED_32.twed
_twed_32.restype = ctypes.c_float
_twed_32.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                     ctypes.c_int,
                     np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                     np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                     ctypes.c_int,
                     np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                     ctypes.c_float,
                     ctypes.c_float,
                     ctypes.c_int,
                     ctypes.c_void_p]

_twed_dev = _libcuTWED.twed_dev
_twed_dev.restype = ctypes.c_double
_twed_dev.argtypes = [ctypes.c_void_p,
                      ctypes.c_int,
                      ctypes.c_void_p,
                      ctypes.c_void_p,
                      ctypes.c_int,
                      ctypes.c_void_p,
                      ctypes.c_double,
                      ctypes.c_double,
                      ctypes.c_int,
                      ctypes.c_void_p]

_twed_dev_32 = _libcuTWED_32.twed_dev
_twed_dev_32.restype = ctypes.c_float
_twed_dev_32.argtypes = [ctypes.c_void_p,
                         ctypes.c_int,
                         ctypes.c_void_p,
                         ctypes.c_void_p,
                         ctypes.c_int,
                         ctypes.c_void_p,
                         ctypes.c_float,
                         ctypes.c_float,
                         ctypes.c_int,
                         ctypes.c_void_p]


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
    assert all([x.dtype==A.dtype for x in [A, TA, B, TB]]) # Dtypes should match

    if A.dtype == np.float64:
        func = _twed
    elif A.dtype == np.float32:
        func = _twed_32
    else:
        raise RuntimeError("Expected inputs to be np.float32 or np.float64")
    
    return func(A, nA, TA, B, nB, TB, nu, lamb, degree, None)

def twed_dev(A, TA, B, TB, nu, lamb, degree, DP=None):
    """
    Invokes CUDA based twed using ctypes wrapper.

    A, B  : GPUArrays of time series values.
    TA, TB: GPUArrays of corresponding time series timestamps.
    nu, lamb, degree: algo parameters.
    """

    # This is the "wrong place", but I don't want to force people to install pycuda
    #   if they just want the regular CUDA C wrapper...
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    
    nA = len(A)
    nB = len(B)
    assert nA == len(TA)
    assert nB == len(TB)
    assert degree>0
    assert all([x.dtype==A.dtype for x in [A, TA, B, TB]]) # Dtypes should match

    if DP is None:
        DP = gpuarray.GPUArray((nA+1,nB+1), dtype=A.dtype)

    if A.dtype == np.float64:
        func = _twed_dev
    elif A.dtype == np.float32:
        func = _twed_dev_32
    else:
        raise RuntimeError("Expected inputs to be np.float32 or np.float64")

    return func(A.ptr, nA, TA.ptr, B.ptr, nB, TB.ptr, nu, lamb, degree, DP.ptr)


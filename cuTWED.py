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
_twed.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                  ctypes.c_int,
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                  ctypes.c_int,
                  np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                  ctypes.c_double,
                  ctypes.c_double,
                  ctypes.c_int,
                  ctypes.c_int]

_twedf = _libcuTWED.twedf
_twedf.restype = ctypes.c_float
_twedf.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                   ctypes.c_int,
                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                   ctypes.c_int,
                   np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                   ctypes.c_float,
                   ctypes.c_float,
                   ctypes.c_int,
                   ctypes.c_int]

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
                      ctypes.c_int]

_twed_devf = _libcuTWED.twed_devf
_twed_devf.restype = ctypes.c_float
_twed_devf.argtypes = [ctypes.c_void_p,
                         ctypes.c_int,
                         ctypes.c_void_p,
                         ctypes.c_void_p,
                         ctypes.c_int,
                         ctypes.c_void_p,
                         ctypes.c_float,
                         ctypes.c_float,
                       ctypes.c_int,
                         ctypes.c_int]


def twed(A, TA, B, TB, nu, lamb, degree=2):
    """
    Invokes CUDA based twed using ctypes wrapper.

    A, B  : Arrays of time series values.
    TA, TB: Arrays of corresponding time series timestamps.
    degree: Power used in the Lp norm, default is 2.
    nu, lamb: algo parameters.
    """

    if A.ndim == 1:
        A = A.reshape((A.shape[0], 1))
    elif A.ndim != 2:
        raise RuntimeError("Input A should be 1D, or 2d (Time x dim) array.")

    if B.ndim == 1:
        B = B.reshape((B.shape[0], 1))
    elif B.ndim != 2:
        raise RuntimeError("Input B should be 1D, or 2d (Time x dim) array.")

    nA = A.shape[0]
    nB = B.shape[0]
    dim = A.shape[1]
    
    assert dim == B.shape[1], "A and B can be different length," \
        "but should have same 'dim'."
    assert nA == len(TA)
    assert nB == len(TB)
    assert degree>0
    assert all([x.dtype==A.dtype for x in [A, TA, B, TB]]) # Dtypes should match

    if A.dtype == np.float64:
        func = _twed
    elif A.dtype == np.float32:
        func = _twedf
    else:
        raise RuntimeError("Expected inputs to be np.float32 or np.float64")
    
    return func(A, nA, TA, B, nB, TB, nu, lamb, degree, dim)

def twed_dev(A, TA, B, TB, nu, lamb, degree=2):
    """
    Invokes CUDA based twed using ctypes wrapper.

    A, B  : GPUArrays of time series values.
    TA, TB: GPUArrays of corresponding time series timestamps.
    degree: Power used in the Lp norm, default is 2.
    nu, lamb: algo parameters.
    """

    # This is the "wrong place", but I don't want to force people to install pycuda
    #   if they just want the regular CUDA C wrapper... that may have to change
    #   when I package this up.
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    
    if A.ndim == 1:
        A = A.reshape((A.shape[0], 1))
    elif A.ndim != 2:
        raise RuntimeError("Input A should be 1D, or 2d (Time x dim) array.")

    if B.ndim == 1:
        B = B.reshape((B.shape[0], 1))
    elif B.ndim != 2:
        raise RuntimeError("Input B should be 1D, or 2d (Time x dim) array.")
    
    nA = A.shape[0]
    nB = B.shape[0]
    dim = A.shape[1]

    assert dim == B.shape[1], "A and B can be different length," \
        "but should have same 'dim'."
    assert nA == len(TA)
    assert nB == len(TB)
    assert degree>0
    assert all([x.dtype==A.dtype for x in [A, TA, B, TB]]) # Dtypes should match

    if A.dtype == np.float64:
        func = _twed_dev
    elif A.dtype == np.float32:
        func = _twed_devf
    else:
        raise RuntimeError("Expected inputs to be np.float32 or np.float64")

    return func(A.ptr, nA, TA.ptr, B.ptr, nB, TB.ptr, nu, lamb, degree, dim)


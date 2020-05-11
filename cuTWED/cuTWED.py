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
from _cuTWED import ffi
from _cuTWED import lib


def twed(A, TA, B, TB, nu, lamb, degree=2):
    """
    Invokes CUDA based twed using cffi wrapper.

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
    assert degree > 0
    assert all([x.dtype == A.dtype for x in [A, TA, B, TB]])  # Dtypes should match

    if A.dtype == np.float64:
        func = lib.twed

        def caster(x):
            return ffi.cast("double *", x.ctypes.data)

    elif A.dtype == np.float32:
        func = lib.twedf

        def caster(x):
            return ffi.cast("float *", x.ctypes.data)

    else:
        raise RuntimeError("Expected inputs to be np.float32 or np.float64")

    return func(caster(A), nA, caster(TA), caster(B), nB, caster(TB), nu, lamb, degree, dim)


def twed_dev(A, TA, B, TB, nu, lamb, degree=2):
    """
    Invokes CUDA based twed using ctypes wrapper.

    A, B  : GPUArrays of time series values.
    TA, TB: GPUArrays of corresponding time series timestamps.
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
    assert degree > 0
    assert all([x.dtype == A.dtype for x in [A, TA, B, TB]])  # Dtypes should match

    if A.dtype == np.float64:
        func = lib.twed_dev

        def caster(x):
            return ffi.cast("double *", x.gpudata)

    elif A.dtype == np.float32:
        func = lib.twed_devf

        def caster(x):
            return ffi.cast("float *", x.gpudata)

    else:
        raise RuntimeError("Expected inputs to be np.float32 or np.float64")

    return func(caster(A), nA, caster(TA), caster(B), nB, caster(TB), nu, lamb, degree, dim)

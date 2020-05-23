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
from _CTWED import ffi
from _CTWED import lib


def ctwed(A, TA, B, TB, nu, lamb, degree=2):
    """
    Invokes Reference C TWED using cffi wrapper.

    A, B  : Arrays of time series values.
    TA, TB: Arrays of corresponding time series timestamps.
    degree: Power used in the Lp norm, default is 2.
    nu, lamb: algo parameters.

    Note Marteau's reference implementation did not take
    nth-root for the norm. To reproduce that calculation,
    use a negative degree.  Example, -2 would raise to power 2
    but not take the sqrt, reproducing the original work.

    Also note reference code was not in R^N,
    but it has been extended here.
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
    assert all([x.dtype == A.dtype for x in [A, TA, B, TB]])  # Dtypes should match

    if A.dtype == np.float64:
        func = lib.CTWED

        def caster(x):
            return ffi.cast("double *", x.ctypes.data)

    else:
        raise NotImplementedError("Reference CTWED is doubles (float64) only.")

    # Starting to think ctypes was, and remains, way better.
    la_ptr = ffi.new("int *")
    la_ptr[0] = nA
    lb_ptr = ffi.new("int *")
    lb_ptr[0] = nB
    nu_ptr = ffi.new("double *")
    nu_ptr[0] = nu
    lambda_ptr = ffi.new("double *")
    lambda_ptr[0] = lamb
    degree_ptr = ffi.new("int *")
    degree_ptr[0] = degree
    dim_ptr = ffi.new("int *")
    dim_ptr[0] = dim
    res_ptr = ffi.new("double *")
    res_ptr[0] = -1

    func(caster(A),
         la_ptr,
         caster(TA),
         caster(B),
         lb_ptr,
         caster(TB),
         nu_ptr,
         lambda_ptr,
         degree_ptr,
         res_ptr,
         dim_ptr)

    return res_ptr[0]

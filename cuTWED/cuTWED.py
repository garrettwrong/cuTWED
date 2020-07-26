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

TRI_OPT = {'triu': lib.TRIU,
           'tril': lib.TRIL,
           'nopt': lib.NOPT}


def _get_tri_opt(opt):
    if isinstance(opt, int):
        assert opt in TRI_OPT.values()
    else:
        opt = TRI_OPT.get(opt.lower())
    return opt


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

    result = func(caster(A), nA, caster(TA), caster(B), nB, caster(TB), nu, lamb, degree, dim)

    if result < 0:
        raise RuntimeError(f"cuTWED call failed with {result}.")

    return result


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

    result = func(caster(A), nA, caster(TA), caster(B), nB, caster(TB), nu, lamb, degree, dim)

    if result < 0:
        raise RuntimeError(f"cuTWED call failed with {result}.")

    return result


def twed_batch_dev(AA, TAA, BB, TBB, nu, lamb, degree=2, tri=lib.NOPT):
    """
    Invokes CUDA based batch twed using ctypes wrapper.

    AA, BA  : GPUArrays of time series values. size, (nAA,nA,dim) and (nBB,nA,dim
    TA, TB: GPUArrays of corresponding time series timestamps.
    degree: Power used in the Lp norm, default is 2.
    nu, lamb: algo parameters.
    tri: Triangle Optimization for Square Symmetric batches. defaults 'noopt',
    Permits 'triu', 'tril'.  Probably want tril for square symmetric case.
    """

    if AA.ndim == 2:
        AA = AA.reshape((AA.shape[0], AA.shape[1], 1))
    elif AA.ndim != 3:
        raise RuntimeError("Input AA should be 2D, or 3d (nAA x Time x <dim>) array.")

    if BB.ndim == 2:
        BB = BB.reshape((BB.shape[0], BB.shape[1], 1))
    elif BB.ndim != 3:
        raise RuntimeError("Input BB should be 2D, or 3d (nBB x Time x <dim>) array.")

    nAA = AA.shape[0]
    nA = AA.shape[1]
    nBB = BB.shape[0]
    nB = BB.shape[1]
    dim = AA.shape[2]

    assert dim == BB.shape[2], "AA and BB should have same 'dim'."
    assert nAA, nA == TAA.shape
    assert nBB, nB == TBB.shape
    assert degree > 0
    assert all([x.dtype == AA.dtype for x in [AA, TAA, BB, TBB]])  # Dtypes should match

    RRes = np.zeros((nAA, nBB), AA.dtype)

    if AA.dtype == np.float64:
        func = lib.twed_batch_dev

        RRes_ptr = ffi.cast("double *", RRes.ctypes.data)

        def caster(x):
            return ffi.cast("double *", x.gpudata)

    elif AA.dtype == np.float32:
        func = lib.twed_batch_devf

        RRes_ptr = ffi.cast("float *", RRes.ctypes.data)

        def caster(x):
            return ffi.cast("float *", x.gpudata)

    else:
        raise RuntimeError("Expected inputs to be np.float32 or np.float64")

    ret_code = func(caster(AA), nA, caster(TAA),
                    caster(BB), nB, caster(TBB),
                    nu, lamb, degree, dim,
                    nAA, nBB, RRes_ptr, _get_tri_opt(tri))

    if ret_code != 0:
        raise RuntimeError(f"cuTWED call failed with {ret_code}.")

    return RRes


def twed_batch(AA, TAA, BB, TBB, nu, lamb, degree=2, tri=lib.NOPT):
    """
    Invokes CUDA based batch twed using ctypes wrapper.

    AA, BA  : Numpy C Arrays of time series values.
              size, (nAA,nA,dim) and (nBB,nA,dim)
    TA, TB: Numpy C Arrays of corresponding time series timestamps.
    degree: Power used in the Lp norm, default is 2.
    nu, lamb: algo parameters.
    tri: Triangle Optimization for Square Symmetric batches. defaults 'noopt',
    Permits 'triu', 'tril'. Probably want tril for square symmetric case.
    """

    if AA.ndim == 2:
        AA = AA.reshape((AA.shape[0], AA.shape[1], 1))
    elif AA.ndim != 3:
        raise RuntimeError("Input AA should be 2D, or 3d (nAA x Time x <dim>) array.")

    if BB.ndim == 2:
        BB = BB.reshape((BB.shape[0], BB.shape[1], 1))
    elif BB.ndim != 3:
        raise RuntimeError("Input BB should be 2D, or 3d (nBB x Time x <dim>) array.")

    nAA = AA.shape[0]
    nA = AA.shape[1]
    nBB = BB.shape[0]
    nB = BB.shape[1]
    dim = AA.shape[2]

    assert dim == BB.shape[2], "AA and BB should have same 'dim'."
    assert nAA, nA == TAA.shape
    assert nBB, nB == TBB.shape
    assert degree > 0
    assert all([x.dtype == AA.dtype for x in [AA, TAA, BB, TBB]])  # Dtypes should match

    RRes = np.zeros((nAA, nBB), AA.dtype)

    if AA.dtype == np.float64:
        func = lib.twed_batch

        def caster(x):
            return ffi.cast("double *", x.ctypes.data)

    elif AA.dtype == np.float32:
        func = lib.twed_batchf

        def caster(x):
            return ffi.cast("float *", x.ctypes.data)

    else:
        raise RuntimeError("Expected inputs to be np.float32 or np.float64")

    ret_code = func(caster(AA), nA, caster(TAA),
                    caster(BB), nB, caster(TBB),
                    nu, lamb, degree, dim,
                    nAA, nBB, caster(RRes), _get_tri_opt(tri))

    if ret_code != 0:
        raise RuntimeError(f"cuTWED call failed with {ret_code}.")

    return RRes

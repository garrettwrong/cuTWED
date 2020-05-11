#!/usr/bin/env python
"""
Demonstrates usage of gpuarray based cuTWED calls.

These are a special case where you already have input
time series residing in gpu memory.  The cononical way
to use gpu memory in python is through PyCUDA.

Copyright 2020 Garrett Wright, Gestalt Group LLC
"""

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.gpuarray as gpuarray
from numpy.random import RandomState

# Import the twed_dev function from cuTWED
from cuTWED import twed_dev

# Generate some junk data
n = 10
rng = RandomState(42)
noise1 = rng.randn(n)

TA = np.arange(n, dtype=np.float64)
A = np.sin(TA) + np.sin(TA/10) + noise1

m = 2 * n
noise2 = rng.randn(m)
TB = np.arange(m, dtype=np.float64)
B = np.sin(TB) + np.sin(TB/10) + noise2

# Set algo params
nu = 1.
lamb = 1.
degree = 2

reference_result = 54.827250


def test_basic_dev():
    """ Test calling twed_dev using GPUarrays. """

    A_dev = gpuarray.to_gpu(A)
    TA_dev = gpuarray.to_gpu(TA)
    B_dev = gpuarray.to_gpu(B)
    TB_dev = gpuarray.to_gpu(TB)

    # Call TWED
    dist = twed_dev(A_dev, TA_dev, B_dev, TB_dev, nu, lamb, degree)

    print('Python gpuarray cuTWED distance: {:f}'.format(dist))

    assert np.allclose(dist, reference_result)


def test_basic_dev_float():
    """ Test calling twed_dev using GPUarrays using floats. """

    A_dev = gpuarray.to_gpu(A.astype(np.float32))
    TA_dev = gpuarray.to_gpu(TA.astype(np.float32))
    B_dev = gpuarray.to_gpu(B.astype(np.float32))
    TB_dev = gpuarray.to_gpu(TB.astype(np.float32))

    # Call TWED
    dist = twed_dev(A_dev, TA_dev, B_dev, TB_dev, nu, lamb, degree)

    print('Python gpuarray cuTWED distance (single precision): {:f}'.format(dist))

    assert np.allclose(dist, reference_result)

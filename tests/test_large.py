#!/usr/bin/env python
"""
Demonstates basic cuTWED usage

Copyright 2020 Garrett Wright, Gestalt Group LLC
"""

import numpy as np
from numpy.random import RandomState

# Import the twed function from cuTWED
from cuTWED import twed

# Generate some junk data
n = 10000
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

reference_result = 61445.496235


def test_large_call():
    """ Test calling twed """
    # Call TWED
    dist = twed(A, TA, B, TB, nu, lamb, degree)

    print('Python cuTWED distance: {:f}'.format(dist))

    assert np.allclose(dist, reference_result)


def test_large_call_float():
    """ Test the same call in single precision by feeding different types. """

    dist = twed(A.astype(np.float32), TA.astype(np.float32),
                B.astype(np.float32), TB.astype(np.float32),
                nu, lamb, degree)

    print('Python cuTWED distance (single precision): {:f}'.format(dist))

    assert np.allclose(dist, reference_result)

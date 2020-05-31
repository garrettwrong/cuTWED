#!/usr/bin/env python
"""
Demonstates basic cTWED usage

Copyright 2020 Garrett Wright, Gestalt Group LLC
"""

import numpy as np
import pytest
from numpy.random import RandomState

# Import the twed function from cuTWED
from cuTWED import ctwed

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

original_reference_result = 58.981692
reference_result = 54.54317


def test_basic_call():
    """ Test calling twed """

    dist = ctwed(A, TA, B, TB, nu, lamb, degree)

    print('Python CTWED distance: {:f}'.format(dist))

    assert np.allclose(dist, reference_result)


def test_repro_call():
    """ Test calling twed """
    repro_degree = -2    # repro Marteau's original code.
    dist = ctwed(A, TA, B, TB, nu, lamb, repro_degree)

    print('Python CTWED distance: {:f}'.format(dist))

    assert np.allclose(dist, original_reference_result)


def test_basic_call_float():
    """ Test the same call in single precision by feeding different types. """

    with pytest.raises(NotImplementedError):
        _ = ctwed(A.astype(np.float32), TA.astype(np.float32),
                  B.astype(np.float32), TB.astype(np.float32),
                  nu, lamb, degree)

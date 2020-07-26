#!/usr/bin/env python
"""
Demonstrates usage of twed_batch_dev cuTWED call.

Copyright 2020 Garrett Wright, Gestalt Group LLC
"""

import numpy as np
from numpy.random import RandomState

# Import the function from cuTWED
from cuTWED import twed_batch

# Generate some junk data

# First generate the same time series uses in other tests.
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

# Expected results
reference_result_a_b = 54.54317
reference_result_a_0 = 44.739147

# We now make some batches.
batch_sz = 100

AA = np.tile(A, (batch_sz, 1))
TAA = np.tile(TA, (batch_sz, 1))
BB = np.tile(B, (batch_sz, 1))
# Pack half of B with 0
BB[batch_sz//2:] = 0
TBB = np.tile(TB, (batch_sz, 1))

# And the Reference result for the batch
Ref = np.zeros((batch_sz, batch_sz), dtype=AA.dtype)
for i in range(batch_sz//2):
    # one half batch is A B.
    Ref[:, i] = reference_result_a_b
    # then another half batch is A 0, d(A, 0).
    Ref[:, batch_sz//2 + i] = reference_result_a_0


def test_basic_batch():
    """ Test calling twed_batch using GPUarrays. """

    # Call TWED
    Res = twed_batch(AA, TAA, BB, TBB, nu, lamb, degree)

    print('Python Device Batch cuTWED distances:')
    print(Res)
    # print("Ref\n",Ref)
    assert np.allclose(Ref, Res)


def test_basic_batch_float():
    """ Test calling twed_batch using GPUarrays. """

    AAf = AA.astype(np.float32)
    TAAf = TAA.astype(np.float32)
    BBf = BB.astype(np.float32)
    TBBf = TBB.astype(np.float32)

    # Call TWED
    Res = twed_batch(AAf, TAAf, BBf, TBBf, nu, lamb, degree)

    print('Python Device Batch cuTWED distances (single precision):')
    print(Res)
    # print("Ref\n",Ref)
    assert np.allclose(Ref, Res)


def test_basic_batch_tril():
    """ Test calling twed_batch using GPUarrays, lower triangle optimization.  """

    # Call TWED
    Res = twed_batch(AA, TAA, BB, TBB, nu, lamb, degree, tri='tril')

    print('Python Device Batch cuTWED distances tril:')
    print(Res)
    # print("Ref\n",Ref)
    lower_tri = np.tril(Ref, -1)
    assert np.allclose(lower_tri, Res)


def test_basic_batch_triu():
    """
    Test calling twed_batch using GPUarrays, upper triangle optimization.
    Note uses cuBLAS for transpose, but its kind of stupid to do this
    instead of tril in most cases...
    """

    # Call TWED
    Res = twed_batch(BB, TBB, AA, TAA, nu, lamb, degree, tri='triu')

    print('Python Device Batch cuTWED distances triu:')
    print(Res)
    # print("Ref\n",Ref)
    upper_tri = np.triu(Ref, 1)
    assert np.allclose(upper_tri, Res)

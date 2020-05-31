#!/usr/bin/env python
"""
Demonstrates usage of twed_batch_dev cuTWED call.

Copyright 2020 Garrett Wright, Gestalt Group LLC
"""
import sys

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.gpuarray as gpuarray
from numpy.random import RandomState

# Import the twed_dev function from cuTWED
from cuTWED import twed_batch_dev

# Set algo params
nu = 1.
lamb = 1.
degree = 2

# Expected results
reference_result_a_b = 54.54317
reference_result_a_0 = 44.739147


def generate(n=10, batch_sz=100):
    """ Generate some junk data. """

    # First generate the same time series uses in other tests.
    rng = RandomState(42)
    noise1 = rng.randn(n)

    TA = np.arange(n, dtype=np.float64)
    A = np.sin(TA) + np.sin(TA/10) + noise1

    m = 2 * n
    noise2 = rng.randn(m)
    TB = np.arange(m, dtype=np.float64)
    B = np.sin(TB) + np.sin(TB/10) + noise2

    # We now make some batches.

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

    return AA, TAA, BB, TBB, Ref


def test_basic_batch_dev(n=10, batch_sz=100):
    """ Test calling twed_batch_dev using GPUarrays. """

    AA, TAA, BB, TBB, Ref = generate(n, batch_sz)

    AA_dev = gpuarray.to_gpu(AA)
    TAA_dev = gpuarray.to_gpu(TAA)
    BB_dev = gpuarray.to_gpu(BB)
    TBB_dev = gpuarray.to_gpu(TBB)

    # Call TWED
    Res = twed_batch_dev(AA_dev, TAA_dev, BB_dev, TBB_dev, nu, lamb, degree)

    print('Python Device Batch cuTWED distances:')
    print(Res)
    # print("Ref\n",Ref)
    if n == 10:
        assert np.allclose(Ref, Res)


def test_basic_batch_dev_float(n=10, batch_sz=100):
    """ Test calling twed_batch_dev using GPUarrays. """

    AA, TAA, BB, TBB, Ref = generate(n, batch_sz)

    AA_dev = gpuarray.to_gpu(AA.astype(np.float32))
    TAA_dev = gpuarray.to_gpu(TAA.astype(np.float32))
    BB_dev = gpuarray.to_gpu(BB.astype(np.float32))
    TBB_dev = gpuarray.to_gpu(TBB.astype(np.float32))

    # Call TWED
    Res = twed_batch_dev(AA_dev, TAA_dev, BB_dev, TBB_dev, nu, lamb, degree)

    print('Python Device Batch cuTWED distances (single precision):')
    print(Res)
    # print("Ref\n",Ref)
    if n == 10:
        assert np.allclose(Ref, Res)


if __name__ == "__main__":
    # Probably should just bring in argparse next time.
    if len(sys.argv) > 3:
        raise RuntimeError(f"Usage ./{sys.argv[0]} <n> <batch_sz>")
    elif len(sys.argv) == 3:
        test_basic_batch_dev(n=int(sys.argv[1]), batch_sz=int(sys.argv[2]))
    elif len(sys.argv) == 2:
        test_basic_batch_dev(n=int(sys.argv[1]))
    else:
        test_basic_batch_dev()

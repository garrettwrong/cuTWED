#!/usr/bin/env python
"""
Validates CTWED and cuTWED with a well known synthetic control dataset:

http://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series

We compare the all-pairs distance matrix as computed by
several cuTWED implementations with Marteau's reference code.

Note this collection of tests is skipped by default.

Copyright 2020 Garrett Wright, Gestalt Group LLC
"""

from timeit import default_timer as timer

# import matplotlib.pylab as plt
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.gpuarray as gpuarray
import pytest
# import seaborn as sns
from tqdm import tqdm

# Import functions from cuTWED
from cuTWED import ctwed
from cuTWED import twed
from cuTWED import twed_batch
from cuTWED import twed_batch_dev
from cuTWED import twed_dev

# Load and preprocess the raw data:
fn = 'tests/data/synthetic_control-mld/synthetic_control.data'
TS = np.zeros((600, 60))
with open(fn, 'r') as fh:
    lines = fh.readlines()
    for n, line in enumerate(lines):
        line = line.strip().split()
        for t, item in enumerate(line):
            TS[n][t] = item

# Get shape
nseries, nsamples = TS.shape

# Make a time axis
T = np.arange(nsamples, dtype=np.float64)
# We stack our fabricated time axis to match our nseries batch
TT = np.tile(T, (nseries, 1))

# Set algo params
nu = 1.
lamb = 1.
degree = 2

# Prep for GPU based calls
T_d = gpuarray.to_gpu(T)
TS_d = gpuarray.to_gpu(TS)
TT_d = gpuarray.to_gpu(TT)


@pytest.mark.slow
def test_ref():
    """ Test running the synthetic control dataset"""

    DistanceMatrix = np.zeros((nseries, nseries))

    for row, A in enumerate(tqdm(TS)):
        for col, B in enumerate(TS):
            if col < row:
                continue
            dist = ctwed(A, T, B, T, nu, lamb, degree)
            DistanceMatrix[row][col] = dist
            # print(f'Python CTWED distance:\t{row}\t{col}\t{dist:f}')

    name = 'synthetic_distance_matrix_ref'
    with open(f'{name}.npy', 'wb') as fh:
        np.save(fh, DistanceMatrix)

    # with sns.axes_style("white"):
    #     sns.heatmap(DistanceMatrix, square=True,  cmap="YlGnBu")
    #     plt.savefig(f'{name}.png')

    return DistanceMatrix


@pytest.mark.slow
def test_cutwed():
    """ Test running the synthetic control dataset"""

    DistanceMatrix = np.zeros((nseries, nseries))

    for row, A in enumerate(tqdm(TS)):
        for col, B in enumerate(TS):
            if col < row:
                continue
            dist = twed(A, T, B, T, nu, lamb, degree)
            DistanceMatrix[row][col] = dist
            # print(f'Python CTWED distance:\t{row}\t{col}\t{dist:f}')

    name = 'synthetic_distance_matrix_cutwed'
    with open(f'{name}.npy', 'wb') as fh:
        np.save(fh, DistanceMatrix)

    # with sns.axes_style("white"):
    #     sns.heatmap(DistanceMatrix, square=True,  cmap="YlGnBu")
    #     plt.savefig(f'{name}.png')

    return DistanceMatrix


@pytest.mark.slow
def test_cutwed_dev():
    """ Test running the synthetic control dataset"""

    DistanceMatrix = np.zeros((nseries, nseries))

    for row in tqdm(range(nseries)):
        for col in range(nseries):
            if col < row:
                continue
            dist = twed_dev(TS_d[row], T_d, TS_d[col], T_d, nu, lamb, degree)
            DistanceMatrix[row][col] = dist
            # print(f'Python CTWED distance:\t{row}\t{col}\t{dist:f}')

    name = 'synthetic_distance_matrix_cutwed_dev'
    with open(f'{name}.npy', 'wb') as fh:
        np.save(fh, DistanceMatrix)

    # with sns.axes_style("white"):
    #     sns.heatmap(DistanceMatrix, square=True,  cmap="YlGnBu")
    #     plt.savefig(f'{name}.png')

    return DistanceMatrix


@pytest.mark.slow
def test_batch():
    """ Test running the synthetic control dataset"""

    DistanceMatrix = twed_batch(TS, TT, TS, TT, nu, lamb, degree)

    name = 'synthetic_distance_matrix_cutwed_batch'
    with open(f'{name}.npy', 'wb') as fh:
        np.save(fh, DistanceMatrix)

    # with sns.axes_style("white"):
    #     sns.heatmap(DistanceMatrix, square=True,  cmap="YlGnBu")
    #     plt.savefig(f'{name}.png')

    return DistanceMatrix


@pytest.mark.slow
def test_batch_dev():
    """ Test running the synthetic control dataset"""

    DistanceMatrix = twed_batch_dev(TS_d, TT_d, TS_d, TT_d, nu, lamb, degree)

    name = 'synthetic_distance_matrix_cutwed_batch_dev'
    with open(f'{name}.npy', 'wb') as fh:
        np.save(fh, DistanceMatrix)

    # with sns.axes_style("white"):
    #     sns.heatmap(DistanceMatrix, square=True,  cmap="YlGnBu")
    #     plt.savefig(f'{name}.png')

    return DistanceMatrix


if __name__ == "__main__":
    results = {}
    for f in [test_ref,
              test_cutwed,
              test_cutwed_dev,
              test_batch,
              test_batch_dev]:
        fn = f.__name__
        print(f'Begin {fn}...')
        tic = timer()
        results[fn] = f()
        toc = timer()
        delta = toc - tic
        print(f'Time Elapsed {fn}:\t{delta}')
        diff = np.triu(results[fn]) - np.triu(results[test_ref.__name__])
        mae = np.max(np.abs(diff))
        rms = np.sqrt(np.mean(diff**2))
        print(f'Max Abs Error {fn}:\t{mae}')
        print(f'RMS Error {fn}:\t{rms}')
        print('...end')
    print('Done.')

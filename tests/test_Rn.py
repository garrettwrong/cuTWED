#!/usr/bin/env python
"""
Performs three way validation between:
* Wiki TWED python code (originally supports R^N norms).
* twed.c that has been modified to support R^N
* cutwed in single and batch modes.
"""

import os

import numpy as np
from CTWED import ctwed

from cuTWED import twed
from cuTWED import twed_batch


# Reference Implementation from wiki:
# https://en.wikipedia.org/wiki/Time_Warp_Edit_Distance
def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1 / p)
    # return np.linalg.norm(A-B)
    # cost = np.sum(np.power(np.abs(A - B), p))
    # return np.sqrt(cost)


def pytwed(A, timeSA, B, timeSB, nu, _lambda):
    # [distance, DP] = TWED( A, timeSA, B, timeSB, lambda, nu )
    # Compute Time Warp Edit Distance (TWED) for given time series A and B
    #
    # A      := Time series A (e.g. [ 10 2 30 4])
    # timeSA := Time stamp of time series A (e.g. 1:4)
    # B      := Time series B
    # timeSB := Time stamp of time series B
    # lambda := Penalty for deletion operation
    # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # Reference :
    #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    #    http://people.irisa.fr/Pierre-Francois.Marteau/

    # Check if input arguments
    if len(A) != len(timeSA):
        print("The length of A is not equal length of timeSA")
        return None, None

    if len(B) != len(timeSB):
        print("The length of B is not equal length of timeSB")
        return None, None

    if nu < 0:
        print("nu is negative")
        return None, None

    # Add padding
    A = np.array([0] + list(A))
    timeSA = np.array([0] + list(timeSA))
    B = np.array([0] + list(B))
    timeSB = np.array([0] + list(timeSB))

    n = len(A)
    m = len(B)
    # Dynamical programming
    DP = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            C = np.ones((3, 1)) * np.inf
            # Deletion in A
            C[0] = (
                DP[i - 1, j]
                + Dlp(A[i - 1], A[i])
                + nu * (timeSA[i] - timeSA[i - 1])
                + _lambda
            )
            # Deletion in B
            C[1] = (
                DP[i, j - 1]
                + Dlp(B[j - 1], B[j])
                + nu * (timeSB[j] - timeSB[j - 1])
                + _lambda
            )
            # Keep data points in both time series
            C[2] = (
                DP[i - 1, j - 1]
                + Dlp(A[i], B[j])
                + Dlp(A[i - 1], B[j - 1])
                + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
            )
            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = np.min(C)
    distance = DP[n - 1, m - 1]
    return distance, DP


# Load reference data
fn = os.path.join(os.path.dirname(__file__), 'data', 'mnist_4x4.npz')
data = np.load(fn)
A = data['A'].astype(np.float64)
AA = data['AA'].astype(np.float64)
B = data['B'].astype(np.float64)
BB = data['BB'].astype(np.float64)
T = data['T'].astype(np.float64)
DIST = data['DIST']

assert len(A) == len(B)
assert len(A) == len(T)
assert len(AA) == len(BB)

TT = np.tile(T, (len(AA), 1))
sz = (AA.shape[0], BB.shape[0])

# Set algo params
nu = 1.
lamb = 1.
degree = 2

single_ref = 7948.0187956562495


def test_python():
    print("Computing pytwed")
    dist, _ = pytwed(A, T, B, T, nu, lamb)
    print(f"Distance: {dist}")
    assert np.allclose(single_ref, dist)


def test_single_ctwed():
    print("Computing ctwed")
    dist = ctwed(A, T, B, T, nu, lamb, degree)
    print(f"Distance: {dist}")
    assert np.allclose(single_ref, dist)


def test_single_cutwed():
    print("Computing cutwed")
    dist = twed(A, T, B, T, nu, lamb, degree)
    print(f"Distance: {dist}")
    assert np.allclose(single_ref, dist)


def test_batch():
    D = np.zeros(sz)
    print("Computing twed batch")
    D = twed_batch(AA, TT, BB, TT, nu, lamb, degree)
    assert np.allclose(np.triu(D), DIST)


def test_multi_twed():
    D = np.zeros(sz)
    print("Computing ctwed batch")
    for row, A in enumerate(AA):
        for col, B in enumerate(BB):
            if col < row:
                continue
            dist = twed(A, T, B, T, nu, lamb, degree)
            D[row][col] = dist
    assert np.allclose(D, DIST)


def test_multi_pytwed():
    D = np.zeros(sz)
    print("Computing ctwed batch")
    for row, A in enumerate(AA):
        for col, B in enumerate(BB):
            if col < row:
                continue
            dist, _ = pytwed(A, T, B, T, nu, lamb)
            D[row][col] = dist
    assert np.allclose(D, DIST)


def test_multi_ctwed():
    D = np.zeros(sz)
    print("Computing ctwed batch")
    for row, A in enumerate(AA):
        for col, B in enumerate(BB):
            if col < row:
                continue
            dist = ctwed(A, T, B, T, nu, lamb, degree)
            D[row][col] = dist
    assert np.allclose(D, DIST)

#!/usr/bin/env python

import numpy as np
from numpy.random import RandomState

# Import the twed function from cuTWED
from cuTWED import twed


# Generate some junk data
n = 10000
rng = RandomState(42)
noise1 = rng.randn(n)
noise2 = rng.randn(n)
    
TA = np.arange(n, dtype=np.float64)
A = np.sin(TA) + np.sin(TA/10) + noise1
    
TB = np.arange(n, dtype=np.float64)
B = np.sin(TB) + np.sin(TB/10) + noise2

# Set algo params
nu = 1.
lamb = 1.
degree = 2

# Call TWED
dist = twed(A, TA, B, TB, nu, lamb, degree)

print('Python cuTWED distance: {:f}'.format(dist))
# ref: 29869

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
"""
Validates CTWED and cuTWED with a well known synthetic dataset:

http://archive.ics.uci.edu/ml/datasets/Pseudo+Periodic+Synthetic+Time+Series

We compare the all-pairs distance matrix as computed by
several cuTWED implementations with Marteau's reference code.

Note this collection of tests is skipped by default.

Copyright 2020 Garrett Wright, Gestalt Group LLC
"""


# In[2]:


# Note, in addition to the regular cuTWED depeneds you may need to
#%pip install matplotlib seaborn

from timeit import default_timer as timer

import gzip
import matplotlib.pylab as plt
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.gpuarray as gpuarray
import pytest
import seaborn as sns
from itertools import product
from tqdm import tqdm

# Import functions from cuTWED
from CTWED import ctwed
from cuTWED import twed
from cuTWED import twed_batch
from cuTWED import twed_batch_dev
from cuTWED import twed_dev


# In[ ]:





# In[3]:


# Load and preprocess the raw data:
fn = 'synthetic.data.gz'

with gzip.open(fn, 'r') as fh:
    lines = fh.readlines()

# Get Shape
# Note this data is stored as (10) column vectors.
# In the process of assigning into array, we will also put in C order.
# This yields 10 rows, where each row is a time series of 100001 points.
# Your data may vary.  The other synthetic control set is opposite of this, for example.
nsamples = len(lines)
nseries = len(lines[0].split())

# Process the data into an array
TS = np.zeros((nseries, nsamples),dtype=np.float64)
for t, line in enumerate(lines):
    line = line.strip().split()
    for n, item in enumerate(line):
        TS[n][t] = item

# Make a time axis
T = np.linspace(0, 1, nsamples, dtype=np.float64)
# For batched calls we stack our fabricated time axis to match our nseries batch
TT = np.tile(T, (nseries, 1))
# Note this is a little clunky, but it generalizes for cases where your times are not identical,
# like event driven data.  Its easier to just stack an identical vector then be confused with too many API methods...


# If you want to run a smaller problem (use only some of the data), reduce subset...
#subset = False
subset=20000
if subset:
    T = np.linspace(0, 1, subset, dtype=np.float64)
    TT = np.tile(T, (nseries, 1))
    TS = TS[:, 0:subset].copy() # note the copy is important, if you get a "view" from np, your results will be crap.


# In[4]:


TS.shape


# In[5]:


# Set algo params
nu = 1.
lamb = 1.
degree = 2


# In[6]:


# Compute all TWED between series in the synthetic index dataset, and plot

# This takes a while for the full dataset, about enough time to go for a walk,
#    but fits fine; uses only about 150MiB of GPU memory for the full size..
# I doubt this was reasonably computable at all prior to cuTWED....
DistanceMatrix_batch = twed_batch(TS, TT, TS, TT, nu, lamb, degree)

with sns.axes_style("white"):
    sns.heatmap(np.triu(DistanceMatrix_batch), square=True,  cmap="YlGnBu")
    plt.plot()


# In[7]:


DistanceMatrix_batch[0,-1]


# In[8]:


# NOTE YOU WILL NEED A HIGH MEM MACHINE TO RUN THE FULL DATASET.....
from psutil import virtual_memory
mem = virtual_memory()
ram = mem.total
if ram < subset**2 * 8  * 1.1:
    raise MemoryError("Sorry I don't think you have enough RAM for this using ctwed..."
                      " Try adjusting the variable 'subset' to a smaller amount to run a reduced problem."
                      " I chose to emit this poorly worded message instead of thrashing you computer...")

# Test running the synthetic index dataset"""

DistanceMatrix_ref = np.zeros((nseries, nseries))
for (row, A), (col,B) in tqdm(product(enumerate(TS), enumerate(TS)), total=len(TS)**2):
        if col < row:
            continue
        dist = ctwed(A, T, B, T, nu, lamb, degree)
        DistanceMatrix_ref[row][col] = dist

with sns.axes_style("white"):
    sns.heatmap(DistanceMatrix_ref, square=True,  cmap="YlGnBu")
    plt.plot()


# In[9]:


DistanceMatrix_ref[0,-1]


# In[10]:


# Test running the synthetic index dataset"""

DistanceMatrix_cu = np.zeros((nseries, nseries))
for (row, A), (col,B) in tqdm(product(enumerate(TS), enumerate(TS)), total=len(TS)**2):
        if col < row:
            continue
        dist = twed(A, T, B, T, nu, lamb, degree)
        DistanceMatrix_cu[row][col] = dist

with sns.axes_style("white"):
    sns.heatmap(DistanceMatrix_cu, square=True,  cmap="YlGnBu")
    plt.plot()


# In[11]:


DistanceMatrix_cu[0,-1]


# In[12]:


np.max(np.abs(DistanceMatrix_cu - DistanceMatrix_ref))


# In[13]:


np.max(np.triu(DistanceMatrix_batch - DistanceMatrix_ref))


# In[ ]:





# In[ ]:





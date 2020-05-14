#!/usr/bin/env python

import os
import pickle
import sys
import ctypes
import numpy as np

import matplotlib.pyplot as plt

from collections import defaultdict
from time import perf_counter

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.gpuarray as gpuarray
    import cuTWED

    # we will assume you have compiled twed into a shared library
    ## maybe with `gcc -g -fPIC -shared -Wall -O2 twed.c -o twed.so`
    # bring it in with ctypes
    twedc = ctypes.CDLL('twed.so')
    
    twedc.CTWED.restype = None
    twedc.CTWED.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                            ctypes.c_void_p,
                            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                            ctypes.c_void_p,
                            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
    ]

    twedc.CTWED_nd.restype = None
    twedc.CTWED_nd.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                               ctypes.c_void_p,
                               np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                               ctypes.c_void_p,
                               np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                               ctypes.c_void_p,
                               ctypes.c_void_p,
                               ctypes.c_void_p,
                               ctypes.c_void_p,
                               ctypes.c_int
    ]

except Exception as e:
    print(e)
    pass

#repro
np.random.seed(42)

class Argus:
    def __init__(self, nA, nB=None, ndim=1, dtype=np.float64):
        self.dtype = dtype
        self.ndim = ndim
        if nB is None:
            nB = nA
        self.nA = nA
        self.nB = nB

        self.TA = np.arange(nA, dtype=self.dtype)
        self.TB = np.arange(nB, dtype=self.dtype)

        self._noise1 = np.random.randn(nA)
        self._noise2 = np.random.randn(nB)

        self.A = np.sin(self.TA) + np.sin(self.TA/10) + self._noise1
        self.B = np.sin(self.TB) + np.sin(self.TB/10) + self._noise2

        self.A = np.stack([self.A]*self.ndim, axis=1)
        self.B = np.stack([self.B]*self.ndim, axis=1)
        
class Mollusk(Argus):
    def __init__(self, nA, nB=None, ndim=1, dtype=np.float64):
        Argus.__init__(self, nA=nA, nB=nB, ndim=ndim, dtype=dtype)
        self.TA = gpuarray.to_gpu(self.TA)
        self.TB = gpuarray.to_gpu(self.TB)

        self.A = gpuarray.to_gpu(self.A)
        self.B = gpuarray.to_gpu(self.B)


def time_twed(dim, p=17):
    D = {}
    for x in ['twedc', 'cuTWED', 'cuTWED_dev']:
        D[x] = {'times':  defaultdict(list),
                'distances':  defaultdict(list)}

    for p in range(p)[::-1]:
        n = 2**p
        print(f'Size {n}')

        data = Argus(n, ndim=dim)
        #algo params
        nu = 1.
        lamb = 1.
        degree = 2
        dist = ctypes.c_double(0.)

        print("reference iter: ", end = '', flush=True)
        for i in range(10):
            print('.', end = '', flush=True)
            if data.ndim == 1:
                tic = perf_counter()
                twedc.CTWED(data.A,
                            ctypes.byref(ctypes.c_int(data.nA)),
                            data.TA,
                            data.B,
                            ctypes.byref(ctypes.c_int(data.nB)),                        
                            data.TB,
                            ctypes.byref(ctypes.c_double(nu)),
                            ctypes.byref(ctypes.c_double(lamb)),
                            ctypes.byref(ctypes.c_int(degree)),
                            ctypes.byref(dist)
                )
                toc = perf_counter()
            else:
                tic = perf_counter()
                twedc.CTWED_nd(data.A,
                               ctypes.byref(ctypes.c_int(data.nA)),
                               data.TA,
                               data.B,
                               ctypes.byref(ctypes.c_int(data.nB)),                        
                               data.TB,
                               ctypes.byref(ctypes.c_double(nu)),
                               ctypes.byref(ctypes.c_double(lamb)),
                               ctypes.byref(ctypes.c_int(degree)),
                               ctypes.byref(dist),
                               data.ndim
                )
                toc = perf_counter()

            t = toc - tic
            D['twedc']['times'][n].append(t)
            D['twedc']['distances'][n].append(dist)
        print()
        
        print("Managed iter: ", end = '', flush=True)
        for i in range(10):
            print('.', end = '', flush=True)
            tic = perf_counter()
            cud = cuTWED.twed(data.A, data.TA, data.B, data.TB, nu, lamb, degree)
            toc = perf_counter()
            t = toc - tic
            D['cuTWED']['times'][n].append(t)
            D['cuTWED']['distances'][n].append(dist)
        print()
        
        print("Device iter: ", end = '', flush=True)            
        data = Mollusk(n, ndim=dim)
        for i in range(10):
            print('.', end = '', flush=True)
            tic = perf_counter()
            cud_d = cuTWED.twed_dev(data.A, data.TA, data.B, data.TB, nu, lamb, degree)
            toc = perf_counter()
            t = toc - tic
            D['cuTWED_dev']['times'][n].append(t)
            D['cuTWED_dev']['distances'][n].append(dist)
        print()

    return D
    

if __name__ == "__main__":

    dim = 1
    if len(sys.argv)>1:
        dim = int(sys.argv[1])
    raw_data_fn = f'raw_{dim}d.dat'
    if os.path.exists(raw_data_fn):
        print(f'Previous {raw_data_fn} found, loading...')
        # load raw data
        with open(raw_data_fn, 'rb') as fh:
            D = pickle.load(fh)
    else:
        
        print(f'Generating {raw_data_fn}... (might take a while)')
        # generate it        
        D = time_twed(dim=dim, p=17)
        with open(raw_data_fn, 'wb') as fh:
            pickle.dump(D, fh)

    # wall time comp 1d
    for x, v in D.items():
        D[x]['tavg'] = {}
        print(x)
        for n, vvv in v['times'].items():
            D[x]['tavg'][n] = np.average(vvv)
            print(n, D[x]['tavg'][n])

        X = sorted(v['tavg'].keys())
        Y = [v['tavg'][n] for n in X]

        linestyle = '-'
        if x == 'cuTWED_dev':
            linestyle = '--'
        plt.plot(X, Y, label=x, marker='.', linestyle=linestyle)
        
    plt.title("Time to Solution")
    plt.xlabel(f'Input Time-Series Length, {dim}-Dimensional')
    plt.ylabel('Time (secs)')
    plt.legend()
    plt.show()

    speedup = [D['twedc']['tavg'][n]/D['cuTWED']['tavg'][n] for n in X]
    plt.plot(X, speedup, marker='.')
    plt.title("Speedup")
    plt.xlabel(f'Input Time-Series Length, {dim}-Dimensional')
    #plt.ylabel('')
    plt.show()
    

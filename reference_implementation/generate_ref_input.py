#!/usr/bin/env python

import sys

import numpy as np
from numpy.random import RandomState


def main(n):
    rng = RandomState(42)
    noise1 = rng.randn(n)

    TA = np.arange(n, dtype=np.int32)
    A = np.sin(TA) + np.sin(TA/10) + noise1

    m = 2 * n
    noise2 = rng.randn(m)
    TB = np.arange(m, dtype=np.int32)
    B = np.sin(TB) + np.sin(TB/10) + noise2

    f = open('reference_arrays.h', 'w')

    s = ''
    s = '#ifndef REAL_t'
    s = '#define REAL_t double'
    s = '#endif'
    s = '\n\n'

    s += 'int nA = {};\n'.format(n)
    s += 'REAL_t TA[{}] = {{'.format(n)
    s += ', '.join([str(x)+'.' for x in TA])
    s += '};\n\n'
    f.write(s)

    s = ''
    s += 'int nB = {};\n'.format(m)
    s += 'REAL_t TB[{}] = {{'.format(m)
    s += ', '.join([str(x)+'.' for x in TB])
    s += '};\n\n'
    f.write(s)

    s = ''
    s += 'REAL_t A[{}] = {{'.format(n)
    s += ','.join([str(x) for x in A])
    s += '};\n\n'
    f.write(s)

    s = ''
    s += 'REAL_t B[{}] = {{'.format(m)
    s += ','.join([str(x) for x in B])
    s += '};\n\n'
    f.write(s)

    f.close()


if __name__ == "__main__":
    n = 10000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    main(n)

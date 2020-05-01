#!/usr/bin/env python

import sys
import numpy as np
from numpy.random import RandomState

def main(n=10000):
    rng = RandomState(42)
    noise1 = rng.randn(n)
    noise2 = rng.randn(n)
    
    TA = np.arange(n, dtype=np.int32)
    A = np.sin(TA) + np.sin(TA/10) + noise1
    
    TB = np.arange(n, dtype=np.int32)
    B = np.sin(TB) + np.sin(TB/10) + noise2

    f = open('reference_arrays.h', 'w')
    
    s = ''
    s += 'int nA = {};\n'.format(n);
    s += 'double TA[{}] = {{'.format(n)
    s += ', '.join([str(x)+'.' for x in TA])
    s += '};\n\n'
    f.write(s)

    s = ''
    s += 'int nB = {};\n'.format(n);
    s += 'double TB[{}] = {{'.format(n)
    s += ', '.join([str(x)+'.' for x in TB])
    s += '};\n\n'
    f.write(s)

    s = ''
    s += 'double A[{}] = {{'.format(n)
    s += ','.join([str(x) for x in A])
    s += '};\n\n'
    f.write(s)

    s = ''
    s += 'double B[{}] = {{'.format(n)
    s += ','.join([str(x) for x in B])
    s += '};\n\n'
    f.write(s)

    f.close()
    

if __name__ == "__main__":
    n = 10000
    if len(sys.argv)>1:
        n = int(sys.argv[1])

    main(n)

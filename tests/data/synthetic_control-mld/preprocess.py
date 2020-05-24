#!/usr/bin/python
"""
Preprocess synthetic_control-mld text dataset into np array.
"""

import numpy as np


def main():
    fn = 'synthetic_control.data'
    with open(fn, 'r') as fh:
        lines = fh.readlines()

    TS = np.zeros((600, 60))
    for n, line in enumerate(lines):
        line = line.strip().split()
        for t, item in enumerate(line):
            TS[n][t] = item

    print(TS)
    with open('synthetic_control.npy', 'wb') as fh:
        np.save(fh, TS)


if __name__ == "__main__":
    main()

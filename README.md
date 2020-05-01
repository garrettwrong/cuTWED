# cuTWED

CUDA adaptation of the Time Warp Edit Distance algorithm.

## About

This is parallel implementation of the Time Warp Edit Distance.
The original algorithm was described by Marteau in:
    `Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching (2009)`.
A variant of the code from that paper is included in `reference_implementation`,
and was used to compare results.

The original TWED algorithm is `O(n^2)`, while here we are roughly `O(n * n/p)` for p (CUDA) cores.
Unfortunately, the access patterns are not optimal, and we also must alloc/memcpy to the device,
so we only achieve speedups closer to 4-10x for practical problem sizes.

## Getting Started

### Requirements

You will need NVCC and a CUDA capable card that can fit your problem.
The problem is `O(n^2)` in memory for time series of length n.

### Building

```
make -j
```

The makefile is intentionally simple, so it can be edited to suit your needs.

This builds on OSX and multiple flavors of Linux with more/less default installs.
If you would like to pursue Windows, please do and report back, I can try to integrate the effort.

### Using in other programs

#### C/C++

In C/C++ you should be able to `include "cuTWED.h"` and link with the shared library `libcuTWED.so`.

#### Python

For Python I have included basic bindings in `cuTWED.py` and I use it in `example/test.py`.
This requires that you have built the library, and have it available in your `LD_LIBRARY_PATH`.
The following works for me out of the box on linux for quick development:
```
make -j
python examples/test.py
```

## License

GPLv3

Copyright 2020 Garrett Wright, Gestalt Group LLC

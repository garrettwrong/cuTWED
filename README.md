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

The makefile is pretty basic so it can be edited to suit your needs.

This builds on OSX and multiple flavors of Linux with more/less default installs.
If you would like to pursue Windows, please do and report back, I can try to integrate the effort.

### Using in other programs

#### C/C++

In C/C++ you should be able to `include "cuTWED.h"` and link with the shared library `libcuTWED.so`.
This is what I do in `test.x`.

A 32bit library is also built `libcuTWED_32.so`, containing functions with the same names.
You may choose which one is suitable for your application.  This is what I use in `test_32.x`.

There are two main ways to invoke the cuTWED alogorithm, `twed` and `twed_dev`.
First is the most common way, where you pass C arrays on the host,
and the library manages device memory and transfers for you.

Alternatively, if you are already managing GPU memory,
you may use twed_dev which expects pointers to memory to reside on the gpu.
I have also provided malloc, copy, and free helpers in case it makes sense to reuse memory.
_All logic and size checks for such advanced cases are expected to be owned by the user._

#### Python

For Python I have included basic bindings in `cuTWED.py` and I use it in `example/test.py`.
This requires that you have built the library, and have it available in your `LD_LIBRARY_PATH`.
The following works for me out of the box on linux for quick development:
```
make -j
python examples/test.py
```

I have also wrapped up the GPU only memory methods in python, using PyCUDA gpuarrays.  Examples in
double and single precision are in `example/test_dev.py`.

## License

GPLv3

Copyright 2020 Garrett Wright, Gestalt Group LLC

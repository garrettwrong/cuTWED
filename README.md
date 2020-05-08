# cuTWED

A linear memory CUDA adaptation of the Time Warp Edit Distance algorithm.

## About

This is a novel parallel implementation of the Time Warp Edit Distance.
The original algorithm was described by Marteau in:
    `Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching (2009)`.
A variant of the code from that paper is included in `reference_implementation`,
and was used to compare results. There is also a
[wiki page](https://en.wikipedia.org/wiki/Time_Warp_Edit_Distance).

The original TWED algorithm is `O(n^2)` in time and space.
This algorithm is roughly `O(n * n/p)` in time for p (CUDA) cores.
Most importantly, cuTWED is linear in memory,
requiring storage for roughly `6*nA + 6*nB` elements.

In the process of understanding the dynamic program data dependencies in order to parallelize
towards the CUDA architecture, I devised a method with improved memory access patterns
and the ability to massively parallelize over large problems.
This is accomplished by a procession of a three diagonal band moving across the dynamic
program matrix in `nA+nB-1` steps.  No `O(n^2)` matrix is required anywhere.
Note, this is not an approximation method.  The full and complete TWED dynamic program
computation occurs with linear storage.

The code is provided in a library that has methods for `double` and `float` precision.
It admits input time series in `R^N` as arrays of N-dimensional arrays in C-order
(time is the slow moving axis).

For typical problems computable by the original TWED implementation,
utilizing cuTWED and thousands of CUDA cores achieves great speedups.
More so, the linear memory footprint allows for the computation
of previously intractable problems.  Large problems, large systems
of inputs can be computed much more effectively now.

Some speed comparisons and a more formal explanation will follow.

## Getting Started

### Requirements

You will need NVCC and a CUDA capable card that can fit your problem.
The problem is roughly 12*n in memory for time series of length n.

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
This is what I do in `test.x`.  The public methods are extern C mangled and should be usable
from both C and C++ without issue.

Float (32bit) versions of all the public methods are included in the shared library.
They simply have an `f` appended, for example, `twedf` is the float version of `twed`.
You may choose which one is suitable for your application.  I use floats in `testf.x`.

There are currently two main ways to invoke the cuTWED alogorithm, `twed` and `twed_dev`.
First `twed` is the most common way, where you pass C arrays on the host,
and the library manages device memory and transfers for you.

Alternatively, if you are already managing GPU memory,
you may use twed_dev which expects pointers to memory that resides on the gpu.
I have also provided malloc, copy, and free helpers in case it makes sense to reuse memory.
See `cuTWED.h`.
_All logic and size checks for such advanced cases are expected to be owned by the user._

Future plans include a mode for streaming batched mode optimized for large systems.

#### Python

For Python I have included basic bindings in `cuTWED.py` and I use it in `example/test.py`.
This requires that you have built the library, and have it available in your `LD_LIBRARY_PATH`.
You may not need the LD_LIBRARY_PATH line, but I have included it here for completeness.
The following works for me out of the box on my linux for quick development:

```
make -j
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
python examples/test.py
```

I have also wrapped up the GPU only memory methods in python, using PyCUDA gpuarrays.
Examples in double and single precision are in `example/test_dev.py`.

## License

GPLv3

Copyright 2020 Garrett Wright, Gestalt Group LLC

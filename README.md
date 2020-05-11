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

For the CUDA code you will need NVCC, a CUDA capable card and CMake.
Otherwise, the CUDA code has no dependencies.

For the Python binding `pip` manages the specific depends and installation of the Python
interface after you have built the main CUDA C library.  Generally requires
`numpy`, `pycuda`, and `cffi`.  I recommend you use virtualenv or conda to manage python.

### Building

Building has two stages.  First the CUDA C library is built.  It can be permanently installed to your system,
or just append the path to  `libcuTWED.so` onto your `LD_LIBRARY_PATH`.  That can be either temorarily or in your
`.bashrc` etc.

If you would like the python bindings, I have (with great pain) formed a pip installable package for the bindings.

#### Building the core CUDA C library

Note you may customize the call to `cmake` below with flags like `-DCMAKE_INSTALL_PREFIX=/opt/`, or other flags
you might require.

```
# git a copy of the source code
git clone https://github.com/garrettwrong/cuTWED
cd cuTWED

# setup a build area
mkdir build
cd build
cmake ../ # configures/generates makefiles

# make
make -j  # builds the software
```

This should create several files in the `build` directory including `libcuTWED.so`, `*.h` headers, and some other stuff.
To install to your system (may require sudo):

```
make install
```

If you just want to temporarily have the library available on your linux machine you can just use the LD path.
This makes no changes to your system outside the repo and this shell process.
```
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
```

#### Python
Once you have the CUDA C library readied, we can use `pip` for the python bindings.

```
pip install cuTWED
```

If you are a developer you might prefer pip use your local checkout instead.
```
pip install -e .
```

### Using cuTWED in other programs

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

```
from cuTWED import twed
```

For Python I have included basic pip installable python bindings.  I use it in `example/test.py`.
If you are curious, these are implemented by a `cffi` backend which parses the C header.
which is built for you by `setuptools`. The main python interface is in `cuTWED.py`.
This requires that you have built the library, and have it available in your `LD_LIBRARY_PATH`.

I have also wrapped up the GPU only memory methods in python, using PyCUDA gpuarrays.
Examples in double and single precision are in `example/test_dev.py`.

```
from cuTWED import twed_dev
```

## License

GPLv3

Copyright 2020 Garrett Wright, Gestalt Group LLC

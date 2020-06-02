/*  Copyright 2020 Garrett Wright, Gestalt Group LLC

    This file is part of cuTWED.

    cuTWED is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    cuTWED is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with cuTWED.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cublas_v2.h"

#include "cuTWED.h"

/* Flip on debug prints.
   Warning, I don't recomend for large inputs, 10x20 ish is fine...
*/
/* #define DEBUG */


/* Note this DIMENSION_LIMIT is easily changed with some care.
   But you want to stay in fast memory...
   Small values can live in registers...
   Medium, __shared__...
   Absurd, global...

   I figure most commone values might be 1,2, or 3 dim...
*/
static const int DIMENSION_LIMIT = 32;
static const int BATCH_LIMIT = 65535;

/*
  CUDA Utility
*/
#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

/*
  These are just some helper utilities,
  mainly to help me remember how I translate between the index systems
*/

typedef struct rcIdx {
  int row;
  int col;
} rcIdx_t;

typedef struct diagIdx {
  int orth_diag;  // the "left" diagonals
  int idx; // index along the diag
} diagIdx_t;

static __inline__ __host__ __device__ rcIdx_t map_diag_to_rc(int orth_diag, int idx){
  /* orth_diag is the zero based ortho diagonal ("left" diagonals),
     idx is the zero based index into orth_diag */
  return { orth_diag - idx, idx};
}

static __inline__ __host__ __device__ diagIdx_t map_rc_to_diag(int row, int col){
  /* orth_diag is the zero based ortho diagonal,
     idx is the zero based index into orth_diag */
  return {row+col, col};
}

/*
  The core alogorithm is expanded here for doubles then single precision.
  See cuTWED_core.h
*/

#define REAL_t double
#define _TWED_MALLOC_DEV twed_malloc_dev
#define _TWED_FREE_DEV twed_free_dev
#define _TWED_COPY_TO_DEV twed_copy_to_dev
#define _TWED_DEV twed_dev
#define _TWED twed
#define _TWED_BATCH_DEV twed_batch_dev
#define _TWED_BATCH twed_batch
#define _GEAM cublasDgeam
#include "cuTWED_core.h"
#undef REAL_t
#undef _TWED_MALLOC_DEV
#undef _TWED_FREE_DEV
#undef _TWED_COPY_TO_DEV
#undef _TWED_DEV
#undef _TWED
#undef _TWED_BATCH_DEV
#undef _TWED_BATCH
#undef _GEAM

#define REAL_t float
#define _TWED_MALLOC_DEV twed_malloc_devf
#define _TWED_FREE_DEV twed_free_devf
#define _TWED_COPY_TO_DEV twed_copy_to_devf
#define _TWED_DEV twed_devf
#define _TWED twedf
#define _TWED_BATCH_DEV twed_batch_devf
#define _TWED_BATCH twed_batchf
#define _GEAM cublasSgeam
#include "cuTWED_core.h"
#undef REAL_t
#undef _TWED_MALLOC_DEV
#undef _TWED_FREE_DEV
#undef _TWED_COPY_TO_DEV
#undef _TWED_DEV
#undef _TWED
#undef _TWED_BATCH_DEV
#undef _TWED_BATCH
#undef _GEAM

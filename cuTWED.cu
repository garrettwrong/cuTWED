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

#include "cuTWED.h"

#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void local_distance_kernel(const REAL_t* __restrict__ A, int nA, int degree, REAL_t* __restrict__ DA){
  // implicitly assumed D can hold nA + 1 elements.
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  REAL_t d;

  if( tid > nA ) return;

  if(tid == 0){
    d = 0.;
  }
  else if(tid == 1) {
    d = pow( fabs( A[tid - 1]), degree);
  }
  else {
    d = pow( fabs( A[tid - 1] - A[tid - 2] ), degree);
  }
  DA[tid] = d;
}


typedef struct rcIdx {
  int row;
  int col;
} rcIdx_t;

typedef struct diagIdx {
  int orth_diag;
  int idx;
} diagIdx_t;

static __inline__ __host__ __device__ rcIdx_t map_diag_to_rc(int orth_diag, int idx){
  /* orth_diag is the zero based ortho diagonal,
     idx is the zero based index into orth_diag */
  return { orth_diag - idx, idx};
}

static __inline__ __host__ __device__ diagIdx_t map_rc_to_diag(int row, int col){
  /* orth_diag is the zero based ortho diagonal,
     idx is the zero based index into orth_diag */
  return {row+col, col};
}


__global__ void evalZ_kernel(int diagIdx,
                             REAL_t* DP_diag_lag_2,
                             REAL_t* DP_diag_lag,
                             REAL_t* DP_diag,
                             const REAL_t* __restrict__ A, const REAL_t* __restrict__ DA, int nA, const REAL_t* __restrict__ TA,
                             const REAL_t* __restrict__ B, const REAL_t* __restrict__ DB, int nB, const REAL_t* __restrict__ TB,
                             REAL_t nu, int degree, REAL_t lambda){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid > diagIdx) return;

  // map from the diagonal index and thread into the DP row/col
  const rcIdx_t id = map_diag_to_rc(diagIdx, tid);
  const int row = id.row;
  const int col = id.col; //diagIdx - tid;

  // get computing DP indexes out of the way
  // lag one row
  const size_t tidDrm1 = map_rc_to_diag(row-1, col).idx;
  assert( map_rc_to_diag(row-1, col).orth_diag == diagIdx -1);
  // lag one col
  const size_t tidDcm1 = map_rc_to_diag(row, col-1).idx;
  assert( map_rc_to_diag(row, col-1).orth_diag == diagIdx -1);
  // lag one row and one col
  const size_t tidDrm1cm1 = map_rc_to_diag(row-1, col-1).idx;
  assert( map_rc_to_diag(row-1, col-1).orth_diag == diagIdx -2);

  if(row > nA || col > nB) return;

  //dp dist
  REAL_t d;
  if(row==0 && col==0){
    d = 0;
  } else if(row==0 || col==0){
    d = INFINITY;
  }
  else{
    d = pow( fabs( A[row - 1] - B[col - 1]), degree);
    if(row>1 && col>1){
      d += pow( fabs( A[row - 2] - B[col - 2]), degree);
    }
  }

  //DBG printf("d [%d] = %f;\n", tid, d);
  if(row<1 || col <1) {
    DP_diag[tid] = d;
    return;
  }


  REAL_t htrans;
  REAL_t dmin;
  REAL_t dist;

  // case 1
  htrans = fabs( (REAL_t)(TA[row-1] - TB[col-1]));
  if(col>1 && row>1){
    htrans += fabs((REAL_t)(TA[row-2] - TB[col-2]));
  }
  //DBG printf("DP_diag_lag_2[tidDrm1cm1= %ld] = %f\n", tidDrm1cm1, DP_diag_lag_2[tidDrm1cm1]);
  dmin = DP_diag_lag_2[tidDrm1cm1] + d + nu * htrans;

  // case 2
  if(row>1)
    htrans = ((REAL_t)(TA[row-1] - TA[row-2]));
  else htrans = (REAL_t)TA[row-1];
  dist = DA[row] + DP_diag_lag[tidDrm1] + lambda + nu * htrans;
  // check if we need to assign new min
  dmin = fmin(dmin, dist);

  // case 3
  if(col>1)
    htrans = ((REAL_t)(TB[col-1] - TB[col-2]));
  else htrans = (REAL_t)TB[col-1];
  dist = DB[col] + DP_diag_lag[tidDcm1] + lambda + nu * htrans;
  // check if we need to assign new min
  dmin = fmin(dmin, dist);

  // assign result to dynamic program matrix
  DP_diag[tid] = dmin;
  //DBG DP_diag[tid] = diagIdx;

}


static void evalZ(REAL_t DP_diag[],
                  const REAL_t* __restrict__ A, const REAL_t* __restrict__ DA, int nA, const REAL_t* __restrict__ TA,
                  const REAL_t* __restrict__ B, const REAL_t* __restrict__ DB, int nB, const REAL_t* __restrict__ TB,
                  REAL_t nu, int degree, REAL_t lambda){
  const int n = (nA+1) + (nB+1) -1;
  dim3 block_dim(32); // note this particular var might be sensitive to tuning and architectures...
  int diagIdx;

  REAL_t* tmp_ptr=NULL;
  REAL_t* DP_diag_lag;
  REAL_t* DP_diag_lag_2;
  const size_t sz = n * sizeof(*DP_diag_lag);
  HANDLE_ERROR(cudaMalloc(&DP_diag_lag, sz));
  HANDLE_ERROR(cudaMalloc(&DP_diag_lag_2, sz));

  HANDLE_ERROR(cudaPeekAtLastError());

  for(diagIdx=0; diagIdx < n; diagIdx++){

    tmp_ptr = DP_diag_lag_2;
    DP_diag_lag_2 = DP_diag_lag;
    DP_diag_lag = DP_diag;
    DP_diag = tmp_ptr;

    dim3 grid_dim((diagIdx + block_dim.x)/ block_dim.x);
    evalZ_kernel<<<grid_dim, block_dim>>>(diagIdx, DP_diag_lag_2, DP_diag_lag, DP_diag, A, DA, nA, TA, B, DB, nB, TB, nu, degree, lambda);
    HANDLE_ERROR(cudaPeekAtLastError());


    /*
    // DBG
    REAL_t* tmp = (REAL_t*)calloc(n, sizeof(REAL_t));
    HANDLE_ERROR(cudaMemcpy(tmp, DP_diag_lag_2, sz, cudaMemcpyDeviceToHost));
    printf("\n\n\nDiag LAG-2 (%d)\n", diagIdx-2);
    for(int r=0; r<= nA; r++){
      for(int c=0; c<= nB; c++){
        if(map_rc_to_diag(r, c).orth_diag == diagIdx-2){
          //printf("%d %d \t %d %d\n", r, c, map_rc_to_diag(r,c).orth_diag, map_rc_to_diag(r,c).idx);
          printf("%f, ", tmp[map_rc_to_diag(r,c).idx]);
        } else printf("_, ");
      }
      printf("\n");
    }

    HANDLE_ERROR(cudaMemcpy(tmp, DP_diag_lag, sz, cudaMemcpyDeviceToHost));
    printf("\nDiag LAG-1 (%d)\n", diagIdx-1);
    for(int r=0; r<= nA; r++){
      for(int c=0; c<= nB; c++){
        if(map_rc_to_diag(r, c).orth_diag == diagIdx-1){
          //printf("%d %d \t %d %d\n", r, c, map_rc_to_diag(r,c).orth_diag, map_rc_to_diag(r,c).idx);
          printf("%f, ", tmp[map_rc_to_diag(r,c).idx]);
        } else printf("_, ");
      }
      printf("\n");
    }

    HANDLE_ERROR(cudaMemcpy(tmp, DP_diag, sz, cudaMemcpyDeviceToHost));
    printf("\nDiag (%d)\n", diagIdx);
    for(int r=0; r<= nA; r++){
      for(int c=0; c<= nB; c++){
        if(map_rc_to_diag(r, c).orth_diag == diagIdx){
        //printf("%d %d \t %d %d\n", r, c, map_rc_to_diag(r,c).orth_diag, map_rc_to_diag(r,c).idx);
          printf("%f, ", tmp[map_rc_to_diag(r,c).idx]);
        } else printf("_, ");
      }
      printf("\n");
    }
    free(tmp);
    // DBG
    */

  }

  int r = n;
  while (r%3 !=0){
    // we need to cycle our pointers...
    tmp_ptr = DP_diag_lag_2;
    DP_diag_lag_2 = DP_diag_lag;
    DP_diag_lag = DP_diag;
    DP_diag = tmp_ptr;
    HANDLE_ERROR(cudaMemcpy(DP_diag, DP_diag_lag, sz, cudaMemcpyDeviceToDevice));
    r++;
  }
}


#ifdef __cplusplus
extern "C" {
#endif
void twed_malloc_dev(int nA, REAL_t **A_dev, REAL_t  **TA_dev,
                     int nB, REAL_t **B_dev, REAL_t  **TB_dev,
                     REAL_t **DP_dev){
  //malloc on gpu and copy
  const size_t sza = (nA+1) * sizeof(**A_dev);
  HANDLE_ERROR(cudaMalloc(A_dev, sza));
  HANDLE_ERROR(cudaMalloc(TA_dev, sza));

  const size_t szb = (nB+1) * sizeof(**B_dev);
  HANDLE_ERROR(cudaMalloc(B_dev, szb));
  HANDLE_ERROR(cudaMalloc(TB_dev, szb));

  const size_t sz = ((nA+1) + (nB+1) -1) * sizeof(**DP_dev);
  HANDLE_ERROR(cudaMalloc(DP_dev, sz));
}
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
void twed_free_dev(REAL_t *A_dev, REAL_t  *TA_dev,
                   REAL_t *B_dev, REAL_t  *TB_dev,
                   REAL_t *DP_dev){
  //cleanup
  HANDLE_ERROR(cudaFree(A_dev));
  HANDLE_ERROR(cudaFree(TA_dev));
  HANDLE_ERROR(cudaFree(B_dev));
  HANDLE_ERROR(cudaFree(TB_dev));
  HANDLE_ERROR(cudaFree(DP_dev));
}
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
void twed_copy_to_dev(int nA, REAL_t A[], REAL_t A_dev[], REAL_t TA[], REAL_t TA_dev[],
                      int nB, REAL_t B[], REAL_t B_dev[], REAL_t TB[], REAL_t TB_dev[]){
  const size_t sza = nA*sizeof(*A);
  HANDLE_ERROR(cudaMemcpy(A_dev, A, sza, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(TA_dev, TA, sza, cudaMemcpyHostToDevice));
  const size_t szb = nB*sizeof(*B);
  HANDLE_ERROR(cudaMemcpy(B_dev, B , szb, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(TB_dev, TB, szb, cudaMemcpyHostToDevice));
}
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
REAL_t twed_dev(REAL_t A_dev[], int nA, REAL_t TA_dev[],
                REAL_t B_dev[], int nB, REAL_t TB_dev[],
                REAL_t nu, REAL_t lambda, int degree,
                REAL_t DP_dev[]){
  REAL_t *DA_dev, *DB_dev;
  REAL_t result;

  dim3 block_dim;
  dim3 grid_dim;

  const int nstreams = 2;
  int s;
  cudaStream_t streams[nstreams];
  for(s=0; s<nstreams; s++){
    HANDLE_ERROR(cudaStreamCreate(&streams[s]));
  }

  const size_t sza = (nA+1) * sizeof(*A_dev);
  const size_t szb = (nB+1) * sizeof(*B_dev);
  HANDLE_ERROR(cudaMalloc(&DA_dev, sza));
  HANDLE_ERROR(cudaMalloc(&DB_dev, szb));


  // compute initial distance A
  block_dim.x = 256;
  grid_dim.x = (nA + block_dim.x - 1) / block_dim.x;
  local_distance_kernel<<<grid_dim, block_dim, 0, streams[0]>>>(A_dev, nA, degree, DA_dev);
  HANDLE_ERROR(cudaPeekAtLastError());

  // compute initial distance B
  block_dim.x = 256;
  grid_dim.x = (nB + block_dim.x - 1) / block_dim.x;
  local_distance_kernel<<<grid_dim, block_dim, 0, streams[1]>>>(B_dev, nB, degree, DB_dev);
  HANDLE_ERROR(cudaPeekAtLastError());

  HANDLE_ERROR(cudaDeviceSynchronize());

  // iteratively update the DP matrix
  //   we process diagonals moving the diagonal from upper left to lower right,
  //         each element of a diag can is done in parallel.
  evalZ(DP_dev, A_dev, DA_dev, nA, TA_dev, B_dev, DB_dev, nB, TB_dev, nu, degree, lambda);

  // the algo result should be the final distance stored in DP
  HANDLE_ERROR(cudaMemcpy(&result, &DP_dev[map_rc_to_diag(nA, nB).idx], sizeof(result), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(DA_dev));
  HANDLE_ERROR(cudaFree(DB_dev));

  for(s=0; s<nstreams; s++){
    HANDLE_ERROR(cudaStreamDestroy(streams[s]));
  }

  return result;
}
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
REAL_t twed(REAL_t A[], int nA, REAL_t TA[],
            REAL_t B[], int nB, REAL_t TB[],
            REAL_t nu, REAL_t lambda, int degree,
            REAL_t* DP){
  REAL_t *A_dev, *TA_dev;
  REAL_t *B_dev, *TB_dev;
  REAL_t *DP_dev;
  REAL_t result;

  // malloc gpu arrays
  twed_malloc_dev(nA, &A_dev, &TA_dev,
              nB, &B_dev, &TB_dev,
              &DP_dev);

  // copy inputs to device
  twed_copy_to_dev(nA, A, A_dev, TA, TA_dev,
                   nB, B, B_dev, TB, TB_dev);

  // compute TWED on device
  result = twed_dev(A_dev, nA, TA_dev,
                    B_dev, nB, TB_dev,
                    nu, lambda, degree,
                    DP_dev);

  /// this isnt write for diagZ
  // // optionally copy back DP matrix
  // if(DP != NULL){
  //   const size_t sz = (nA+1) * (nB+1) * sizeof(*DP_dev);
  //   HANDLE_ERROR(cudaMemcpy(&result, DP_dev, sz, cudaMemcpyDeviceToHost));
  // }

  // free device memory
  twed_free_dev(A_dev, TA_dev,
            B_dev, TB_dev,
            DP_dev);

  return result;
}
#ifdef __cplusplus
}
#endif

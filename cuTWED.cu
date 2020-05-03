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

__global__ void local_distance_kernel(REAL_t A[], int nA, int degree, REAL_t DA[]){
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

__global__ void dp_distance_kernel(REAL_t A[], int nA, REAL_t B[], int nB, int degree, REAL_t DP[]){
  const int tidA = blockIdx.x * blockDim.x + threadIdx.x;
  const int tidB = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t tidD = tidA * (nB + 1) + tidB;
  REAL_t d;

  if(tidA >nA || tidB > nB) return;

  if(tidA==0 && tidB==0){
    d = 0;
  } else if(tidA==0 || tidB==0){
    d = INFINITY;
  }
  else{
    d = pow( fabs( A[tidA - 1] - B[tidB - 1]), degree);
    if(tidA>1 && tidB>1){
      d += pow( fabs( A[tidA - 2] - B[tidB - 2]), degree);
    }
  }

  DP[tidD] = d;
}

typedef struct diagIdx {
  int row;
  int col;
} diagIdx_t;

static __inline__ __host__ __device__ diagIdx_t map_diag_to_mat(int orth_diag, int idx){
  /* orth_diag is the zero based ortho diagonal,
     idx is the zero based index into orth_diag */
  return {idx, orth_diag - idx};
}

__global__ void evalZ_kernel(int diagIdx,
                             REAL_t DP[],
                             REAL_t DA[], int nA, REAL_t TA[],
                             REAL_t DB[], int nB, REAL_t TB[],
                             REAL_t nu, REAL_t lambda){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // bound, consider for non square later
  if(tid > diagIdx) return;

  // map from the diagonal index and thread into the DP row/col
  const diagIdx_t id = map_diag_to_mat(diagIdx, tid);
  const int row = id.row;
  const int col = id.col; //diagIdx - tid;
  if(row<1 || col <1) return;
  if(row > nA || col > nB) return;

  // get computing DP indexes out of the way
  const size_t tidD = row * (nB+1) + col;
  // lag one row
  const size_t tidDrm1 = (row-1) * (nB+1) + col;
  // lag one col
  const size_t tidDcm1 = tidD - 1;
  // lag one row and one col
  const size_t tidDrm1cm1 = tidDrm1 - 1;

  REAL_t htrans;
  REAL_t dmin;
  REAL_t dist;

  // case 1
  htrans = fabs( (REAL_t)(TA[row-1] - TB[col-1]));
  if(col>1 && row>1){
    htrans += fabs((REAL_t)(TA[row-2] - TB[col-2]));
  }
  dmin = DP[tidDrm1cm1] + DP[tidD] + nu * htrans;

  // case 2
  if(row>1)
    htrans = ((REAL_t)(TA[row-1] - TA[row-2]));
  else htrans = (REAL_t)TA[row-1];
  dist = DA[row] + DP[tidDrm1] + lambda + nu * htrans;
  // check if we need to assign new min
  if(dist<dmin){
    dmin = dist;
  }

  // case 3
  if(col>1)
    htrans = ((REAL_t)(TB[col-1] - TB[col-2]));
  else htrans = (REAL_t)TB[col-1];
  dist = DB[col] + DP[tidDcm1] + lambda + nu * htrans;
  if(dist<dmin){
    dmin = dist;
  }

  // assign result to dynamic program matrix
  DP[tidD] = dmin;
}


static void evalZ(REAL_t DP[],
                  REAL_t DA[], int nA, REAL_t TA[],
                  REAL_t DB[], int nB, REAL_t TB[],
                  REAL_t nu, REAL_t lambda){
  int blocksz = 32;  // note this particular var might be sensitive to tuning and architectures...
  int diagIdx;
  int n;

  n = (nA+1) + (nB+1);

  for(diagIdx=1; diagIdx < n; diagIdx++){
    dim3 block_dim(blocksz);
    dim3 grid_dim((diagIdx + block_dim.x)/ block_dim.x);
    evalZ_kernel<<<grid_dim, block_dim>>>(diagIdx,DP, DA, nA, TA, DB, nB, TB, nu, lambda);
    HANDLE_ERROR(cudaPeekAtLastError());
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

  const size_t sz = (nA+1) * (nB+1) * sizeof(**DP_dev);
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

  const size_t sza = (nA+1) * sizeof(*A_dev);
  const size_t szb = (nB+1) * sizeof(*B_dev);
  HANDLE_ERROR(cudaMalloc(&DA_dev, sza));
  HANDLE_ERROR(cudaMalloc(&DB_dev, szb));

  // compute initial distance A
  block_dim.x = 256;
  grid_dim.x = (nA + block_dim.x - 1) / block_dim.x;
  local_distance_kernel<<<grid_dim, block_dim>>>(A_dev, nA, degree, DA_dev);
  HANDLE_ERROR(cudaPeekAtLastError());

  // compute initial distance B
  block_dim.x = 256;
  grid_dim.x = (nB + block_dim.x - 1) / block_dim.x;
  local_distance_kernel<<<grid_dim, block_dim>>>(B_dev, nB, degree, DB_dev);
  HANDLE_ERROR(cudaPeekAtLastError());

  // compute initial dynamic program matrix D
  block_dim.x = 32;
  block_dim.y = 32;
  // recall DP is nA+1 x nB+1
  grid_dim.x = (nA + block_dim.x) / block_dim.x;
  grid_dim.y = (nB + block_dim.y) / block_dim.y;
  dp_distance_kernel<<<grid_dim, block_dim>>>(A_dev, nA, B_dev, nB, degree, DP_dev);
  HANDLE_ERROR(cudaPeekAtLastError());

  // iteratively update the DP matrix
  //   we process diagonals moving the diagonal from upper left to lower right,
  //         each element of a diag can is done in parallel.
  evalZ(DP_dev, DA_dev, nA, TA_dev, DB_dev, nB, TB_dev, nu, lambda);

  // the algo result should be the final distance stored in DP
  HANDLE_ERROR(cudaMemcpy(&result, &DP_dev[(nA+1) * (nB+1) - 1], sizeof(result), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(DA_dev));
  HANDLE_ERROR(cudaFree(DB_dev));

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

  // optionally copy back DP matrix
  if(DP != NULL){
    const size_t sz = (nA+1) * (nB+1) * sizeof(*DP_dev);
    HANDLE_ERROR(cudaMemcpy(&result, DP_dev, sz, cudaMemcpyDeviceToHost));
  }

  // free device memory
  twed_free_dev(A_dev, TA_dev,
            B_dev, TB_dev,
            DP_dev);

  return result;
}
#ifdef __cplusplus
}
#endif

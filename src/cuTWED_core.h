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

/*
  This file is meant to be included after defining REAL_t and function name _TOKENS
  for the preprocessor, thus covering both single and double precision methods with
  one core source code file (this one).
*/


__device__ REAL_t lpnorm(const int p,
                         const int dim,
                         const REAL_t* __restrict__ P){
  /* Compute Lp norm for point P in R^dim */
  const REAL_t pf = (REAL_t)p;
  int d;

  REAL_t s=0;
  for(d=0; d<dim; d++){
    s += pow(fabs(P[d]), pf);
  }

  /* this is not strictly necessary for our purposes... */
  s = pow(s, 1./pf); 

  return s;      
}

__device__ void vsub(const int dim,
                     const REAL_t* __restrict__ P1,
                     const REAL_t* __restrict__ P2,
                     REAL_t* __restrict__ P3){
  /* Compute subraction of two points P1, P1 in R^dim
     Stores in P3
   */
  int d;
  #pragma unroll
  for(d=0; d<dim; d++){
    P3[d] = P1[d] - P2[d];
  }
}


__global__ void local_distance_kernel(const REAL_t* __restrict__ A, const int nA, const int degree,
                                      const int dim,
                                      REAL_t* __restrict__ DA){
  /* Implicitly assumed D can hold nA + 1 elements. */
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  REAL_t tmp[DIMENSION_CUTOVER];
  
  REAL_t d;

  if( tid > nA ) return;

  if(tid == 0){
    d = 0.;
  }
  else if(tid == 1) {
    d = lpnorm(degree, dim, &A[dim_offset(tid-1,dim)]);
  }
  else {
    vsub(dim, &A[dim_offset(tid-1, dim)], &A[dim_offset(tid-2, dim)], tmp);
    d = lpnorm(degree, dim, tmp);
  }

  DA[tid] = d;
}

__global__ void local_distance_shared_mem_kernel(const REAL_t* __restrict__ A, const int nA, const int degree,
                                                 const int dim,
                                      REAL_t* __restrict__ DA){
  /* Implicitly assumed D can hold nA + 1 elements. */
  const int dimid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = blockIdx.y * blockDim.y + threadIdx.y;
  const REAL_t pf = (REAL_t)degree;
  int i;
  REAL_t d;

  __shared__ REAL_t tmp[DIMENSION_CUTOVER * DIMENSION_CUTOVER];
  
  if( tid > nA ) return;

  if(tid == 0 || dimid >=dim){
    /*d = 0.; */
    tmp[threadIdx.y*DIMENSION_CUTOVER + dimid] = 0;
  }
  else if(tid == 1) {
    tmp[threadIdx.y*DIMENSION_CUTOVER + dimid] = A[tid-1*dim];
  }
  else {
    tmp[threadIdx.y*DIMENSION_CUTOVER + dimid] = A[tid-1*dim] - A[tid-2*dim];
  }

  /* raise to power */
  tmp[threadIdx.y*DIMENSION_CUTOVER + threadIdx.x] = pow(tmp[threadIdx.y*DIMENSION_CUTOVER + threadIdx.x], pf);
  /* each block sums */
  if(threadIdx.x==0){
    for(i=1; i<blockDim.x; i++){
      tmp[threadIdx.y*DIMENSION_CUTOVER] += tmp[threadIdx.y*DIMENSION_CUTOVER + i];
    }
    /* then we sum the blocks */
    if(dimid==0){
      d=0;
      for(i=0; i < gridDim.x; i++){
      d += tmp[threadIdx.y*DIMENSION_CUTOVER + i];
      }
      /* assign the accumulated sum */
      DA[tid] = d;
    }
  }  
}




__global__ void evalZ_kernel(int diagIdx,
                             REAL_t* DP_diag_lag_2,
                             REAL_t* DP_diag_lag,
                             REAL_t* DP_diag,
                             const REAL_t* __restrict__ A, const REAL_t* __restrict__ DA, int nA, const REAL_t* __restrict__ TA,
                             const REAL_t* __restrict__ B, const REAL_t* __restrict__ DB, int nB, const REAL_t* __restrict__ TB,
                             REAL_t nu, int degree, REAL_t lambda, int dim){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid > diagIdx) return;

  /* map from the diagonal index and thread into the DP row/col */
  const rcIdx_t id = map_diag_to_rc(diagIdx, tid);
  const int row = id.row;
  const int col = id.col;

  /*
    Get computing DP indexes out of the way
  */

  /* lag one row */
  const size_t tidDrm1 = map_rc_to_diag(row-1, col).idx;
  //DBG assert( map_rc_to_diag(row-1, col).orth_diag == diagIdx -1);

  /* lag one col */
  const size_t tidDcm1 = map_rc_to_diag(row, col-1).idx;
  //DBG assert( map_rc_to_diag(row, col-1).orth_diag == diagIdx -1);

  /* lag one row and one col, note this is _two_ diag vectors behind DP diag */
  const size_t tidDrm1cm1 = map_rc_to_diag(row-1, col-1).idx;
  //DBG assert( map_rc_to_diag(row-1, col-1).orth_diag == diagIdx -2);

  if(row > nA || col > nB) return;

  /*
    Compute the initial DP disance for this entry.
  */
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

  /*
    Compute the procession of DP updates.
  */

  REAL_t htrans;
  REAL_t dmin;
  REAL_t dist;

  /* case 1, Keep Both */
  htrans = fabs( (REAL_t)(TA[row-1] - TB[col-1]));
  if(col>1 && row>1){
    htrans += fabs((REAL_t)(TA[row-2] - TB[col-2]));
  }
  //DBG printf("DP_diag_lag_2[tidDrm1cm1= %ld] = %f\n", tidDrm1cm1, DP_diag_lag_2[tidDrm1cm1]);
  dmin = DP_diag_lag_2[tidDrm1cm1] + d + nu * htrans;

  /* case 2, Delete point in A */
  if(row>1)
    htrans = ((REAL_t)(TA[row-1] - TA[row-2]));
  else htrans = (REAL_t)TA[row-1];
  dist = DA[row] + DP_diag_lag[tidDrm1] + lambda + nu * htrans;
  /* check if we need to assign new min */
  dmin = fmin(dmin, dist);

  /* case 3, Delete Point in B */
  if(col>1)
    htrans = ((REAL_t)(TB[col-1] - TB[col-2]));
  else htrans = (REAL_t)TB[col-1];
  dist = DB[col] + DP_diag_lag[tidDcm1] + lambda + nu * htrans;
  /* check if we need to assign new min */
  dmin = fmin(dmin, dist);

  /* assign minimal result to dynamic program matrix */
  DP_diag[tid] = dmin;

  /* Note, the following line is a handy way to debug the procession
     across the DP matrix (in conjuction with DEBUG prints).
     /\* DBG DP_diag[tid] = diagIdx; *\/
  */

}


static REAL_t evalZ(const REAL_t* __restrict__ A, const REAL_t* __restrict__ DA, int nA, const REAL_t* __restrict__ TA,
                    const REAL_t* __restrict__ B, const REAL_t* __restrict__ DB, int nB, const REAL_t* __restrict__ TB,
                    REAL_t nu, int degree, REAL_t lambda, int dim){
  const int n = (nA+1) + (nB+1) -1;
  dim3 block_dim(32); /* note this particular var might be sensitive to tuning and architectures... */
  int diagIdx;

  REAL_t* tmp_ptr=NULL;
  REAL_t* DP_diag;
  REAL_t* DP_diag_lag;
  REAL_t* DP_diag_lag_2;
  const size_t sz = n * sizeof(*DP_diag);
  HANDLE_ERROR(cudaMalloc(&DP_diag, sz));
  HANDLE_ERROR(cudaMalloc(&DP_diag_lag, sz));
  HANDLE_ERROR(cudaMalloc(&DP_diag_lag_2, sz));

  HANDLE_ERROR(cudaPeekAtLastError());

  for(diagIdx=0; diagIdx < n; diagIdx++){

    tmp_ptr = DP_diag_lag_2;
    DP_diag_lag_2 = DP_diag_lag;
    DP_diag_lag = DP_diag;
    DP_diag = tmp_ptr;

    dim3 grid_dim((diagIdx + block_dim.x)/ block_dim.x);
    evalZ_kernel<<<grid_dim, block_dim>>>(diagIdx, DP_diag_lag_2, DP_diag_lag, DP_diag, A, DA, nA, TA, B, DB, nB, TB, nu, degree, lambda, dim);
    HANDLE_ERROR(cudaPeekAtLastError());


#ifdef DEBUG
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
#endif

  }

  REAL_t result;
  HANDLE_ERROR(cudaMemcpy(&result, &DP_diag[map_rc_to_diag(nA, nB).idx], sizeof(*DP_diag), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(DP_diag));
  HANDLE_ERROR(cudaFree(DP_diag_lag));
  HANDLE_ERROR(cudaFree(DP_diag_lag_2));
  return result;
}


#ifdef __cplusplus
extern "C" {
#endif
  void _TWED_MALLOC_DEV(int nA, REAL_t **A_dev, REAL_t  **TA_dev,
                        int nB, REAL_t **B_dev, REAL_t  **TB_dev){
    /* malloc on gpu and copy */
    const size_t sza = (nA+1) * sizeof(**A_dev);
    HANDLE_ERROR(cudaMalloc(A_dev, sza));
    HANDLE_ERROR(cudaMalloc(TA_dev, sza));

    const size_t szb = (nB+1) * sizeof(**B_dev);
    HANDLE_ERROR(cudaMalloc(B_dev, szb));
    HANDLE_ERROR(cudaMalloc(TB_dev, szb));
  }
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
  void _TWED_FREE_DEV(REAL_t *A_dev, REAL_t  *TA_dev,
                      REAL_t *B_dev, REAL_t  *TB_dev){
    /* In a minute I'll be free */
    HANDLE_ERROR(cudaFree(A_dev));
    HANDLE_ERROR(cudaFree(TA_dev));
    HANDLE_ERROR(cudaFree(B_dev));
    HANDLE_ERROR(cudaFree(TB_dev));
  }
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
  void _TWED_COPY_TO_DEV(int nA, REAL_t A[], REAL_t A_dev[], REAL_t TA[], REAL_t TA_dev[],
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
  REAL_t _TWED_DEV(REAL_t A_dev[], int nA, REAL_t TA_dev[],
                   REAL_t B_dev[], int nB, REAL_t TB_dev[],
                   REAL_t nu, REAL_t lambda, int degree, int dim){
    REAL_t *DA_dev;
    REAL_t *DB_dev;
    REAL_t result;

    dim3 block_dim;
    dim3 grid_dim;

    /*
      Sanity Check
    */
    /*    if(dim > DIMENSION_CUTOVER){
      printf("Error, supplied dimension %d is greater than compiled DIMENSION_CUTOVER %d\n. Exiting",
             dim, DIMENSION_CUTOVER);
      exit(1);
    }
    */

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

    /* compute initial distance A */
    if(dim < DIMENSION_CUTOVER){
      block_dim.x = 256;
      grid_dim.x = (nA + block_dim.x - 1) / block_dim.x;
      local_distance_kernel<<<grid_dim, block_dim, 0, streams[0]>>>(A_dev, nA, degree, dim, DA_dev);
      HANDLE_ERROR(cudaPeekAtLastError());

      /* compute initial distance B */
      block_dim.x = 256;
      grid_dim.x = (nB + block_dim.x - 1) / block_dim.x;
      local_distance_kernel<<<grid_dim, block_dim, 0, streams[1]>>>(B_dev, nB, degree, dim, DB_dev);
      HANDLE_ERROR(cudaPeekAtLastError());
    }
    else{
      size_t shmem = DIMENSION_CUTOVER * DIMENSION_CUTOVER * sizeof(REAL_t);
      block_dim.x = DIMENSION_CUTOVER;
      block_dim.y = 32;
      grid_dim.x = (dim + block_dim.x - 1) / block_dim.x;
      grid_dim.y = (nA + block_dim.y - 1) / block_dim.y;
      local_distance_shared_mem_kernel<<<grid_dim, block_dim, shmem, streams[0]>>>(A_dev, nA, degree, dim, DA_dev);
      HANDLE_ERROR(cudaPeekAtLastError());

      /* compute initial distance B */
      block_dim.x = DIMENSION_CUTOVER;
      block_dim.y = 32;
      grid_dim.x = (dim + block_dim.x - 1) / block_dim.x;
      grid_dim.y = (nB + block_dim.y - 1) / block_dim.y;      
      local_distance_shared_mem_kernel<<<grid_dim, block_dim, shmem, streams[1]>>>(B_dev, nB, degree, dim, DB_dev);
      HANDLE_ERROR(cudaPeekAtLastError());
    }

    HANDLE_ERROR(cudaDeviceSynchronize());

    /*
      This method iteratively updates the DP matrix.
      We process diagonals moving the diagonal from upper left to lower right.
      Each element of a diag is done in parallel.
    */
    result = evalZ(A_dev, DA_dev, nA, TA_dev, B_dev, DB_dev, nB, TB_dev, nu, degree, lambda, dim);

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
  REAL_t _TWED(REAL_t A[], int nA, REAL_t TA[],
               REAL_t B[], int nB, REAL_t TB[],
               REAL_t nu, REAL_t lambda, int degree, int dim){
    REAL_t *A_dev, *TA_dev;
    REAL_t *B_dev, *TB_dev;
    REAL_t result;

    /* malloc gpu arrays */
    _TWED_MALLOC_DEV(nA, &A_dev, &TA_dev,
                     nB, &B_dev, &TB_dev);

    /* copy inputs to device */
    _TWED_COPY_TO_DEV(nA, A, A_dev, TA, TA_dev,
                      nB, B, B_dev, TB, TB_dev);

    /* compute TWED on device */
    result = _TWED_DEV(A_dev, nA, TA_dev,
                       B_dev, nB, TB_dev,
                       nu, lambda, degree, dim);

    /* cleanup device memory initialized here */
    _TWED_FREE_DEV(A_dev, TA_dev,
                   B_dev, TB_dev);

    return result;
  }
#ifdef __cplusplus
}
#endif

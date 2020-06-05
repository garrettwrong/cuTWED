/*  Copyrigaidht 2020 Garrett Wright, Gestalt Group LLC

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
  s = pow(s, (REAL_t)1./pf);

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


__global__ void local_distance_kernel(const REAL_t* __restrict__ X, const int n, const int degree,
                                      const int dim,
                                      REAL_t* __restrict__ D, const int nBatch){
  /* Implicitly assumed D can hold n + 1 elements. */
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bat = blockIdx.y * blockDim.y + threadIdx.y;
  const int gtid = (n+1) * bat + tid;

  REAL_t tmp[DIMENSION_LIMIT];

  REAL_t d;

  if( tid > n ) return;
  if( bat >= nBatch) return;

  if(tid == 0){
    d = 0.;
  }
  else if(tid == 1) {
    d = lpnorm(degree, dim, &X[bat*n*dim + (tid-1)*dim]);
  }
  else {
    vsub(dim,
         &X[bat*n*dim + (tid-1)*dim],
         &X[bat*n*dim + (tid-2)*dim], tmp);
    d = lpnorm(degree, dim, tmp);
  }

  D[gtid] = d;
}


__global__ void evalZ_kernel(int diagIdx,
                             REAL_t* DP_diag_lag_2,
                             REAL_t* DP_diag_lag,
                             REAL_t* DP_diag,
                             const REAL_t* __restrict__ A, const REAL_t* __restrict__ DA, int nA, const REAL_t* __restrict__ TA,
                             const REAL_t* __restrict__ B, const REAL_t* __restrict__ DB, int nB, const REAL_t* __restrict__ TB,
                             REAL_t nu, int degree, REAL_t lambda, int dim, int nBB){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int Bid = blockIdx.y * blockDim.y + threadIdx.y;

  if(Bid >= nBB) return;
  if(tid > diagIdx) return; /* note also diagIdx<n */

  /* map from the diagonal index and thread into the DP row/col */
  const rcIdx_t id = map_diag_to_rc(diagIdx, tid);
  const int row = id.row;
  const int col = id.col;
  const int n = (nA+1) + (nB+1)-1;

  /*
    Get computing DP indexes out of the way
  */

  /* lag one row */
  const size_t tidDrm1 = Bid*n + map_rc_to_diag(row-1, col).idx;
  assert( map_rc_to_diag(row-1, col).orth_diag == diagIdx -1);

  /* lag one col */
  const size_t tidDcm1 = Bid*n + map_rc_to_diag(row, col-1).idx;
  assert( map_rc_to_diag(row, col-1).orth_diag == diagIdx -1);

  /* lag one row and one col, note this is _two_ diag vectors behind DP diag */
  const size_t tidDrm1cm1 = Bid*n + map_rc_to_diag(row-1, col-1).idx;
  assert( map_rc_to_diag(row-1, col-1).orth_diag == diagIdx -2);

  if(row > nA || col > nB) return;

  const REAL_t* Bptr = &B[Bid*nB*dim];
  const REAL_t* TBptr = &TB[Bid*nB];
  const REAL_t* DBpter = &DB[Bid*(nB+1)];
  /*
    Compute the initial DP distance for this entry.
  */
  int i;
  REAL_t d;
  REAL_t d2;
  const REAL_t recip = (REAL_t)1. / degree;

  if(row==0 && col==0){
    d = 0;
  } else if(row==0 || col==0){
    d = INFINITY;
  }
  else{

    d=0;
    d2=0;
    for(i=0; i<dim; i++){
      d += pow( fabs( A[(row - 1)*dim + i] - Bptr[(col - 1)*dim + i]), degree);
      if(row>1 && col>1){
        d2 += pow( fabs( A[(row - 2)*dim + i] - Bptr[(col - 2)*dim + i]), degree);
      }
    }
    d = pow(d, recip) + pow(d2, recip);
  }

  //DBG printf("d [%d] = %f;\n", tid, d);
  if(row<1 || col <1) {
    DP_diag[Bid*n + tid] = d;
    return;
  }

  /*
    Compute the procession of DP updates.
  */

  REAL_t htrans;
  REAL_t dmin;
  REAL_t dist;

  /* case 1, Keep Both */
  htrans = fabs( (REAL_t)(TA[row-1] - TBptr[col-1]));
  if(col>1 && row>1){
    htrans += fabs((REAL_t)(TA[row-2] - TBptr[col-2]));
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
    htrans = ((REAL_t)(TBptr[col-1] - TBptr[col-2]));
  else htrans = (REAL_t)TBptr[col-1];
  dist = DBpter[col] + DP_diag_lag[tidDcm1] + lambda + nu * htrans;
  /* check if we need to assign new min */
  dmin = fmin(dmin, dist);

  /* assign minimal result to dynamic program matrix */
  DP_diag[Bid*n + tid] = dmin;

  /* Note, the following line is a handy way to debug the procession
     across the DP matrix (in conjuction with DEBUG prints).
     /\* DBG DP_diag[tid] = diagIdx; *\/
  */
}


static __host__ REAL_t evalZ(const REAL_t* __restrict__ A, const REAL_t* __restrict__ DA, int nA, const REAL_t* __restrict__ TA,
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
    evalZ_kernel<<<grid_dim, block_dim>>>(diagIdx, DP_diag_lag_2, DP_diag_lag, DP_diag, A, DA, nA, TA, B, DB, nB, TB, nu, degree, lambda, dim, 1);
    HANDLE_ERROR(cudaPeekAtLastError());


#ifdef DEBUG

    REAL_t tmpB[100];
    cudaMemcpy(&tmpB, DB, (nB+1)*sizeof(REAL_t), cudaMemcpyDeviceToHost);
    printf("DB");
    for(int b=0; b<=nB; b++){
      printf(" %f", tmpB[b]);
    };printf("\n");

    REAL_t tmpA[100];
    cudaMemcpy(&tmpA, DA, (nA+1)*sizeof(REAL_t), cudaMemcpyDeviceToHost);
    printf("DA");
    for(int a=0; a<=nA; a++){
      printf(" %f", tmpA[a]);
    };printf("\n");

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


__global__ void result_agg_kernel(REAL_t* __restrict__ res,
                                  const REAL_t* __restrict__ DPP,
                                  int nBB,
                                  int res_offest,
                                  int stride){

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid >= nBB) return;

  res[tid] = DPP[tid*stride + res_offest];

}


static void grid_evalZ(const REAL_t* __restrict__ A, const REAL_t* __restrict__ DA, int nA, const REAL_t* __restrict__ TA,
                       const REAL_t* __restrict__ BB, const REAL_t* __restrict__ DBB, int nB, const REAL_t* __restrict__ TBB,
                       REAL_t nu, int degree, REAL_t lambda, int dim, REAL_t* __restrict__ Res_dev, int nAA, int nBB, int tril){
  const int n = (nA+1) + (nB+1) -1;
  dim3 block_dim(32, 32); /* note this particular var might be sensitive to tuning and architectures... */
  int diagIdx;

  REAL_t* tmp_ptr=NULL;
  REAL_t* DP_diag;
  REAL_t* DP_diag_lag;
  REAL_t* DP_diag_lag_2;
  const size_t sz = nBB * n * sizeof(*DP_diag);
  HANDLE_ERROR(cudaMalloc(&DP_diag, sz));
  HANDLE_ERROR(cudaMalloc(&DP_diag_lag, sz));
  HANDLE_ERROR(cudaMalloc(&DP_diag_lag_2, sz));

  HANDLE_ERROR(cudaPeekAtLastError());

  if(tril != -1 ){
    nBB = tril;
  }

  for(diagIdx=0; diagIdx < n; diagIdx++){

    tmp_ptr = DP_diag_lag_2;
    DP_diag_lag_2 = DP_diag_lag;
    DP_diag_lag = DP_diag;
    DP_diag = tmp_ptr;

    /* note grid/block dims are 1 based */
    dim3 grid_dim((diagIdx + block_dim.x)/ block_dim.x,
                  (nBB + block_dim.y)/block_dim.y);
    evalZ_kernel<<<grid_dim, block_dim>>>(diagIdx, DP_diag_lag_2, DP_diag_lag, DP_diag,
                                          A, DA, nA, TA, BB, DBB, nB, TBB,
                                          nu, degree, lambda, dim, nBB);
    HANDLE_ERROR(cudaPeekAtLastError());

  }

  result_agg_kernel<<<(nBB+256)/256, 256>>>(Res_dev, DP_diag, nBB, map_rc_to_diag(nA, nB).idx, n);
  HANDLE_ERROR(cudaPeekAtLastError());


  HANDLE_ERROR(cudaFree(DP_diag));
  HANDLE_ERROR(cudaFree(DP_diag_lag));
  HANDLE_ERROR(cudaFree(DP_diag_lag_2));
}


#ifdef __cplusplus
extern "C" {
#endif

  void _TWED_MALLOC_DEV(const int nA, REAL_t **A_dev, REAL_t  **TA_dev,
                        const int nB, REAL_t **B_dev, REAL_t  **TB_dev,
                        const int dim, const int nAA, const int nBB){
    /* malloc on gpu and copy */
    const size_t sza = nAA*(nA+1) * sizeof(**A_dev);
    HANDLE_ERROR(cudaMalloc(A_dev, sza*dim));
    HANDLE_ERROR(cudaMalloc(TA_dev, sza));

    const size_t szb = nBB*(nB+1) * sizeof(**B_dev);
    HANDLE_ERROR(cudaMalloc(B_dev, szb*dim));
    HANDLE_ERROR(cudaMalloc(TB_dev, szb));
  }


  void _TWED_FREE_DEV(REAL_t *A_dev, REAL_t  *TA_dev,
                      REAL_t *B_dev, REAL_t  *TB_dev){
    /* In a minute I'll be free */
    HANDLE_ERROR(cudaFree(A_dev));
    HANDLE_ERROR(cudaFree(TA_dev));
    HANDLE_ERROR(cudaFree(B_dev));
    HANDLE_ERROR(cudaFree(TB_dev));
  }


  void _TWED_COPY_TO_DEV(const int nA, REAL_t A[], REAL_t A_dev[], REAL_t TA[], REAL_t TA_dev[],
                         const int nB, REAL_t B[], REAL_t B_dev[], REAL_t TB[], REAL_t TB_dev[],
                         const int dim, const int nAA, const int nBB){
    const size_t sza = nAA*nA*sizeof(*A);
    HANDLE_ERROR(cudaMemcpy(A_dev, A, sza*dim, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(TA_dev, TA, sza, cudaMemcpyHostToDevice));

    const size_t szb = nBB*nB*sizeof(*B);
    HANDLE_ERROR(cudaMemcpy(B_dev, B , szb*dim, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(TB_dev, TB, szb, cudaMemcpyHostToDevice));
  }


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
    if(dim > DIMENSION_LIMIT){
      printf("Error, supplied dimension %d is greater than compiled DIMENSION_LIMIT %d.\n" \
             "  If encountered during units tests, this is probably safe to ignore, (different stream).\n" \
             "  If that was not a mistake, you may change DIMENSION_LIMIT and recomplile. Exiting.\n",
             dim, DIMENSION_LIMIT);
      return -2.;
    }

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
    block_dim.x = 256;
    grid_dim.x = (nA + block_dim.x) / block_dim.x;
    local_distance_kernel<<<grid_dim, block_dim, 0, streams[0]>>>(A_dev, nA, degree, dim, DA_dev, 1);
    HANDLE_ERROR(cudaPeekAtLastError());

    /* compute initial distance B */
    block_dim.x = 256;
    grid_dim.x = (nB + block_dim.x) / block_dim.x;
    local_distance_kernel<<<grid_dim, block_dim, 0, streams[1]>>>(B_dev, nB, degree, dim, DB_dev, 1);
    HANDLE_ERROR(cudaPeekAtLastError());

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


  int _TWED_BATCH(REAL_t AA[], int nA, REAL_t TAA[],
                  REAL_t BB[], int nB, REAL_t TBB[],
                  REAL_t nu, REAL_t lambda, int degree, int dim,
                  int nAA, int nBB, REAL_t* RRes, TRI_OPT_t tri){
    REAL_t *AA_dev, *TAA_dev;
    REAL_t *BB_dev, *TBB_dev;
    int result;


    /* malloc gpu arrays */
    _TWED_MALLOC_DEV(nA, &AA_dev, &TAA_dev,
                     nB, &BB_dev, &TBB_dev, dim, nAA, nBB);

    /* copy inputs to device */
    _TWED_COPY_TO_DEV(nA, AA, AA_dev, TAA, TAA_dev,
                      nB, BB, BB_dev, TBB, TBB_dev, dim, nAA, nBB);

    /* compute TWED on device */
    result = _TWED_BATCH_DEV(AA_dev, nA, TAA_dev,
                             BB_dev, nB, TBB_dev,
                             nu, lambda, degree, dim,
                             nAA, nBB, RRes, tri);

    /* cleanup device memory initialized here */
    _TWED_FREE_DEV(AA_dev, TAA_dev,
                   BB_dev, TBB_dev);

    return result;

  }


  int _TWED_BATCH_DEV(REAL_t AA_dev[], int nA, REAL_t TAA_dev[],
                      REAL_t BB_dev[], int nB, REAL_t TBB_dev[],
                      REAL_t nu, REAL_t lambda, int degree, int dim,
                      int nAA, int nBB, REAL_t* RRes, TRI_OPT_t tri){
    REAL_t *DA_dev;
    REAL_t *DBB_dev;
    REAL_t *Res_dev_write;
    REAL_t *Res_dev_read;
    int a;
    int tril;
    /*
      tri= 0 for complete matrix (typical). Note -1, -2 require symmetric batch (dist matrix).
      tri=-1 for lower (tril)
      tri=-2 triu for upper (triu). Note, triu will compute tril then transpose, which is
      not very effecient, but I offer for convenience.
    */

    tril = -1;  /* defaul tri optimizations off */
    if(tri == TRIL || tri == TRIU){
      if(nAA != nBB){
        fprintf(stderr, "Error. To use the triangular optimization, you must request a symmetric batch.\n");
        return -2;
      }
      tril = 0;  /* tri opt on */
    }


    /*
      Sanity Check
    */
    if(nBB > BATCH_LIMIT || nAA > BATCH_LIMIT){
      fprintf(stderr, "Error, a supplied batch dimension nAA %d nBB %d is greater than BATCH_LIMIT %d.\n" \
             "  If encountered during units tests, this is probably safe to ignore, (different stream).\n" \
             "  Try running a few batches instead of one large one.",
             nAA, nBB, BATCH_LIMIT);
      return -BATCH_LIMIT;
    }


    /*
      Sanity Check Dimension
    */
    if(dim > DIMENSION_LIMIT){
      fprintf(stderr, "Error, supplied dimension %d is greater than compiled DIMENSION_LIMIT %d.\n" \
             "  If encountered during units tests, this is probably safe to ignore, (different stream).\n" \
             "  If that was not a mistake, you may change DIMENSION_LIMIT and recomplile. Exiting.\n",
             dim, DIMENSION_LIMIT);
      return -DIMENSION_LIMIT;
    }

    const int nstreams = 2;
    int s;
    cudaStream_t streams[nstreams];
    for(s=0; s<nstreams; s++){
      HANDLE_ERROR(cudaStreamCreate(&streams[s]));
    }

    const size_t sza = (nA+1) * sizeof(*AA_dev);
    const size_t szbb = nBB*(nB+1) * sizeof(*BB_dev);
    HANDLE_ERROR(cudaMalloc(&DA_dev, sza));
    HANDLE_ERROR(cudaMalloc(&DBB_dev, szbb));
    const size_t szr = nBB * sizeof(*BB_dev);
    HANDLE_ERROR(cudaMalloc(&Res_dev_write, szr));
    HANDLE_ERROR(cudaMalloc(&Res_dev_read, szr));
    HANDLE_ERROR(cudaMemset(Res_dev_write, 0, szr));
    HANDLE_ERROR(cudaMemset(Res_dev_read, 0, szr));
    REAL_t* tmp_ptr;


    /* compute initial distance B */
    dim3 block_dim(32, 32);
    dim3 grid_dim((nB + block_dim.x) / block_dim.x,
                  (nBB + block_dim.y) / block_dim.y);
    local_distance_kernel<<<grid_dim, block_dim, 0, streams[0]>>>(BB_dev, nB, degree, dim, DBB_dev, nBB);
    HANDLE_ERROR(cudaPeekAtLastError());

    /* A and B now have diff block dims block/grid dims */
    dim3 block_dimA(256);
    dim3 grid_dimA((nA + block_dimA.x) / block_dimA.x);
    for(a=0; a<nAA; a++){
      /* compute initial distance A */
      /* note I offset into the (nAA,nA,dim) array  AA, but DAA is (nA, dim) */
      local_distance_kernel<<<grid_dimA, block_dimA, 0, streams[1]>>>(&AA_dev[a*nA*dim], nA, degree, dim, DA_dev, 1);
      HANDLE_ERROR(cudaPeekAtLastError());

      HANDLE_ERROR(cudaDeviceSynchronize());

    /*
      Same as eval but over all B, up to 65k

      This method iteratively updates the DP matrix.
      We process diagonals moving the diagonal from upper left to lower right.
      Each element of a diag is done in parallel.
    */
      if(tril != -1){
        tril = a;
      }

      grid_evalZ(&AA_dev[a*nA*dim], DA_dev, nA, &TAA_dev[a*nA],
                 BB_dev, DBB_dev, nB, TBB_dev,
                 nu, degree, lambda, dim,
                 Res_dev_write, nAA, nBB, tril);
      HANDLE_ERROR(cudaPeekAtLastError());

      HANDLE_ERROR(cudaDeviceSynchronize());

      /* Res_dev now has results for a,B ;
          Cycle the pointers
          then write out in a seperate stream1 */
      tmp_ptr = Res_dev_read;
      Res_dev_read = Res_dev_write;
      Res_dev_write = tmp_ptr;

      // copy out, do something async later
      HANDLE_ERROR(cudaMemcpy(&RRes[a*nBB], Res_dev_read, nBB*sizeof(*RRes), cudaMemcpyDeviceToHost));

    }
    HANDLE_ERROR(cudaFree(DA_dev));
    HANDLE_ERROR(cudaFree(DBB_dev));
    HANDLE_ERROR(cudaFree(Res_dev_read));
    HANDLE_ERROR(cudaFree(Res_dev_write));

    for(s=0; s<nstreams; s++){
      HANDLE_ERROR(cudaStreamDestroy(streams[s]));
    }

    if(tri == TRIU){
      //transpose
      //cudaError_t cudaStat;
      cublasStatus_t stat;
      cublasHandle_t handle;
      REAL_t *B = NULL;
      REAL_t alpha = 1.0;
      REAL_t beta = 0.0;
      REAL_t* devPtrA = NULL;
      REAL_t* devPtrC = NULL;
      HANDLE_ERROR(cudaMalloc(&devPtrA, nBB*nBB*sizeof(*devPtrA)));
      HANDLE_ERROR(cudaMalloc(&devPtrC, nBB*nBB*sizeof(*devPtrC)));

      stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return -EXIT_FAILURE;
      }

      stat = cublasSetMatrix (nBB, nBB, sizeof(*RRes), RRes, nBB, devPtrA, nBB);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        printf(_cudaGetErrorEnum(stat));
        printf ("\ndata download failed\n");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return -EXIT_FAILURE;
      }

      /* call geam */
      stat = _GEAM(handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   nBB, nBB,
                   &alpha,
                   devPtrA, nBB,
                   &beta,
                   B, nBB,
                   devPtrC, nBB);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        printf(_cudaGetErrorEnum(stat));
        printf ("\nCUBLAS geam failed\n");
        return -EXIT_FAILURE;
      }


      stat = cublasGetMatrix (nBB, nBB, sizeof(*RRes), devPtrC, nBB, RRes, nBB);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return -EXIT_FAILURE;
      }

      cublasDestroy(handle);
      HANDLE_ERROR(cudaFree(devPtrA));
      HANDLE_ERROR(cudaFree(devPtrC));
    }

    return 0;
  }


  REAL_t _TWED(REAL_t A[], int nA, REAL_t TA[],
               REAL_t B[], int nB, REAL_t TB[],
               REAL_t nu, REAL_t lambda, int degree, int dim){
    REAL_t *A_dev, *TA_dev;
    REAL_t *B_dev, *TB_dev;
    REAL_t result;

    /* malloc gpu arrays */
    _TWED_MALLOC_DEV(nA, &A_dev, &TA_dev,
                     nB, &B_dev, &TB_dev, dim, 1, 1);

    /* copy inputs to device */
    _TWED_COPY_TO_DEV(nA, A, A_dev, TA, TA_dev,
                      nB, B, B_dev, TB, TB_dev, dim, 1, 1);

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

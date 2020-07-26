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

#ifndef __cuTWED_H_
#define __cuTWED_H_

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum TRI_OPT {TRIU=-2,
                        TRIL=-1,
                        NOPT=0} TRI_OPT_t;

  /*
    A, B are arrays of time series values.
    TA, TB are the respective time stamps.
    nA, nB are number of elements in A and B
    nu, lambda, and degree are algo params.
  */
  double twed(double A[], int nA, double TA[],
              double B[], int nB, double TB[],
              double nu, double lambda, int degree, int dim);

  float twedf(float A[], int nA, float TA[],
              float B[], int nB, float TB[],
              float nu, float lambda, int degree, int dim);


  /*
    Same a twed, just expect CU (gpu) arrays,
    You may use twed_malloc_dev if you want to right logic to reuse gpu memory etc.
  */
  double twed_dev(double A_dev[], int nA, double TA_dev[],
                  double B_dev[], int nB, double TB_dev[],
                  double nu, double lambda, int degree, int dim);

  float twed_devf(float A_dev[], int nA, float TA_dev[],
                  float B_dev[], int nB, float TB_dev[],
                  float nu, float lambda, int degree, int dim);


  /*
    Mallocs memory on device, approximately (6*nA + 6*nB) * sizeof(REAL_t)
  */
  void twed_malloc_dev(const int nA, double **A_dev, double  **TA_dev,
                       const int nB, double **B_dev, double  **TB_dev,
                       const int dim, const int nAA, const int nBB);

  void twed_malloc_devf(const int nA, float **A_dev, float  **TA_dev,
                        const int nB, float **B_dev, float  **TB_dev,
                        const int dim, const int nAA, const int nBB);



  /*
    Frees memory malloc'd in twed_malloc_dev
  */
  void twed_free_dev(double *A_dev, double  *TA_dev,
                     double *B_dev, double  *TB_dev);

  void twed_free_devf(float *A_dev, float  *TA_dev,
                      float *B_dev, float  *TB_dev);

  /*
    Copies data from host to device. You would only use this function if you
    are writing logic to reuse gpu memory.
  */
  void twed_copy_to_dev(const int nA, double A[], double A_dev[], double TA[], double TA_dev[],
                        const int nB, double B[], double B_dev[], double TB[], double TB_dev[],
                        const int dim, const int nAA, const int nBB);

  void twed_copy_to_devf(const int nA, float A[], float A_dev[], float TA[], float TA_dev[],
                         const int nB, float B[], float B_dev[], float TB[], float TB_dev[],
                         const int dim, const int nAA, const int nBB);


  /*
    Batch calls for TWED.
      Expects AA as (nAA, nA, dim) array for A.
      Expects BB as (nBB, nB, dim) array for B.
      RRes is a pointer to result scratch space.
  */
  int twed_batch(double AA_dev[], int nA, double TAA_dev[],
                 double BB_dev[], int nB, double TBB_dev[],
                 double nu, double lambda, int degree, int dim,
                 int nAA, int nBB, double* RRes, TRI_OPT_t tri);

  int twed_batchf(float AA_dev[], int nA, float TAA_dev[],
                  float BB_dev[], int nB, float TBB_dev[],
                  float nu, float lambda, int degree, int dim,
                  int nAA, int nBB, float* RRes, TRI_OPT_t tri);

  int twed_batch_dev(double AA_dev[], int nA, double TAA_dev[],
                     double BB_dev[], int nB, double TBB_dev[],
                     double nu, double lambda, int degree, int dim,
                     int nAA, int nBB, double* RRes, TRI_OPT_t tri);

  int twed_batch_devf(float AA_dev[], int nA, float TAA_dev[],
                      float BB_dev[], int nB, float TBB_dev[],
                      float nu, float lambda, int degree, int dim,
                      int nAA, int nBB, float* RRes, TRI_OPT_t tri);


#ifdef __cplusplus
}
#endif


#endif

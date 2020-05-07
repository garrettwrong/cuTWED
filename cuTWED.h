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
  void twed_malloc_dev(int nA, double **A_dev, double  **TA_dev,
                       int nB, double **B_dev, double  **TB_dev);

  void twed_malloc_devf(int nA, float **A_dev, float  **TA_dev,
                        int nB, float **B_dev, float  **TB_dev);


  
  /*
    Frees memory malloc'd in twed_malloc_dev
  */
  void twed_free_dev(double *A_dev, double  *TA_dev,
                     double *B_dev, double  *TB_dev);

  void twed_free_devf(float *A_dev, float  *TA_dev,
                      float *B_dev, float  *TB_dev);

  /*
    Copies data from host to device. You would only use this function if you
    are writing logic to resuse gpu memory.
  */
  void twed_copy_to_dev(int nA, double A[], double A_dev[], double TA[], double TA_dev[],
                        int nB, double B[], double B_dev[], double TB[], double TB_dev[]);

  void twed_copy_to_devf(int nA, float A[], float A_dev[], float TA[], float TA_dev[],
                         int nB, float B[], float B_dev[], float TB[], float TB_dev[]);


#ifdef __cplusplus
}
#endif
  

#endif

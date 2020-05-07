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

/* CPP Macro for floating point type */
#ifndef REAL_t
#define REAL_t double
#endif

#ifdef __cplusplus  
extern "C" { 
#endif

  /*
    A, B are arrays of time series values.
    TA, TB are the respective time stamps.
    nA, nB are number of elements in A and B
    nu, lambda, and degree are algo params.
  */
REAL_t twed(REAL_t A[], int nA, REAL_t TA[],
            REAL_t B[], int nB, REAL_t TB[],
            REAL_t nu, REAL_t lambda, int degree);
#ifdef __cplusplus  
}
#endif 


#ifdef __cplusplus
extern "C" {
#endif
  /*
    Same a twed, just expect CU (gpu) arrays,
    You may use twed_malloc_dev if you want to right logic to reuse gpu memory etc.
   */
REAL_t twed_dev(REAL_t A_dev[], int nA, REAL_t TA_dev[],
                REAL_t B_dev[], int nB, REAL_t TB_dev[],
                REAL_t nu, REAL_t lambda, int degree);
#ifdef __cplusplus  
}
#endif 


#ifdef __cplusplus
extern "C" {
#endif
void twed_malloc_dev(int nA, REAL_t **A_dev, REAL_t  **TA_dev,
                     int nB, REAL_t **B_dev, REAL_t  **TB_dev);
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
  /*
    Frees memory malloc'd in twed_malloc_dev
   */
void twed_free_dev(REAL_t *A_dev, REAL_t  *TA_dev,
                   REAL_t *B_dev, REAL_t  *TB_dev);
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif
  /*
    Copies data from host to device. You would only use this function if you
    are writing logic to resuse gpu memory.
   */
void twed_copy_to_dev(int nA, REAL_t A[], REAL_t A_dev[], REAL_t TA[], REAL_t TA_dev[],
                      int nB, REAL_t B[], REAL_t B_dev[], REAL_t TB[], REAL_t TB_dev[]);
#ifdef __cplusplus
}
#endif

  

#endif

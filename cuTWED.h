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
            double nu, double lambda, int degree, double* DP);

#ifdef __cplusplus  
}
#endif 

#endif

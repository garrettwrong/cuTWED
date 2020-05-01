#include <stdio.h>

#include "cuTWED.h"

#include "reference_implementation/reference_arrays.h"

int main(int argc, char** argv){

  double nu = 1.;
  double lambda = 1.;
  int degree = 2;
  double dist = -1;
  
  dist = twed(A, nA, TA, B, nB, TB, nu, lambda, degree, NULL);

  /* to return DP for inspection
  double* DP;
  DP = (double*)calloc((nA+1)*(nB+1), sizeof(*DP));
  dist = twed(A, nA, TA, B, nB, TB, nu, lambda, degree, DP);
  */
  printf("ctwed dist: %lf\n", dist);
  
}


  

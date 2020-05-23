#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "twed.h"

int main(int argc, char** argv){

#include "reference_arrays.h"

  double nu = 1.;
  double lambda = 1.;
  int degree = 2;
  double dist = -1;
  int dimension = 1;


  CTWED(A, &nA, TA, B, &nB, TB, &nu, &lambda, &degree, &dist, &dimension);
  printf("CTWED dist: %f\n", dist);

  return 0;
}


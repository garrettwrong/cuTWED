/*
  Demonstates basic cuTWED usage in C.

  Copyright 2020 Garrett Wright, Gestalt Group LLC
*/

#include <stdio.h>

#include "cuTWED.h"

/* this just loads some static arrays */
#define REAL_t float
#include "../reference_implementation/reference_arrays.h"

int main(int argc, char** argv){

  int degree = 2;

  float nu = 1.;
  float lambda = 1.;
  float dist = -1;
  int dim = 1;

  dist = twedf(A, nA, TA, B, nB, TB, nu, lambda, degree, dim);

  printf("cuTWED dist: %f\n", dist);

  return 0;
}

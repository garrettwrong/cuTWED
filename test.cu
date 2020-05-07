/*
  Demonstates basic cuTWED usage in C.

  Copyright 2020 Garrett Wright, Gestalt Group LLC
*/

#include <stdio.h>

#include "cuTWED.h"

#include "reference_implementation/reference_arrays.h"

int main(int argc, char** argv){

  int degree = 2;

  double nu = 1.;
  double lambda = 1.;
  double dist = -1;

  dist = twed(A, nA, TA, B, nB, TB, nu, lambda, degree);

  printf("cuTWED dist: %lf\n", dist);

}

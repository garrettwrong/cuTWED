/* Note the original code was extended to cover inputs of R^N.

   This requires an extra argument (int *dim),
   and in that case expects ta and tb to be c-ordered shape
   (la, N) and (lb, N) respectively.

   I also add the root into the norm calls, because it
   is more consistent with implementations that followed this.

   Otherwise up to white space and some optional debug printing,
   this is taken right from Marteau as a reference implementation.

   Garrett Wright, 2020
*/

/*
  Filename: twed.h
  source code for the Time Warp Edit Distance in ANSI C.
  Author: Pierre-Francois Marteau, adapted for R integration by Jorg Schaber
  Version: V1.2.a du 25/08/2014, radix addition line 101, thanks to Benjamin Herwig from University of Kassel, Germany
  Licence: GPL
  ******************************************************************
  This software and description is free delivered "AS IS" with no
  guaranties for work at all. Its up to you testing it modify it as
  you like, but no help could be expected from me due to lag of time
  at the moment. I will answer short relevant questions and help as
  my time allow it. I have tested it played with it and found no
  problems in stability or malfunctions so far.
  Have fun.
  *****************************************************************
  Please cite as:
  @article{Marteau:2009:TWED,
  author = {Marteau, Pierre-Francois},
  title = {Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching},
  journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
  issue_date = {February 2009},
  volume = {31},
  number = {2},
  month = feb,
  year = {2009},
  issn = {0162-8828},
  pages = {306--318},
  numpages = {13},
  url = {http://dx.doi.org/10.1109/TPAMI.2008.76},
  doi = {10.1109/TPAMI.2008.76},
  acmid = {1496043},
  publisher = {IEEE Computer Society},
  address = {Washington, DC, USA},
  keywords = {Dynamic programming, Pattern recognition, Pattern recognition, time series, algorithms, similarity measures,
  Similarity measures, algorithms, similarity measures., time series},
  }
*/

/*
  INPUTS
  double ta[]: array containing the first time series; ta[i] is the i^th sample for i in {0, .., la-1}
  int *la: length of the first time series
  double tsa[]: array containing the time stamps for time series ta; tsa[i] is the time stamp for sample ta[i]. The length of tsa array is expected to be la.
  double tb[]: array containing the second time series; tb[i] is the i^th sample for i in {0, .., lb-1}
  int *lb: length of the second time series
  double tsb[]: array containing the time stamps for time series tb; tsb[j] is the time stamp for sample tb[j]. The length of tsb array is expected to be lb.
  double *nu: value for parameter nu
  double *lambda: value for parameter lambda
  int *degree: power degree for the evaluation of the local distance between samples: degree>0 required
  OUTPUT
  double: the TWED distance between time series ta and tb.
*/


void CTWED(double ta[], int *la, double tsa[],
           double tb[], int *lb, double tsb[],
           double *nu, double *lambda, int *degree,
           double *dist, int *dimension);

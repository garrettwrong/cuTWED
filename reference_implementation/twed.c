/* Note the original code was extended to cover inputs of R^N.

   This requires an extra argument (int *dimension),
   and in that case expects ta and tb to be c-ordered shape
   (la, N) and (lb, N) respectively.

   I also add the root into the norm calls, because it
   is more consistent with implementations that followed this.

   Otherwise up to white space and some formatting,
   this is taken right from Marteau as a reference implementation.

   Garrett Wright, 2020
*/

/*
  Filename: twed.c
  source code for the Time Warp Edit Distance in ANSI C.
  Author: Pierre-Francois Marteau, adapted for R integration by Jörg Schaber
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "twed.h"

void CTWED(double ta[], int *la, double tsa[],
           double tb[], int *lb, double tsb[],
           double *nu, double *lambda, int *degree,
           double *dist, int *dimension) {
  //
  //    TWED PAMI
  //
  if(*la<0||*lb<0){
    fprintf(stderr, "twed: the lengths of the input timeseries should be greater or equal to 0\n");
    exit(-1);
  }
  int r = *la;
  int c = *lb;
  int deg = abs(*degree); // allow a hack to repro authors original results.
  double disti1, distj1;
  int i,j,n;
  int dim = *dimension;

  // allocations
  double **D = (double **)calloc(r+1, sizeof(double*));
  double *Di1 = (double *)calloc(r+1, sizeof(double));
  double *Dj1 = (double *)calloc(c+1, sizeof(double));

  double dmin, htrans, dist0;

  for(i=0; i<=r; i++) {
    D[i]=(double *)calloc(c+1, sizeof(double));
  }
  // local costs initializations
  for(j=1; j<=c; j++) {
    distj1=0;
    for(n=0; n<dim; n++){
      if(j>1){
        distj1+=pow(fabs(tb[(j-2)*dim + n] -
                         tb[(j-1)*dim + n]),deg);
      }
      else distj1+=pow(fabs(tb[(j-1)*dim + n]),deg);
    }

    // NOTE original author did not nth-root
    if(*degree<0){      // I provide "no root" as negative degree, to match any prior results
      Dj1[j]=distj1;    // Consider it a semi hidden feature.
    } else if(deg==2){
      Dj1[j]=sqrt(distj1);
    }
    else Dj1[j]=pow(distj1, 1./deg);
  }

  for(i=1; i<=r; i++) {
    disti1=0;
    for(n=0; n<dim; n++){
      if(i>1)
        disti1+=pow(fabs(ta[(i-2)*dim + n] -
                         ta[(i-1)*dim + n]),deg);
      else disti1+=pow(fabs(ta[(i-1)*dim + n]),deg);
    }

    // NOTE original author did not nth-root
    if(*degree<0){      // I provide "no root" as negative degree, to match any prior results
      Di1[i]=disti1;    // Consider it a semi hidden feature.
    } else if(deg==2){
      Di1[i]=sqrt(disti1);
    }
    else Di1[i]=pow(disti1, 1./deg);

    for(j=1; j<=c; j++) {
      (*dist)=0;
      dist0 = 0;
      for(n=0; n<dim; n++){
        *(dist)+=pow(fabs(ta[(i-1)*dim + n] -
                          tb[(j-1)*dim + n]),deg);
        if(i>1&&j>1)
          dist0 += pow(fabs(ta[(i-2)*dim + n] -
                            tb[(j-2)*dim + n]),deg);
      }

      // NOTE original author did not nth-root
      if(*degree<0){      // I provide "no root" as negative degree, to match any prior results
        D[i][j]=*dist + dist0;    // Consider it a semi hidden feature.
      } else if(deg==2){
        D[i][j]=sqrt(*dist) + sqrt(dist0);
      }
      else D[i][j]=pow(*dist, 1./deg) + pow(dist0, 1./deg);


    }
  } // for i

  // border of the cost matrix initialization
  D[0][0]=0;
  for(i=1; i<=r; i++)
    D[i][0]=INFINITY;
  for(j=1; j<=c; j++)
    D[0][j]=INFINITY;


  for (i=1; i<=r; i++){
    for (j=1; j<=c; j++){
      htrans=fabs((double)(tsa[i-1]-tsb[j-1]));
      if(j>1&&i>1)
        htrans+=fabs((double)(tsa[i-2]-tsb[j-2]));
      dist0=D[i-1][j-1]+D[i][j]+(*nu)*htrans;

      dmin=dist0;

      if(i>1)
        htrans=((double)(tsa[i-1]-tsa[i-2]));
      else htrans=(double)tsa[i-1];

      (*dist)=Di1[i]+D[i-1][j]+(*lambda)+(*nu)*htrans;

      if(dmin>(*dist)){
        dmin=(*dist);
      }


      if(j>1)
        htrans=((double)(tsb[j-1]-tsb[j-2]));
      else htrans=(double)tsb[j-1];
      (*dist)=Dj1[j]+D[i][j-1]+(*lambda)+(*nu)*htrans;
      if(dmin>(*dist)){
        dmin=(*dist);
      }
      D[i][j] = dmin;
    }
  }

  (*dist) = D[r][c];

  // freeing
  for(i=0; i<=r; i++) {
    free(D[i]);
  }
  free(D);
  free(Di1);
  free(Dj1);
}

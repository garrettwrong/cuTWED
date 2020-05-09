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
//#include "twed.h"

void CTWED(double ta[], int *la, double tsa[], double tb[], int *lb, double tsb[], double *nu, double *lambda, int *degree, double *dist) {
        // 
        //    TWED PAMI
        //    
        if(*la<0||*lb<0){
                fprintf(stderr, "twed: the lengths of the input timeseries should be greater or equal to 0\n");
                exit(-1);
                }
        int r = *la;
        int c = *lb;
        int deg = *degree;
        double disti1, distj1;
        int i,j;

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
                if(j>1){
                        distj1+=pow(fabs(tb[j-2]-tb[j-1]),deg); 
                }
                else distj1+=pow(fabs(tb[j-1]),deg);
                
                //Dj1[j]=distj1;
                Dj1[j]=sqrt(distj1);  // NOTE original author did not sqrt
                //printf("Dj1[ %d ] = %f\n", j, distj1);
        }
        
        for(i=1; i<=r; i++) { 
                disti1=0;
                if(i>1)
                        disti1+=pow(fabs(ta[i-2]-ta[i-1]),deg);
                else disti1+=pow(fabs(ta[i-1]),deg);

                //Di1[i]=disti1;
                Di1[i]=sqrt(disti1); // NOTE original author did not sqrt
  
                for(j=1; j<=c; j++) {
                        (*dist)=0;
                        (*dist)+=pow(fabs(ta[i-1]-tb[j-1]),deg);        
                        if(i>1&&j>1)
                                (*dist)+=pow(fabs(ta[i-2]-tb[j-2]),deg);

                D[i][j]=*dist;
                //printf("D[ %d ][ %d ] = %f\n", i, j, *dist);
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

        /* for(int row=0; row<= *la; row++){ */
        /*   for(c=0; c<= *lb; c++){ */
        /*     printf("%f, ", D[row][c]); */
        /*   } */
        /*   printf("\n"); */
        /* } */


        // freeing
        for(i=0; i<=r; i++) {
                free(D[i]);
        }
        free(D);
        free(Di1);
        free(Dj1);
}


int main(int argc, char** argv){

#include "reference_arrays.h"

  double nu = 1.;
  double lambda = 1.;
  int degree = 2;
  double dist = -1;
  
  CTWED(A, &nA, TA, B, &nB, TB, &nu, &lambda, &degree, &dist);
  printf("CTWED dist: %f\n", dist);
  
}


  

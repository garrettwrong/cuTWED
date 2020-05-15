  double twed(double A[], int nA, double TA[],
              double B[], int nB, double TB[],
              double nu, double lambda, int degree, int dim);
  float twedf(float A[], int nA, float TA[],
              float B[], int nB, float TB[],
              float nu, float lambda, int degree, int dim);
  double twed_dev(double A_dev[], int nA, double TA_dev[],
                  double B_dev[], int nB, double TB_dev[],
                  double nu, double lambda, int degree, int dim);
  float twed_devf(float A_dev[], int nA, float TA_dev[],
                  float B_dev[], int nB, float TB_dev[],
                  float nu, float lambda, int degree, int dim);
  void twed_malloc_dev(int nA, double **A_dev, double **TA_dev,
                       int nB, double **B_dev, double **TB_dev);
  void twed_malloc_devf(int nA, float **A_dev, float **TA_dev,
                        int nB, float **B_dev, float **TB_dev);
  void twed_free_dev(double *A_dev, double *TA_dev,
                     double *B_dev, double *TB_dev);
  void twed_free_devf(float *A_dev, float *TA_dev,
                      float *B_dev, float *TB_dev);
  void twed_copy_to_dev(int nA, double A[], double A_dev[], double TA[], double TA_dev[],
                        int nB, double B[], double B_dev[], double TB[], double TB_dev[]);
  void twed_copy_to_devf(int nA, float A[], float A_dev[], float TA[], float TA_dev[],
                         int nB, float B[], float B_dev[], float TB[], float TB_dev[]);

  typedef enum TRI_OPT {TRIU=-2,
                        TRIL=-1,
                        NOPT=0} TRI_OPT_t;
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
  void twed_malloc_dev(const int nA, double **A_dev, double **TA_dev,
                       const int nB, double **B_dev, double **TB_dev,
                       const int dim, const int nAA, const int nBB);
  void twed_malloc_devf(const int nA, float **A_dev, float **TA_dev,
                        const int nB, float **B_dev, float **TB_dev,
                        const int dim, const int nAA, const int nBB);
  void twed_free_dev(double *A_dev, double *TA_dev,
                     double *B_dev, double *TB_dev);
  void twed_free_devf(float *A_dev, float *TA_dev,
                      float *B_dev, float *TB_dev);
  void twed_copy_to_dev(const int nA, double A[], double A_dev[], double TA[], double TA_dev[],
                        const int nB, double B[], double B_dev[], double TB[], double TB_dev[],
                        const int dim, const int nAA, const int nBB);
  void twed_copy_to_devf(const int nA, float A[], float A_dev[], float TA[], float TA_dev[],
                         const int nB, float B[], float B_dev[], float TB[], float TB_dev[],
                         const int dim, const int nAA, const int nBB);
  int twed_batch(double AA_dev[], int nA, double TAA_dev[],
                 double BB_dev[], int nB, double TBB_dev[],
                 double nu, double lambda, int degree, int dim,
                 int nAA, int nBB, double* RRes, TRI_OPT_t tri);
  int twed_batchf(float AA_dev[], int nA, float TAA_dev[],
                  float BB_dev[], int nB, float TBB_dev[],
                  float nu, float lambda, int degree, int dim,
                  int nAA, int nBB, float* RRes, TRI_OPT_t tri);
  int twed_batch_dev(double AA_dev[], int nA, double TAA_dev[],
                     double BB_dev[], int nB, double TBB_dev[],
                     double nu, double lambda, int degree, int dim,
                     int nAA, int nBB, double* RRes, TRI_OPT_t tri);
  int twed_batch_devf(float AA_dev[], int nA, float TAA_dev[],
                      float BB_dev[], int nB, float TBB_dev[],
                      float nu, float lambda, int degree, int dim,
                      int nAA, int nBB, float* RRes, TRI_OPT_t tri);

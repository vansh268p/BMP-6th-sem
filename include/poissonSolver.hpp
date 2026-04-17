#include "types.hpp"
// extern double *A;						///< Non zero coefficients for Poisson Solver (CSR format)			
// extern int *IA;						///< Number of non zero values per row in A (CSR format)
// extern int *JA;						///< Column index of each element in A (CSR format)

// extern int n;							///< no of rows or cols in Sparse matrix(A) n = GRID_X * GRID_Y 
// extern void *pt[64];					///< Pardiso Integer array (Refer to PARDSIO Manual)
// extern int idum;						///< Integer dummy (Refer to PARDSIO Manual)
// extern int iparm[64];					///< Pardiso parameters (Refer to PARDSIO Manual)
// extern double dparm[64];				///< pardiso control parameters (Refer to PARDISO Manual)
// extern int mtype;						///< Matrix type for Pardiso solver (Refer to PARDSIO Manual)
// extern int maxfct;						///< Maximum number of numerical factorization (Refer to PARDSIO Manual)
// extern int mnum;						///< Which factorization to use (Refer to PARDSIO Manual)
// extern int nrhs;						///< Number of RHS (Refer to PARDSIO Manual)
// extern int phase;						///< Only reordering and symbolic factorization (Refer to PARDSIO Manual)
// extern int error;						///< Initialize error flag (Refer to PARDSIO Manual)
// extern int msglvl;						///< Print statistical information (Refer to PARDSIO Manual)
// extern int solver;						///< Pardiso Parameter (Refer to PARDISO Manual)
/** Initializes all needed for poisson solver (basically call initCoeffs and initPardiso function) (Helper Function) */
void initPoissonSolver(sycl::queue &q);
int initCoeffs(sycl::queue &q);
int initPardiso();
void setPotentialBoundary(char *potentialBoundary);
int generateCoeffs(char *potentialBoundary,sycl::queue &q);
int generateCoeffsPeriodic(char *potentialBoundary,sycl::queue &q);
void printMatrixToFile();
void overwritePhi(Geometry *geometries, double *phi);
void poissonSolver(double *rho, double *phi, int iteration,sycl::queue &q);
void overwriteRho(Geometry *geometries, double *rho);
void calculateElectricFieldPeriodic(double *phi, Field *electricField);
void calculateElectricField(double *phi, Field *electricField);


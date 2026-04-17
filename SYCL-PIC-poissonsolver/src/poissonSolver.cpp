#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <mkl.h>
#include <sycl/sycl.hpp>
#include "poissonSolver.hpp"


double *A;						///< Non zero coefficients for Poisson Solver (CSR format)			
MKL_INT *IA;						///< Number of non zero values per row in A (CSR format)
MKL_INT *JA;						///< Column index of each element in A (CSR format)

MKL_INT n;							///< no of rows or cols in Sparse matrix(A) n = GRID_X * GRID_Y 
void *pt[64];					///< Pardiso Integer array (Refer to PARDSIO Manual)
MKL_INT idum;						///< Integer dummy (Refer to PARDSIO Manual)
MKL_INT iparm[64];					///< Pardiso parameters (Refer to PARDSIO Manual)
double dparm[64];				///< pardiso control parameters (Refer to PARDISO Manual)
MKL_INT mtype;						///< Matrix type for Pardiso solver (Refer to PARDSIO Manual)
MKL_INT maxfct;						///< Maximum number of numerical factorization (Refer to PARDSIO Manual)
MKL_INT mnum;						///< Which factorization to use (Refer to PARDSIO Manual)
MKL_INT nrhs;						///< Number of RHS (Refer to PARDSIO Manual)
MKL_INT phase;						///< Only reordering and symbolic factorization (Refer to PARDSIO Manual)
MKL_INT error;						///< Initialize error flag (Refer to PARDSIO Manual)
MKL_INT msglvl;						///< Print statistical information (Refer to PARDSIO Manual)
MKL_INT solver;						///< Pardiso Parameter (Refer to PARDISO Manual)

void printMatrixToFile(){

	int i = 0;
	FILE *fd = fopen("pardiso_matrix.out","w");

	if(fd == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/field.out\n");
		exit(1);
	}

	fprintf(fd, "A:[");
	for(i = 0; i < 5 * params1.GRID_X * (params1.GRID_Y-1); i++){
		fprintf(fd, "%lf\t", A[i]);
	}
	fprintf(fd, "]\n");
	
	fprintf(fd, "IA:[");
	for(i = 0; i < params1.GRID_X * (params1.GRID_Y-1) + 1; i++){
		fprintf(fd, "%d\t", IA[i]);
	}
	fprintf(fd, "]\n");

	fprintf(fd, "JA:[");
	for(i = 0; i < 5 * params1.GRID_X * (params1.GRID_Y-1); i++){
		fprintf(fd, "%d\t", JA[i]);
	}
	fprintf(fd, "]\n");

	fclose(fd);

}


void setPotentialBoundary(char *potentialBoundary){

	int i,j,k;

	for(i = 0 ;i < geometryCount ;i++){

		int x1 = geometries[i].x1 , y1 = geometries[i].y1 , x2 = geometries[i].x2 , y2 = geometries[i].y2;
		//printf("%d %d %d %d\n",geometries[i].x1 , geometries[i].y1 , geometries[i].x2 , geometries[i].y2);

		for(k = y1; k <= y2; k++){
			for(j = x1; j <= x2; j++){	
				potentialBoundary[k*GRID_X+j] = 1;
			}
		}
	}
}


int generateCoeffsPeriodic(char *potentialBoundary, sycl::queue &q) {

    const MKL_INT NX = params1.GRID_X;
    const MKL_INT NY = params1.GRID_Y;
    const MKL_INT N  = NX * (NY - 1);   // as your original code used (maybe exclude last row)

    // allocate up to 5 entries per row (worst-case)
    A  = sycl::malloc_shared<double>(5 * N, q);
    IA = sycl::malloc_shared<MKL_INT>(N + 1, q);
    JA = sycl::malloc_shared<MKL_INT>(5 * N, q);

	if(A == NULL || IA == NULL || JA == NULL){
		printf("Couldn't allocate memory : A | IA | JA\n");
		exit(3);
	}

	int i = 0, index = 0;
	double inv_dx2 = 1.0 / (dx * dx);
	double inv_dy2 = 1.0 / (dy * dy);
	IA[0]=1;

	/* Check if top and bottom boundary geometries are provided. Then it shouldn't use periodic Poisson solver. */

	for(i = 0; i < (GRID_X * (GRID_Y-1)); i++){

		if(potentialBoundary[i] == 1){
			
			A[index] = -1;
			IA[i+1] = IA[i] + 1;
			JA[index] = i+1;
			index++;

		}
		else{

			/* Use two different loops for boundary to avoid control divergence in CUDA */

			if(i - GRID_X < 0){
				A[index+4] = -inv_dy2;
				A[index] = -inv_dx2;
				A[index+1] = 2 * (inv_dx2 + inv_dy2);
				A[index+2] = -inv_dx2;
				A[index+3] = -inv_dy2;


				JA[index+4] = ((i - GRID_X) + GRID_X * (GRID_Y-1)) % (GRID_X * (GRID_Y-1)) + 1;
				JA[index] = i - 1 + 1;
				JA[index+1] = i + 1;
				JA[index+2] = i + 1 + 1;
				JA[index+3] = (i + GRID_X) + 1;
			}

			else if(i + GRID_X > GRID_X * (GRID_Y-1)){

				A[index+1] = -inv_dy2;
				A[index+2] = -inv_dx2;
				A[index+3] = 2 * (inv_dx2 + inv_dy2);
				A[index+4] = -inv_dx2;
				A[index] = -inv_dy2;


				JA[index+1] = (i - GRID_X) + 1;
				JA[index+2] = i - 1 + 1;
				JA[index+3] = i + 1;
				JA[index+4] = i + 1 + 1;
				JA[index] = (i + GRID_X) % (GRID_X * (GRID_Y-1)) + 1;

			}

			else{

				A[index] = -inv_dy2;
				A[index+1] = -inv_dx2;
				A[index+2] = 2 * (inv_dx2 + inv_dy2);
				A[index+3] = -inv_dx2;
				A[index+4] = -inv_dy2;


				JA[index] = (i - GRID_X) + 1;
				JA[index+1] = i - 1 + 1;
				JA[index+2] = i + 1;
				JA[index+3] = i + 1 + 1;
				JA[index+4] = (i + GRID_X) + 1;

			}	
		
			IA[i+1] = IA[i] + 5;

			index = index + 5;

		}

	}

	printMatrixToFile();

	return GRID_X*(GRID_Y-1);

}


int generateCoeffs(char *potentialBoundary, sycl::queue &q) {

    const MKL_INT NX = params1.GRID_X;
    const MKL_INT NY = params1.GRID_Y;
    const MKL_INT N  = NX * NY;

    // allocate up to 5 entries per row (worst-case)
    A  = sycl::malloc_shared<double>(5 * N, q);
    IA = sycl::malloc_shared<MKL_INT>(N + 1, q);
    JA = sycl::malloc_shared<MKL_INT>(5 * N, q);

	if(A == NULL || IA == NULL || JA == NULL){
		printf("Couldn't allocate memory : A | IA | JA\n");
		exit(3);
	}

	int i = 0, index = 0;
	double inv_dx2 = 1.0 / (dx * dx);
	double inv_dy2 = 1.0 / (dy * dy);
	IA[0]=1;

	for(i = 0; i < (GRID_X * GRID_Y); i++){

		if(potentialBoundary[i] == 1){
			
			A[index] = -1;
			IA[i+1] = IA[i] + 1;
			JA[index] = i+1;
			index++;

		}
		else{

			A[index] = -inv_dy2;
			A[index+1] = -inv_dx2;
			A[index+2] = 2 * (inv_dx2 + inv_dy2);
			A[index+3] = -inv_dx2;
			A[index+4] = -inv_dy2;


			JA[index] = i - (GRID_X-1)-1 +1;
			JA[index+1] = i-1 +1;
			JA[index+2] = i +1;
			JA[index+3] = i+1 +1;
			JA[index+4] = i + (GRID_X - 1) + 1 +1;
			
		
			IA[i+1] = IA[i] + 5;

			index = index + 5;

		}

	}
	
	printMatrixToFile();
	
	return GRID_X*GRID_Y;
}

int initCoeffs(sycl::queue &q){
	char *potentialBoundary = sycl::malloc_shared<char>(params1.GRID_X * (params1.GRID_Y-1),q);
	int size;
	setPotentialBoundary(potentialBoundary);
	if(periodicBoundary == 0)
		size = generateCoeffs(potentialBoundary,q);
	else
		size = generateCoeffsPeriodic(potentialBoundary,q);
	sycl::free(potentialBoundary,q);

	return size;
}

void initPardiso(int size){
	/* Zero out the Pardiso handle array — MKL docs require this */
	memset(pt, 0, sizeof(pt));
	memset(iparm, 0, sizeof(iparm));
	memset(dparm, 0, sizeof(dparm));

	n = static_cast<MKL_INT>(size);
	solver = 0;					// used sparsed direct solver

	mtype = 11;					// For non-symmetric matrix
	iparm[0] = 0;				// Use all default iparm
	iparm[2] = 1;				// Number of processors

	maxfct = 1;
	mnum = 1;
	nrhs = 1;
	msglvl = 0;
	error = 0;
	phase = 13;

	iparm[32] = 1;				// Calculate determinant of the matrix. Gives output in dparm[32]


	// Use the below for host



	//pardisoinit(pt,&mtype,&solver,iparm,dparm,&error);

	// Use the below for copro


	pardisoinit(pt, &mtype, iparm);


	if(error != 0){
		if(error == -10)
			printf("Solver Initialization Error: No license file found\n");
		if(error == -11)
			printf("Solver Initialization Error: License expired\n");
		if (error == -12)
			printf("Solver Initialization Error: Wrong username or host name\n");
		exit(5);
	}

	//pardiso_chkmatrix(&mtype, &n, A, IA, JA, &error);

	if(error != 0){
		printf("Solver Initialization Error: Inconsistent matrix\n");
		exit(5);
	}

	//pardiso_chkvec(&n,&nrhs,rho,&error);
	if(error != 0){
		printf("Solver Initialization Error: Incorrect chargeDensity array format\n");
		exit (5);
	}

	//pardiso_printstats(&mtype,&n,A,IA,JA,&nrhs,rho,&error);
	if(error != 0){
		printf("Warning: Problem in printing statistics\n");
	}
}


void overwritePhi_Benchmark(Geometry *geometries, double *phi){
	int i,j,k;

	if(periodicBoundary)
		memcpy(phi + ((GRID_Y-1)*GRID_X), phi, GRID_X * sizeof(double));

	for(i = 0 ;i < geometryCount ;i++){

		int x1 = geometries[i].x1 , y1 = geometries[i].y1 , x2 = geometries[i].x2 , y2 = geometries[i].y2;
		double pot = geometries[i].potential;
		for(j=y1;j <= y2; j++){

			for(k = x1; k <= x2; k++){
				phi[j*GRID_X+k] = pot;
			}
		}
	}
	
	// calculate average azimuthal potential at emmision plane 0.024m
	
	int x1 = (int) (GRID_X-1) * 0.96 , y1 = (int) (GRID_Y-1) * 0.0 , x2 = (int) (GRID_X-1) * 0.96 , y2 = (int) (GRID_Y-1) * 1.0;
	double  pot=0.0;
	for(j=y1;j <= y2; j++){
		for(k = x1; k <= x2; k++){
			pot += phi[j*GRID_X+k];
			}
		}
		
	
	pot=pot/GRID_Y;
	
	// adjusting the poisson solvers solution
	
	for(i = 0; i < GRID_Y; i++){
		for(j = 0; j < GRID_X; j++){
			
			phi[i*GRID_X+j] = phi[i*GRID_X+j] - (double) ( (double) j / (double) x1 ) * pot;
			
		}
	}
}

/** Overwrites the potential matrix with boundary values provided in geometry.conf(Helper Function) */
void overwritePhi(Geometry *geometries, double *phi){
	int i,j,k;

	if(periodicBoundary)
		memcpy(phi + ((GRID_Y-1)*GRID_X), phi, GRID_X * sizeof(double));

	for(i = 0 ;i < geometryCount ;i++){

		int x1 = geometries[i].x1 , y1 = geometries[i].y1 , x2 = geometries[i].x2 , y2 = geometries[i].y2;
		double pot = geometries[i].potential;
		for(j=y1;j <= y2; j++){

			for(k = x1; k <= x2; k++){
				phi[j*GRID_X+k] = pot;
			}
		}
	}

}

void overwriteRho(Geometry *geometries, double *rho){
	int i,j,k;

	for(i = 0 ;i < geometryCount ;i++){

		int x1 = geometries[i].x1 , y1 = geometries[i].y1 , x2 = geometries[i].x2 , y2 = geometries[i].y2;
		double pot = geometries[i].potential;
		for(j=y1;j <= y2; j++){

			for(k = x1; k <= x2; k++){
				rho[j*GRID_X+k] = pot;
			}
		}
	}

	for(i = 0; i < GRID_X*GRID_Y; i++){
		rho[i]=-rho[i];
	}
}

void initPoissonSolver(sycl::queue &q){
	int size = initCoeffs(q);
	initPardiso(size);
	overwritePhi(geometries , phi);
}


void calculateElectricFieldPeriodic(double *phi, Field *electricField){


	int i, j;
	
	for(i = 0; i < params1.GRID_Y; i++){
		electricField[i*params1.GRID_X].fx = -((phi[i*params1.GRID_X+1] - phi[i*params1.GRID_X])/params1.dx);
		for(j = 1; j < params1.GRID_X-1; j++){
			electricField[i * params1.GRID_X + j].fx = -((phi[i * params1.GRID_X + (j+1)] - phi[i * params1.GRID_X + (j-1)]) / (2*params1.dx));
		}
		electricField[(i+1)*params1.GRID_X - 1].fx = -((phi[(i+1)*params1.GRID_X-1] - phi[(i+1)*params1.GRID_X-2])/params1.dx);
	}


	for(j=0;j<params1.GRID_X;j++){
		for(i=0;i<params1.GRID_Y;i++){
			electricField[i * params1.GRID_X + j].fy = -((phi[(((i+1) % (params1.GRID_Y-1))  * params1.GRID_X + j)] - phi[(((i-1) + params1.GRID_Y-1) % (params1.GRID_Y-1)) * params1.GRID_X + j]) / (2*params1.dy));
		}
	}
}

/** Calculates new electric field using the new potential (non-periodic boundary case)*/
void calculateElectricField(double *phi, Field *electricField){
	
	int i, j;
	
	for(i = 1; i < GRID_Y-1; i++){
		for(j = 1; j < GRID_X-1; j++){
			electricField[i * GRID_X + j].fx = -((phi[i * GRID_X + (j+1)] - phi[i * GRID_X + (j-1)]) / (2*dx));
			electricField[i * GRID_X + j].fy = -((phi[(i+1)  * GRID_X + j] - phi[(i-1) * GRID_X + j]) / (2*dy));
		}
	}

	//memcpy(electricField, electricField + GRID_X, GRID_X * sizeof(Field));
	//memcpy(electricField + (GRID_X * (GRID_Y-1) ), electricField + ( GRID_X * (GRID_Y-2) ), GRID_X * sizeof(Field));

	for(i = 0 ;i < GRID_Y ;i++){
		electricField[i*GRID_X] = electricField [i*GRID_X + 1];
		electricField[(i+1)*GRID_X -1] = electricField[(i+1)*GRID_X -2];
	}

	for(i = 0 ;i < GRID_X ;i++){
		electricField[i] = electricField [GRID_X + i];
		electricField[GRID_X*(GRID_Y-1)+i] = electricField[GRID_X*(GRID_Y-2) +i ];
	}
}


void poissonSolver(double *rho, double *phi, int iteration, sycl::queue &q) {

    //std::cout << "Starting Poisson Solver for iteration " << iteration << std::endl;
    overwriteRho(geometries, rho);

    if (!A || !IA || !JA) {
        std::cerr << "PARDISO input arrays not allocated (A/IA/JA)\n";
        exit(3);
    }
    if (n <= 0) {
        std::cerr << "PARDISO matrix size n <= 0: " << (long)n << "\n";
        exit(3);
    }

    // Basic sanity checks on CSR arrays
    if (IA[0] != 1) {
        std::cerr << "IA[0] must be 1 for 1-based indexing; found " << (long)IA[0] << "\n";
        exit(3);
    }
    MKL_INT nnz = IA[n] - 1; // IA[n] = nnz + 1 for 1-based CSR
    if (nnz <= 0) {
        std::cerr << "IA[n] invalid, computed nnz <= 0: IA[n]=" << (long)IA[n] << "\n";
        exit(3);
    }
    // for (MKL_INT ii = 0; ii < n; ++ii) {
    //     if (IA[ii] > IA[ii+1]) {
    //         std::cerr << "IA not monotonic at row " << (long)ii << " IA[ii]=" << (long)IA[ii] << " IA[ii+1]=" << (long)IA[ii+1] << "\n";
    //         exit(3);
    //     }
    // }
    // for (MKL_INT k = 0; k < nnz; ++k) {
    //     if (JA[k] < 1 || JA[k] > n) {
    //         std::cerr << "JA[" << (long)k << "] out of bounds: " << (long)JA[k]
    //                   << " valid range 1.." << (long)n << "\n";
    //         exit(3);
    //     }
    // }

    // Use NULL permutation (most common usage)
    MKL_INT *perm = nullptr;

    // Make sure iparm is set; you already call pardisoinit somewhere else
    // but ensure iparm[2] sets threads, etc. mkl_set_num_threads(iparm[2]); if desired.

    //std::cout << "Starting PARDISO Solver for iteration " << iteration << std::endl;

    if (iteration == 0) {
        phase = 11;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase,
                &n, A, IA, JA, perm, &nrhs, iparm, &msglvl, rho, phi, &error);

        //std::cout << "Phase 11 completed, error=" << (long)error << "\n";
        if (error != 0) {
            std::cerr << "PARDISO phase 11 error: " << (long)error << "\n";
            exit(6);
        }

        phase = 22;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase,
                &n, A, IA, JA, perm, &nrhs, iparm, &msglvl, rho, phi, &error);

        //std::cout << "Phase 22 completed, error=" << (long)error << "\n";
        if (error != 0) {
            std::cerr << "PARDISO phase 22 error: " << (long)error << "\n";
            exit(6);
        }
    }

    phase = 33;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &n, A, IA, JA, perm, &nrhs, iparm, &msglvl, rho, phi, &error);

    //std::cout << "Phase 33 completed, error=" << (long)error << "\n";
    if (error != 0) {
        std::cerr << "PARDISO phase 33 error: " << (long)error << "\n";
        exit(6);
    }

    //std::cout << "PARDISO Solver finished for iteration " << iteration << std::endl;
    overwritePhi(geometries, phi);
    if (periodicBoundary)
        calculateElectricFieldPeriodic(phi, electricField);
    else
        calculateElectricField(phi, electricField);
    //std::cout << "Electric Field calculated for iteration " << iteration << std::endl;
}

//  #include "mddft3d.cu"		## File to include is defined on command line with --pre-include
#include <stdio.h>
#include <cufft.h>
#include <helper_cuda.h>

//  Size will be defined when compiling 
//  #define M		100
//  #define N		224
//  #define K		224
//  #define FUNCNAME		mddft3d

static void buildInputBuffer(double *host_X, double *X)
{
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				host_X[(k + n*K + m*N*K)*2 + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
				host_X[(k + n*K + m*N*K)*2 + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
			}
		}
	}

	cudaMemcpy(X, host_X, M*N*K*2*sizeof(double), cudaMemcpyHostToDevice);
	return;
}

int main() {

//	int M, N, K;
//	M=80;
//	N=80;
//	K=80;

	double *X, *Y;

	cudaMalloc(&X,M*N*K*2*sizeof(double));
	cudaMalloc(&Y,M*N*K*2*sizeof(double));

	double *host_X = new double[M*N*K*2];

	cudaEvent_t start, stop, custart, custop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);

//	cudaEventRecord(start);
//	FUNCNAME(Y, X);
//	cudaEventRecord(stop);

	#ifndef N_ITERS
	#define N_ITERS 20
	#endif
	
//	checkCudaErrors ( cudaEventRecord(start) );
	int iters = N_ITERS;		// = 100;  // use smaller number due to overhead of initializing buffers
	float milliseconds[N_ITERS];
	float cumilliseconds[N_ITERS];

	cufftDoubleComplex *cufft_Y; 
	cudaMalloc(&cufft_Y, M*N*K * sizeof(cufftDoubleComplex));

	cufftHandle plan;
	if (cufftPlan3d(&plan, M, N, K,  CUFFT_Z2Z) != CUFFT_SUCCESS) {
		exit(-1);
	}

	for ( int ii = 0; ii < iters; ii++ ) {
		init_mddft3d();
		checkCudaErrors ( cudaGetLastError () );
 
		// set up data in input buffer
		buildInputBuffer(host_X, X);

		checkCudaErrors ( cudaEventRecord(start) );
		FUNCNAME(Y, X);
		checkCudaErrors ( cudaGetLastError () );
		checkCudaErrors( cudaEventRecord(stop) );
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds[ii], start, stop);
		destroy_mddft3d();
		checkCudaErrors ( cudaGetLastError () );

		cudaEventRecord(custart);
		if (cufftExecZ2Z(
				plan,
				(cufftDoubleComplex *) X,
				(cufftDoubleComplex *) cufft_Y,
				CUFFT_INVERSE
				) != CUFFT_SUCCESS) {
			printf("cufftExecZ2Z launch failed\n");
			exit(-1);
		}
		cudaEventRecord(custop);
		cudaEventSynchronize(custop);

		cudaEventElapsedTime(&cumilliseconds[ii], custart, custop);

	}

	cudaDeviceSynchronize();

	if (cudaGetLastError() != cudaSuccess) {
		printf("cufftExecZ2Z failed\n");
		exit(-1);
	}

	printf("cube = [ %d, %d, %d ]\t\t ##PICKME## \n", M, N, K);
	for ( int ii = 0; ii < iters; ii++ ) { 
		printf("%f\tms (SPIRAL) vs\t%f\tms (cufft), iterations [%d] ##PICKME## \n",
		   milliseconds[ii], cumilliseconds[ii], ii);
	} 
 
	cufftDoubleComplex *host_Y       = new cufftDoubleComplex[M*N*K];
	cufftDoubleComplex *host_cufft_Y = new cufftDoubleComplex[M*N*K];

	cudaMemcpy(host_Y      ,       Y, M*N*K*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_cufft_Y, cufft_Y, M*N*K*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	bool correct = true;
	int errCount = 0;

	for (int m = 0; m < 1; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				cufftDoubleComplex s = host_Y      [k + n*K + m*N*K];
				cufftDoubleComplex c = host_cufft_Y[k + n*K + m*N*K];
	    
				bool elem_correct =
					(abs(s.x - c.x) < 1e-7) &&
					(abs(s.y - c.y) < 1e-7);
				correct &= elem_correct;
				if (!elem_correct && errCount < 10) 
				{
					correct = false;
					errCount++;
					//  printf("error at (%d,%d,%d): %f+%fi instead of %f+%fi\n", k, n, m, s.x, s.y, c.x, c.y);
				}
			}
		}
	}

end_check:
	printf("Correct: %s\t\t##PICKME## \n", (correct ? "True" : "False") );
}

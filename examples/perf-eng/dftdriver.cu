//  #include "mddft3d.cu"		## File to include is defined on command line with --pre-include
#include <stdio.h>
#include <cufft.h>
#include <helper_cuda.h>

//  Size will be defined when compiling 
//  #define M		100
//  #define N		224
//  #define K		224
//  #define FUNCNAME		mddft3d

int main() {

//	int M, N, K;
//	M=80;
//	N=80;
//	K=80;

	double *X, *Y;

	cudaMalloc(&X,M*N*K*2*sizeof(double));
	cudaMalloc(&Y,M*N*K*2*sizeof(double));

	double *host_X = new double[M*N*K*2];
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				host_X[(k + n*K + m*N*K)*2 + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
				host_X[(k + n*K + m*N*K)*2 + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
			}
		}
	}

	cudaMemcpy(X, host_X, M*N*K*2*sizeof(double), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop, custart, custop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);

//	cudaEventRecord(start);
//	FUNCNAME(Y, X);
//	cudaEventRecord(stop);

	checkCudaErrors ( cudaEventRecord(start) );
	int iters = 100;
	for ( int ii = 0; ii < iters; ii++ ) {
		FUNCNAME(Y, X);
		checkCudaErrors ( cudaGetLastError () );
	}
	checkCudaErrors( cudaEventRecord(stop) );
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cufftDoubleComplex *cufft_Y; 
	cudaMalloc(&cufft_Y, M*N*K * sizeof(cufftDoubleComplex));

	cufftHandle plan;
	if (cufftPlan3d(&plan, M, N, K,  CUFFT_Z2Z) != CUFFT_SUCCESS) {
		exit(-1);
	}
 
	cudaEventRecord(custart);
	for ( int ii = 0; ii < iters; ii++ ) {
		if (cufftExecZ2Z(
				plan,
				(cufftDoubleComplex *) X,
				(cufftDoubleComplex *) cufft_Y,
				CUFFT_INVERSE
				) != CUFFT_SUCCESS) {
			printf("cufftExecZ2Z launch failed\n");
			exit(-1);
		}
	}
	cudaEventRecord(custop);
	cudaEventSynchronize(custop);

	float cumilliseconds = 0;
	cudaEventElapsedTime(&cumilliseconds, custart, custop);
 
	cudaDeviceSynchronize();


	printf("%f\tms (SPIRAL) vs\t%f\tms (cufft), averaged over %d iterations ##PICKME## \n",
		   milliseconds / iters, cumilliseconds / iters, iters);
 
	if (cudaGetLastError() != cudaSuccess) {
		printf("cufftExecZ2Z failed\n");
		exit(-1);
	}
 
	cufftDoubleComplex *host_Y       = new cufftDoubleComplex[M*N*K];
	cufftDoubleComplex *host_cufft_Y = new cufftDoubleComplex[M*N*K];

	cudaMemcpy(host_Y      ,       Y, M*N*K*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_cufft_Y, cufft_Y, M*N*K*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	bool correct = true;

	for (int m = 0; m < 1; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				cufftDoubleComplex s = host_Y      [k + n*K + m*N*K];
				cufftDoubleComplex c = host_cufft_Y[k + n*K + m*N*K];
	    
				bool elem_correct =
					(abs(s.x - c.x) < 1e-7) &&
					(abs(s.y - c.y) < 1e-7);
				correct &= elem_correct;
				if (!elem_correct) 
				{
					correct = false;
					// printf("error at (%d,%d,%d): %f+%fi instead of %f+%fi\n", k, n, m, s.x, s.y, c.x, c.y);
				}
			}
		}
	}
end_check:
	printf("Correct: %s\t\t##PICKME## \n", (correct ? "True" : "False") );
}

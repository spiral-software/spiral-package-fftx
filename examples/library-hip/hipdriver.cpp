#include "mddft3d.cu"			// will be defined on command line with --pre-include
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hipfft.h>
#include "rocfft.h"
#include <stdlib.h>
#include <string.h>

#ifndef M
#define M 32
#define N 32
#define K 32
#endif

//  extern void mddft3d(double *Y, double *X);

//  generate file name

static char * generateFileName ( const char *type )
{
	// type is: input ==> random input data; output ==> spiral output data; roc ==> rocFFT output data
	static char fileNameBuff[100];
	sprintf ( fileNameBuff, "mddft3d-%s-%dx%dx%d.dat", type, M, N, K );
	return fileNameBuff;
}

//  write data to file(s) for test repeatability.

static void writeBufferToFile ( const char *type, double *datap )
{
	char *fname = generateFileName ( type );
	FILE *fhandle = fopen ( fname, "w" );
	fprintf ( fhandle, "[ \n" );
	for ( int mm = 0; mm < M; mm++ ) {
		for ( int nn = 0; nn < N; nn++ ) {
			for ( int kk = 0; kk < K; kk++ ) {
				fprintf ( fhandle, "FloatString(\"%.12g\"), FloatString(\"%.12g\"), ", 
						  datap[(kk + nn*K + mm*N*K)*2 + 0], datap[(kk + nn*K + mm*N*K)*2 + 1] );
				if ( kk > 0 && kk % 8 == 0 )
					fprintf ( fhandle, "\n" );
			}
			fprintf ( fhandle, "\n" );
		}
	}
	fprintf ( fhandle, "];\n" );
	
	//  fwrite ( datap, sizeof(double) * 2, M * N * K, fhandle );
	fclose ( fhandle );
	return;
}

static void buildInputBuffer ( double *host_X, double *X, int genData )
{
	if ( genData ) {					// generate a new data input buffer
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				for (int k = 0; k < K; k++) {
					host_X[(k + n*K + m*N*K)*2 + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
					host_X[(k + n*K + m*N*K)*2 + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
				}
			}
		}
	}

	hipMemcpy(X, host_X, M*N*K*2*sizeof(double), hipMemcpyHostToDevice);
	return;
}

	
int main( int argc, char** argv) {

	double *X, *Y;
	int writefiles = 0;

	hipMalloc(&X,M*N*K*2*sizeof(double));
	hipMalloc(&Y,M*N*K*2*sizeof(double));

	double *host_X = new double[M*N*K*2];

	printf ( "Usage: %s: [ writefiles ]\n", argv[0] );
	if ( argc > 1 ) {
		// if an argument is specified on the command line write all
		// data to files -- spiral input data, spiral output data, and
		// rocFFT output [input to spiral & roc is of course the same]
		printf ( "%s: argc = %d, will write in/out/roc files\n", argv[0], argc );
		writefiles = 1;
	}
	
	//  setup the library
	rocfft_setup();

	hipEvent_t start, stop, custart, custop;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	hipEventCreate(&custart);
	hipEventCreate(&custop);

	//  want to run and time: 1st iteration; 2nd iteration; then N iterations
	//  Report 1st time, 2nd time, and average of N further iterations
	//  Will use different random data for all iterations

	#ifndef NUM_ITERS
	#define NUM_ITERS 100
	#endif

	int iters = NUM_ITERS + 10;
	float milliseconds[iters];
	float cumilliseconds[iters];

	hipfftDoubleComplex *hipfft_Y;
	hipMalloc(&hipfft_Y, M*N*K * sizeof(hipfftDoubleComplex));

	hipfftHandle plan;
	if (hipfftPlan3d(&plan, M, N, K,  HIPFFT_Z2Z) != HIPFFT_SUCCESS) {
		exit(-1);
	}

	hipError_t  lastErr;		// last error

	// set up data in input buffer
	buildInputBuffer(host_X, X, 1);
	if ( writefiles ) {
		printf ( "Write input buffer to a file..." );
		writeBufferToFile ( (const char *)"input", host_X );
		printf ( "done\n" );
	}

	for ( int ii = 0; ii < iters; ii++ ) {
		init_mddft3d();
		lastErr = hipGetLastError();
		if (lastErr != hipSuccess) {
			printf("Error occured during transform execution: %s\n", hipGetErrorString(lastErr) );
			exit(-1);
		}

		hipEventRecord(start);
		FUNCNAME(Y, X);
		hipEventRecord(stop);

		lastErr = hipGetLastError();
		if (lastErr != hipSuccess) {
			printf("Error occured during transform execution: %s\n", hipGetErrorString(lastErr) );
			exit(-1);
		}
		
		hipEventSynchronize(stop);
		hipEventElapsedTime(&milliseconds[ii], start, stop);

		destroy_mddft3d();
		lastErr = hipGetLastError();
		if (lastErr != hipSuccess) {
			printf("Error occured during transform execution: %s\n", hipGetErrorString(lastErr) );
			exit(-1);
		}

		hipEventRecord(custart);
		if (hipfftExecZ2Z(
				plan,
				(hipfftDoubleComplex *) X,
				(hipfftDoubleComplex *) hipfft_Y,
				HIPFFT_BACKWARD
				) != HIPFFT_SUCCESS) {
			printf("hipfftExecZ2Z launch failed\n");
			exit(-1);
		}
		hipEventRecord(custop);
		hipEventSynchronize(custop);
		hipEventElapsedTime(&cumilliseconds[ii], custart, custop);

		if (hipGetLastError() != hipSuccess) {
			printf("hipfftExecZ2Z failed\n");
			exit(-1);
		}

#ifdef USE_DIFF_DATA
		buildInputBuffer(host_X, X, 1);
#else
		buildInputBuffer(host_X, X, 0);
#endif
	}
	hipDeviceSynchronize();

	printf("cube = [ %d, %d, %d ]\t\t ##PICKME## \n", M, N, K);
	printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft),\t\tFIRST iteration\t##PICKME## \n",
			   milliseconds[0], cumilliseconds[0]);
	printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft),\t\tSECOND iteration\t##PICKME## \n",
			   milliseconds[1], cumilliseconds[1]);

	float cumulSpiral = 0.0, cumulHip = 0.0;
	for ( int ii = 10; ii < iters; ii++ ) {
		cumulSpiral += milliseconds[ii];
		cumulHip    += cumilliseconds[ii];
	} 
	printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft), AVERAGE over %d iterations (range: 11 - %d) ##PICKME## \n",
		   cumulSpiral / NUM_ITERS, cumulHip / NUM_ITERS, NUM_ITERS, (10 + NUM_ITERS) );

	hipfftDoubleComplex *host_Y        = new hipfftDoubleComplex[M*N*K];
	hipfftDoubleComplex *host_hipfft_Y = new hipfftDoubleComplex[M*N*K];

	hipMemcpy(host_Y       ,        Y, M*N*K*sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
	hipMemcpy(host_hipfft_Y, hipfft_Y, M*N*K*sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
	if ( writefiles ) {
		writeBufferToFile ( (const char *)"spiral-out", (double *)host_Y );
		writeBufferToFile ( (const char *)"rocFFT",     (double *)host_hipfft_Y );
	}

	bool correct = true;
	int nerrors = 0;

	for (int m = 0; m < 1; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				hipfftDoubleComplex s = host_Y       [k + n*K + m*N*K];
				hipfftDoubleComplex c = host_hipfft_Y[k + n*K + m*N*K];

				bool elem_correct =
					(abs(s.x - c.x) < 1e-7) &&
					(abs(s.y - c.y) < 1e-7);

				correct &= elem_correct;
				if (!elem_correct)
				{
					correct = false;
					if ( nerrors < 10 )
						// printf("error at (%d,%d,%d): %f+%fi instead of %f+%fi\n", k, n, m, s.x, s.y, c.x, c.y);
						printf("error at (%d,%d,%d): %e+%ei instead of %e+%ei; delta: %e+%ei\n", k, n, m, s.x, s.y, c.x, c.y, abs(s.x - c.x), abs(s.y - c.y) );
					nerrors += 1;
				}
			}
		}
	}

	printf("Correct: %s\t\t##PICKME## \n", (correct ? "True" : "False") );

	//  cleanup the library
	rocfft_cleanup();
}

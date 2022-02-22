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

#define checkLastHipError(str)	{ hipError_t err = hipGetLastError();	if (err != hipSuccess) {  printf("%s: %s\n", (str), hipGetErrorString(err) );  exit(-1);	} }

static int writefiles = 0;


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
				fprintf ( fhandle, "FloatString(\"%g\"), FloatString(\"%g\"), ", 
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
	if ( genData ) {
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

static void mergeBuffers ( double *merged, double *inp1, double *inp2 )
{
	for ( int ii = 0; ii < 2 * M * N * K; ii++)
		merged[ii] = inp1[ii] + inp2[ii];
	
	return;
}

static void unitInputBuffer ( double *host_X, double *X, int mm, int nn )
{
	static int doOnce = 1;
	if ( doOnce ) {
		for ( int ii = 0; ii < 2 * M * N * K; ii++)
			host_X[ii] = 0.0;
		doOnce = 0;
	}
	if ( mm > 0 || nn > 0 ) {
		int mp = mm, np = nn - 1;
		if ( np < 0 ) {
			np = 0; mp--;
		}
		for (int kk = 0; kk < K; kk++) {
			host_X[(kk + np*K + mp*N*K)*2 + 0] = 0.0;   // - ((double) rand()) / (double) (RAND_MAX/2);
			host_X[(kk + np*K + mp*N*K)*2 + 1] = 0.0;  // - ((double) rand()) / (double) (RAND_MAX/2);
		}
	}

	for (int kk = 0; kk < K; kk++) {
		host_X[(kk + nn*K + mm*N*K)*2 + 0] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
		host_X[(kk + nn*K + mm*N*K)*2 + 1] = 1 - ((double) rand()) / (double) (RAND_MAX/2);
	}
	
	hipMemcpy(X, host_X, M*N*K*2*sizeof(double), hipMemcpyHostToDevice);
	return;
}

static double maxdelta = 0.0;

static bool compareBuffers ( hipfftDoubleComplex *b1, hipfftDoubleComplex *b2 )
{
	int nerrors = 0;
	bool correct = true;

	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				hipfftDoubleComplex s = b1[k + n*K + m*N*K];
				hipfftDoubleComplex c = b2[k + n*K + m*N*K];

				bool elem_correct =
					(abs(s.x - c.x) < 1e-7) &&
					(abs(s.y - c.y) < 1e-7);
				maxdelta = maxdelta < (double)(abs(s.x -c.x)) ? (double)(abs(s.x -c.x)) : maxdelta ;
				maxdelta = maxdelta < (double)(abs(s.y -c.y)) ? (double)(abs(s.y -c.y)) : maxdelta ;

				correct &= elem_correct;
				if (!elem_correct)
				{
					correct = false;
					if ( nerrors < 10 )
						//  printf("error at (%d,%d,%d): %f+%fi instead of %f+%fi\n", k, n, m, s.x, s.y, c.x, c.y);
						//  printf("error at (%d,%d,%d): %e+%ei instead of %e+%ei; delta: %e+%ei\n", k, n, m, s.x, s.y, c.x, c.y, abs(s.x - c.x), abs(s.y - c.y) );
					nerrors += 1;
				}
			}
		}
	}

	return correct;
}

static hipfftDoubleComplex *host_Y[3];
static hipfftDoubleComplex *host_hipfft_Y[3];

static void checkOutputBuffers ( double *Y, double *hipfft_Y, int pass, int choice )
{
	static int doOnce = 1;
	if ( doOnce ) {			// create buffers when requested
		host_Y[0]        = new hipfftDoubleComplex[M*N*K];
		host_hipfft_Y[0] = new hipfftDoubleComplex[M*N*K];
		host_Y[1]        = new hipfftDoubleComplex[M*N*K];
		host_hipfft_Y[1] = new hipfftDoubleComplex[M*N*K];
		host_Y[2]        = new hipfftDoubleComplex[M*N*K];
		host_hipfft_Y[2] = new hipfftDoubleComplex[M*N*K];
		doOnce = 0;
	}

	if (choice < 0 || choice > 2 ) {
		printf ( "checkOutputBuffers: choice (%d is invalid) must be between 0 and 2\n", choice );
		return;
	}

	hipfftDoubleComplex *spiralY = host_Y[choice], *rocoutY = host_hipfft_Y[choice];
	hipMemcpy( spiralY,        Y, M*N*K*sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
	hipMemcpy( rocoutY, hipfft_Y, M*N*K*sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
	//  if ( writefiles ) {
	//  	writeBufferToFile ( (const char *)"spiral-out", (double *)host_Y );
	//  	writeBufferToFile ( (const char *)"rocFFT",     (double *)host_hipfft_Y );
	//  }

	bool correct = compareBuffers( spiralY, rocoutY );
	printf("Correct: %s, pass = %d\t\t##PICKME## \n", (correct ? "True" : "False"), pass );
	if ( !correct && writefiles ) {
		writeBufferToFile ( (const char *)"orig-input", (double *)spiralY );
		writeBufferToFile ( (const char *)"copy-input", (double *)rocoutY );
		exit (-1);
	}

	if ( pass > 0 && pass % 100 == 0 )
		printf ( "Completed %d passes\n", pass );
	return;
}
	
int main( int argc, char** argv) {

	double *X1, *Y1,  *X2, *Y2, *X12, *Y12;

	hipMalloc(&X1,M*N*K*2*sizeof(double));
	hipMalloc(&Y1,M*N*K*2*sizeof(double));
	hipMalloc(&X2,M*N*K*2*sizeof(double));
	hipMalloc(&Y2,M*N*K*2*sizeof(double));
	hipMalloc(&X12,M*N*K*2*sizeof(double));
	hipMalloc(&Y12,M*N*K*2*sizeof(double));

	double *host_X1 = new double[M*N*K*2];
	double *host_X2 = new double[M*N*K*2];
	double *host_X12 = new double[M*N*K*2];

	//  printf ( "Usage: %s: [ writefiles ]\n", argv[0] );
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

	hipfftDoubleComplex *hipfft_Y1,  *hipfft_Y2,  *hipfft_Y12;
	hipMalloc(&hipfft_Y1, M*N*K * sizeof(hipfftDoubleComplex));
	hipMalloc(&hipfft_Y2, M*N*K * sizeof(hipfftDoubleComplex));
	hipMalloc(&hipfft_Y12, M*N*K * sizeof(hipfftDoubleComplex));

	hipfftHandle plan;
	if (hipfftPlan3d(&plan, M, N, K,  HIPFFT_Z2Z) != HIPFFT_SUCCESS) {
		exit(-1);
	}

	hipError_t  lastErr;		// last error

	// set up data in input buffer
	buildInputBuffer(host_X1, X1, iters);
	//  writeBufferToFile ( (const char *)"orig-input0", host_X1 );
	buildInputBuffer(host_X2, X2, iters);
	mergeBuffers ( host_X12, host_X1, host_X2 ); 
	hipMemcpy(X12, host_X12, M*N*K*2*sizeof(double), hipMemcpyHostToDevice);
	//  memcpy ( host_X2, host_X1, M*N*K*2*sizeof(double) );  // make copy of input buffer
	//  hipMemcpy(X2, host_X2, M*N*K*2*sizeof(double), hipMemcpyHostToDevice); // write copy to device
	//  checkOutputBuffers ( X1, X2, 0 );
	//  printf ( "before 1st kernel call\n" );
	//  writeBufferToFile ( (const char *)"input", host_X );
	//  unitInputBuffer(host_X, X, 0, 0);
	//  if ( writefiles ) {
	// 	printf ( "Write input buffer to a file..." );
	// 	writeBufferToFile ( (const char *)"input", host_X );
	// 	printf ( "done\n" );
	// }

	init_mddft3d();
	checkLastHipError ("Error occured during transform execution");
	
	for ( int ii = 0; ii < iters; ii++ ) {

	//  iters = 2 * M * N * K;
	//  for ( int ii = 0; ii < iters; ii++ ) {
	//  for ( int mm = 0; mm < M; mm++ ) {
	//  	for ( int nn = 0; nn < N; nn++ ) {
	//  		unitInputBuffer ( host_X, X, mm, nn );
			//  hipEventRecord(start);
		buildInputBuffer( host_X1, X1, 1 );				// copy input buffer to device
		/* if ( ii == 0 ) */
		FUNCNAME(Y1, X1);
		//  hipEventRecord(stop);
		checkLastHipError ("Error occured during transform execution");

		buildInputBuffer( host_X2, X2, 1 );				// copy input buffer to device
		FUNCNAME(Y2, X2);
		checkLastHipError ("Error occured during transform execution");
		
		mergeBuffers ( host_X12, host_X1, host_X2 );	// merge X1 and X2
		buildInputBuffer( host_X12, X12, 0 );			// copy input buffer to device (don't change it)
		FUNCNAME(Y12, X12);
		checkLastHipError ("Error occured during transform execution");

		//  hipEventSynchronize(stop);
		//  hipEventElapsedTime(&milliseconds[ii], start, stop);

		//  destroy_mddft3d();
		//  checkLastHipError ("Error occured during transform execution");

		//  hipEventRecord(custart);
		//	if ( ii == (iters - 1) ) {
			if ( hipfftExecZ2Z ( plan, (hipfftDoubleComplex *) X1,
								 (hipfftDoubleComplex *) hipfft_Y1,
								 HIPFFT_BACKWARD	) != HIPFFT_SUCCESS) {
				printf("hipfftExecZ2Z launch failed\n");
				exit(-1);
			}
			if ( hipfftExecZ2Z ( plan, (hipfftDoubleComplex *) X2,
								 (hipfftDoubleComplex *) hipfft_Y2,
								 HIPFFT_BACKWARD	) != HIPFFT_SUCCESS) {
				printf("hipfftExecZ2Z launch failed\n");
				exit(-1);
			}
			if ( hipfftExecZ2Z ( plan, (hipfftDoubleComplex *) X12,
								 (hipfftDoubleComplex *) hipfft_Y12,
								 HIPFFT_BACKWARD	) != HIPFFT_SUCCESS) {
				printf("hipfftExecZ2Z launch failed\n");
				exit(-1);
			}
				//	}
		//  hipEventRecord(custop);
		//  hipEventSynchronize(custop);
		//  hipEventElapsedTime(&cumilliseconds[ii], custart, custop);

			checkLastHipError ("hipfftExecZ2Z failed");
			
#ifdef USE_DIFF_DATA
			buildInputBuffer(host_X, X);
#endif
			hipDeviceSynchronize();
			//  checkOutputBuffers ( X1, X2, ii );		// check input & copy are the same
			
			checkOutputBuffers ( Y1, (double *)hipfft_Y1, /* mm * M + nn */ ii, 0 );
			checkOutputBuffers ( Y2, (double *)hipfft_Y2, /* mm * M + nn */ ii, 1 );
			checkOutputBuffers ( Y12, (double *)hipfft_Y12, /* mm * M + nn */ ii, 2 );
			// merge output buffers 1 & 2
			mergeBuffers ( (double *)host_Y[2], (double *)host_Y[0], (double *)host_Y[1] );
			if ( !compareBuffers ( host_Y[2], host_hipfft_Y[2] ) ) {
				// transform (X1) + transform (X2) != rocFFT (X1 + X2) -- linearity failed
				printf ( "Linearity test failed: tr(X1) + tr(X2) != rocFFT(X1 + X2)\n" );
			}
			checkOutputBuffers ( Y12, (double *)hipfft_Y12, ii, 2 ); // pull results from gpu
			// merge output buffers 1 & 2
			mergeBuffers ( (double *)host_hipfft_Y[2], (double *)host_hipfft_Y[0], (double *)host_hipfft_Y[1] );
			if ( !compareBuffers ( host_Y[2], host_hipfft_Y[2] ) ) {
				// transform (X1 + x2) != rocFFT (X1) + rocFFT(X2) -- linearity failed
				printf ( "Linearity test failed: tr(X1 + X2) != rocFFT(X1) + rocFFT(X2)\n" );
			}
	//  	}
	}

	printf("cube = [ %d, %d, %d ]\tMax delta (over all iterations) = %E\t\t##PICKME## \n", M, N, K, maxdelta);
	// printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft),\t\tFIRST iteration\t##PICKME## \n",
	// 		   milliseconds[0], cumilliseconds[0]);
	// printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft),\t\tSECOND iteration\t##PICKME## \n",
	// 		   milliseconds[1], cumilliseconds[1]);

	float cumulSpiral = 0.0, cumulHip = 0.0;
	for ( int ii = 10; ii < iters; ii++ ) {
		cumulSpiral += milliseconds[ii];
		cumulHip    += cumilliseconds[ii];
	} 
	//  printf("%f\tms (SPIRAL) vs\t%f\tms (hipfft), AVERAGE over %d iterations (range: 11 - %d) ##PICKME## \n",
	//  	   cumulSpiral / NUM_ITERS, cumulHip / NUM_ITERS, NUM_ITERS, (10 + NUM_ITERS) );

	//  cleanup the library
	rocfft_cleanup();
}

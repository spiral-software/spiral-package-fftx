
//  Copyright (c) 2018-2021, Carnegie Mellon University
//  See LICENSE for details

#include <iostream>
#include <fstream>
#include <cmath>

#ifdef WIN64
#include <winsock.h>			// defines struct timeval -- go figure?
#include <sys/timeb.h>
#endif							// WIN64

#include <cufft.h>
#include <cufftXt.h>

#include <helper_cuda.h>

#include "testsizes.h"			// define sizes

//  2 macros needed to get the proper expansion -- one of the arguments to GEN_FUNC_NAME
//  will be a macro with the function name (extracted from the spiral code).  If the
//  concatenation occurs at that stage that macro will not be expanded properly

#define GEN_FUNC_NAME2(root, stem)   (root ## stem)
#define GEN_FUNC_NAME(root, stem)  GEN_FUNC_NAME2(root, stem)

#include CODEFILE

static int cpu_code_run = 0;

#include "WarpXConst.H"

#ifdef RUN_CPU_CODE
#include PSATDCODE
#endif							// RUN_CPU_CODE

#define DIM 3

size_t product3(size_t* vec)
{
  return (vec[0] * vec[1] * vec[2]);
}

void readComponent(std::ifstream& is,
                   double*& data,
                   size_t len)
{
  data = new double[len];
  size_t bytes = len * sizeof(double);
  is.read((char*) data, bytes);
}

void readAll(const std::string& str,
             double**& dataPtr,
             int nfields,
             const size_t* lengths,
			 int validate)
{
	dataPtr = new double*[nfields];
	if (validate) {
		// read from files when validate is true
		std::ifstream is(str, std::ios::binary);
		if (!is.is_open()) {
			std::cout << "Error: missing file " << str << std::endl;
		}
		for (int comp = 0; comp < nfields; comp++) {
			readComponent(is, dataPtr[comp], lengths[comp]);
		}
		is.close();
	}
	else {
		for (int comp = 0; comp < nfields; comp++) {
			dataPtr[comp] = new double[lengths[comp]];
		}
	}
}

void deleteAll(double**& dataPtr, int nfields)
{
  for (int comp = 0; comp < nfields; comp++)
    {
      delete[] dataPtr[comp];
    }
  delete[] dataPtr;
}

static void FreeCudaMemory ( double **dev_ptr, double **host_ptr, int nf)
{
	// clean up all cuda memory allocated
	cudaFree ( dev_ptr );
	for ( int ix = 0; ix < nf; ix++)
		cudaFree ( host_ptr[ix] );

	cudaFreeHost ( host_ptr );
}

static void check_buf_values(char *step, double *buf, int len)
{
	double max1 = 0.0, min1 = 1e56;
	int    nz1 = 0, zeros1 = 0;
	
	for (int ii = 0; ii < len; ii++) {
		if (buf[ii] > max1 ) max1 = buf[ii];
		if (buf[ii] < min1 ) min1 = buf[ii];
		(buf[ii] == 0.0 ) ? zeros1++ : nz1++ ;
	}

	printf("%s: max = %g, min = %g, # non-zero = %d, # zero entries = %d\n",
		   step, max1, min1, nz1, zeros1);
}

int main(int argc, char* argv[])
{
	//  CUDA WaprX harness provides the ability to test/time CUDA code gereated for
	//  different sizes of the WarpX code.  Inputs and symbol are all zero;
	//  outputs are not checked.  This harness is derived from the initial
	//  cudaspiral code that validated the original problem size (with an 80^3)

#define VALIDATE 0				// don't validate; 1 ==> validate values with file data 

	int nbase = 64;
	int ng = 8;
	//  int n = nbase + 2*ng;
	int n = cubeN;
	//  int np = n + 1;
	int np = cubeNP;
	//  int nf = (n + 2)/2;
	int nf = cubeNF;

	size_t nodesAll[DIM] = {np, np, np};
	size_t nodesR2C[DIM] = {nf, n, n};
	size_t edges[DIM][DIM] = {{n, np, np},
							  {np, n, np},
							  {np, np, n}};
	size_t faces[DIM][DIM] = {{np, n, n},
							  {n, np, n},
							  {n, n, np}};

#ifdef WIN64
	struct timeb     start, finish, startcpu, finishcpu;
    ftime(&start);
#else
	struct timespec  start, finish, startcpu, finishcpu;
	clock_gettime(CLOCK_MONOTONIC, &start);
#endif					 // WIN64

	int iters = 100;
  
	const int inFields = INFIELDS;			//  3*DIM + 2;
	size_t inputLength[inFields] = {
		product3(edges[0]), product3(edges[1]), product3(edges[2]), // E
		product3(faces[0]), product3(faces[1]), product3(faces[2]), // B
		product3(edges[0]), product3(edges[1]), product3(edges[2]), // J
		product3(nodesAll), product3(nodesAll) // rho_old and rho_new
	};
	double** inputPtr; //  = new double*[inFields];
	readAll("input.bin", inputPtr, inFields, inputLength, VALIDATE);

	// spiral generated cuda code assumes in/out/symbol will be in device memory
  
	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	cufftDoubleReal **cudain, **hostin;
	cudaMalloc     ( &cudain, sizeof(cufftDoubleReal) * inFields );
	cudaMallocHost ( &hostin, sizeof(cufftDoubleReal) * inFields );
	for (int comp = 0; comp < inFields; comp++) {
		cudaMalloc ( &hostin[comp], sizeof(cufftDoubleReal) * inputLength[comp] );
		cudaMemcpy ( hostin[comp], inputPtr[comp], sizeof(cufftDoubleReal) * inputLength[comp],
					 cudaMemcpyHostToDevice );
	}
	cudaMemcpy ( cudain, hostin, sizeof(cufftDoubleReal) * inFields, cudaMemcpyHostToDevice );
	checkCudaErrors(cudaGetLastError());

	const int symFields = DIM + 5;
	size_t n3DR2C = product3(nodesR2C);
	size_t symLength[symFields] =
		{ nf, n, n, n3DR2C, n3DR2C, n3DR2C, n3DR2C, n3DR2C };
	double** symPtr; // = new double*[symFields];
	readAll("sym.bin", symPtr, symFields, symLength, VALIDATE);
  
	cufftDoubleReal **cudasym, **hostsym;
	cudaMalloc     ( &cudasym, sizeof(cufftDoubleReal) * symFields );
	cudaMallocHost ( &hostsym, sizeof(cufftDoubleReal) * symFields );
	for (int comp = 0; comp < symFields; comp++) {
		cudaMalloc ( &hostsym[comp], sizeof(cufftDoubleReal) * symLength[comp] );
		cudaMemcpy ( hostsym[comp], symPtr[comp], sizeof(cufftDoubleReal) * symLength[comp],
					 cudaMemcpyHostToDevice );
	}
	cudaMemcpy ( cudasym, hostsym, sizeof(cufftDoubleReal) * symFields, cudaMemcpyHostToDevice );
	checkCudaErrors(cudaGetLastError());

	const int outFields = OUTFIELDS;			//  2*DIM;
	size_t** outputDims = new size_t*[outFields];
	size_t* outputLength = new size_t[outFields];
	for (int idir = 0; idir < DIM; idir++) {
		outputDims[idir] = edges[idir]; // E
		outputDims[idir + DIM] = faces[idir]; // B
		outputLength[idir] = product3(outputDims[idir]);
		outputLength[idir + DIM] = product3(outputDims[idir + DIM]);
    }
	double** outputWarpXPtr; // = new double*[outFields];
	readAll("output.bin", outputWarpXPtr, outFields, outputLength, VALIDATE);

	double** outputSpiralPtr = new double*[outFields];
	for (int comp = 0; comp < outFields; comp++) {
		outputSpiralPtr[comp] = new double[outputLength[comp]];
    }

	cufftDoubleReal **cudaout, **hostout;
	cudaMalloc     ( &cudaout, sizeof(cufftDoubleReal) * outFields );
	cudaMallocHost ( &hostout, sizeof(cufftDoubleReal) * outFields );
	for (int comp = 0; comp < outFields; comp++) {
		cudaMalloc ( &hostout[comp], sizeof(cufftDoubleReal) * outputLength[comp] );
	}
	cudaMemcpy ( cudaout, hostout, sizeof(cufftDoubleReal) * outFields, cudaMemcpyHostToDevice );
	checkCudaErrors(cudaGetLastError());

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; device++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d.\n",
			   device, deviceProp.major, deviceProp.minor);
	}

	/* size_t heap; */
	/* cudaDeviceGetLimit ( &heap, cudaLimitMallocHeapSize); */
	/* // printf("Current heap size = %p (%lu); size_t length = %d\n", heap, heap, sizeof(heap)); */

	/* heap = 1024 * 1024 * 1024; */
	/* cudaDeviceSetLimit ( cudaLimitMallocHeapSize, heap ); */
	/* cudaDeviceGetLimit ( &heap, cudaLimitMallocHeapSize); */
	/* printf("Heap size set = %p (%llu)\n", heap, heap); */

#ifdef WIN64
	ftime(&startcpu);
#else
	clock_gettime(CLOCK_MONOTONIC, &startcpu);
#endif					 // WIN64

	double elapsed;
#ifdef RUN_CPU_CODE
	cpu_code_run = 1;
	GEN_FUNC_NAME(init_, FUNCNAME)();
	for (int ixx = 0; ixx < iters; ixx++)
		FUNCNAME(outputSpiralPtr, inputPtr, symPtr);		// , intbuf1, intbuf2

	GEN_FUNC_NAME(destroy_, FUNCNAME)();
	
#ifdef WIN64
	ftime(&finishcpu);
	elapsed = (1000.0 * (finishcpu.time - startcpu.time)) + (finishcpu.millitm - startcpu.millitm);
	printf("%f;\t\t## CPU based SPIRAL code exec time [ms] / iteration, over %d iterations\n",
		   elapsed / iters, iters);
#else
	clock_gettime(CLOCK_MONOTONIC, &finishcpu);
	elapsed = ( ( (double)finishcpu.tv_sec * 1e9 + (double)finishcpu.tv_nsec) -
					   ( (double)startcpu.tv_sec * 1e9 + (double)startcpu.tv_nsec ) );
	printf("%f;\t\t## CPU based SPIRAL code exec time [ms] / iteration, over %d iterations\n",
		   elapsed * 1e-6 / iters, iters );
#endif					// WIN64
#endif					// RUN_CPU_CODE

	cudaEvent_t      begin, end;
	cudaEventCreate ( &begin );
	cudaEventCreate ( &end );

#ifdef WIN64
	ftime(&startcpu);
#else
	clock_gettime(CLOCK_MONOTONIC, &startcpu);
#endif					 // WIN64

	checkCudaErrors( cudaEventRecord(begin) );
	for (int izz = 0; izz < iters; izz++) {
		GEN_FUNC_NAME(init_, CUDAFUNCNAME)();

		CUDAFUNCNAME(cudaout, cudain, cudasym);				//  , cudabufout
		checkCudaErrors(cudaGetLastError());
		cudaDeviceSynchronize();

		GEN_FUNC_NAME(destroy_, CUDAFUNCNAME)();
		// cudaDeviceReset();
	}

	checkCudaErrors( cudaEventRecord(end) );
	cudaDeviceSynchronize();

	float milli = 0.0;
	checkCudaErrors ( cudaEventElapsedTime ( &milli, begin, end ) );
	printf("%f;\t\t## GPU based SPIRAL kernel execution [ms] / iteration, over %d iterations\n",
		   milli / iters, iters );

#ifdef WIN64
	ftime(&finishcpu);
	elapsed = (1000.0 * (finishcpu.time - startcpu.time)) + (finishcpu.millitm - startcpu.millitm);
	//  printf("%f;\t\t## GPU based SPIRAL kernel alt measure [ms] / iteration, over %d iterations\n",
	//  	   elapsed / iters, iters);
#else
	clock_gettime(CLOCK_MONOTONIC, &finishcpu);
	elapsed = ( ( (double)finishcpu.tv_sec * 1e9 + (double)finishcpu.tv_nsec) -
				( (double)startcpu.tv_sec * 1e9 + (double)startcpu.tv_nsec ) );
	//  printf("%f;\t\t## GPU based SPIRAL kernel alt measure [ms] / iteration, over %d iterations\n",
	//  	   elapsed * 1e-6 / iters, iters );
#endif					 // WIN64

	// copy the output data from device memory to host
	cudaMemcpy ( hostout, cudaout, sizeof(cufftDoubleReal) * outFields, cudaMemcpyDeviceToHost );
	for (int comp = 0; comp < outFields; comp++) {
		cudaMemcpy ( outputSpiralPtr[comp], hostout[comp],
					 sizeof(cufftDoubleReal) * outputLength[comp], cudaMemcpyDeviceToHost );
		checkCudaErrors(cudaGetLastError());
	}

	//  We only need to do data validation when we are using files with values
	//  for input & symbol and values foir the expected outputs

	if (VALIDATE) {
		std::string names[6] = { "Ex", "Ey", "Ez", "Bx", "By", "Bz" };
		double E2max = 0.;
		double diffE2max = 0.;
		double B2max = 0.;
		double diffB2max = 0.;
		for (int comp = 0; comp < outFields; comp++) {
			std::string name = names[comp];
			// int idir = comp % DIM;
			size_t* dims = outputDims[comp];
			double* WarpX = outputWarpXPtr[comp];
			double* Spiral = outputSpiralPtr[comp];
			double maxWarpX = 0.;
			double maxdiff = 0.;
			size_t pt = 0;
			for (int k = 0; k < dims[2]; k++)
				for (int j = 0; j < dims[1]; j++)
					for (int i = 0; i < dims[0]; i++) {
						// We need to compare the valid parts of the output only.
						if ((k >= ng) && (k < dims[2]-ng) &&
							(j >= ng) && (j < dims[1]-ng) &&
							(i >= ng) && (i < dims[0]-ng)) {
							double absWarpX = std::abs(WarpX[pt]);
							if (absWarpX > maxWarpX) {
								maxWarpX = absWarpX;
							}
							double diff = std::abs(WarpX[pt] - Spiral[pt]);
							if (diff > maxdiff)	{
								maxdiff = diff;
							}
						}
						pt++;
					}
			std::cout << "|diff(" << name << ")| <= " << maxdiff
					  << " of |" << name << "| <= " << maxWarpX
					  << " relative " << (maxdiff/maxWarpX)
					  << std::endl;
			if (comp < DIM) {
				E2max += maxWarpX * maxWarpX;
				diffE2max += maxdiff * maxdiff;
			}
			else {
				B2max += maxWarpX * maxWarpX;
				diffB2max += maxdiff * maxdiff;
			}
		}

		double diffEmax = std::sqrt(diffE2max);
		double diffBmax = std::sqrt(diffB2max);
		double Emax = std::sqrt(E2max);
		double Bmax = std::sqrt(B2max);
		double const c2 = PhysConst::c * PhysConst::c;
		double compositeEmax = std::sqrt(E2max + c2*B2max);
		double compositeBmax = std::sqrt(E2max/c2 + B2max);

		std::cout << "||diff(E)|| <= " << diffEmax
				  << " of ||E|| <= " << Emax
				  << " relative " << (diffEmax / Emax)
				  << std::endl;
		std::cout << "||diff(E)|| <= " << diffEmax
				  << " of sqrt(||E||^2 + c^2*||B||^2) <= " << compositeEmax
				  << " relative " << (diffEmax / compositeEmax)
				  << std::endl;

		std::cout << "||diff(B)|| <= " << diffBmax
				  << " of ||B|| <= " << Bmax
				  << " relative " << (diffBmax / Bmax)
				  << std::endl;
		std::cout << "||diff(B)|| <= " << diffBmax
				  << " of sqrt(||E||^2/c^2 + ||B||^2) <= " << compositeBmax
				  << " relative " << (diffBmax / compositeBmax)
				  << std::endl;
	}
  
	deleteAll(inputPtr, inFields);
	deleteAll(symPtr, symFields);
	deleteAll(outputWarpXPtr, outFields);
	deleteAll(outputSpiralPtr, outFields);

	// clean up all cuda memory allocated
	FreeCudaMemory ( cudain,  hostin,  inFields );
	FreeCudaMemory ( cudasym, hostsym, symFields );
	FreeCudaMemory ( cudaout, hostout, outFields );
  
#ifdef WIN64
	ftime(&finish);
	double totelapse = (1000.0 * (finish.time - start.time)) + (finish.millitm - start.millitm);
	printf("%f;\t\t## Total elapsed time for test [ms], (completed %d iterations of %s GPU code)\n",
		   totelapse, iters, (cpu_code_run ? "both CPU & " : "") );
#else
	clock_gettime(CLOCK_MONOTONIC, &finish);
	double totelapse = ( ( (double)finish.tv_sec * 1e9 + (double)finish.tv_nsec) -
						 ( (double)start.tv_sec  * 1e9 + (double)start.tv_nsec ) );
	printf("%f;\t\t## Total elapsed time for test [ms], (completed %d iterations of %s GPU code)\n",
		   totelapse * 1e-6, iters, (cpu_code_run ? "both CPU & " : "") );
#endif					 // WIN64

}

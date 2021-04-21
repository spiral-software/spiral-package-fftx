;; This buffer is for notes you don't want to save, and for Lisp evaluation.
;; If you want to create a file, visit that file with C-x C-f,
;; then enter the text in that file's own buffer.


void plan_2Dgrid_3D_stage1(int M, int N, int K, const Grid& g, cufftHandle* plan_stage_1)
  {
    int size_x = M;
    int size_y = N/g.Dimension(0);
    int size_z = K/g.Dimension(1);
    int sizes[1]   = {size_x};
    int inembed[1] = {size_x};
    int onembed[1] = {size_x};
    int idist      = size_x;       // distance to next dft
    int istride = 1;               // distance to next element;
    int odist   = 1;               // distance to next dft
    int ostride = size_y*size_z;   // distance to next element;
    //rotate on the way out
    if (
      cufftPlanMany(plan_stage_1,  1, sizes,
				       inembed, istride, idist,
				           onembed, ostride, odist,
					       CUFFT_Z2Z, size_z * size_y)
        != CUFFT_SUCCESS) {
          fprintf(stderr, "CUFFT error: Stage 1 3D Plan creation failed");
          exit(-1);
    }
    
  }

void plan_2Dgrid_3D_stage2(int M, int N, int K, const Grid& g, cufftHandle* plan_stage_2)
{
    int size_x = M/g.Dimension(0);
    int size_y = N;
    int size_z = K/g.Dimension(1);
    int sizes[1]   = {size_y};
    int inembed[1] = {size_y};
    int onembed[1] = {size_y};
    int idist      = size_y;       // distance to next dft
    int istride = 1;               // distance to next element;
    int odist   = 1;               // distance to next dft
    int ostride = size_x*size_z;   // distance to next element;
    //rotate on the way out
    if (
      cufftPlanMany(plan_stage_2,  1, sizes,
        inembed, istride, idist,
        onembed, ostride, odist,
        CUFFT_Z2Z, size_x * size_z)
        != CUFFT_SUCCESS) {
          fprintf(stderr, "CUFFT error: Stage 1 3D Plan creation failed");
          exit(-1);
        }    
}

void plan_2Dgrid_3D_stage3(int M, int N, int K, const Grid& g, cufftHandle* plan_stage_3)
  {
    int size_x = M/g.Dimension(0);
    int size_y = N/g.Dimension(1); 
    int size_z = K;
    int sizes[1]   = {size_z};
    int inembed[1] = {size_z};
    int onembed[1] = {size_z};
    int idist      = size_z;       // distance to next dft
    int istride = 1;               // distance to next element;
    int odist   = 1;               // distance to next dft
    int ostride = size_x*size_y;   // distance to next element;
    //rotate on the way out
    if (
      cufftPlanMany(plan_stage_3,  1, sizes,
        inembed, istride, idist,
        onembed, ostride, odist,
        CUFFT_Z2Z, size_x * size_y)
        != CUFFT_SUCCESS) {
          fprintf(stderr, "CUFFT error: Stage 1 3D Plan creation failed");
          exit(-1);
        }
  }

cudaError_t execute_cuFFT_plan(
			       cufftHandle& plan,
			       std::complex<double>* dev_in,
			       std::complex<double>* dev_out
){
  cudaError_t cudaStatus;

  cufftExecZ2Z(plan, (cufftDoubleComplex*) dev_in, (cufftDoubleComplex*) dev_out, CUFFT_FORWARD);


  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    return cudaStatus;
  }

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching stage 1!\n", cudaStatus);
    return cudaStatus;
  }

  return cudaStatus;
}
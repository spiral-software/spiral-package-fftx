#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fftw3.h>
#include <complex.h>
#include <math.h>

#define nof_repetitions 1000

#define num_fft_sizes 21
static uint32_t fft_sizes[num_fft_sizes] = {
    // OFDM sizes
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,

    // PRACH sizes
    139,
    839,

    4608,  // 12x384
    6144,  // 12x512
    9216,  // 12x768
    12288, // 12x1024
    18432, // 12x1536
    24576, // 12x2048
    36864, // 12x3072
    49152, // 12x4096
    2304,  // 3x768

    // optional sizes: all M < 3300 such that M = 2^a * 2^b * 2^c where a,b,c are integers

};

static double elapsed_us(struct timeval* ts_start, struct timeval* ts_end)
{
  if (ts_end->tv_usec > ts_start->tv_usec) {
    return ((double)ts_end->tv_sec - (double)ts_start->tv_sec) * 1000000 + (double)ts_end->tv_usec -
        (double)ts_start->tv_usec;
  } else {
    return ((double)ts_end->tv_sec - (double)ts_start->tv_sec - 1) * 1000000 + ((double)ts_end->tv_usec + 1000000) -
        (double)ts_start->tv_usec;
  }
}

int main() {
  struct timeval start, end;

  for (uint32_t k = 0; k < num_fft_sizes; k++) {

    // allocate in/out buffers
    fftwf_complex* in  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_sizes[k]);
    fftwf_complex* tmp = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_sizes[k]);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_sizes[k]);

    // generate backwards and forward plans
    fftwf_plan p_bw = fftwf_plan_dft_1d(fft_sizes[k], in, tmp, FFTW_BACKWARD, FFTW_MEASURE);
    fftwf_plan p_fw = fftwf_plan_dft_1d(fft_sizes[k], tmp, out, FFTW_FORWARD, FFTW_MEASURE);

    // Calculate random complex input
    for (uint32_t i = 0; i < fft_sizes[k]; i++) {
      for (uint32_t j = 0; j < 2; j++) {
        in[i][j] = (float) rand()/INT32_MAX;
      }
    }

    printf("FFT size: %d ", fft_sizes[k]);

    // Execute Tx
    gettimeofday(&start, NULL);
    for (uint32_t i = 0; i < nof_repetitions; i++) {
      fftwf_execute(p_bw);
    }
    gettimeofday(&end, NULL);
    printf(" Tx@%.1fMsps", (float)(fft_sizes[k] * nof_repetitions) / elapsed_us(&start, &end));

    // Normalize
    for (uint32_t i = 0; i < fft_sizes[k]; i++) {
      for (uint32_t j = 0; j < 2; j++) {
        tmp[i][j] /= sqrtf(fft_sizes[k]);
      }
    }

    // Execute Rx
    gettimeofday(&start, NULL);
    for (uint32_t i = 0; i < nof_repetitions; i++) {
      fftwf_execute(p_fw);
    }
    gettimeofday(&end, NULL);
    printf(" Rx@%.1fMsps", (double)(fft_sizes[k] * nof_repetitions) / elapsed_us(&start, &end));

    // Normalize
    for (uint32_t i = 0; i < fft_sizes[k]; i++) {
      for (uint32_t j = 0; j < 2; j++) {
        out[i][j] /= sqrtf(fft_sizes[k]);
      }
    }

    // compute Mean Square Error
    float err = 0;
    for (uint32_t i = 0; i < fft_sizes[k]; i++) {
      for (uint32_t j = 0; j < 2; j++) {
        err += powf(fabs(in[i][j] - out[i][j]), 2);
      }
    }
    float mse = err/(2*fft_sizes[k]);

    printf(" MSE=%e\n", mse);

    if (mse >= 0.0001) {
      printf("MSE too large\n");
      exit(-1);
    }

    fftwf_free(in);
    fftwf_free(out);
    fftwf_destroy_plan(p_bw);
    fftwf_destroy_plan(p_fw);
  }

}

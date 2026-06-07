#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>
#include <cuda.h>

// ---- kernels.h (inlined) ----

__global__ void compute_probs(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  // assign overall id/index of the thread = id of row
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    double maxval;
    int m, k;
    int maxind;
    double M_d = (double) M;
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){   // initialize probs (though already done on CPU)
      probs[i*K + k] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){   // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        w[k] = alphas[i*K + k] + rands[m*K + k];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        }
      }
      probs[i*K + maxind] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      probs[i*K + k] /= M_d;
    }
  }
}

__global__ void compute_probs_unitStrides(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  // assign overall id/index of the thread = id of row
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n) {
    double maxval;
    int m, k;
    int maxind;
    double M_d = (double) M;
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){  // initialize probs (though already done on CPU)
      probs[k*n + i] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){    // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        // with +i we now have unit strides in inner loop
        w[k] = alphas[k*n + i] + rands[k*M + m];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        }
      }
      probs[maxind*n + i] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      // unit strides
      probs[k*n + i] /= M_d;
    }
  }
}

__global__ void compute_probs_unitStrides_sharedMem(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  // assign overall id/index of the thread = id of row
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int threads_per_block = blockDim.x;

  // set up shared memory: half for probs and half for w
  extern __shared__ double shared[];
  double* probs_shared = shared;

  // shared mem is one big block, so need to index into latter portion of it to use for w
  double* w = &shared[K*threads_per_block];

  double maxval;
  int m, k;
  int maxind;
  double M_d = (double) M;

  // initialize shared memory probs
  for(k = 0; k < K; ++k) {
    probs_shared[k*threads_per_block + threadIdx.x] = 0.0;
  }

  // core computation
  for(m = 0; m < M; ++m){     // loop over Monte Carlo iterations
    for(k = 0; k < K; ++k){   // generate W ~ N(alpha, 1)
      w[k*threads_per_block + threadIdx.x] = alphas[k*n + i] + rands[k*M + m];
    }
    maxind = K-1;
    maxval = w[(K-1)*threads_per_block + threadIdx.x];
    for(k = 0; k < (K-1); ++k){
      if(w[k*threads_per_block + threadIdx.x] > maxval){
        maxind = k;
        maxval = w[k*threads_per_block + threadIdx.x];
      }
    }
    probs_shared[maxind*threads_per_block + threadIdx.x] += 1.0;
  }

  for(k = 0; k < K; ++k) {
    probs_shared[k*threads_per_block + threadIdx.x] /= M_d;
  }

  // copy to device memory so can be returned to CPU
  for(k = 0; k < K; ++k) {
    probs[k*n + i] = probs_shared[k*threads_per_block + threadIdx.x];
  }
}

// ---- reference.h (inlined) ----

void verify(const double *probs, const double *probs_ref, int alphas_size) {
  bool error = false;
  for (int i = 0; i < alphas_size; i++) {
    if (fabs(probs[i] - probs_ref[i]) > 1e-3) {
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
}

void reference(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  for (int i = 0; i < n; i++) {
    double maxval;
    int m, k;
    int maxind;
    double M_d = (double) M;
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){   // initialize probs (though already done on CPU)
      probs[i*K + k] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){   // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        w[k] = alphas[i*K + k] + rands[m*K + k];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        }
      }
      probs[i*K + maxind] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      probs[i*K + k] /= M_d;
    }
  }
}

void reference_unitStrides(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  for (int i = 0; i < n; i++) {
    double maxval;
    int m, k;
    int maxind;
    double M_d = (double) M;
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){  // initialize probs (though already done on CPU)
      probs[k*n + i] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){    // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        // with +i we now have unit strides in inner loop
        w[k] = alphas[k*n + i] + rands[k*M + m];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        }
      }
      probs[maxind*n + i] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      // unit strides
      probs[k*n + i] /= M_d;
    }
  }
}

// ---- main.cu body ----

// transpose
double* t(const double *idata, const int width, const int height)
{
  double *odata = (double*) malloc (sizeof(double) * width * height);
  for (int yIndex = 0; yIndex < height; yIndex++) {
    for (int xIndex = 0; xIndex < width; xIndex++) {
      int index_in  = xIndex + width * yIndex;
      int index_out = yIndex + height * xIndex;
      odata[index_out] = idata[index_in];
    }
  }
  return odata;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <path to filename> <repeat>\n", argv[0]);
    return 1;
  }
  char *filename = argv[1];
  const int repeat = atoi(argv[2]);

  // n and K should match the dimension of the dataset in the csv file
  const int n = 26280, K = 21, M = 10000;

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Error: failed to open file alphas.csv. Exit\n");
    return 1;
  }

  int alphas_size = n * K; // n rows and K cols
  int alphas_size_byte = n * K * sizeof(double);

  int rands_size = M * K;  // M rows and K cols
  int rands_size_byte = M * K * sizeof(double);

  double *alphas, *rands, *probs, *probs_ref;
  alphas = (double*) malloc (alphas_size_byte);
  rands = (double*) malloc (rands_size_byte);
  probs = (double*) malloc (alphas_size_byte);
  probs_ref = (double*) malloc (alphas_size_byte);

  // load the csv file
  for (int i = 0; i < alphas_size; i++)
    fscanf(fp, "%lf", &alphas[i]);
  fclose(fp);

  // normal distribution (mean: 0 and var: 1)
  std::mt19937 gen(19937);
  std::normal_distribution<double> norm_dist(0.0,1.0);
  for (int i = 0; i < rands_size; i++) rands[i] = norm_dist(gen);

  reference(alphas, rands, probs_ref, n, K, M);

  double *d_alphas, *d_rands, *d_probs;
  cudaMalloc((void**)&d_rands, rands_size_byte);
  cudaMalloc((void**)&d_alphas, alphas_size_byte);
  cudaMalloc((void**)&d_probs, alphas_size_byte);

  cudaMemcpy(d_rands, rands, rands_size_byte, cudaMemcpyHostToDevice);
  cudaMemcpy(d_alphas, alphas, alphas_size_byte, cudaMemcpyHostToDevice);

  // kernel 1
  int threads_per_block = 192;
  dim3 threads (threads_per_block);
  dim3 blocks(ceil(1.0 * n / threads_per_block));

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    compute_probs<<<blocks, threads>>>(d_alphas, d_rands, d_probs, n, K, M);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of compute_probs kernel: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(probs, d_probs, alphas_size_byte, cudaMemcpyDeviceToHost);
  verify(probs, probs_ref, alphas_size);

  // kernel 2
  double *t_rands = t(rands, K, M);
  double *t_alphas = t(alphas, K, n);

  reference_unitStrides(t_alphas, t_rands, probs_ref, n, K, M);

  cudaMemcpy(d_rands, t_rands, rands_size_byte, cudaMemcpyHostToDevice);
  cudaMemcpy(d_alphas, t_alphas, alphas_size_byte, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    compute_probs_unitStrides<<<blocks, threads>>>(
      d_alphas, d_rands, d_probs, n, K, M);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of compute_probs_unitStrides kernel: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(probs, d_probs, alphas_size_byte, cudaMemcpyDeviceToHost);
  verify(probs, probs_ref, alphas_size);

  // kernel 3
  threads_per_block = 96;
  dim3 threads2 (threads_per_block);
  dim3 blocks2 (ceil(1.0 * n / threads_per_block));

  const int sm_size = sizeof(double) * K * threads_per_block * 2;

  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    compute_probs_unitStrides_sharedMem<<<blocks2, threads2, sm_size, 0>>>(
      d_alphas, d_rands, d_probs, n, K, M);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of compute_probs_unitStrides_sharedMem kernel: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(probs, d_probs, alphas_size_byte, cudaMemcpyDeviceToHost);
  verify(probs, probs_ref, alphas_size);

  // free memory
  cudaFree(d_alphas);
  cudaFree(d_rands);
  cudaFree(d_probs);
  free(alphas);
  free(rands);
  free(t_alphas);
  free(t_rands);
  free(probs);
  free(probs_ref);
  return 0;
}

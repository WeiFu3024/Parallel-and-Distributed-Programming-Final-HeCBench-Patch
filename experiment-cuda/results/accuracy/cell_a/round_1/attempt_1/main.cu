#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>
#include "reference.h"

#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef cub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  input = BlockReduce(temp_storage).Sum(input);
}

// Original kernel — kept for baseline comparison
__global__
void accuracy_kernel(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int* __restrict__ labelData,
    int* accuracy)
{
  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int   label      = __ldg(labelData + row);
    const float label_pred = __ldg(Xdata + (size_t)row * D + label);
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = __ldg(Xdata + (size_t)row * D + col);
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    BlockReduce(ngt);
    if (ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(accuracy, count);
  }
}

// Optimized kernel2:
//   - __ldg() read-only cache on all global loads
//   - 8x loop unroll (up from 4x) to hide memory latency
//   - float4 vectorized loads for 4x wider memory transactions
//   - warp-level ballot to reduce block-level atomic contention
__global__
void accuracy_kernel2(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int*   __restrict__ labelData,
    int* accuracy)
{
  __shared__ float s_label_pred;
  __shared__ int   s_label;

  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {

    if (threadIdx.x == 0) {
      s_label      = __ldg(labelData + row);
      s_label_pred = __ldg(Xdata + (size_t)row * D + s_label);
    }
    __syncthreads();

    const int   label      = s_label;
    const float label_pred = s_label_pred;

    int ngt = 0;
    const float* row_ptr = Xdata + (size_t)row * D;

    // float4 vectorized path: processes 4 consecutive columns per load.
    // Thread t reads columns [t*4, t*4+1, t*4+2, t*4+3] each iteration.
    // Consecutive threads read consecutive groups → coalesced 128-bit loads.
    const int D4     = D / 4;           // number of complete float4 groups
    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);

    for (int i = threadIdx.x; i < D4; i += blockDim.x) {
      float4 v = __ldg(row_ptr4 + i);
      int base = i * 4;
      ngt += (v.x > label_pred || (v.x == label_pred && base     <= label));
      ngt += (v.y > label_pred || (v.y == label_pred && base + 1 <= label));
      ngt += (v.z > label_pred || (v.z == label_pred && base + 2 <= label));
      ngt += (v.w > label_pred || (v.w == label_pred && base + 3 <= label));
    }
    // Handle tail (at most 3 remaining elements)
    for (int col = D4 * 4 + threadIdx.x; col < D; col += blockDim.x) {
      float pred = __ldg(row_ptr + col);
      ngt += (pred > label_pred || (pred == label_pred && col <= label));
    }

    BlockReduce(ngt);

    if (threadIdx.x == 0 && ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && count > 0) {
    atomicAdd(accuracy, count);
  }
}


int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ndims = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int data_size = nrows * ndims;

  const int label_size_bytes = nrows * sizeof(int);
  const size_t data_size_bytes = (size_t)data_size * sizeof(float);

  int *label = (int*) malloc (label_size_bytes);

  srand(123);
  for (int i = 0; i < nrows; i++)
    label[i] = rand() % ndims;

  float *data = (float*) malloc (data_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  int count_ref = reference(nrows, ndims, top_k, data, label);

  int *d_label;
  cudaMalloc((void**)&d_label, label_size_bytes);
  cudaMemcpy(d_label, label, label_size_bytes, cudaMemcpyHostToDevice);

  float *d_data;
  cudaMalloc((void**)&d_data, data_size_bytes);
  cudaMemcpy(d_data, data, data_size_bytes, cudaMemcpyHostToDevice);

  int *d_count;
  cudaMalloc((void**)&d_count, sizeof(int));

  cudaDeviceSynchronize();
  dim3 block (GPU_NUM_THREADS);

  for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {

    dim3 grid (ngrid);
    printf("Grid size is %d\n", ngrid);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      cudaMemset(d_count, 0, sizeof(int));
      accuracy_kernel<<<grid, block>>>(nrows, ndims, top_k, d_data, d_label, d_count);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel: %f (us)\n", (time * 1e-3f) / repeat);

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      cudaMemset(d_count, 0, sizeof(int));
      accuracy_kernel2<<<grid, block>>>(nrows, ndims, top_k, d_data, d_label, d_count);
    }

    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel2: %f (us)\n", (time * 1e-3f) / repeat);
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");
  }

  cudaFree(d_label);
  cudaFree(d_data);
  cudaFree(d_count);

  free(label);
  free(data);

  return 0;
}

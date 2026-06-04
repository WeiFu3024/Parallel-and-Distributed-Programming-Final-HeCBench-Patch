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
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = Xdata[row * D + col];
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

// Round 2: eliminate shared-memory label broadcast + 4x float4 unrolling per iteration
// All threads read labelData[row] directly — same address => L1 broadcast, no __syncthreads needed.
// Warp shuffle reduction. 2 syncs per row (down from 3 in round 1).
__launch_bounds__(GPU_NUM_THREADS, 4)   // allow compiler up to 64 regs/thread for unrolling
__global__
void accuracy_kernel2(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int*   __restrict__ labelData,
    int* accuracy)
{
  __shared__ int s_warp_ngt[GPU_NUM_THREADS / 32]; // 8 ints = 32 bytes

  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {

    // All 256 threads read the same label address — hardware broadcasts from L1 after first hit
    const int   label      = __ldg(labelData + row);
    const float label_pred = __ldg(Xdata + (size_t)row * D + label);

    int ngt = 0;

    const float4* row4 = reinterpret_cast<const float4*>(Xdata + (size_t)row * D);
    const int D4 = D >> 2; // number of float4 elements

    // 4x unrolled float4 loop: 4 independent load-compute chains per iteration
    // This exposes 4x more memory-level parallelism to the load pipelines.
    int i = threadIdx.x;
    for (; i + 3 * blockDim.x < D4; i += 4 * blockDim.x) {
      float4 v0 = __ldg(row4 + i);
      float4 v1 = __ldg(row4 + i + blockDim.x);
      float4 v2 = __ldg(row4 + i + 2 * blockDim.x);
      float4 v3 = __ldg(row4 + i + 3 * blockDim.x);

      int c0 = i * 4;
      int c1 = (i + blockDim.x) * 4;
      int c2 = (i + 2 * blockDim.x) * 4;
      int c3 = (i + 3 * blockDim.x) * 4;

      ngt += (int)(v0.x > label_pred || (v0.x == label_pred && c0     <= label));
      ngt += (int)(v0.y > label_pred || (v0.y == label_pred && c0 + 1 <= label));
      ngt += (int)(v0.z > label_pred || (v0.z == label_pred && c0 + 2 <= label));
      ngt += (int)(v0.w > label_pred || (v0.w == label_pred && c0 + 3 <= label));

      ngt += (int)(v1.x > label_pred || (v1.x == label_pred && c1     <= label));
      ngt += (int)(v1.y > label_pred || (v1.y == label_pred && c1 + 1 <= label));
      ngt += (int)(v1.z > label_pred || (v1.z == label_pred && c1 + 2 <= label));
      ngt += (int)(v1.w > label_pred || (v1.w == label_pred && c1 + 3 <= label));

      ngt += (int)(v2.x > label_pred || (v2.x == label_pred && c2     <= label));
      ngt += (int)(v2.y > label_pred || (v2.y == label_pred && c2 + 1 <= label));
      ngt += (int)(v2.z > label_pred || (v2.z == label_pred && c2 + 2 <= label));
      ngt += (int)(v2.w > label_pred || (v2.w == label_pred && c2 + 3 <= label));

      ngt += (int)(v3.x > label_pred || (v3.x == label_pred && c3     <= label));
      ngt += (int)(v3.y > label_pred || (v3.y == label_pred && c3 + 1 <= label));
      ngt += (int)(v3.z > label_pred || (v3.z == label_pred && c3 + 2 <= label));
      ngt += (int)(v3.w > label_pred || (v3.w == label_pred && c3 + 3 <= label));
    }
    // Tail: single float4 per iteration
    for (; i < D4; i += blockDim.x) {
      float4 v = __ldg(row4 + i);
      int c0 = i * 4;
      ngt += (int)(v.x > label_pred || (v.x == label_pred && c0     <= label));
      ngt += (int)(v.y > label_pred || (v.y == label_pred && c0 + 1 <= label));
      ngt += (int)(v.z > label_pred || (v.z == label_pred && c0 + 2 <= label));
      ngt += (int)(v.w > label_pred || (v.w == label_pred && c0 + 3 <= label));
    }
    // Scalar tail for D not divisible by 4
    for (int col = D4 * 4 + threadIdx.x; col < D; col += blockDim.x) {
      float p = __ldg(Xdata + (size_t)row * D + col);
      ngt += (int)(p > label_pred || (p == label_pred && col <= label));
    }

    // Warp-level reduction (5 shuffle steps, no shared memory read after)
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      ngt += __shfl_down_sync(0xffffffff, ngt, mask);

    // Lane 0 of each warp posts its partial sum
    if ((threadIdx.x & 31) == 0)
      s_warp_ngt[threadIdx.x >> 5] = ngt;
    __syncthreads(); // all warp partial sums now written

    // Thread 0 aggregates
    if (threadIdx.x == 0) {
      int total = 0;
      #pragma unroll
      for (int w = 0; w < GPU_NUM_THREADS / 32; w++)
        total += s_warp_ngt[w];
      if (total <= top_k)
        ++count;
    }
    __syncthreads(); // before overwriting s_warp_ngt in next row iteration
  }

  if (threadIdx.x == 0 && count > 0)
    atomicAdd(accuracy, count);
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
  const size_t data_size_bytes = data_size * sizeof(float);

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

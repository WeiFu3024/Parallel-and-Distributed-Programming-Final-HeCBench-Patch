#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>
// ---- INLINED: kernels.h (from /home/WillFu/parallel/final/HeCBench/src/attention-cuda/kernels.h) ----
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

__global__
void attention_kernel1 (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    atomicAdd(exp_sum, __expf(sum));
  }
}

__global__
void attention_kernel2 (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    float*__restrict__ score,
    const int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    score[i] = __expf(dot_product[i]) / exp_sum[0];
}

__global__
void attention_kernel3 (
    const float*__restrict__ score,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}

__global__
void attention_kernel1_blockReduce (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  // each i iteration is assigned to a block
  int i = blockIdx.x;
  float sum = 0;
  for (int j = threadIdx.x; j < d; j += blockDim.x) {
    sum += key[i * d + j] * query[j];
  }
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  sum = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    dot_product[i] = sum;
    atomicAdd(exp_sum, __expf(sum));
  }
}

__global__
void attention_kernel1_warpReduce (
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  // each i iteration is assigned to a warp
  // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
  int i = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (i < n) {
    float sum = 0;
    for (int j = warp.thread_rank(); j < d; j += warp.size()) {
      sum += key[i * d + j] * query[j];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0) {
      dot_product[i] = sum;
      atomicAdd(exp_sum, __expf(sum));
    }
  }
}

__global__
void attention_kernel2_blockReduce (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  int j = blockIdx.x;
  float sum = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float score = __expf(dot_product[i]) / exp_sum[0];
    sum += score * value[i * d + j];
  }
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  sum = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0)
    output[j] = sum;
}

__global__
void attention_kernel2_warpReduce (
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int j = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (j < d) {
    float sum = 0;
    for (int i = warp.thread_rank(); i < n; i += warp.size()) {
      float score = __expf(dot_product[i]) / exp_sum[0];
      sum += score * value[i * d + j];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0)
      output[j] = sum;
  }
}

// ---- END INLINED: kernels.h ----

// ---- INLINED: reference.h (from /home/WillFu/parallel/final/HeCBench/src/attention-cuda/reference.h) ----
float* attention_host(const float* key, const float* value, const float* query,
                      const int n, const int d) 
{
  // intermediate
  float* dot_product = (float*) malloc (n * sizeof(float));
  float* score = (float*) malloc (n * sizeof(float));
  // result
  float* output = (float*) malloc (d * sizeof(float));

  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
  }

  float sum = 0;
  for (int i = 0; i < n; i++)
    sum += expf(dot_product[i]);

  for (int i = 0; i < n; i++)
    score[i] = expf(dot_product[i]) / sum;

  for (int j = 0; j < d; j++) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }

  free(dot_product);
  free(score);
  return output;
}



// ---- END INLINED: reference.h ----


float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int impl_num, const int repeat)
{
  size_t q_size = (size_t)d;
  size_t r_size = (size_t)n;
  size_t kv_size = (size_t)d * n;
  size_t r_size_bytes = r_size * sizeof(float);
  size_t q_size_bytes = q_size * sizeof(float);
  size_t kv_size_bytes = kv_size * sizeof(float);

  // input
  float *d_key;
  cudaMalloc((void**)&d_key, kv_size_bytes);
  cudaMemcpy(d_key, key, kv_size_bytes, cudaMemcpyHostToDevice);

  float *d_value;
  cudaMalloc((void**)&d_value, kv_size_bytes);
  cudaMemcpy(d_value, value, kv_size_bytes, cudaMemcpyHostToDevice);

  float *d_query;
  cudaMalloc((void**)&d_query, q_size_bytes);
  cudaMemcpy(d_query, query, q_size_bytes, cudaMemcpyHostToDevice);

  // intermediate
  float *d_dot_product;
  cudaMalloc((void**)&d_dot_product, r_size_bytes);

  float *d_exp_sum;
  cudaMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (q_size_bytes);
  float *d_output;
  cudaMalloc((void**)&d_output, q_size_bytes);

  cudaDeviceSynchronize();

  if (impl_num == 3) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      attention_kernel1_warpReduce<<<(n+7)/8, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      attention_kernel2_blockReduce<<<d, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 2) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      attention_kernel1_warpReduce<<<(n+7)/8, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      attention_kernel2_warpReduce<<<(d+7)/8, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 1) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      attention_kernel1_blockReduce<<<n, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      attention_kernel2_blockReduce<<<d, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else {
    float *d_score;
    cudaMalloc((void**)&d_score, r_size_bytes);

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      attention_kernel1<<<(n+255)/256, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      attention_kernel2<<<(n+255)/256, 256>>>(d_exp_sum, d_dot_product, d_score, n);
      attention_kernel3<<<(d+255)/256, 256>>>(d_score, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
    cudaFree(d_score);
  }

  cudaMemcpy(output, d_output, q_size_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_value);
  cudaFree(d_output);
  cudaFree(d_key);
  cudaFree(d_dot_product);
  cudaFree(d_exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <rows> <columns> <implementation> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: fused kernels with block reduce\n");
    printf("implementation 2: fused kernels with warp reduce\n");
    printf("implementation 3: fused kernels with mixed reduce\n");
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int r = atoi(argv[4]);

  // input
  size_t q_size = (size_t)d;
  size_t kv_size = (size_t)d * n;
  size_t q_size_bytes = q_size * sizeof(float);
  size_t kv_size_bytes = kv_size * sizeof(float);

  float* key = (float*) malloc (kv_size_bytes);
  float* value = (float*) malloc (kv_size_bytes);
  float* query = (float*) malloc (q_size_bytes);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  for (size_t i = 0; i < kv_size; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
    query[i % d] = dist(gen);
  }

  float* hout = attention_host(key, value, query, n, d);
  float* dout = attention_device(key, value, query, n, d, k, r);

  bool ok = true;
  for (int i = 0; i < d; i++) {
    if (fabsf(hout[i] - dout[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "utils.h"

// Tiled GEMM: C = alpha * A * B + beta * C
//
// Template parameters
//   BM, BN : output tile dimensions per thread-block
//   BK     : K-dimension strip processed per shared-memory phase
//   TM, TN : per-thread output tile (register blocking)
//
// Thread-block shape: (BN/TN, BM/TM) — always 256 threads for all types.
//
// Shared memory layout (with +1 padding column to reduce bank conflicts):
//   sA[BM][BK+1]  —  A sub-tile
//   sB[BK][BN+1]  —  B sub-tile
//
// Arithmetic intensity per global byte:
//   baseline : 2 FLOPs / (2 * sizeof(T)) bytes  = O(1)
//   this kernel: 2*BM*BN*BK FLOPs / shared-reload traffic = O(BK) improvement
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_tiled(const T* __restrict__ A, const T* __restrict__ B, T* C,
           int M, int K, int N, T alpha, T beta)
{
    constexpr int THREADS_X  = BN / TN;
    constexpr int THREADS_Y  = BM / TM;
    constexpr int NUM_THREADS = THREADS_X * THREADS_Y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;

    // +1 padding reduces shared-memory bank conflicts on column reads
    __shared__ T sA[BM][BK + 1];
    __shared__ T sB[BK][BN + 1];

    // Accumulator lives entirely in registers
    T acc[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; m++)
        #pragma unroll
        for (int n = 0; n < TN; n++)
            acc[m][n] = T(0);

    // Main K loop: each iteration loads one BM×BK strip of A
    // and one BK×BN strip of B into shared memory, then accumulates.
    for (int k_off = 0; k_off < K; k_off += BK) {

        // --- Cooperative load of A tile (BM × BK) ---
        for (int i = tid; i < BM * BK; i += NUM_THREADS) {
            int r = i / BK, c = i % BK;
            int gr = row_base + r, gc = k_off + c;
            sA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : T(0);
        }

        // --- Cooperative load of B tile (BK × BN) ---
        for (int i = tid; i < BK * BN; i += NUM_THREADS) {
            int r = i / BN, c = i % BN;
            int gr = k_off + r, gc = col_base + c;
            sB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : T(0);
        }

        __syncthreads();

        // --- Accumulate TM×TN outer products across BK ---
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            T ra[TM], rb[TN];
            // Load a column of A into registers (broadcast: all warp threads share ty)
            #pragma unroll
            for (int m = 0; m < TM; m++)
                ra[m] = sA[ty * TM + m][k];
            // Load a row of B into registers
            #pragma unroll
            for (int n = 0; n < TN; n++)
                rb[n] = sB[k][tx * TN + n];
            // Outer product accumulation
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += ra[m] * rb[n];
        }

        __syncthreads();
    }

    // --- Write TM×TN results with alpha/beta scaling ---
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int gr = row_base + ty * TM + m;
            int gc = col_base + tx * TN + n;
            if (gr < M && gc < N)
                C[gr * N + gc] = alpha * acc[m][n] + beta * C[gr * N + gc];
        }
    }
}

// Dispatch gemm_tiled with type-appropriate tile sizes.
//
// float  : BM=BN=128, BK=8,  TM=TN=8  → 256 threads, ~8.5 KB shmem
// double : BM=BN=64,  BK=8,  TM=TN=4  → 256 threads, ~8.6 KB shmem
// __half : BM=BN=128, BK=16, TM=TN=8  → 256 threads, ~8.3 KB shmem
template <typename T>
void run_simple_gemm(T *a, T *b, T *c, int M, int K, int N, T alpha, T beta) {
    if constexpr (std::is_same_v<T, double>) {
        constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
        dim3 threads(BN / TN, BM / TM);
        dim3 grids((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_tiled<T, BM, BN, BK, TM, TN><<<grids, threads>>>(a, b, c, M, K, N, alpha, beta);
    } else if constexpr (std::is_same_v<T, float>) {
        constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
        dim3 threads(BN / TN, BM / TM);
        dim3 grids((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_tiled<T, BM, BN, BK, TM, TN><<<grids, threads>>>(a, b, c, M, K, N, alpha, beta);
    } else {
        // __half: wider K strip to amortize shared-memory overhead
        constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
        dim3 threads(BN / TN, BM / TM);
        dim3 grids((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_tiled<T, BM, BN, BK, TM, TN><<<grids, threads>>>(a, b, c, M, K, N, alpha, beta);
    }
}

//
// Main example for Gemm consisting of
// initialization of A, B and C matrices as well as
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
template <typename fp>
void run_gemm_example(int m, int k, int n, int repeat) {

  //
  // Initialize data for Gemm
  //
  // C = alpha * op(A) * op(B)  + beta * C
  //

  // set scalar fp values
  const fp alpha = fp(2.0);
  const fp beta  = fp(0.5);

  const size_t A_size = sizeof(fp) * m * k;
  const size_t B_size = sizeof(fp) * k * n;
  const size_t C_size = sizeof(fp) * m * n;

  // prepare matrix data
  fp* a = (fp *) aligned_alloc(64, A_size);
  fp* b = (fp *) aligned_alloc(64, B_size);
  fp* c = (fp *) aligned_alloc(64, C_size);
  fp* r = (fp *) aligned_alloc(64, C_size);

  srand(2);
  rand_matrix(a, m, k);
  rand_matrix(b, k, n);
  rand_matrix(c, m, n);

  fp *da, *db, *dc, *dr;
  cudaMalloc((void**)&da, A_size);
  cudaMalloc((void**)&db, B_size);
  cudaMalloc((void**)&dc, C_size);
  cudaMalloc((void**)&dr, C_size);
  cudaMemcpy(da, a, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, c, C_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dr, c, C_size, cudaMemcpyHostToDevice);

  // create execution queue and buffers of matrix data
  cublasHandle_t h;
  cublasCreate(&h);

  std::cout << "Checking BLAS GEMM.. ";
  run_simple_gemm(da, db, dr, m, k, n, alpha, beta);

  if constexpr (std::is_same_v<fp, __half>)
    cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, db, n, da, k, &beta, dc, n);
  else if constexpr (std::is_same_v<fp, float>)
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, db, n, da, k, &beta, dc, n);
  else if constexpr (std::is_same_v<fp, double>)
    cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, db, n, da, k, &beta, dc, n);

  cudaMemcpy(c, dc, C_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(r, dr, C_size, cudaMemcpyDeviceToHost);
  int error = memcmp(c, r, C_size);
  std::cout << (error ? "FAIL" : "PASS") << std::endl;

  // Benchmark our tiled kernel (not cuBLAS)
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    run_simple_gemm(da, db, dc, m, k, n, alpha, beta);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  performance(m, n, k, false, time / repeat);

  //
  // Post Processing
  //

#ifdef DEBUG
  std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

  // output the top 2x2 block of A matrix
  print_2x2_matrix_values(a, k, "A");

  // output the top 2x2 block of B matrix
  print_2x2_matrix_values(b, n, "B");

  // output the top 2x2 block of C matrix
  cudaMemcpy(c, dc, C_size, cudaMemcpyDeviceToHost);
  print_2x2_matrix_values(c, n, "C");
#endif

  cublasDestroy(h);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dr);

  free(a);
  free(b);
  free(c);
  free(r);
}

//
// Main entry point for example.
//
int main (int argc, char ** argv) {
  if (argc != 5) {
    printf("Usage: %s <m> <k> <n> <repeat>\n", argv[0]);
    return 1;
  }
  const int m = atoi(argv[1]);
  const int k = atoi(argv[2]);
  const int n = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  std::cout << "\tRunning with half precision data type:" << std::endl;
  run_gemm_example<__half>(m, k, n, repeat);

  std::cout << "\tRunning with single precision data type:" << std::endl;
  run_gemm_example<float>(m, k, n, repeat);

  std::cout << "\tRunning with double precision data type:" << std::endl;
  run_gemm_example<double>(m, k, n, repeat);

  return 0;
}

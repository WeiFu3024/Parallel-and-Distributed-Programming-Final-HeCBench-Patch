/*
  Reference
  Chapter 16 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <cuda.h>

#define TILE_WIDTH 16

#define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
#define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
#define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

#ifdef CUDNN_CONV
#include <cudnn.h>
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
    }                                                        \
  }
#endif

template <typename T>
void verify (const T* Y, T* Y_ref, size_t Y_size)
{
  bool ok = true;
  for (size_t i = 0; i < Y_size; i++) {
    if (fabs(Y[i] - Y_ref[i]) > 1e-3f) {
      printf("%f (device) != %f (reference)\n", Y[i], Y_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

template<typename T>
__global__
void conv3d_s1(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int n = blockIdx.x;
  int m = blockIdx.y;
  int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}

template<typename T>
__global__
void conv3d_s2(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int m = blockIdx.x;
  int h = blockIdx.y / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.y % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.z;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}

template<typename T>
__global__
void conv3d_s3(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int h = blockIdx.x / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.x % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.y;
  int m = blockIdx.z;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}


// Hin = Hout-1+K; max(h+p) is Hin - 1 as max(h) = Hout-1 and max(p) = K-1
template <typename T>
void reference(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int N,
               const int M,
               const int C,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout)
{
  for(int n = 0; n < N; n++)
    for(int m = 0; m < M; m++)
      for(int h = 0; h < Hout; h++)
        for(int w = 0; w < Wout; w++) {
          Y[OI(n, m, h, w)] = 0;
          for(int c = 0; c < C; c++)
            for(int p = 0; p < K; p++)
              for(int q = 0; q < K; q++)
                Y[OI(n, m, h, w)] += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
}

template <typename T>
void conv3D(const int N, const int C, const int M, const int Win, const int Hin, const int K, const int repeat)
{
  const int Hout = Hin-K+1;
  const int Wout = Win-K+1;

  size_t X_size = N * C * Hin * Win;
  size_t W_size = M * C * K * K;
  size_t Y_size = N * M * Hout * Wout;
  size_t X_bytes = X_size * sizeof(T);
  size_t W_bytes = W_size * sizeof(T);
  size_t Y_bytes = Y_size * sizeof(T);

  T *X, *W, *Y, *Y_ref;
  X = (T *)malloc(X_bytes); // input
  W = (T *)malloc(W_bytes); // filter
  Y = (T *)malloc(Y_bytes); // output
  Y_ref = (T *)malloc(Y_bytes);

  srand(123);

  for (size_t i = 0; i < W_size; i++) W[i] = rand() % 31;
  for (size_t i = 0; i < X_size; i++) X[i] = rand() % 13;

  for (size_t i = 0; i < Y_size; i++) {
    Y[i] = -1;
    Y_ref[i] = -1;
  }

  reference(X, W, Y_ref, N, M, C, K, Hin, Win, Hout, Wout);

  T *dX, *dW, *dY;
  cudaMalloc((void **)&dX, X_bytes);
  cudaMalloc((void **)&dW, W_bytes);
  cudaMalloc((void **)&dY, Y_bytes);

  cudaMemcpy(dX, X, X_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dW, W, W_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dY, Y, Y_bytes, cudaMemcpyHostToDevice);

  int W_grid = (Wout + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_grid = (Hout + TILE_WIDTH - 1) / TILE_WIDTH;
  int Z = H_grid * W_grid;

  printf("input dimensions: C=%d Win=%d Hin=%d\n", C, Win, Hin);
  printf("output dimensions: M=%d Wout=%d Hout=%d\n", M, Wout, Hout);
  printf("3D grid dimensions: N=%d M=%d Z=%d\n", N, M, Z);

  // try grid organizations
  dim3 grids_s1 (N, M, Z);
  dim3 grids_s2 (M, Z, N);
  dim3 grids_s3 (Z, N, M);
  dim3 blocks (TILE_WIDTH, TILE_WIDTH, 1);

  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv3d_s1 <<< grids_s1, blocks >>> (dX, dW, dY, C, M, K, Hin, Win, Hout, Wout, W_grid);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s1 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv3d_s2 <<< grids_s2, blocks >>> (dX, dW, dY, C, M, K, Hin, Win, Hout, Wout, W_grid);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s2 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv3d_s3 <<< grids_s3, blocks >>> (dX, dW, dY, C, M, K, Hin, Win, Hout, Wout, W_grid);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s3 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);

#ifdef CUDNN_CONV
// ---- INLINED: conv3d_s4.cu (from /home/WillFu/parallel/final/HeCBench/src/convolution3D-cuda/conv3d_s4.cu) ----
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/N,
                                        /*channels=*/C,
                                        /*image_height=*/Hin,
                                        /*image_width=*/Win));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/M,
                                        /*in_channels=*/C,
                                        /*kernel_height=*/K,
                                        /*kernel_width=*/K));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/0,
                                             /*pad_width=*/0,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));

  int batch_size{0}, channels{0}, height{0}, width{0};
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  #ifdef DEBUG
  std::cerr << "Output Image(NxHxWxC): " << batch_size << " x "
            << height << " x " << width << " x " << channels << std::endl;
  #endif

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/N,
                                        /*channels=*/M,
                                        /*image_height=*/Hout,
                                        /*image_width=*/Wout));

  int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
  int returnedAlgoCount = -1;
  cudnnConvolutionFwdAlgoPerf_t results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
                                                  input_descriptor,
                                                  kernel_descriptor,
                                                  convolution_descriptor,
                                                  output_descriptor,
                                                  requestedAlgoCount,
                                                  &returnedAlgoCount,
                                                  results));

  #ifdef DEBUG
  std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
  for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
    printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n",
           cudnnGetErrorString(results[algoIndex].status),
           results[algoIndex].algo, results[algoIndex].time,
           (unsigned long long)results[algoIndex].memory);
  }
  #endif

  cudnnConvolutionFwdAlgo_t convolution_algorithm = results[0].algo;

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
  void* d_workspace{nullptr};
  if (workspace_bytes > 0) {
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;
    cudaMalloc(&d_workspace, workspace_bytes);
  }

  const float alpha = 1.0f, beta = 0.0f;

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       dX, //d_input,
                                       kernel_descriptor,
                                       dW, //d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       dY )); //d_output
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s4 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);

// ---- END INLINED: conv3d_s4.cu ----

  cudaMemcpy(Y, dY, Y_bytes, cudaMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);
#endif

  free(X);
  free(W);
  free(Y);
  free(Y_ref);
  cudaFree(dX);
  cudaFree(dW);
  cudaFree(dY);
}

int main(int argc, char* argv[]) {
  if (argc != 8) {
    printf("Usage: %s <batch size:N> <input channels:C> <output feature maps:M>", argv[0]);
    printf(" <input width:Win> <input height:Hin> <kernel size:K> <repeat>\n");
    return 1;
  }

  int N = atoi(argv[1]);
  int C = atoi(argv[2]);
  int M = atoi(argv[3]);
  int W = atoi(argv[4]);
  int H = atoi(argv[5]);
  int K = atoi(argv[6]);
  int repeat = atoi(argv[7]);

  printf("3D convolution (FP32)\n");
  printf("\n========== Warmup start ==========\n");
  conv3D<float>(N, C, M, W, H, K, 1000);
  printf("\n========== Warmup done ==========\n");
  conv3D<float>(N, C, M, W, H, K, repeat);

  return 0;
}

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// System includes (deduplicated)
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

// Local dep includes (kept as separate dep files)
#include "binomialOptions.h"
#include "realtype.h"

// ---- kernel.cu body (inlined; __global__ and extern functions marked weak
//      to avoid duplicate symbol errors when compile.sh also compiles
//      kernel.cu as a separate TU; static/inline/device-inline functions
//      have internal or non-external linkage and need no weak guard) ----

//Preprocessed input option data
typedef struct
{
  real S;
  real X;
  real vDt;
  real puByDf;
  real pdByDf;
} __TOptionData;


// Overloaded shortcut functions for different precision modes
// (__device__ inline functions have no external host linkage; no weak needed)
#ifndef DOUBLE_PRECISION
__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
  float d = S * __expf(vDt * (2.0f * i - NUM_STEPS)) - X;
  return (d > 0.0F) ? d : 0.0F;
}
#else
__device__ inline double expiryCallValue(double S, double X, double vDt, int i)
{
  double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
  return (d > 0.0) ? d : 0.0;
}
#endif


// GPU kernel
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif

__attribute__((weak)) __global__ void binomialOptionsKernel(const __TOptionData *__restrict d_OptionData,
                                      real *__restrict d_CallValue)
{
  __shared__ real call_exchange[THREADBLOCK_SIZE + 1];

  const int     tid = threadIdx.x;
  const real      S = d_OptionData[blockIdx.x].S;
  const real      X = d_OptionData[blockIdx.x].X;
  const real    vDt = d_OptionData[blockIdx.x].vDt;
  const real puByDf = d_OptionData[blockIdx.x].puByDf;
  const real pdByDf = d_OptionData[blockIdx.x].pdByDf;

  real call[ELEMS_PER_THREAD + 1];
#pragma unroll
  for(int i = 0; i < ELEMS_PER_THREAD; ++i)
    call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

  if (tid == 0)
    call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

  int final_it = max(0, tid * ELEMS_PER_THREAD - 1);

#pragma unroll 16
  for(int i = NUM_STEPS; i > 0; --i)
  {
    call_exchange[tid] = call[0];
    __syncthreads();
    call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
    __syncthreads();

    if (i > final_it)
    {
#pragma unroll
      for(int j = 0; j < ELEMS_PER_THREAD; ++j)
        call[j] = puByDf * call[j + 1] + pdByDf * call[j];
    }
  }

  if (tid == 0)
  {
    d_CallValue[blockIdx.x] = call[0];
  }
}

// Host-side interface to GPU binomialOptions
extern "C" __attribute__((weak)) void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN,
    int numIterations
    )
{
  __TOptionData h_OptionData[MAX_OPTIONS];

  for (int i = 0; i < optN; i++)
  {
    const real      T = optionData[i].T;
    const real      R = optionData[i].R;
    const real      V = optionData[i].V;

    const real     dt = T / (real)NUM_STEPS;
    const real    vDt = V * sqrt(dt);
    const real    rDt = R * dt;
    //Per-step interest and discount factors
    const real     If = exp(rDt);
    const real     Df = exp(-rDt);
    //Values and pseudoprobabilities of upward and downward moves
    const real      u = exp(vDt);
    const real      d = exp(-vDt);
    const real     pu = (If - d) / (u - d);
    const real     pd = (real)1.0 - pu;
    const real puByDf = pu * Df;
    const real pdByDf = pd * Df;

    h_OptionData[i].S      = (real)optionData[i].S;
    h_OptionData[i].X      = (real)optionData[i].X;
    h_OptionData[i].vDt    = (real)vDt;
    h_OptionData[i].puByDf = (real)puByDf;
    h_OptionData[i].pdByDf = (real)pdByDf;
  }

  __TOptionData *d_OptionData;
  cudaMalloc ((void**)&d_OptionData, sizeof(__TOptionData) * MAX_OPTIONS);
  cudaMemcpy(d_OptionData, h_OptionData, optN * sizeof(__TOptionData), cudaMemcpyHostToDevice);

  real *d_CallValue;
  cudaMalloc ((void**)&d_CallValue, sizeof(real) * MAX_OPTIONS);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; i++)
    binomialOptionsKernel<<<optN, THREADBLOCK_SIZE>>>(d_OptionData, d_CallValue);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (us)\n", time * 1e-3f / numIterations);

  cudaMemcpy(callValue, d_CallValue, optN *sizeof(real), cudaMemcpyDeviceToHost);
  cudaFree(d_OptionData);
  cudaFree(d_CallValue);
}

// ---- reference.cu body (inlined; static functions have internal linkage
//      and need no weak guard; extern "C" functions use weak) ----

// Polynomial approximation of cumulative normal distribution function
// (static = internal linkage, no external symbol conflict)
static real CND(real d)
{
  const real       A1 = 0.31938153;
  const real       A2 = -0.356563782;
  const real       A3 = 1.781477937;
  const real       A4 = -1.821255978;
  const real       A5 = 1.330274429;
  const real RSQRT2PI = 0.39894228040143267793994605993438;

  real
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  real
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
    (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0)
    cnd = 1.0 - cnd;

  return cnd;
}

extern "C" __attribute__((weak)) void BlackScholesCall(
    real &callResult,
    TOptionData optionData
    )
{
  real S = optionData.S;
  real X = optionData.X;
  real T = optionData.T;
  real R = optionData.R;
  real V = optionData.V;

  real sqrtT = sqrt(T);
  real    d1 = (log(S / X) + (R + (real)0.5 * V * V) * T) / (V * sqrtT);
  real    d2 = d1 - V * sqrtT;
  real CNDD1 = CND(d1);
  real CNDD2 = CND(d2);

  //Calculate Call and Put simultaneously
  real expRT = exp(- R * T);
  callResult   = (real)(S * CNDD1 - X * expRT * CNDD2);
}

// Process an array of OptN options on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
// (static = internal linkage; name reused from kernel.cu but different type: host vs device)
static real expiryCallValueCPU(real S, real X, real vDt, int i)
{
  real d = S * exp(vDt * (real)(2 * i - NUM_STEPS)) - X;
  return (d > (real)0) ? d : (real)0;
}

extern "C" __attribute__((weak)) void binomialOptionsCPU(
    real &callResult,
    TOptionData optionData
    )
{
  static real Call[NUM_STEPS + 1];

  const real       S = optionData.S;
  const real       X = optionData.X;
  const real       T = optionData.T;
  const real       R = optionData.R;
  const real       V = optionData.V;

  const real      dt = T / (real)NUM_STEPS;
  const real     vDt = V * sqrt(dt);
  const real     rDt = R * dt;
  //Per-step interest and discount factors
  const real      If = exp(rDt);
  const real      Df = exp(-rDt);
  //Values and pseudoprobabilities of upward and downward moves
  const real       u = exp(vDt);
  const real       d = exp(-vDt);
  const real      pu = (If - d) / (u - d);
  const real      pd = 1.0 - pu;
  const real  puByDf = pu * Df;
  const real  pdByDf = pd * Df;

  // Compute values at expiration date:
  // call option value at period end is V(T) = S(T) - X
  // if S(T) is greater than X, or zero otherwise.
  // The computation is similar for put options.
  for (int i = 0; i <= NUM_STEPS; i++)
    Call[i] = expiryCallValueCPU(S, X, vDt, i);

  // Walk backwards up binomial tree
  for (int i = NUM_STEPS; i > 0; i--)
    for (int j = 0; j <= i - 1; j++)
      Call[j] = puByDf * Call[j + 1] + pdByDf * Call[j];

  callResult = (real)Call[0];
}

// ---- main.cu body ----

// Helper function, returning uniformly distributed
// random float in [low, high] range
real randData(real low, real high)
{
  real t = (real)rand() / (real)RAND_MAX;
  return ((real)1.0 - t) * low + t * high;
}

int main(int argc, char **argv)
{
  printf("[%s] - Starting...\n", argv[0]);

  const int OPT_N = MAX_OPTIONS;

  TOptionData optionData[MAX_OPTIONS];
  real
    callValueBS[MAX_OPTIONS],
    callValueGPU[MAX_OPTIONS],
    callValueCPU[MAX_OPTIONS];

  real
    sumDelta, sumRef, gpuTime, errorVal;

  int i;

  printf("Generating input data...\n");
  srand(123);

  for (i = 0; i < OPT_N; i++)
  {
    optionData[i].S = randData(5.0f, 30.0f);
    optionData[i].X = randData(1.0f, 100.0f);
    optionData[i].T = randData(0.25f, 10.0f);
    optionData[i].R = 0.06f;
    optionData[i].V = 0.10f;
    BlackScholesCall(callValueBS[i], optionData[i]);
  }

  printf("Running GPU binomial tree...\n");

  auto start = std::chrono::high_resolution_clock::now();

  binomialOptionsGPU(callValueGPU, optionData, OPT_N, NUM_ITERATIONS);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<real> elapsed_seconds = end - start;
  gpuTime = (real)elapsed_seconds.count();

  printf("Options count            : %i     \n", OPT_N);
  printf("Time steps               : %i     \n", NUM_STEPS);
  printf("Total binomialOptionsGPU() time: %f msec\n", gpuTime * 1000);
  printf("Options per second       : %f     \n", OPT_N / (gpuTime));

  printf("Running CPU binomial tree...\n");

  for (i = 0; i < OPT_N; i++)
  {
    binomialOptionsCPU(callValueCPU[i], optionData[i]);
  }

  printf("Comparing the results...\n");
  sumDelta = 0;
  sumRef   = 0;
  printf("GPU binomial vs. Black-Scholes\n");

  for (i = 0; i < OPT_N; i++)
  {
    sumDelta += fabs(callValueBS[i] - callValueGPU[i]);
    sumRef += fabs(callValueBS[i]);
  }

  if (sumRef >1E-5)
  {
    printf("L1 norm: %E\n", (double)(sumDelta / sumRef));
  }
  else
  {
    printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
  }

  printf("CPU binomial vs. Black-Scholes\n");
  sumDelta = 0;
  sumRef   = 0;

  for (i = 0; i < OPT_N; i++)
  {
    sumDelta += fabs(callValueBS[i]- callValueCPU[i]);
    sumRef += fabs(callValueBS[i]);
  }

  if (sumRef >1E-5)
  {
    printf("L1 norm: %E\n", sumDelta / sumRef);
  }
  else
  {
    printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
  }

  printf("CPU binomial vs. GPU binomial\n");
  sumDelta = 0;
  sumRef   = 0;

  for (i = 0; i < OPT_N; i++)
  {
    sumDelta += fabs(callValueGPU[i] - callValueCPU[i]);
    sumRef += callValueCPU[i];
  }

  printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
  printf("L1 norm: %E\n", errorVal = sumDelta / sumRef);

  if (errorVal > 5e-4)
  {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}

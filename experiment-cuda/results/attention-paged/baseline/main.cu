/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_paged_attention.py
 *
 *
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <chrono>
// ---- INLINED: cuda_compat.h (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/cuda_compat.h) ----
#pragma once

#ifdef USE_ROCM

#include <hip/hip_runtime.h>
#define GPU_CHECK(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error %s:%d: %s\n", \
               __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(err); \
    } \
} while (0)

#else

#include <cuda_runtime.h>
#define GPU_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", \
               __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
} while (0)

#endif

#ifdef USE_ROCM
struct Utils {
  static __host__ int get_warp_size() {
    static bool is_cached = false;
    static int result;

    if (!is_cached) {
      int warp_size;
      GPU_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
      result = warp_size;
      is_cached = true;
    }

    return result;
  }

  static __device__ constexpr int get_warp_size() {
  #ifdef __GFX9__
    return 64;
  #else
    return 32;
  #endif
  }
};

  #define WARP_SIZE Utils::get_warp_size()
#else
  #define WARP_SIZE 32
#endif

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor(var, lane_mask, width)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_DOWN_SYNC(var, lane_delta) \
    __shfl_down_sync(uint32_t(-1), var, lane_delta)
#else
  #define VLLM_SHFL_DOWN_SYNC(var, lane_delta) __shfl_down(var, lane_delta)
#endif

#ifndef USE_ROCM
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    GPU_CHECK(cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL))
#else
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    GPU_CHECK(hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL))
#endif

// ---- END INLINED: cuda_compat.h ----

// ---- INLINED: attention_kernels.cuh (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/attention_kernels.cuh) ----
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <algorithm>
// [merge] skipped duplicate include "cuda_compat.h"
// ---- INLINED: attention_dtypes.h (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/attention_dtypes.h) ----
#pragma once
// ---- INLINED: attention_generic.cuh (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/attention_generic.cuh) ----
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <stdint.h>

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE>
struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T>
struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template <typename T>
inline __device__ float sum(T v);

template <typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

template <typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

// ---- END INLINED: attention_generic.cuh ----

// ---- INLINED: dtype_float32.cuh (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/dtype_float32.cuh) ----
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
// [merge] skipped duplicate include "attention_generic.cuh"

#include <stdint.h>

//namespace vllm {

// Define custom FP32 vector data types.
struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

// FP32 vector types for Q, K, V.
template <>
struct Vec<float, 1> {
  using Type = float;
};
template <>
struct Vec<float, 2> {
  using Type = float2;
};
template <>
struct Vec<float, 4> {
  using Type = float4;
};

// FP32 accumulator vector types corresponding to Vec.
template <>
struct FloatVec<float> {
  using Type = float;
};
template <>
struct FloatVec<float2> {
  using Type = float2;
};
template <>
struct FloatVec<float4> {
  using Type = float4;
};

// Vector addition.
inline __device__ float add(float a, float b) { return a + b; }

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

// Vector multiplication.
template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

template <>
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}

// Vector fused multiply-add.
inline __device__ float vfma(float a, float b, float c) { return a * b + c; }

inline __device__ float2 vfma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = vfma(a.x, b.x, c.x);
  d.y = vfma(a.y, b.y, c.y);
  return d;
}

inline __device__ float2 vfma(float a, float2 b, float2 c) {
  float2 d;
  d.x = vfma(a, b.x, c.x);
  d.y = vfma(a, b.y, c.y);
  return d;
}

inline __device__ float4 vfma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = vfma(a.x, b.x, c.x);
  d.y = vfma(a.y, b.y, c.y);
  d.z = vfma(a.z, b.z, c.z);
  d.w = vfma(a.w, b.w, c.w);
  return d;
}

inline __device__ float4 vfma(float a, float4 b, float4 c) {
  float4 d;
  d.x = vfma(a, b.x, c.x);
  d.y = vfma(a, b.y, c.y);
  d.z = vfma(a, b.z, c.z);
  d.w = vfma(a, b.w, c.w);
  return d;
}

inline __device__ Float4_ vfma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = vfma(a, b.x, c.x);
  d.y = vfma(a, b.y, c.y);
  return d;
}

inline __device__ Float8_ vfma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = vfma(a, b.x, c.x);
  d.y = vfma(a, b.y, c.y);
  d.z = vfma(a, b.z, c.z);
  d.w = vfma(a, b.w, c.w);
  return d;
}

// Vector sum.
template <>
inline __device__ float sum(float v) {
  return v;
}

template <>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

template <>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template <>
inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

template <>
inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

// Vector dot product.
inline __device__ float dot(float a, float b) { return a * b; }

inline __device__ float dot(float2 a, float2 b) {
  float2 c = mul<float2, float2, float2>(a, b);
  return c.x + c.y;
}

inline __device__ float dot(Float4_ a, Float4_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = vfma(a.y, b.y, acc);
  return acc.x + acc.y;
}

inline __device__ float dot(Float8_ a, Float8_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = vfma(a.y, b.y, acc);
  acc = vfma(a.z, b.z, acc);
  acc = vfma(a.w, b.w, acc);
  return acc.x + acc.y;
}

// From float to float.
inline __device__ void from_float(float& dst, float src) { dst = src; }

inline __device__ void from_float(float2& dst, float2 src) { dst = src; }

inline __device__ void from_float(float4& dst, float4 src) { dst = src; }

// From float to float.
inline __device__ float to_float(float u) { return u; }

inline __device__ float2 to_float(float2 u) { return u; }

inline __device__ float4 to_float(float4 u) { return u; }

inline __device__ Float4_ to_float(Float4_ u) { return u; }

inline __device__ Float8_ to_float(Float8_ u) { return u; }

// Zero-out a variable.
inline __device__ void zero(float& dst) { dst = 0.f; }

//}  // namespace vllm

// ---- END INLINED: dtype_float32.cuh ----

// ---- INLINED: dtype_bfloat16.cuh (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/dtype_bfloat16.cuh) ----
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
// [merge] skipped duplicate include "attention_generic.cuh"
// [merge] skipped duplicate include "dtype_float32.cuh"

#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16 __nv_bfloat16;
#endif

#include <stdint.h>

//namespace vllm {

// Define custom BF16 vector data types.
struct bf16_4_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
};

struct bf16_8_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
};

// BF16 vector types for Q, K, V.
template <>
struct Vec<__nv_bfloat16, 1> {
  using Type = __nv_bfloat16;
};
template <>
struct Vec<__nv_bfloat16, 2> {
  using Type = __nv_bfloat162;
};
template <>
struct Vec<__nv_bfloat16, 4> {
  using Type = bf16_4_t;
};
template <>
struct Vec<__nv_bfloat16, 8> {
  using Type = bf16_8_t;
};

// FP32 accumulator vector types corresponding to Vec.
template <>
struct FloatVec<__nv_bfloat16> {
  using Type = float;
};
template <>
struct FloatVec<__nv_bfloat162> {
  using Type = float2;
};
template <>
struct FloatVec<bf16_4_t> {
  using Type = Float4_;
};
template <>
struct FloatVec<bf16_8_t> {
  using Type = Float8_;
};

// Utility functions for type conversions.
inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __bfloat1622float2(val);
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __bfloat162bfloat162(val);
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

// Vector addition.
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  #ifndef USE_ROCM
  return a + b;
  #else
  return __hadd(a, b);
  #endif
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hadd2(a, b);
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

inline __device__ bf16_4_t add(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ bf16_8_t add(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ float2 add(__nv_bfloat162 a, float2 fb) {
  float2 fa = bf1622float2(a);
  return add(fa, fb);
}

inline __device__ Float4_ add(bf16_4_t a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline __device__ Float8_ add(bf16_8_t a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

// Vector multiplication.
template <>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hmul(a, b);
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hmul2(a, b);
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat16 a, __nv_bfloat162 b) {
  return mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

template <>
inline __device__ bf16_4_t mul(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return c;
}

template <>
inline __device__ bf16_4_t mul(__nv_bfloat16 a, bf16_4_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  return c;
}

template <>
inline __device__ bf16_8_t mul(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return c;
}

template <>
inline __device__ bf16_8_t mul(__nv_bfloat16 a, bf16_8_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.z);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.w);
  return c;
}

template <>
inline __device__ float mul(__nv_bfloat16 a, __nv_bfloat16 b) {
  float fa = __bfloat162float(a);
  float fb = __bfloat162float(b);
  return fa * fb;
}

template <>
inline __device__ float2 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
  float2 fa = bf1622float2(a);
  float2 fb = bf1622float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float2 mul(__nv_bfloat16 a, __nv_bfloat162 b) {
  return mul<float2, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

template <>
inline __device__ Float4_ mul(bf16_4_t a, bf16_4_t b) {
  Float4_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return fc;
}

template <>
inline __device__ Float4_ mul(__nv_bfloat16 a, bf16_4_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  Float4_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  return fc;
}

template <>
inline __device__ Float8_ mul(bf16_8_t a, bf16_8_t b) {
  Float8_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return fc;
}

template <>
inline __device__ Float8_ mul(__nv_bfloat16 a, bf16_8_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  Float8_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.z);
  fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.w);
  return fc;
}

// Vector fused multiply-add.
inline __device__ __nv_bfloat162 vfma(__nv_bfloat162 a, __nv_bfloat162 b,
                                     __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hfma2(a, b, c);
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

inline __device__ __nv_bfloat162 vfma(__nv_bfloat16 a, __nv_bfloat162 b,
                                     __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  return __hfma2(bf162bf162(a), b, c);
#endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

inline __device__ bf16_4_t vfma(bf16_4_t a, bf16_4_t b, bf16_4_t c) {
  bf16_4_t d;
  d.x = vfma(a.x, b.x, c.x);
  d.y = vfma(a.y, b.y, c.y);
  return d;
}

inline __device__ bf16_4_t vfma(__nv_bfloat16 a, bf16_4_t b, bf16_4_t c) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_4_t d;
  d.x = vfma(s, b.x, c.x);
  d.y = vfma(s, b.y, c.y);
  return d;
}

inline __device__ bf16_8_t vfma(bf16_8_t a, bf16_8_t b, bf16_8_t c) {
  bf16_8_t d;
  d.x = vfma(a.x, b.x, c.x);
  d.y = vfma(a.y, b.y, c.y);
  d.z = vfma(a.z, b.z, c.z);
  d.w = vfma(a.w, b.w, c.w);
  return d;
}

inline __device__ bf16_8_t vfma(__nv_bfloat16 a, bf16_8_t b, bf16_8_t c) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_8_t d;
  d.x = vfma(s, b.x, c.x);
  d.y = vfma(s, b.y, c.y);
  d.z = vfma(s, b.z, c.z);
  d.w = vfma(s, b.w, c.w);
  return d;
}

inline __device__ float vfma(__nv_bfloat16 a, __nv_bfloat16 b, float fc) {
  return __bfloat162float(a) * __bfloat162float(b) + fc;
}

inline __device__ float2 vfma(__nv_bfloat162 a, __nv_bfloat162 b, float2 fc) {
  float2 fa = bf1622float2(a);
  float2 fb = bf1622float2(b);
  return vfma(fa, fb, fc);
}

inline __device__ float2 vfma(__nv_bfloat16 a, __nv_bfloat162 b, float2 fc) {
  return vfma(bf162bf162(a), b, fc);
}

inline __device__ Float4_ vfma(bf16_4_t a, bf16_4_t b, Float4_ fc) {
  Float4_ fd;
  fd.x = vfma(a.x, b.x, fc.x);
  fd.y = vfma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ vfma(__nv_bfloat16 a, bf16_4_t b, Float4_ fc) {
  __nv_bfloat162 s = bf162bf162(a);
  Float4_ fd;
  fd.x = vfma(s, b.x, fc.x);
  fd.y = vfma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ vfma(bf16_8_t a, bf16_8_t b, Float8_ fc) {
  Float8_ fd;
  fd.x = vfma(a.x, b.x, fc.x);
  fd.y = vfma(a.y, b.y, fc.y);
  fd.z = vfma(a.z, b.z, fc.z);
  fd.w = vfma(a.w, b.w, fc.w);
  return fd;
}

inline __device__ Float8_ vfma(__nv_bfloat16 a, bf16_8_t b, Float8_ fc) {
  __nv_bfloat162 s = bf162bf162(a);
  Float8_ fd;
  fd.x = vfma(s, b.x, fc.x);
  fd.y = vfma(s, b.y, fc.y);
  fd.z = vfma(s, b.z, fc.z);
  fd.w = vfma(s, b.w, fc.w);
  return fd;
}

// Vector sum.
template <>
inline __device__ float sum(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
inline __device__ float sum(__nv_bfloat162 v) {
  float2 vf = bf1622float2(v);
  return vf.x + vf.y;
}

template <>
inline __device__ float sum(bf16_4_t v) {
  return sum(v.x) + sum(v.y);
}

template <>
inline __device__ float sum(bf16_8_t v) {
  return sum(v.x) + sum(v.y) + sum(v.z) + sum(v.w);
}

// From float32 to bfloat16.
inline __device__ void from_float(__nv_bfloat16& dst, float src) {
  dst = __float2bfloat16(src);
}

inline __device__ void from_float(__nv_bfloat162& dst, float2 src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst = __float22bfloat162_rn(src);
#endif
}

inline __device__ void from_float(bf16_4_t& dst, Float4_ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
#endif
}

inline __device__ void from_float(bf16_8_t& dst, Float8_ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
  dst.z = __float22bfloat162_rn(src.z);
  dst.w = __float22bfloat162_rn(src.w);
#endif
}

// From bfloat16 to float32.
inline __device__ float to_float(__nv_bfloat16 u) {
  return __bfloat162float(u);
}

// Zero-out a variable.
inline __device__ void zero(__nv_bfloat16& dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
#else
  // Same as CUDART_ZERO_BF16 introduced in CUDA 12.2.
  dst = __ushort_as_bfloat16((unsigned short)0x0000U);
#endif
}

//}  // namespace vllm

// ---- END INLINED: dtype_bfloat16.cuh ----


// ---- END INLINED: attention_dtypes.h ----

// ---- INLINED: attention_utils.cuh (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/attention_utils.cuh) ----
/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
// [merge] skipped duplicate include "cuda_compat.h"
// [merge] skipped duplicate include "attention_dtypes.h"

#include <float.h>
#include <type_traits>

// Q*K^T operation.
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = vfma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += VLLM_SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// ---- END INLINED: attention_utils.cuh ----



#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, bool IS_BLOCK_SPARSE, int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,// [num_seqs, num_heads, // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions, // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step)
{
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];

  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);

  static_assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;

  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  //using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;

  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) * (-blocksparse_head_sliding_step) + 1;
  }

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  //using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
          !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
        continue;
      }
    }
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx));

    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]

    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step)
{
    paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, IS_BLOCK_SPARSE>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, block_tables, seq_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP

// ---- END INLINED: attention_kernels.cuh ----

// ---- INLINED: kvcache.h (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/kvcache.h) ----
/*
def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    set_random_seed(seed)

    dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches
struct KVCaches {
    void** key_caches;    // array of device pointers
    void** value_caches;
};
*/


__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ float uint32_to_uniform(uint32_t x) {
    // Convert to (0,1)
    return (x >> 8) * 0x1.0p-24f;
}

__device__ __forceinline__ float normal_from_uniform( uint32_t& rng) {
    float u1 = uint32_to_uniform(xorshift32(rng));
    float u2 = uint32_to_uniform(xorshift32(rng));

    // Box–Muller
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 6.28318530718f * u2;
    return r * cosf(theta);           // N(0,1)
}

template <typename T>
__global__ void uniform_fill_kernel(
    T* data,
    int64_t n,
    float low,
    float high,
    uint32_t seed
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t rng = seed ^ idx;

    uint32_t r = xorshift32(rng);
    float u = uint32_to_uniform(r);
    float v = low + (high - low) * u;
    data[idx] = (T)v;
}

template <typename T>
__global__ void norm_fill_kernel(
    T* data,
    int n,
    uint32_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t rng = seed ^ idx;
    data[idx] = (T)normal_from_uniform(rng);
}

template<typename T>
struct KVCaches {
    T** key_caches;
    T** value_caches;
};

template <typename T>
KVCaches<T> create_kv_caches_with_random(
    int num_blocks,
    int block_size,
    int num_layers,
    int num_heads,
    int head_size,
    unsigned long seed,
    int &kv_block_stride,
    int &kv_head_stride
) {
    float scale = std::pow((float)head_size, -0.5f);

    int element_size = sizeof(T);
    int x = 16 / element_size;

    //printf("Key cache shape:\n");
    //printf("[%d %d %d %d %d]\n", num_blocks, num_heads, head_size/x, block_size, x);
    int64_t key_elems = (int64_t) num_blocks * num_heads * (head_size / x) * block_size * x;
    kv_block_stride = key_elems / num_blocks;
    kv_head_stride = kv_block_stride / num_heads;

    //printf("Value cache shape:\n");
    //printf("[%d %d %d %d]\n", num_blocks, num_heads, head_size, block_size);
    int64_t value_elems = (int64_t)num_blocks * num_heads * head_size * block_size;

    T** key_caches_h = (T**)malloc(num_layers * sizeof(T*));
    T** value_caches_h = (T**)malloc(num_layers * sizeof(T*));

    int threads = 256;

    for (int l = 0; l < num_layers; ++l) {
        T* key_cache_d;
        T* value_cache_d;
        GPU_CHECK(cudaMalloc(&key_cache_d, key_elems * element_size));
        GPU_CHECK(cudaMalloc(&value_cache_d, value_elems * element_size));

        int64_t key_blocks = (key_elems + threads - 1) / threads;
        int64_t val_blocks = (value_elems + threads - 1) / threads;

        uniform_fill_kernel<T>
            <<<key_blocks, threads>>>(
                (T*)key_cache_d,
                key_elems,
                -scale,
                scale,
                seed + l);

        uniform_fill_kernel<T>
            <<<val_blocks, threads>>>(
                (T*)value_cache_d,
                value_elems,
                -scale,
                scale,
                seed + l);

        key_caches_h[l] = key_cache_d;
        value_caches_h[l] = value_cache_d;
    }

    KVCaches<T> out;
    out.key_caches = key_caches_h;
    out.value_caches = value_caches_h;
    return out;
}

// ---- END INLINED: kvcache.h ----

// ---- INLINED: reference.h (from /home/WillFu/parallel/final/HeCBench/src/attention-paged-cuda/reference.h) ----
// a modified reference generated by Gemini

#include <cmath>
#include <vector>
#include <algorithm>

template<typename T>
struct PagedAttentionParams {
    T* out;                // [num_seqs, num_query_heads, head_size]
    const T* query;        // [num_seqs, num_query_heads, head_size]
    const T* key_cache;    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const T* value_cache;  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* block_tables; 
    const int* seq_lens;
    const float* alibi_slopes;
    
    int num_seqs;
    int num_query_heads;
    int num_kv_heads;
    int head_size;
    int block_size;
    int max_num_blocks_per_seq;
    float scale;
};

void softmax(float* scores, int length) {
    float max_val = -INFINITY;
    for (int i = 0; i < length; ++i) max_val = std::max(max_val, scores[i]);
    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        scores[i] = std::exp(scores[i] - max_val);
        sum += scores[i];
    }
    for (int i = 0; i < length; ++i) scores[i] /= sum;
}

template<typename T>
void reference(PagedAttentionParams<T> params) {
    int queries_per_kv = params.num_query_heads / params.num_kv_heads;

    for (int64_t i = 0; i < params.num_seqs; ++i) {
        int seq_len = params.seq_lens[i];
        const int* block_table = params.block_tables + i * params.max_num_blocks_per_seq;

        for (int h = 0; h < params.num_query_heads; ++h) {
            int kv_h = h / queries_per_kv;
            std::vector<float> logits(seq_len);

            // 1. Compute Dot Product (Q * K) + ALiBi
            for (int t = 0; t < seq_len; ++t) {
                int64_t block_idx = block_table[t / params.block_size];
                int block_offset = t % params.block_size;

                float score = 0.0f;
                for (int d = 0; d < params.head_size; ++d) {
                    // Accessing Key Cache [block, head, dim, offset]
                    // Note: Layout depends on specific PagedAttention implementation
                    T q_val = params.query[i * params.num_query_heads * params.head_size + h * params.head_size + d];
                    T k_val = params.key_cache[block_idx * params.num_kv_heads * params.head_size * params.block_size + 
                                               kv_h * params.head_size * params.block_size + 
                                               d * params.block_size + block_offset];
                    score += (float)q_val * (float)k_val;
                }
                score *= params.scale;

                // Apply ALiBi Bias
                if (params.alibi_slopes != nullptr) {
                    float slope = params.alibi_slopes[h];
                    float bias = slope * (float)(t - seq_len + 1);
                    score += bias;
                }
                logits[t] = score;
            }

            // 2. Softmax
            softmax(logits.data(), seq_len);

            // 3. Weighted Sum (Logits * V)
            for (int d = 0; d < params.head_size; ++d) {
                float res = 0.0f;
                for (int t = 0; t < seq_len; ++t) {
                    int64_t block_idx = block_table[t / params.block_size];
                    int block_offset = t % params.block_size;

                    T v_val = params.value_cache[block_idx * params.num_kv_heads * params.head_size * params.block_size + 
                                                 kv_h * params.head_size * params.block_size + 
                                                 d * params.block_size + block_offset];
                    res += logits[t] * (float)v_val;
                }
                params.out[i * params.num_query_heads * params.head_size + h * params.head_size + d] = (T)res;
            }
        }
    }
}

// ---- END INLINED: reference.h ----



#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                          \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(               \
      ((void*)paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,\
                                              IS_BLOCK_SPARSE>),      \
      shared_mem_size);                                               \
  paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, IS_BLOCK_SPARSE>       \
      <<<grid, block, shared_mem_size, 0>>>(                          \
          out, query, key_cache, value_cache, num_kv_heads,           \
          scale, block_tables, seq_lens, max_num_blocks_per_seq,      \
          alibi_slopes, q_stride, kv_block_stride, kv_head_stride,    \
          tp_rank, blocksparse_local_blocks,                          \
          blocksparse_vert_stride, blocksparse_block_size,            \
          blocksparse_head_sliding_step);

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          bool IS_BLOCK_SPARSE, int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    T *out,
    T *query,
    CACHE_T* key_cache,
    CACHE_T* value_cache,
    int num_kv_heads,
    float scale,
    int *block_tables,
    int *seq_lens,
    int max_seq_len,
    float *alibi_slopes,
    const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step,
    const int num_seqs,
    const int num_heads,
    const int head_size,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride)
{
  const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len = (max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 32:
      LAUNCH_PAGED_ATTENTION_V1(32);
      break;
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 120:
      LAUNCH_PAGED_ATTENTION_V1(120);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      printf("Error: unsupported head size: %d\n", head_size);
      break;
  }
}

template <typename T, int block_size = 16> 
void attention_page (int num_seqs, 
                     int num_query_heads,
                     int num_kv_heads,
                     int head_size,
                     int max_seq_len,
                     int num_blocks,
                     int repeat)
{
    printf("kv cache block size = %d\n", block_size);

    const unsigned long seed = 1234;
    const bool use_alibi = true;

    assert(num_query_heads % num_kv_heads == 0);
    //int num_queries_per_kv = num_query_heads / num_kv_heads;

    // note kscale and vscale are not used
    const float scale = 1.f / std::sqrt((float)head_size);

    // Allocate query
    const int query_elems = num_seqs * num_query_heads * head_size;
    const int q_stride = num_query_heads * head_size; // query.stride(0)

    T* query_d;
    GPU_CHECK(cudaMalloc(&query_d, query_elems * sizeof(T)));

    T* query_h = (T*) malloc (query_elems * sizeof(T));

    const int threads = 256;
    int blocks = (query_elems + threads - 1) / threads;

    uniform_fill_kernel<T><<<blocks, threads>>>(query_d, query_elems, -scale, scale, seed);
    GPU_CHECK(cudaMemcpy(query_h, query_d, sizeof(T) * query_elems, cudaMemcpyDeviceToHost));

    // Allocate outputs
    T *out_d;
    GPU_CHECK(cudaMalloc(&out_d, query_elems * sizeof(T)));

    T *out_h = (T*) malloc (query_elems * sizeof(T));
    T *out_r = (T*) malloc (query_elems * sizeof(T));

    // ALiBi slopes
    float* alibi_d = nullptr;
    float* alibi_h = nullptr;
    if (use_alibi) {
        GPU_CHECK(cudaMalloc(&alibi_d, num_query_heads * sizeof(float)));
        alibi_h = (float*) malloc (num_query_heads * sizeof(float));

        blocks = (num_query_heads + threads - 1) / threads;
        norm_fill_kernel<float><<<blocks, threads>>>( alibi_d, num_query_heads, seed);
        GPU_CHECK(cudaMemcpy(alibi_h, alibi_d, sizeof(float) * num_query_heads, cudaMemcpyDeviceToHost));
    }

    // seq_lens
    int* seq_lens_h = (int*)malloc(num_seqs * sizeof(int));

    srand(seed);
    for (int i = 0; i < num_seqs; i++) {
        seq_lens_h[i] = rand() % max_seq_len + 1; // variable length [1, max_seq_len]
    }

    int* seq_lens_d;
    GPU_CHECK(cudaMalloc(&seq_lens_d, num_seqs * sizeof(int)));
    GPU_CHECK(cudaMemcpy( seq_lens_d, seq_lens_h, num_seqs * sizeof(int), cudaMemcpyHostToDevice));

    // Block tables
    const int max_num_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;
    const int block_tables_size = num_seqs * max_num_blocks_per_seq;

    int* block_tables_h = (int*)malloc(block_tables_size * sizeof(int));

    for (int s = 0; s < num_seqs; s++) {
        for (int b = 0; b < max_num_blocks_per_seq; b++) {
            block_tables_h[s * max_num_blocks_per_seq + b] = rand() % num_blocks; // [0, num_blocks-1]
        }
    }

    int* block_tables_d;
    GPU_CHECK(cudaMalloc(&block_tables_d, block_tables_size * sizeof(int)));
    GPU_CHECK(cudaMemcpy(block_tables_d, block_tables_h, block_tables_size * sizeof(int), cudaMemcpyHostToDevice));

    // KV cache allocation
    int num_layers = 1;
    int kv_block_stride, kv_head_stride;

    auto caches = create_kv_caches_with_random<T>(
      num_blocks,
      block_size,
      num_layers,
      num_kv_heads,
      head_size,
      seed,
      kv_block_stride,
      kv_head_stride
    );

    // device points
    auto key_cache_d = caches.key_caches[0];
    auto value_cache_d = caches.value_caches[0];

    int64_t kv_elems = (int64_t)num_blocks * num_kv_heads * head_size * block_size;
    T *key_cache_h = (T*) malloc (sizeof(T) * kv_elems);
    T *value_cache_h = (T*) malloc (sizeof(T) * kv_elems);
    GPU_CHECK(cudaMemcpy(key_cache_h, key_cache_d, sizeof(T) * kv_elems, cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(value_cache_h, value_cache_d, sizeof(T) * kv_elems, cudaMemcpyDeviceToHost));

    GPU_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      paged_attention_v1_launcher<T, T, block_size, false>(
        out_d,
        query_d,
        key_cache_d,
        value_cache_d,
        num_kv_heads,
        scale,
        block_tables_d,
        seq_lens_d,
        max_seq_len,
        alibi_d,
        0, //const int tp_rank,
        0, 0, //const int blocksparse_local_blocks, const int blocksparse_vert_stride,
        0, 0, //const int blocksparse_block_size, const int blocksparse_head_sliding_step,
        num_seqs,
        num_kv_heads,
        head_size,
        max_num_blocks_per_seq,
        q_stride,
        kv_block_stride,
        kv_head_stride
      );
    }

    GPU_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

    GPU_CHECK(cudaMemcpy(out_h, out_d, sizeof(T) * query_elems, cudaMemcpyDeviceToHost));

    //printf("Running PagedAttention CPU Reference...\n");
    PagedAttentionParams<T> params;
    params.out = out_r;
    params.query = query_h;
    params.key_cache = key_cache_h;
    params.value_cache = value_cache_h;
    params.block_tables = block_tables_h;
    params.seq_lens = seq_lens_h;
    params.alibi_slopes = alibi_h;
    params.num_seqs = num_seqs;
    params.num_query_heads = num_query_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_size = head_size;
    params.block_size = block_size;
    params.max_num_blocks_per_seq = max_num_blocks_per_seq;
    params.scale = scale;

    reference(params);
   
    const float atol = 1e-3f, rtol = 1e-5f;
    bool ok = true;
    for (int i = 0; i < query_elems; i++) {
      if (std::fabs(float(out_h[i] - out_r[i])) > 
          atol + rtol * std::fabs((float)out_r[i])) {
        printf("Mismatch at index %d: %f %f\n", i, (float)out_h[i], (float)out_r[i]);
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    free(query_h);
    free(out_h);
    free(out_r);
    free(seq_lens_h);
    free(block_tables_h);
    free(key_cache_h);
    free(value_cache_h);
    GPU_CHECK(cudaFree(query_d));
    GPU_CHECK(cudaFree(seq_lens_d));
    GPU_CHECK(cudaFree(block_tables_d));
    if (alibi_d) {
      GPU_CHECK(cudaFree(alibi_d));
      free(alibi_h);
    }

    for (int l = 0; l < num_layers; ++l) {
      GPU_CHECK(cudaFree(caches.key_caches[l]));
      GPU_CHECK(cudaFree(caches.value_caches[l]));
    }
    free(caches.key_caches);
    free(caches.value_caches);
}

int main(int argc, char* argv[])
{
   if (argc != 7) {
     printf("Usage: %s <batch size> <number of query heads> <head size> ", argv[0]);
     printf("<max_seq_len> <number of cache blocks> <repeat>\n");
     printf("head size choices [32, 64, 80, 96, 112, 120, 128, 192, 256]\n");
     return 1;
   }
    int num_seqs = atoi(argv[1]);        // batch size 8
    int num_query_heads = atoi(argv[2]); //32, 64
    int num_kv_heads = num_query_heads;  // there exist bugs in the vllm kernel when num_kv_heads and num_query_heads diff 
    int head_size = atoi(argv[3]);       // choices=[32, 64, 80, 96, 112, 120, 128, 192, 256],
    int max_seq_len = atoi(argv[4]);     // choices=[4096]
    int num_blocks = atoi(argv[5]);      //128*1024
    int repeat = atoi(argv[6]);

    printf("query shape = [%d (num_seqs), %d (num_query_heads), %d (head_size)]\n",
           num_seqs, num_query_heads, head_size);
    printf("Number of kv heads = %d\n", num_kv_heads);
    printf("Number of kv cache blocks = %d\n", num_blocks);
    printf("max_seq_len = %d\n", max_seq_len);
    
    //const int block_size = 16; // choices=[16, 32], a template parameter for paged_attention_v1_launcher
                               // 32 may cause OOM in FP32 kvcache allocation

    
    printf("\n-------------------\nFP32 PageAttention v1\n--------------------\n");
    attention_page<float, 16>(num_seqs, num_query_heads,
      num_kv_heads, head_size, max_seq_len, num_blocks, repeat);

    printf("\n-------------------\nBF16 PageAttention v1\n--------------------\n");
    attention_page<__nv_bfloat16, 32>(num_seqs, num_query_heads,
      num_kv_heads, head_size, max_seq_len, num_blocks, repeat);

    return 0;
}

/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba,
 *        University of Illinois nor the names of its contributors may be used
 *        to endorse or promote products derived from this Software without
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

// System includes (deduplicated)
#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>
#include <chrono>
#include <atomic>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cuda.h>

// Local dep includes (kept as separate dep files).
// Include support/partitioner.h WITHOUT _GPU_COMPILER_ so that the full
// Partitioner struct (with thread_id, n_threads) and the CPU partitioner
// functions are available throughout the entire translation unit.
#include "support/setup.h"
#include "support/common.h"
#include "support/verify.h"
#include "support/partitioner.h"

// ---- device_sc.cu body (inlined) ----
// Define the GPU-side partitioner iterators here, since they were skipped
// when partitioner.h was included without _GPU_COMPILER_.
// These mirror the #ifdef _GPU_COMPILER_ sections of partitioner.h exactly.

__device__ inline int gpu_first(Partitioner *p) {
#ifdef DYNAMIC_PARTITION
    if(p->strategy == DYNAMIC_PARTITIONING) {
        if(threadIdx.y == 0 && threadIdx.x == 0) {
            p->tmp[0] = atomicAdd_system(p->worklist, 1);
        }
        __syncthreads();
        p->current = p->tmp[0];
    } else
#endif
    {
        p->current = p->cut + blockIdx.x;
    }
    return p->current;
}

__device__ inline bool gpu_more(const Partitioner *p) {
    return (p->current < p->n_tasks);
}

__device__ inline int gpu_next(Partitioner *p) {
#ifdef DYNAMIC_PARTITION
    if(p->strategy == DYNAMIC_PARTITIONING) {
        if(threadIdx.y == 0 && threadIdx.x == 0) {
            p->tmp[0] = atomicAdd_system(p->worklist, 1);
        }
        __syncthreads();
        p->current = p->tmp[0];
    } else
#endif
    {
        p->current = p->current + gridDim.x;
    }
    return p->current;
}

// Define the GPU version of partitioner_create (2-arg, __device__ overload)
__device__ inline Partitioner partitioner_create_gpu(int n_tasks, float alpha
#ifdef DYNAMIC_PARTITION
    , int *worklist
    , int *tmp
#endif
    ) {
    Partitioner p;
    p.n_tasks = n_tasks;
    if(alpha >= 0.0 && alpha <= 1.0) {
        p.cut = p.n_tasks * alpha;
#ifdef DYNAMIC_PARTITION
        p.strategy = STATIC_PARTITIONING;
#endif
    } else {
#ifdef DYNAMIC_PARTITION
        p.strategy = DYNAMIC_PARTITIONING;
        p.worklist = worklist;
        p.tmp = tmp;
#endif
    }
    return p;
}

// Device auxiliary functions
__attribute__((weak)) __device__ void reduce(int *l_count, int local_cnt, int *l_data) {
  const int tid       = threadIdx.x;
  const int localSize = blockDim.x;
  // load shared mem
  l_data[tid] = local_cnt;
  __syncthreads();
  // do reduction in shared mem
  for(int s = localSize >> 1; s > 0; s >>= 1) {
    if(tid < s) {
      l_data[tid] += l_data[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if(tid == 0)
    *l_count = l_data[0];
}

__attribute__((weak)) __device__ int block_binary_prefix_sums(int *l_count, int x, int *l_data) {
  l_data[threadIdx.x] = x;
  const int length     = blockDim.x;
  // Build up tree
  int offset = 1;
  for(int l = length >> 1; l > 0; l >>= 1) {
    __syncthreads();
    if(threadIdx.x < l) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;
      l_data[bi] += l_data[ai];
    }
    offset <<= 1;
  }
  if(offset < length) {
    offset <<= 1;
  }
  // Build down tree
  int maxThread = offset >> 1;
  for(int d = 0; d < maxThread; d <<= 1) {
    d += 1;
    offset >>= 1;
    __syncthreads();
    if(threadIdx.x < d) {
      int ai = offset * (threadIdx.x + 1) - 1;
      int bi = ai + (offset >> 1);
      l_data[bi] += l_data[ai];
    }
  }
  __syncthreads();
  int output = l_data[threadIdx.x] + *l_count - x;
  __syncthreads();
  if(threadIdx.x == blockDim.x - 1)
    *l_count += l_data[threadIdx.x];

  return output;
}

__attribute__((weak)) __global__
void StreamCompaction (int size, T value, int n_tasks, float alpha,
                       T *__restrict__ output,
                       const T *__restrict__ input,
                       int *__restrict__ flags
#ifdef DYNAMIC_PARTITION
                       , int *__restrict__ worklist
#endif
    ) {

  extern __shared__ int l_mem[];
  int* l_data = l_mem;
  int* l_count = &l_data[blockDim.x];
#ifdef DYNAMIC_PARTITION
  int* l_tmp = &l_count[1];
#endif

#ifdef DYNAMIC_PARTITION
  Partitioner p = partitioner_create_gpu(n_tasks, alpha, worklist, l_tmp);
#else
  Partitioner p = partitioner_create_gpu(n_tasks, alpha);
#endif

  for(int my_s = gpu_first(&p); gpu_more(&p); my_s = gpu_next(&p)) {

    if(threadIdx.x == 0) {
      l_count[0] = 0;
    }
    __syncthreads();

    int local_cnt = 0;
    // Declare on-chip memory
    T reg[REGS];
#ifdef DYNAMIC_PARTITION
    int pos = my_s * REGS * blockDim.x + threadIdx.x;
#else
    int pos = (my_s - p.cut) * REGS * blockDim.x + threadIdx.x;
#endif
    // Load in on-chip memory
#pragma unroll
    for(int j = 0; j < REGS; j++) {
      if(pos < size) {
        reg[j] = input[pos];
        if(reg[j] != value)
          local_cnt++;
      } else
        reg[j] = value;
      pos += blockDim.x;
    }
    reduce(&l_count[0], local_cnt, &l_data[0]);

    // Set global synch
    if(threadIdx.x == 0) {
      int p_count;
#ifdef DYNAMIC_PARTITION
      while((p_count = atomicAdd_system(&flags[my_s], 0)) == 0) {
      }
      atomicAdd_system(&flags[my_s + 1], p_count + l_count[0]);
#else
      while((p_count = atomicAdd(&flags[my_s], 0)) == 0) {
      }
      atomicAdd(&flags[my_s + 1], p_count + l_count[0]);
#endif
      l_count[0] = p_count - 1;
    }
    __syncthreads();

    // Store to global memory
#pragma unroll
    for(int j = 0; j < REGS; j++) {
      pos = block_binary_prefix_sums(&l_count[0], (int)((reg[j] != value) ? 1 : 0), &l_data[0]);
      if(reg[j] != value) {
        output[pos] = reg[j];
      }
    }
  }
}

__attribute__((weak))
void call_StreamCompaction_kernel(int blocks, int threads, int size, T value, int n_tasks, float alpha,
    T *output, T *input, int *flags, int l_mem_size
#ifdef DYNAMIC_PARTITION
    , int *worklist
#endif
    ){
  dim3 dimGrid(blocks);
  dim3 dimBlock(threads);
  StreamCompaction <<<dimGrid, dimBlock, l_mem_size>>>(size, value, n_tasks, alpha, output, input, flags
#ifdef DYNAMIC_PARTITION
      , worklist
#endif
      );
}

// ---- host_sc.cpp body (inlined) ----

// CPU threads
__attribute__((weak))
void run_cpu_threads(T *output, T *input, std::atomic_int *flags, int size, int value, int n_threads, int ldim,
    int n_tasks, float alpha
#ifdef DYNAMIC_PARTITION
    , std::atomic_int *worklist
#endif
    ) {

    const int REGS_CPU = REGS * ldim;
    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {
        cpu_threads.push_back(std::thread([=]() {

#ifdef DYNAMIC_PARTITION
            Partitioner p = partitioner_create(n_tasks, alpha, i, n_threads, worklist);
#else
            Partitioner p = partitioner_create(n_tasks, alpha, i, n_threads);
#endif

            for(int my_s = cpu_first(&p); cpu_more(&p); my_s = cpu_next(&p)) {

                int l_count = 0;
                // Declare on-chip memory
                T   reg[REGS_CPU];
                int pos = my_s * REGS_CPU;
// Load in on-chip memory
#pragma unroll
                for(int j = 0; j < REGS_CPU; j++) {
                    if(pos < size) {
                        reg[j] = input[pos];
                        if(reg[j] != value)
                            l_count++;
                    } else
                        reg[j] = value;
                    pos++;
                }

                // Set global synch
                int p_count;
                while((p_count = (&flags[my_s])->load()) == 0) {
                }
                (&flags[my_s + 1])->fetch_add(p_count + l_count);
                l_count = p_count - 1;

                // Store to global memory
                pos = l_count;
#pragma unroll
                for(int j = 0; j < REGS_CPU; j++) {
                    if(reg[j] != value) {
                        output[pos] = reg[j];
                        pos++;
                    }
                }
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}

// ---- main.cu body ----

// Params
struct Params {

    int   device;
    int   n_gpu_threads;
    int   n_gpu_blocks;
    int   n_threads;
    int   n_warmup;
    int   n_reps;
    float alpha;
    int   in_size;
    int   compaction_factor;
    int   remove_value;

    Params(int argc, char **argv) {
        n_gpu_threads     = 256;
        n_gpu_blocks      = 1024;
        n_threads         = 4;
        n_warmup          = 5;
        n_reps            = 100;
        alpha             = 0.1;
        in_size           = 8388608;
        compaction_factor = 50;
        remove_value      = 0;
        int opt;
        while((opt = getopt(argc, argv, "hi:g:t:w:r:a:n:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i': n_gpu_threads     = atoi(optarg); break;
            case 'g': n_gpu_blocks      = atoi(optarg); break;
            case 't': n_threads         = atoi(optarg); break;
            case 'w': n_warmup          = atoi(optarg); break;
            case 'r': n_reps            = atoi(optarg); break;
            case 'a': alpha             = atof(optarg); break;
            case 'n': in_size           = atoi(optarg); break;
            case 'c': compaction_factor = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
#ifdef DYNAMIC_PARTITION
            assert((n_gpu_threads > 0 && n_gpu_blocks > 0 || n_threads > 0) && "Invalid # of host + device workers!");
#else
            assert(0 && "Illegal value for -a");
#endif
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./sc [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -i <I>    # of device threads per block (default=256)"
                "\n    -g <G>    # of device blocks (default=1024)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=100)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.1)"
#ifdef DYNAMIC_PARTITION
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
#else
                "\n              NOTE: <A> must be between 0.0 and 1.0"
#endif
                "\n"
                "\nBenchmark-specific options:"
                "\n    -n <N>    input size (default=8388608)"
                "\n    -c <C>    compaction factor (default=50)"
                "\n");
    }
};

// Input Data
void read_input(T *input, const Params &p) {

    // Initialize the host input vectors
    srand(123);
    for(int i = 0; i < p.in_size; i++) {
        input[i] = (T)p.remove_value;
    }
    int M = (p.in_size * p.compaction_factor) / 100;
    int m = M;
    while(m > 0) {
        int x = (int)(p.in_size * (((float)rand() / (float)RAND_MAX)));
        if(x < p.in_size)
            if(input[x] == p.remove_value) {
                input[x] = (T)(x + 2);
                m--;
            }
    }
}

int main(int argc, char **argv) {

    const Params p(argc, argv);

    // Allocate buffers
    const int n_tasks     = divceil(p.in_size, p.n_gpu_threads * REGS);
    const int n_tasks_cpu = n_tasks * p.alpha;
    const int n_tasks_gpu = n_tasks - n_tasks_cpu;
    const int n_flags     = n_tasks + 1;
#ifdef DYNAMIC_PARTITION
    T * h_in_out;
    cudaMallocManaged(&h_in_out, p.in_size * sizeof(T));
    T * d_in_out = h_in_out;
    std::atomic_int *h_flags;
    cudaMallocManaged(&h_flags, n_flags * sizeof(std::atomic_int));
    std::atomic_int *d_flags  = h_flags;
    std::atomic_int * worklist;
    cudaMallocManaged(&worklist, sizeof(std::atomic_int));
#else
    T *    h_in_out = (T *)malloc(n_tasks * p.n_gpu_threads * REGS * sizeof(T));
    T *    d_in_out;
    cudaMalloc((void**)&d_in_out, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T));
    std::atomic_int *h_flags = (std::atomic_int *)malloc(n_flags * sizeof(std::atomic_int));
    int* d_flags;
    cudaMalloc((void**)&d_flags, n_flags * sizeof(int));
    ALLOC_ERR(h_in_out, h_flags);
#endif

    T *h_in_backup = (T *)malloc(p.in_size * sizeof(T));
    ALLOC_ERR(h_in_backup);

    // Initialize
    const int max_gpu_threads = 256;
    read_input(h_in_out, p);
#ifdef DYNAMIC_PARTITION
    h_flags[0].store(1);
#else
    h_flags[0]           = 1;
    h_flags[n_tasks_cpu] = 1;
#endif
    memcpy(h_in_backup, h_in_out, p.in_size * sizeof(T)); // Backup for reuse across iterations

#ifndef DYNAMIC_PARTITION
    // Copy to device
    cudaMemcpy(d_in_out, h_in_out + n_tasks_cpu * p.n_gpu_threads * REGS, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, n_flags * sizeof(int), cudaMemcpyHostToDevice);
#endif

    auto start = std::chrono::steady_clock::now();

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memcpy(h_in_out, h_in_backup, p.in_size * sizeof(T));
        memset(h_flags, 0, n_flags * sizeof(std::atomic_int));
#ifdef DYNAMIC_PARTITION
        h_flags[0].store(1);
        if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
            worklist[0].store(0);
        }
#else
        h_flags[0]           = 1;
        h_flags[n_tasks_cpu] = 1;
        cudaMemcpy(d_in_out, h_in_out + n_tasks_cpu * p.n_gpu_threads * REGS, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_flags, h_flags, n_flags * sizeof(int), cudaMemcpyHostToDevice);
#endif

        // Kernel launch
        if(p.n_gpu_blocks > 0) {
            assert(p.n_gpu_threads <= max_gpu_threads &&
                "The thread block size is greater than the maximum thread block size that can be used on this device");
            call_StreamCompaction_kernel(p.n_gpu_blocks, p.n_gpu_threads, p.in_size, p.remove_value, n_tasks, p.alpha,
                d_in_out, d_in_out, (int*)d_flags,
                p.n_gpu_threads * sizeof(int) + sizeof(int)
#ifdef DYNAMIC_PARTITION
                + sizeof(int), (int*)worklist
#endif
                );
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in_out, h_in_out, h_flags, p.in_size, p.remove_value, p.n_threads,
            p.n_gpu_threads, n_tasks, p.alpha
#ifdef DYNAMIC_PARTITION
            ,
            worklist
#endif
            );

        cudaDeviceSynchronize();
        main_thread.join();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total stream compaction time for %d iterations: %f (ms)\n", p.n_reps + p.n_warmup, time * 1e-6f);

#ifndef DYNAMIC_PARTITION
    // Copy back
    if(p.alpha < 1.0) {
        int offset = n_tasks_cpu == 0 ? 1 : 2;
        cudaMemcpy(h_in_out + h_flags[n_tasks_cpu] - offset, d_in_out, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T), cudaMemcpyDeviceToHost);
    }
#endif

    // Verify answer
    verify(h_in_out, h_in_backup, p.in_size, p.remove_value,
           (p.in_size * p.compaction_factor) / 100);

    // Free memory
#ifdef DYNAMIC_PARTITION
    cudaFree(h_in_out);
    cudaFree(h_flags);
    cudaFree(worklist);
#else
    free(h_in_out);
    free(h_flags);
    cudaFree(d_in_out);
    cudaFree(d_flags);
#endif

    free(h_in_backup);

    return 0;
}

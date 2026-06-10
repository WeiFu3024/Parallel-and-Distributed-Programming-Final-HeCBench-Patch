#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

typedef unsigned int T;
typedef uint4 VECTYPE;

// kernels
// ---- INLINED: sort_reduce.h (from /home/WillFu/parallel/final/HeCBench/src/sort-cuda/sort_reduce.h) ----
__global__ void
reduce (const T* in, T* isums, const size_t size, const unsigned int shift)
{
  __shared__ T lmem[256];
  int group_range = gridDim.x;
  int group = blockIdx.x;
  int lid = threadIdx.x;
  int local_range = blockDim.x;

  int region_size = (size / 4) / group_range * 4;
  int block_start = group * region_size;

  // Give the last block any extra elements
  int block_stop  = (group == group_range - 1) ?
    size : block_start + region_size;

  // Calculate starting index for this thread/work item
  int tid = lid;
  int i = block_start + tid;

  // The per thread histogram, initially 0's.
  int digit_counts[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 };

  // Reduce multiple elements per thread
  while (i < block_stop)
  {
    // This statement
    // 1) Loads the value in from global memory
    // 2) Shifts to the right to have the 4 bits of interest
    //    in the least significant places
    // 3) Masks any more significant bits away. This leaves us
    // with the relevant digit (which is also the index into the
    // histogram). Next increment the histogram to count this occurrence.
    digit_counts[(in[i] >> shift) & 0xFU]++;
    i += local_range;
  }

  for (int d = 0; d < 16; d++)
  {
    // Load this thread's sum into local/shared memory
    lmem[tid] = digit_counts[d];
    __syncthreads();

    // Reduce the contents of shared/local memory
    for (unsigned int s = local_range / 2; s > 0; s >>= 1)
    {
      if (tid < s)
      {
        lmem[tid] += lmem[tid + s];
      }
      __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
      isums[d * group_range + group] = lmem[0];
    }
  }
}

// ---- END INLINED: sort_reduce.h ----

// ---- INLINED: sort_top_scan.h (from /home/WillFu/parallel/final/HeCBench/src/sort-cuda/sort_top_scan.h) ----
__global__ void
top_scan (T* isums, const size_t num_work_groups)
{
  __shared__ T lmem[256*2];
  __shared__ T s_seed;
  int lid = threadIdx.x;
  int local_range = blockDim.x;

  if (lid == 0) s_seed = 0; 
  __syncthreads();

  // Decide if this is the last thread that needs to
  // propagate the seed value
  int last_thread = (lid < num_work_groups &&
                    (lid+1) == num_work_groups) ? 1 : 0;

  for (int d = 0; d < 16; d++)
  {
    T val = 0;
    // Load each block's count for digit d
    if (lid < num_work_groups)
    {
      val = isums[(num_work_groups * d) + lid];
    }
    // Exclusive scan the counts in local memory
    // T res = scanLocalMem(val, lmem, 1);
    int idx = lid;
    lmem[idx] = 0;
    idx += local_range;
    lmem[idx] = val;
    __syncthreads();
    for (int i = 1; i < local_range; i *= 2)
    {
      T t = lmem[idx -  i]; 
      __syncthreads();
      lmem[idx] += t;     
      __syncthreads();
    }
    T res = lmem[idx-1];

    // Write scanned value out to global
    if (lid < num_work_groups)
    {
      isums[(num_work_groups * d) + lid] = res + s_seed;
    }
    __syncthreads();

    if (last_thread)
    {
      s_seed += res + val;
    }
    __syncthreads();
  }
}

// ---- END INLINED: sort_top_scan.h ----

// ---- INLINED: sort_bottom_scan.h (from /home/WillFu/parallel/final/HeCBench/src/sort-cuda/sort_bottom_scan.h) ----
__global__ void
bottom_scan (T* out, const T* in, const T* isums, const size_t size, const unsigned int shift)
{

  __shared__ T lmem[256*2];
  __shared__ T l_scanned_seeds[16];
  __shared__ T l_block_counts[16];

  int group_range = gridDim.x;
  int group = blockIdx.x;
  int lid = threadIdx.x;
  int local_range = blockDim.x;


  // Keep a private histogram as well
  int histogram[16];

  // Prepare for reading 4-element vectors
  // Assume: divisible by 4

  int n4 = size / 4; //vector type is 4 wide

  int region_size = n4 / group_range;
  int block_start = group * region_size;
  // Give the last block any extra elements
  int block_stop  = (group == group_range - 1) ?
    n4 : block_start + region_size;

  // Calculate starting index for this thread/work item
  int i = block_start + lid;
  int window = block_start;

  // Set the histogram in local memory to zero
  // and read in the scanned seeds from gmem
  if (lid < 16)
  {
    l_block_counts[lid] = 0;
    l_scanned_seeds[lid] =
      isums[(lid*group_range)+group];
  }
  __syncthreads();

  // Scan multiple elements per thread
  while (window < block_stop)
  {
    // Reset histogram
    for (int q = 0; q < 16; q++) histogram[q] = 0;
    VECTYPE val_4;
    VECTYPE key_4;

    if (i < block_stop) // Make sure we don't read out of bounds
    {
      val_4 = ((VECTYPE*)in)[i];

      // Mask the keys to get the appropriate digit
      key_4.x = (val_4.x >> shift) & 0xFU;
      key_4.y = (val_4.y >> shift) & 0xFU;
      key_4.z = (val_4.z >> shift) & 0xFU;
      key_4.w = (val_4.w >> shift) & 0xFU;

      // Update the histogram
      histogram[key_4.x]++;
      histogram[key_4.y]++;
      histogram[key_4.z]++;
      histogram[key_4.w]++;
    }

    // Scan the digit counts in local memory
    for (int digit = 0; digit < 16; digit++)
    {
      int idx = lid;
      lmem[idx] = 0;
      idx += local_range;
      lmem[idx] = histogram[digit];
      __syncthreads();
      for (int i = 1; i < local_range; i *= 2)
      {
        T t = lmem[idx -  i]; 
        __syncthreads();
        lmem[idx] += t;     
        __syncthreads();
      }
      histogram[digit] = lmem[idx-1];

      //histogram[digit] = scanLocalMem(histogram[digit], lmem, 1);
      __syncthreads();
    }

    if (i < block_stop) // Make sure we don't write out of bounds
    {
      int address;
      address = histogram[key_4.x] + l_scanned_seeds[key_4.x] + l_block_counts[key_4.x];
      out[address] = val_4.x;
      histogram[key_4.x]++;

      address = histogram[key_4.y] + l_scanned_seeds[key_4.y] + l_block_counts[key_4.y];
      out[address] = val_4.y;
      histogram[key_4.y]++;

      address = histogram[key_4.z] + l_scanned_seeds[key_4.z] + l_block_counts[key_4.z];
      out[address] = val_4.z;
      histogram[key_4.z]++;

      address = histogram[key_4.w] + l_scanned_seeds[key_4.w] + l_block_counts[key_4.w];
      out[address] = val_4.w;
      histogram[key_4.w]++;
    }

    // Before proceeding, make sure everyone has finished their current
    // indexing computations.
    __syncthreads();
    // Now update the seed array.
    if (lid == local_range-1)
    {
      for (int q = 0; q < 16; q++)
      {
        l_block_counts[q] += histogram[q];
      }
    }
    __syncthreads();

    // Advance window
    window += local_range;
    i += local_range;
  }
}


// ---- END INLINED: sort_bottom_scan.h ----


void verifySort(const T *keys, const size_t size)
{
  bool passed = true;
  for (size_t i = 0; i < size - 1; i++)
  {
    if (keys[i] > keys[i + 1])
    {
      passed = false;
#ifdef VERBOSE_OUTPUT
      std::cout << "Idx: " << i;
      std::cout << " Key: " << keys[i] << "\n";
#endif
      break;
    }
  }
  if (passed)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
}

int main(int argc, char** argv) 
{
  if (argc != 3) 
  {
    printf("Usage: %s <problem size> <number of passes>\n.", argv[0]);
    return -1;
  }

  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  // Problem Sizes
  int probSizes[4] = { 1, 8, 32, 64 };
  size_t size = probSizes[select];

  // Convert to MiB
  size = (size * 1024 * 1024) / sizeof(T);

  // Create input data on CPU
  unsigned int bytes = size * sizeof(T);

  T* h_idata = (T*) malloc (bytes); 
  T* h_odata = (T*) malloc (bytes); 

  // Initialize host memory
  std::cout << "Initializing host memory." << std::endl;
  for (unsigned int i = 0; i < size; i++)
  {
    h_idata[i] = i % 16; // Fill with some pattern
    h_odata[i] = size - i;
  }

  std::cout << "Running benchmark with input array length " << size << std::endl;

  // Number of local work items per group
  const size_t local_wsize  = 256;
  // Number of global work items
  const size_t global_wsize = 16384; 
  // 64 work groups
  const size_t num_work_groups = global_wsize / local_wsize;

  // The radix width in bits
  const int radix_width = 4; // Changing this requires major kernel updates
  //const int num_digits = (int)pow((double)2, radix_width); // n possible digits
  const int num_digits = 16;

  T* d_idata;
  T* d_odata;
  T* d_isums;

  cudaMalloc((void**)&d_idata, size * sizeof(T));
  cudaMemcpyAsync(d_idata, h_idata, size * sizeof(T), cudaMemcpyHostToDevice, 0);
  cudaMalloc((void**)&d_odata, size * sizeof(T));
  cudaMalloc((void**)&d_isums, num_work_groups * num_digits * sizeof(T));

  T *d_in, *d_out;
  double time = 0.0;

  for (int k = 0; k < passes; k++)
  {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Assuming an 8 bit byte.
    // shift is uint because Computecpp compiler has no operator>>(unsigned int, int);
    for (unsigned int shift = 0; shift < sizeof(T)*8; shift += radix_width)
    {
      // Like scan, we use a reduce-then-scan approach
      // But before proceeding, update the shift appropriately
      // for each kernel. This is how many bits to shift to the
      // right used in binning.
      // Also, the sort is not in place, so swap the input and output
      // buffers on each pass.
      bool even = ((shift / radix_width) % 2 == 0) ? true : false;
      d_in = even ? d_idata : d_odata;
      d_out = even ? d_odata : d_idata;

      reduce<<<num_work_groups, local_wsize>>> (d_in, d_isums, size, shift);
      top_scan<<<1, local_wsize>>>(d_isums, num_work_groups);
      bottom_scan<<<num_work_groups, local_wsize>>>(d_out, d_in, d_isums, size, shift);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }  // passes

  printf("Average elapsed time of sort: %lf (s)\n", time * 1e-9 / passes);

  cudaMemcpy(h_odata, d_out, size * sizeof(T), cudaMemcpyDeviceToHost);
  verifySort(h_odata, size);

  // reference sort
  time = 0.0;
  for (int k = 0; k < passes; k++) {
    cudaMemcpy(d_odata, h_idata, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    thrust::device_ptr<T> d_out_ptr (d_odata);
    thrust::sort(d_out_ptr, d_out_ptr + size, thrust::less<T>());
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average elapsed time of Thrust::sort: %lf (s)\n", time * 1e-9 / passes);

  cudaMemcpy(h_odata, d_odata, size * sizeof(T), cudaMemcpyDeviceToHost);
  verifySort(h_odata, size);

  cudaFree(d_idata);
  cudaFree(d_odata);
  cudaFree(d_isums);

  free(h_idata);
  free(h_odata);
  return 0;
}

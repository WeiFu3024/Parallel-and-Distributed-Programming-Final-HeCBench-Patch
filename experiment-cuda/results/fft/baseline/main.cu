#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cuda.h>

#ifdef SINGLE_PRECISION
#define T float 
#define T2 float2
#define EPISON 1e-4
#else
#define T double
#define T2 double2
#define EPISON 1e-6
#endif

#ifndef M_SQRT1_2
# define M_SQRT1_2      0.70710678118654752440f
#endif

#define exp_1_8   (T2){  1, -1 }//requires post-multiply by 1/sqrt(2)
#define exp_1_4   (T2){  0, -1 }
#define exp_3_8   (T2){ -1, -1 }//requires post-multiply by 1/sqrt(2)

#define iexp_1_8   (T2){  1, 1 }//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   (T2){  0, 1 }
#define iexp_3_8   (T2){ -1, 1 }//requires post-multiply by 1/sqrt(2)

__host__ __device__
T2 exp_i( T phi ) {
  return (T2){ cos(phi), sin(phi) };
}

__host__ __device__
T2 cmplx_mul( T2 a, T2 b ) { return (T2){ a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x }; }
__host__ __device__
T2 cm_fl_mul( T2 a, T  b ) { return (T2){ b*a.x, b*a.y }; }
__host__ __device__
T2 cmplx_add( T2 a, T2 b ) { return (T2){ a.x + b.x, a.y + b.y }; }
__host__ __device__
T2 cmplx_sub( T2 a, T2 b ) { return (T2){ a.x - b.x, a.y - b.y }; }


#define FFT2(a0, a1)                     \
{                                        \
  T2 c0 = *a0;                           \
  *a0 = cmplx_add(c0,*a1);               \
  *a1 = cmplx_sub(c0,*a1);               \
}

#define FFT4(a0, a1, a2, a3)             \
{                                        \
  FFT2( a0, a2 );                        \
  FFT2( a1, a3 );                        \
  *a3 = cmplx_mul(*a3,exp_1_4);          \
  FFT2( a0, a1 );                        \
  FFT2( a2, a3 );                        \
}

#define FFT8(a)                                               \
{                                                             \
  FFT2( &a[0], &a[4] );                                       \
  FFT2( &a[1], &a[5] );                                       \
  FFT2( &a[2], &a[6] );                                       \
  FFT2( &a[3], &a[7] );                                       \
  \
  a[5] = cm_fl_mul( cmplx_mul(a[5],exp_1_8) , M_SQRT1_2 );    \
  a[6] =  cmplx_mul( a[6] , exp_1_4);                         \
  a[7] = cm_fl_mul( cmplx_mul(a[7],exp_3_8) , M_SQRT1_2 );    \
  \
  FFT4( &a[0], &a[1], &a[2], &a[3] );                         \
  FFT4( &a[4], &a[5], &a[6], &a[7] );                         \
}

#define IFFT2 FFT2

#define IFFT4( a0, a1, a2, a3 )               \
{                                             \
  IFFT2( a0, a2 );                            \
  IFFT2( a1, a3 );                            \
  *a3 = cmplx_mul(*a3 , iexp_1_4);            \
  IFFT2( a0, a1 );                            \
  IFFT2( a2, a3);                             \
}

#define IFFT8( a )                                            \
{                                                             \
  IFFT2( &a[0], &a[4] );                                      \
  IFFT2( &a[1], &a[5] );                                      \
  IFFT2( &a[2], &a[6] );                                      \
  IFFT2( &a[3], &a[7] );                                      \
  \
  a[5] = cm_fl_mul( cmplx_mul(a[5],iexp_1_8) , M_SQRT1_2 );   \
  a[6] = cmplx_mul( a[6] , iexp_1_4);                         \
  a[7] = cm_fl_mul( cmplx_mul(a[7],iexp_3_8) , M_SQRT1_2 );   \
  \
  IFFT4( &a[0], &a[1], &a[2], &a[3] );                        \
  IFFT4( &a[4], &a[5], &a[6], &a[7] );                        \
}

// GPU kernels
// ---- INLINED: fft1D_512.h (from /home/WillFu/parallel/final/HeCBench/src/fft-cuda/fft1D_512.h) ----
__global__ void fft1D_512 (T2* work) 
{
  __shared__  T smem[8*8*9];

  int tid = threadIdx.x;
  int gid = blockIdx.x * 512 + tid;
  int hi = tid>>3;
  int lo = tid&7;
  T2 data[8];
  //__local T smem[8*8*9];
  const int reversed[] = {0,4,2,6,1,5,3,7};

  // starting index of data to/from global memory
  // globalLoads8(data, work, 64)
  for( int i = 0; i < 8; i++ ) data[i] = work[gid+i*64];

  FFT8( data );

  //twiddle8( data, tid, 512 );
  for( int j = 1; j < 8; j++ ){                                       
      data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)512)*tid) ); 
  }                                                                   

  //transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].x;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].x = smem[lo*66+hi+i*8]; 
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].y;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].y= smem[lo*66+hi+i*8]; 
  __syncthreads(); 

  FFT8( data );

  //twiddle8( data, hi, 64 );
  for( int j = 1; j < 8; j++ ){                                       
      data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)64)*hi) ); 
  }                                                                   

  //transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].x;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].x = smem[hi*72+lo+i*8]; 
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].y;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].y= smem[hi*72+lo+i*8]; 

  FFT8( data );

  //globalStores8(data, work, 64);
  for( int i = 0; i < 8; i++ )
    work[gid+i*64] = data[reversed[i]];
}

// ---- END INLINED: fft1D_512.h ----

// ---- INLINED: ifft1D_512.h (from /home/WillFu/parallel/final/HeCBench/src/fft-cuda/ifft1D_512.h) ----
__global__ void ifft1D_512 (T2* work) 
{
  __shared__  T smem[8*8*9];

  int tid = threadIdx.x;
  int gid = blockIdx.x * 512 + tid;
  int hi = tid>>3;
  int lo = tid&7;
  T2 data[8];
  //__local T smem[8*8*9];
  const int reversed[] = {0,4,2,6,1,5,3,7};

  // starting index of data to/from global memory
  for( int i = 0; i < 8; i++ ) data[i] = work[gid+i*64];

  IFFT8( data );

  //itwiddle8( data, tid, 512 );
  for( int j = 1; j < 8; j++ )
      data[j] = cmplx_mul(data[j] , exp_i(((T)2*(T)M_PI*reversed[j]/(T)512)*(tid)) );

  //transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].x;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].x = smem[lo*66+hi+i*8]; 
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].y;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].y= smem[lo*66+hi+i*8]; 
  __syncthreads(); 

  IFFT8( data );

  //itwiddle8( data, hi, 64 );
  for( int j = 1; j < 8; j++ )
      data[j] = cmplx_mul(data[j] , exp_i(((T)2*(T)M_PI*reversed[j]/(T)64)*hi) );


  //transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].x;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].x = smem[hi*72+lo+i*8]; 
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].y;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].y= smem[hi*72+lo+i*8]; 

  IFFT8( data );

  for( int i = 0; i < 8; i++ ) {
      data[i].x = data[i].x/(T)512;
      data[i].y = data[i].y/(T)512;
  }

  //globalStores8(data, work, 64);
  for( int i = 0; i < 8; i++ )
    work[gid+i*64] = data[reversed[i]];
}

// ---- END INLINED: ifft1D_512.h ----


// CPU kernel
// ---- INLINED: reference.h (from /home/WillFu/parallel/final/HeCBench/src/fft-cuda/reference.h) ----
template <int SIZE>
void fft1D_512_reference (T2* work, const int n_ffts)
{
  const int reversed[] = {0,4,2,6,1,5,3,7};
  for (int n = 0; n < n_ffts; n++) {
    T smem[9*64];
    T2 buffer[8*SIZE];
    T2 *data;
    int i, j, gid, tid, hi, lo;
    for (tid = 0; tid < SIZE; tid++) {
      gid = n * 512 + tid;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i] = work[gid+i*64];

      FFT8( data );

      for(j = 1; j < 8; j++ ) {
        data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)512)*tid) );
      }
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].x;
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].x = smem[lo*66+hi+i*8];
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].y;
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].y= smem[lo*66+hi+i*8];
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      data = buffer + tid * 8;
      FFT8( data );
      for(j = 1; j < 8; j++ ) {
        data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)64)*hi) );
      }
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].x;
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].x = smem[hi*72+lo+i*8];
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].y;
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].y= smem[hi*72+lo+i*8];
    }

    for (tid = 0; tid < SIZE; tid++) {
      data = buffer + tid * 8;
      FFT8( data );
      for(i = 0; i < 8; i++ ) {
        gid = n * 512 + tid;
        work[gid+i*64] = data[reversed[i]];
      }
    }
  }
}

// ---- END INLINED: reference.h ----


int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <problem size> <number of passes>\n", argv[0]);
    printf("Problem size [0-3]: 0=1M, 1=8M, 2=96M, 3=256M\n");
    return 1;
  }

  srand(2);
  int i;
  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  // Convert to MB
  int probSizes[4] = { 1, 8, 96, 256 };
  unsigned long bytes = probSizes[select];
  bytes *= 1024 * 1024;

  // now determine how much available memory will be used
  const int half_n_ffts = bytes / (512*sizeof(T2)*2);
  const int n_ffts = half_n_ffts * 2;
  const int half_n_cmplx = half_n_ffts * 512;
  const unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
  const int n_cmplx = half_n_cmplx*2;

  fprintf(stdout, "used_bytes=%lu, n_cmplx=%d\n", used_bytes, n_cmplx);

  // allocate host memory, in-place FFT/iFFT operations
  T2 *source = (T2*) malloc (used_bytes);
  T2 *reference = (T2*) malloc (used_bytes);

  // init host memory...
  for (i = 0; i < half_n_cmplx; i++) {
    source[i].x = sinf(i / powf(10000, i % 768 / 384));
    source[i].y = cosf(i / powf(10000, i % 768 / 384));
    source[i+half_n_cmplx].x = source[i].x;
    source[i+half_n_cmplx].y = source[i].y;
  }

  memcpy(reference, source, used_bytes);

  T2 *d_source;
  cudaMalloc((void**)&d_source, used_bytes);
  cudaMemcpy(d_source, source, used_bytes, cudaMemcpyHostToDevice);

  fft1D_512<<<n_ffts, 64>>>(d_source);

  // verify FFT
  fft1D_512_reference<64>(reference, n_ffts);
  cudaMemcpy(source, d_source, used_bytes, cudaMemcpyDeviceToHost);
  bool error = false;
  for (int i = 0; i < n_cmplx; i++) {
    if (fabs((T)source[i].x - (T)reference[i].x) > EPISON) {
      //std::cout << i << " " << (T)source[i].x << " " << (T)reference[i].x << std::endl;
      error = true;
      break;
    }
    if (fabs((T)source[i].y - (T)reference[i].y) > EPISON) {
      //std::cout << i << " " << (T)source[i].y << " " << (T)reference[i].y << std::endl;
      error = true;
      break;
    }
  }
  std::cout << "FFT " << (error ? "FAIL" : "PASS")  << std::endl;
 
  // execute iFFT
  ifft1D_512<<<n_ffts, 64>>>(d_source);

  // verify iFFT
  cudaMemcpy(source, d_source, used_bytes, cudaMemcpyDeviceToHost);
  error = false;
  for (int i = 0; i < n_cmplx; i++) {
    int j = i % half_n_cmplx;
    if (fabs((T)source[i].x - (T)sinf(j / powf(10000, j%768/384))) > EPISON) {
      error = true;
      break;
    }
    if (fabs((T)source[i].y - (T)cosf(j / powf(10000, j%768/384))) > EPISON) {
      error = true;
      break;
    }
  }
  std::cout << "iFFT " << (error ? "FAIL" : "PASS")  << std::endl;

  auto start = std::chrono::steady_clock::now();

  for (int k=0; k<passes; k++) {
    fft1D_512<<<n_ffts, 64>>>(d_source);
    ifft1D_512<<<n_ffts, 64>>>(d_source);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / passes << " (s)\n";

  cudaFree(d_source);

  free(reference);
  free(source);

  return 0;
}

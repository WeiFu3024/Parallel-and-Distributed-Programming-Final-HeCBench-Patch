/******************************************************************************
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
// ---- INLINED: test_util.h (from /home/WillFu/parallel/final/HeCBench/src/histogram-cuda/test_util.h) ----
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
#endif

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>
#include <float.h>

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <limits>
// ---- INLINED: mersenne.h (from /home/WillFu/parallel/final/HeCBench/src/histogram-cuda/mersenne.h) ----
/*
 A C-program for MT19937, with initialization improved 2002/1/26.
 Coded by Takuji Nishimura and Makoto Matsumoto.

 Before using, initialize the state by using init_genrand(seed)
 or init_by_array(init_key, key_length).

 Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:

 1. Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 3. The names of its contributors may not be used to endorse or promote
 products derived from this software without specific prior written
 permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


 Any feedback is very welcome.
 http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
 */

#include <stdio.h>

namespace mersenne {

/* Period parameters */
const unsigned int N          = 624;
const unsigned int M          = 397;
const unsigned int MATRIX_A   = 0x9908b0df; /* constant vector a */
const unsigned int UPPER_MASK = 0x80000000; /* most significant w-r bits */
const unsigned int LOWER_MASK = 0x7fffffff; /* least significant r bits */

static unsigned int mt[N];  /* the array for the state vector  */
static int mti = N + 1;     /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
void init_genrand(unsigned int s)
{
    mt[0] = s & 0xffffffff;
    for (mti = 1; mti < N; mti++)
    {
        mt[mti] = (1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);

        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for mtiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */

        mt[mti] &= 0xffffffff;
        /* for >32 bit machines */
    }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned int init_key[], int key_length)
{
    int i, j, k;
    init_genrand(19650218);
    i = 1;
    j = 0;
    k = (N > key_length ? N : key_length);
    for (; k; k--)
    {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525))
            + init_key[j] + j;  /* non linear */
        mt[i] &= 0xffffffff;    /* for WORDSIZE > 32 machines */
        i++;
        j++;
        if (i >= N)
        {
            mt[0] = mt[N - 1];
            i = 1;
        }
        if (j >= key_length) j = 0;
    }
    for (k = N - 1; k; k--)
    {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i; /* non linear */
        mt[i] &= 0xffffffff; /* for WORDSIZE > 32 machines */
        i++;
        if (i >= N)
        {
            mt[0] = mt[N - 1];
            i = 1;
        }
    }

    mt[0] = 0x80000000; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned int genrand_int32(void)
{
    unsigned int y;
    static unsigned int mag01[2] = { 0x0, MATRIX_A };

    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N)
    { /* generate N words at one time */
        int kk;

        if (mti == N + 1) /* if init_genrand() has not been called, */
        init_genrand(5489); /* a defat initial seed is used */

        for (kk = 0; kk < N - M; kk++)
        {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < N - 1; kk++)
        {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }

    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
}



} // namespace mersenne

// ---- END INLINED: mersenne.h ----




/******************************************************************************
 * Assertion macros
 ******************************************************************************/

/**
 * Assert equals
 */
#define AssertEquals(a, b) if ((a) != (b)) { std::cerr << "\n(" << __FILE__ << ": " << __LINE__ << ")\n"; exit(1);}


/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * Utility for parsing command line arguments
 */
struct CommandLineArgs
{

    std::vector<std::string>    keys;
    std::vector<std::string>    values;
    std::vector<std::string>    args;
    cudaDeviceProp              deviceProp;
    float                       device_giga_bandwidth;
    size_t                      device_free_physmem;
    size_t                      device_total_physmem;

    /**
     * Constructor
     */
    CommandLineArgs(int argc, char **argv) :
        keys(10),
        values(10)
    {
        using namespace std;

        // Initialize mersenne generator
        unsigned int mersenne_init[4]=  {0x123, 0x234, 0x345, 0x456};
        mersenne::init_by_array(mersenne_init, 4);

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-'))
            {
                args.push_back(arg);
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find('=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }

            keys.push_back(key);
            values.push_back(val);
        }
    }


    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;

        for (int i = 0; i < int(keys.size()); ++i)
        {
            if (keys[i] == string(arg_name))
                return true;
        }
        return false;
    }


    /**
     * Returns number of naked (non-flag and non-key-value) commandline parameters
     */
    template <typename T>
    int NumNakedArgs()
    {
        return args.size();
    }


    /**
     * Returns the commandline parameter for a given index (not including flags)
     */
    template <typename T>
    void GetCmdLineArgument(int index, T &val)
    {
        using namespace std;
        if (index < args.size()) {
            istringstream str_stream(args[index]);
            str_stream >> val;
        }
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        using namespace std;

        for (int i = 0; i < int(keys.size()); ++i)
        {
            if (keys[i] == string(arg_name))
            {
                istringstream str_stream(values[i]);
                str_stream >> val;
            }
        }
    }


    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
    {
        using namespace std;

        if (CheckCmdLineFlag(arg_name))
        {
            // Clear any default values
            vals.clear();

            // Recover from multi-value string
            for (int i = 0; i < keys.size(); ++i)
            {
                if (keys[i] == string(arg_name))
                {
                    string val_string(values[i]);
                    istringstream str_stream(val_string);
                    string::size_type old_pos = 0;
                    string::size_type new_pos = 0;

                    // Iterate comma-separated values
                    T val;
                    while ((new_pos = val_string.find(',', old_pos)) != string::npos)
                    {
                        if (new_pos != old_pos)
                        {
                            str_stream.width(new_pos - old_pos);
                            str_stream >> val;
                            vals.push_back(val);
                        }

                        // skip over comma
                        str_stream.ignore(1);
                        old_pos = new_pos + 1;
                    }

                    // Read last value
                    str_stream >> val;
                    vals.push_back(val);
                }
            }
        }
    }


    /**
     * The number of pairs parsed
     */
    int ParsedArgc()
    {
        return (int) keys.size();
    }

    /**
     * Initialize device
     */
    /*
    cudaError_t DeviceInit(int dev = -1)
    {
        cudaError_t error = cudaSuccess;

        do
        {
            int deviceCount;
            error = cudaGetDeviceCount(&deviceCount);
            if (error) break;

            if (deviceCount == 0) {
                fprintf(stderr, "No devices supporting CUDA.\n");
                exit(1);
            }
            if (dev < 0)
            {
                GetCmdLineArgument("device", dev);
            }
            if ((dev > deviceCount - 1) || (dev < 0))
            {
                dev = 0;
            }

            error = cudaSetDevice(dev);
            if (error) break;

            cudaMemGetInfo(&device_free_physmem, &device_total_physmem);

            error = cudaGetDeviceProperties(&deviceProp, dev);
            if (error) break;

            if (deviceProp.major < 1) {
                fprintf(stderr, "Device does not support CUDA.\n");
                exit(1);
            }

            device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;

            if (!CheckCmdLineFlag("quiet"))
            {
                printf(
                        "Using device %d: %s (%d SMs, "
                        "%lld free / %lld total MB physmem, "
                        "%.3f GB/s @ %d kHz mem clock)\n",
                    dev,
                    deviceProp.name,
                    deviceProp.multiProcessorCount,
                    (unsigned long long) device_free_physmem / 1024 / 1024,
                    (unsigned long long) device_total_physmem / 1024 / 1024,
                    device_giga_bandwidth,
                    deviceProp.memoryClockRate);
                fflush(stdout);
            }

        } while (0);

        return error;
    }
    */
};

/******************************************************************************
 * Random bits generator
 ******************************************************************************/

int g_num_rand_samples = 0;


template <typename T>
bool IsNaN(T val) { return false; }

template<>
 bool IsNaN<float>(float val)
{
    volatile unsigned int bits = reinterpret_cast<unsigned int &>(val);

    return (((bits >= 0x7F800001) && (bits <= 0x7FFFFFFF)) || 
        ((bits >= 0xFF800001) && (bits <= 0xFFFFFFFF)));
}

/*
template<>
 bool IsNaN<float1>(float1 val)
{
    return (IsNaN(val.x));
}
*/

template<>
 bool IsNaN<float2>(float2 val)
{
    return (IsNaN(val.y) || IsNaN(val.x));
}

template<>
 bool IsNaN<float3>(float3 val)
{
    return (IsNaN(val.z) || IsNaN(val.y) || IsNaN(val.x));
}

template<>
 bool IsNaN<float4>(float4 val)
{
    return (IsNaN(val.y) || IsNaN(val.x) || IsNaN(val.w) || IsNaN(val.z));
}

template<>
 bool IsNaN<double>(double val)
{
    volatile unsigned long long bits = *reinterpret_cast<unsigned long long *>(&val);

    return (((bits >= 0x7FF0000000000001) && (bits <= 0x7FFFFFFFFFFFFFFF)) || 
        ((bits >= 0xFFF0000000000001) && (bits <= 0xFFFFFFFFFFFFFFFF)));
}

/*
template<>
 bool IsNaN<double1>(double1 val)
{
    return (IsNaN(val.x));
}
*/

template<>
 bool IsNaN<double2>(double2 val)
{
    return (IsNaN(val.y) || IsNaN(val.x));
}

template<>
 bool IsNaN<double3>(double3 val)
{
    return (IsNaN(val.z) || IsNaN(val.y) || IsNaN(val.x));
}

template<>
 bool IsNaN<double4>(double4 val)
{
    return (IsNaN(val.y) || IsNaN(val.x) || IsNaN(val.w) || IsNaN(val.z));
}







/******************************************************************************
 * Console printing utilities
 ******************************************************************************/

/**
 * Helper for casting character types to integers for cout printing
 */
template <typename T>
T CoutCast(T val) { return val; }

int CoutCast(char val) { return val; }

int CoutCast(unsigned char val) { return val; }

int CoutCast(signed char val) { return val; }



/******************************************************************************
 * Test value initialization utilities
 ******************************************************************************/

/**
 * Test problem generation options
 */
enum GenMode
{
    UNIFORM,            // Assign to '2', regardless of integer seed
    INTEGER_SEED,       // Assign to integer seed
};

/**
 * Initialize value
 */
template <typename T>
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)
{
    switch (gen_mode)
    {
     case UNIFORM:
        value = 2;
        break;
    case INTEGER_SEED:
    default:
         value = (T) index;
        break;
    }
}


/**
 * Initialize value (bool)
 */
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, bool &value, int index = 0)
{
    switch (gen_mode)
    {
     case UNIFORM:
        value = true;
        break;
    case INTEGER_SEED:
    default:
        value = (index > 0);
        break;
    }
}


/**
 * Generates random keys.
 *
 * We always take the second-order byte from rand() because the higher-order
 * bits returned by rand() are commonly considered more uniformly distributed
 * than the lower-order bits.
 *
 * We can decrease the entropy level of keys by adopting the technique
 * of Thearling and Smith in which keys are computed from the bitwise AND of
 * multiple random samples:
 *
 * entropy_reduction    | Effectively-unique bits per key
 * -----------------------------------------------------
 * -1                   | 0
 * 0                    | 32
 * 1                    | 25.95 (81%)
 * 2                    | 17.41 (54%)
 * 3                    | 10.78 (34%)
 * 4                    | 6.42 (20%)
 * ...                  | ...
 *
 */
template <typename K>
void RandomBits(
    K &key,
    int entropy_reduction = 0,
    int begin_bit = 0,
    int end_bit = sizeof(K) * 8)
{
    const int NUM_BYTES = sizeof(K);
    const int WORD_BYTES = sizeof(unsigned int);
    const int NUM_WORDS = (NUM_BYTES + WORD_BYTES - 1) / WORD_BYTES;

    unsigned int word_buff[NUM_WORDS];

    if (entropy_reduction == -1)
    {
        memset((void *) &key, 0, sizeof(key));
        return;
    }

    if (end_bit < 0)
        end_bit = sizeof(K) * 8;

    while (true) 
    {
        // Generate random word_buff
        for (int j = 0; j < NUM_WORDS; j++)
        {
            int current_bit = j * WORD_BYTES * 8;

            unsigned int word = 0xffffffff;
            word &= 0xffffffff << std::max(0, begin_bit - current_bit);
            word &= 0xffffffff >> std::max(0, (current_bit + (WORD_BYTES * 8)) - end_bit);

            for (int i = 0; i <= entropy_reduction; i++)
            {
                // Grab some of the higher bits from rand (better entropy, supposedly)
                word &= mersenne::genrand_int32();
                g_num_rand_samples++;                
            }

            word_buff[j] = word;
        }

        memcpy(&key, word_buff, sizeof(K));

        K copy = key;
        if (!IsNaN(copy))
            break;          // avoids NaNs when generating random floating point numbers
    }
}



/******************************************************************************
 * Helper routines for list comparison and display
 ******************************************************************************/


/**
 * Compares the equivalence of two arrays
 */
template <typename S, typename T, typename OffsetT>
int CompareResults(T* computed, S* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                << CoutCast(computed[i]) << " != "
                << CoutCast(reference[i]);
            return 1;
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(float* computed, float* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            float difference = std::abs(computed[i]-reference[i]);
            float fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << "(computed) " << CoutCast(computed[i]) << " != "
                    << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(double* computed, double* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            double difference = std::abs(computed[i]-reference[i]);
            double fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << CoutCast(computed[i]) << " != "
                    << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}



/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename S, typename T>
int CompareDeviceResults(
    S *h_reference,
    T *d_data,
    size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data)
    {
        printf("Reference:\n");
        for (int i = 0; i < int(num_items); i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nComputed:\n");
        for (int i = 0; i < int(num_items); i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_data) free(h_data);

    return retval;
}

/******************************************************************************
 * Timing
 ******************************************************************************/


struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&ll_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&ll_start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&ll_stop);
    }

    float ElapsedMillis()
    {
        double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
        double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

        return float((stop - start) * 1000);
    }

#else

    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }

#endif
};

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// ---- END INLINED: test_util.h ----


// Decode float4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__device__ __forceinline__ void DecodePixel(float4 &pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    float samples[4];
    samples[0] = pixel.x;
    samples[1] = pixel.y;
    samples[2] = pixel.z;
    samples[3] = pixel.w;

    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
}

// Decode uchar4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__device__ __forceinline__ void DecodePixel(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    unsigned char samples[4];
    samples[0] = pixel.x;
    samples[1] = pixel.y;
    samples[2] = pixel.z;
    samples[3] = pixel.w;

    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
}

// Decode uchar1 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__device__ __forceinline__ void DecodePixel(uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    bins[0] = (unsigned int) pixel.x;
}
// ---- INLINED: histogram_gmem_atomics.h (from /home/WillFu/parallel/final/HeCBench/src/histogram-cuda/histogram_gmem_atomics.h) ----
/******************************************************************************
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


    // First-pass histogram kernel (binning into privatized counters)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS,
        typename    PixelType>
    __global__ void histogram_gmem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out)
    {
        // global position and size
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int nx = blockDim.x * gridDim.x;
        int ny = blockDim.y * gridDim.y;

        // threads in workgroup
        int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
        int nt = blockDim.x * blockDim.y; // total threads in workgroup

        // group index in 0..ngroups-1
        int g = blockIdx.x + blockIdx.y * gridDim.x;

        // initialize global memory
        unsigned int *gmem = out + g * NUM_PARTS;
        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
            gmem[i] = 0;
        __syncthreads();

        // process pixels (updates our group's partial histogram in gmem)
        for (int col = x; col < width; col += nx)
        {
            for (int row = y; row < height; row += ny)
            {
                PixelType pixel = in[row * width + col];

                unsigned int bins[ACTIVE_CHANNELS];
                DecodePixel<NUM_BINS>(pixel, bins);

                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                    atomicAdd(&gmem[(NUM_BINS * CHANNEL) + bins[CHANNEL]], 1);
            }
        }
    }

    // Second pass histogram kernel (accumulation)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS>
    __global__ void histogram_gmem_accum(
        const unsigned int *in,
        int n,
        unsigned int *out)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i > ACTIVE_CHANNELS * NUM_BINS)
            return; // out of range

        unsigned int total = 0;
        for (int j = 0; j < n; j++)
            total += in[i + NUM_PARTS * j];

        out[i] = total;
    }




template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_gmem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist,
    bool warmup)
{
    enum
    {
        NUM_PARTS = 1024
    };

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    dim3 block(32, 4);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((3 * NUM_BINS + block.x - 1) / block.x);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    histogram_gmem_atomics<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS><<<grid, block>>>(
        d_image,
        width,
        height,
        d_part_hist);

    histogram_gmem_accum<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS><<<grid2, block2>>>(
        d_part_hist,
        total_blocks,
        d_hist);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    cudaFree(d_part_hist);

    return elapsed_millis;
}


// ---- END INLINED: histogram_gmem_atomics.h ----

// ---- INLINED: histogram_smem_atomics.h (from /home/WillFu/parallel/final/HeCBench/src/histogram-cuda/histogram_smem_atomics.h) ----
/******************************************************************************
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


    // First-pass histogram kernel (binning into privatized counters)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS,
        typename    PixelType>
    __global__ void histogram_smem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out)
    {
        // global position and size
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int nx = blockDim.x * gridDim.x;
        int ny = blockDim.y * gridDim.y;

        // threads in workgroup
        int t = threadIdx.x + threadIdx.y * blockDim.x; // thread index in workgroup, linear in 0..nt-1
        int nt = blockDim.x * blockDim.y; // total threads in workgroup

        // group index in 0..ngroups-1
        int g = blockIdx.x + blockIdx.y * gridDim.x;

        // initialize smem
        __shared__ unsigned int smem[ACTIVE_CHANNELS * NUM_BINS + 3];
        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS + 3; i += nt)
            smem[i] = 0;
        __syncthreads();

        // process pixels
        // updates our group's partial histogram in smem
        for (int col = x; col < width; col += nx)
        {
            for (int row = y; row < height; row += ny)
            {
                PixelType pixel = in[row * width + col];

                unsigned int bins[ACTIVE_CHANNELS];
                DecodePixel<NUM_BINS>(pixel, bins);

                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                    atomicAdd(&smem[(NUM_BINS * CHANNEL) + bins[CHANNEL] + CHANNEL], 1);
            }
        }

        __syncthreads();

        // move to our workgroup's slice of output
        out += g * NUM_PARTS;

        // store local output to global
        for (int i = t; i < NUM_BINS; i += nt)
        {
            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                out[i + NUM_BINS * CHANNEL] = smem[i + NUM_BINS * CHANNEL + CHANNEL];
        }
    }

    // Second pass histogram kernel (accumulation)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS>
    __global__ void histogram_smem_accum(
        const unsigned int *in,
        int n,
        unsigned int *out)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i > ACTIVE_CHANNELS * NUM_BINS) return; // out of range
        unsigned int total = 0;
        for (int j = 0; j < n; j++)
            total += in[i + NUM_PARTS * j];
        out[i] = total;
    }



template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_smem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist, 
    bool warmup)
{
    enum
    {
        NUM_PARTS = 1024
    };

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    dim3 block(32, 4);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    cudaMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((ACTIVE_CHANNELS * NUM_BINS + block2.x - 1) / block2.x);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    histogram_smem_atomics<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS><<<grid, block>>>(
        d_image,
        width,
        height,
        d_part_hist);

    histogram_smem_accum<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS><<<grid2, block2>>>(
        d_part_hist,
        total_blocks,
        d_hist);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    cudaFree(d_part_hist);

    return elapsed_millis;
}


// ---- END INLINED: histogram_smem_atomics.h ----

//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

// Ensure printing of CUDA runtime errors to console

bool                    g_verbose = false;  // Whether to display input/output to console
bool                    g_report = false;   // Whether to display a full report in CSV format

struct less_than_value
{
    inline bool operator()(
        const std::pair<std::string, double> &a,
        const std::pair<std::string, double> &b)
    {
        return a.second < b.second;
    }
};


//---------------------------------------------------------------------
// Targa (.tga) image file parsing
//---------------------------------------------------------------------

/**
 * TGA image header info
 */
struct TgaHeader
{
    char idlength;
    char colormaptype;
    char datatypecode;
    short colormaporigin;
    short colormaplength;
    char colormapdepth;
    short x_origin;
    short y_origin;
    short width;
    short height;
    char bitsperpixel;
    char imagedescriptor;

    void Parse (FILE *fptr)
    {
        idlength = fgetc(fptr);
        colormaptype = fgetc(fptr);
        datatypecode = fgetc(fptr);
        fread(&colormaporigin, 2, 1, fptr);
        fread(&colormaplength, 2, 1, fptr);
        colormapdepth = fgetc(fptr);
        fread(&x_origin, 2, 1, fptr);
        fread(&y_origin, 2, 1, fptr);
        fread(&width, 2, 1, fptr);
        fread(&height, 2, 1, fptr);
        bitsperpixel = fgetc(fptr);
        imagedescriptor = fgetc(fptr);
    }

    void Display (FILE *fptr)
    {
        fprintf(fptr, "ID length:           %d\n", idlength);
        fprintf(fptr, "Color map type:      %d\n", colormaptype);
        fprintf(fptr, "Image type:          %d\n", datatypecode);
        fprintf(fptr, "Color map offset:    %d\n", colormaporigin);
        fprintf(fptr, "Color map length:    %d\n", colormaplength);
        fprintf(fptr, "Color map depth:     %d\n", colormapdepth);
        fprintf(fptr, "X origin:            %d\n", x_origin);
        fprintf(fptr, "Y origin:            %d\n", y_origin);
        fprintf(fptr, "Width:               %d\n", width);
        fprintf(fptr, "Height:              %d\n", height);
        fprintf(fptr, "Bits per pixel:      %d\n", bitsperpixel);
        fprintf(fptr, "Descriptor:          %d\n", imagedescriptor);
    }
};


/**
 * Decode image byte data into pixel
 */
void ParseTgaPixel(uchar4 &pixel, unsigned char *tga_pixel, int bytes)
{
    if (bytes == 4)
    {
        pixel.x = tga_pixel[2];
        pixel.y = tga_pixel[1];
        pixel.z = tga_pixel[0];
        pixel.w = tga_pixel[3];
    }
    else if (bytes == 3)
    {
        pixel.x = tga_pixel[2];
        pixel.y = tga_pixel[1];
        pixel.z = tga_pixel[0];
        pixel.w = 0;
    }
    else if (bytes == 2)
    {
        pixel.x = (tga_pixel[1] & 0x7c) << 1;
        pixel.y = ((tga_pixel[1] & 0x03) << 6) | ((tga_pixel[0] & 0xe0) >> 2);
        pixel.z = (tga_pixel[0] & 0x1f) << 3;
        pixel.w = (tga_pixel[1] & 0x80);
    }
}


/**
 * Reads a .tga image file
 */
void ReadTga(uchar4* &pixels, int &width, int &height, const char *filename)
{
    // Open the file
    FILE *fptr;
    if ((fptr = fopen(filename, "rb")) == NULL)
    {
        fprintf(stderr, "File open failed\n");
        exit(-1);
    }

    // Parse header
    TgaHeader header;
    header.Parse(fptr);
//    header.Display(stdout);
    width = header.width;
    height = header.height;

    // Verify compatibility
    if (header.datatypecode != 2 && header.datatypecode != 10)
    {
        fprintf(stderr, "Can only handle image type 2 and 10\n");
        exit(-1);
    }
    if (header.bitsperpixel != 16 && header.bitsperpixel != 24 && header.bitsperpixel != 32)
    {
        fprintf(stderr, "Can only handle pixel depths of 16, 24, and 32\n");
        exit(-1);
    }
    if (header.colormaptype != 0 && header.colormaptype != 1)
    {
        fprintf(stderr, "Can only handle color map types of 0 and 1\n");
        exit(-1);
    }

    // Skip unnecessary header info
    int skip_bytes = header.idlength + (header.colormaptype * header.colormaplength);
    fseek(fptr, skip_bytes, SEEK_CUR);

    // Read the image
    int pixel_bytes = header.bitsperpixel / 8;

    // Allocate and initialize pixel data
    size_t image_bytes = width * height * sizeof(uchar4);
    if ((pixels == NULL) && ((pixels = (uchar4*) malloc(image_bytes)) == NULL))
    {
        fprintf(stderr, "malloc of image failed\n");
        exit(-1);
    }
    memset(pixels, 0, image_bytes);

    // Parse pixels
    unsigned char   tga_pixel[5];
    int             current_pixel = 0;
    while (current_pixel < header.width * header.height)
    {
        if (header.datatypecode == 2)
        {
            // Uncompressed
            if (fread(tga_pixel, 1, pixel_bytes, fptr) != pixel_bytes)
            {
                fprintf(stderr, "Unexpected end of file at pixel %d  (uncompressed)\n", current_pixel);
                exit(-1);
            }
            ParseTgaPixel(pixels[current_pixel], tga_pixel, pixel_bytes);
            current_pixel++;
        }
        else if (header.datatypecode == 10)
        {
            // Compressed
            if (fread(tga_pixel, 1, pixel_bytes + 1, fptr) != pixel_bytes + 1)
            {
                fprintf(stderr, "Unexpected end of file at pixel %d (compressed)\n", current_pixel);
                exit(-1);
            }
            int run_length = tga_pixel[0] & 0x7f;
            ParseTgaPixel(pixels[current_pixel], &(tga_pixel[1]), pixel_bytes);
            current_pixel++;

            if (tga_pixel[0] & 0x80)
            {
                // RLE chunk
                for (int i = 0; i < run_length; i++)
                {
                    ParseTgaPixel(pixels[current_pixel], &(tga_pixel[1]), pixel_bytes);
                    current_pixel++;
                }
            }
            else
            {
                // Normal chunk
                for (int i = 0; i < run_length; i++)
                {
                    if (fread(tga_pixel, 1, pixel_bytes, fptr) != pixel_bytes)
                    {
                        fprintf(stderr, "Unexpected end of file at pixel %d (normal)\n", current_pixel);
                        exit(-1);
                    }
                    ParseTgaPixel(pixels[current_pixel], tga_pixel, pixel_bytes);
                    current_pixel++;
                }
            }
        }
    }

    // Close file
    fclose(fptr);
}



//---------------------------------------------------------------------
// Random image generation
//---------------------------------------------------------------------

/**
 * Generate a random image with specified entropy
 */
void GenerateRandomImage(uchar4* &pixels, int width, int height, int entropy_reduction)
{
    int num_pixels = width * height;
    size_t image_bytes = num_pixels * sizeof(uchar4);
    if ((pixels == NULL) && ((pixels = (uchar4*) malloc(image_bytes)) == NULL))
    {
        fprintf(stderr, "malloc of image failed\n");
        exit(-1);
    }

    for (int i = 0; i < num_pixels; ++i)
    {
        RandomBits(pixels[i].x, entropy_reduction);
        RandomBits(pixels[i].y, entropy_reduction);
        RandomBits(pixels[i].z, entropy_reduction);
        RandomBits(pixels[i].w, entropy_reduction);
    }
}



//---------------------------------------------------------------------
// Histogram verification
//---------------------------------------------------------------------

// Decode float4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(float4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    float* samples = reinterpret_cast<float*>(&pixel);

    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
}

// Decode uchar4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
}

// Decode uchar1 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    bins[0] = (unsigned int) pixel.x;
}


// Compute reference histogram.  Specialized for uchar4
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void HistogramGold(PixelType *image, int width, int height, unsigned int* hist)
{
    memset(hist, 0, ACTIVE_CHANNELS * NUM_BINS * sizeof(unsigned int));

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            PixelType pixel = image[i + j * width];

            unsigned int bins[ACTIVE_CHANNELS];
            DecodePixelGold<NUM_BINS>(pixel, bins);

            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            {
                hist[(NUM_BINS * CHANNEL) + bins[CHANNEL]]++;
            }
        }
    }
}


//---------------------------------------------------------------------
// Test execution
//---------------------------------------------------------------------

/**
 * Run a specific histogram implementation
 */
template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void RunTest(
    std::vector<std::pair<std::string, double> >&   timings,
    PixelType*                                      d_pixels,
    const int                                       width,
    const int                                       height,
    unsigned int *                                  d_hist,
    unsigned int *                                  h_hist,
    int                                             timing_iterations,
    const char *                                    long_name,
    const char *                                    short_name,
    double (*f)(PixelType*, int, int, unsigned int*, bool))
{
    if (!g_report) printf("%s ", long_name); fflush(stdout);

    // Run single test to verify (and code cache)
    (*f)(d_pixels, width, height, d_hist, !g_report);

    int compare = CompareDeviceResults(h_hist, d_hist, ACTIVE_CHANNELS * NUM_BINS, true, g_verbose);
    if (!g_report) printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);

    double elapsed_ms = 0;
    for (int i = 0; i < timing_iterations; i++)
    {
        elapsed_ms += (*f)(d_pixels, width, height, d_hist, false);
    }
    double avg_us = (elapsed_ms / timing_iterations) * 1000;    // average in us
    timings.push_back(std::pair<std::string, double>(short_name, avg_us));

    if (!g_report)
    {
        printf("Avg time %.3f us (%d iterations)\n", avg_us, timing_iterations); fflush(stdout);
    }
    else
    {
        printf("%.3f, ", avg_us); fflush(stdout);
    }

    AssertEquals(0, compare);
}


/**
 * Evaluate corpus of histogram implementations
 */
template <
    int         NUM_CHANNELS,
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void TestMethods(
    PixelType*  h_pixels,
    int         height,
    int         width,
    int         timing_iterations,
    double      bandwidth_GBs)
{
    // Copy data to gpu
    PixelType* d_pixels;
    size_t pixel_bytes = width * height * sizeof(PixelType);
    cudaMalloc((void**) &d_pixels, pixel_bytes);
    cudaMemcpy(d_pixels, h_pixels, pixel_bytes, cudaMemcpyHostToDevice);

    if (g_report) printf("%.3f, ", double(pixel_bytes) / bandwidth_GBs / 1000);

    // Allocate results arrays on cpu/gpu
    unsigned int *h_hist;
    unsigned int *d_hist;
    size_t histogram_bytes = NUM_BINS * ACTIVE_CHANNELS * sizeof(unsigned int);
    h_hist = (unsigned int *) malloc(histogram_bytes);
    cudaMalloc((void **) &d_hist, histogram_bytes);

    // Compute reference cpu histogram
    HistogramGold<ACTIVE_CHANNELS, NUM_BINS>(h_pixels, width, height, h_hist);

    // Store timings
    std::vector<std::pair<std::string, double> > timings;

    // Run experiments
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, d_pixels, width, height, d_hist, h_hist, timing_iterations,
        "Shared memory atomics", "smem atomics", run_smem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>);
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, d_pixels, width, height, d_hist, h_hist, timing_iterations,
        "Global memory atomics", "gmem atomics", run_gmem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>);

    // Report timings
    if (!g_report)
    {
        std::sort(timings.begin(), timings.end(), less_than_value());
        printf("Timings (us):\n");
        for (int i = 0; i < timings.size(); i++)
        {
            double bandwidth = height * width * sizeof(PixelType) / timings[i].second / 1000;
            printf("\t %.3f %s (%.3f GB/s, %.3f%% peak)\n", timings[i].second, timings[i].first.c_str(), bandwidth, bandwidth / bandwidth_GBs * 100);
        }
        printf("\n");
    }

    // Free data
    cudaFree(d_pixels);
    cudaFree(d_hist);
    free(h_hist);
}


/**
 * Test different problem genres
 */
void TestGenres(
    uchar4*     uchar4_pixels,
    int         height,
    int         width,
    int         timing_iterations,
    double      bandwidth_GBs)
{
    int num_pixels = width * height;

    {
        if (!g_report) printf("1 channel uchar1 tests (256-bin):\n\n"); fflush(stdout);

        size_t      image_bytes     = num_pixels * sizeof(uchar1);
        uchar1*     uchar1_pixels   = (uchar1*) malloc(image_bytes);

        // Convert to 1-channel (averaging first 3 channels)
        for (int i = 0; i < num_pixels; ++i)
        {
            uchar1_pixels[i].x = (unsigned char)
                (((unsigned int) uchar4_pixels[i].x +
                  (unsigned int) uchar4_pixels[i].y +
                  (unsigned int) uchar4_pixels[i].z) / 3);
        }

        TestMethods<1, 1, 256>(uchar1_pixels, width, height, timing_iterations, bandwidth_GBs);
        free(uchar1_pixels);
        if (g_report) printf(", ");
    }

    {
        if (!g_report) printf("3/4 channel uchar4 tests (256-bin):\n\n"); fflush(stdout);
        TestMethods<4, 3, 256>(uchar4_pixels, width, height, timing_iterations, bandwidth_GBs);
        if (g_report) printf(", ");
    }

    {
        if (!g_report) printf("3/4 channel float4 tests (256-bin):\n\n"); fflush(stdout);
        size_t      image_bytes     = num_pixels * sizeof(float4);
        float4*     float4_pixels   = (float4*) malloc(image_bytes);

        // Convert to float4 with range [0.0, 1.0)
        for (int i = 0; i < num_pixels; ++i)
        {
            float4_pixels[i].x = float(uchar4_pixels[i].x) / 256;
            float4_pixels[i].y = float(uchar4_pixels[i].y) / 256;
            float4_pixels[i].z = float(uchar4_pixels[i].z) / 256;
            float4_pixels[i].w = float(uchar4_pixels[i].w) / 256;
        }
        TestMethods<4, 3, 256>(float4_pixels, width, height, timing_iterations, bandwidth_GBs);
        free(float4_pixels);
        if (g_report) printf("\n");
    }
}


/**
 * Main
 */
int main(int argc, char **argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--device=<device-id>] "
            "[--v] "
            "[--i=<timing iterations>] "
            "\n\t"
                "--file=<.tga filename> "
            "\n\t"
                "--entropy=<-1 (0%%), 0 (100%%), 1 (81%%), 2 (54%%), 3 (34%%), 4 (20%%), ..."
                "[--height=<default: 1080>] "
                "[--width=<default: 1920>] "
            "\n", argv[0]);
        exit(0);
    }

    std::string         filename;
    int                 timing_iterations   = 100;
    int                 entropy_reduction   = 0;
    int                 height              = 1080;
    int                 width               = 1920;

    g_verbose = args.CheckCmdLineFlag("v");
    g_report = args.CheckCmdLineFlag("report");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("file", filename);
    args.GetCmdLineArgument("height", height);
    args.GetCmdLineArgument("width", width);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Initialize device
    //args.DeviceInit();

    // Get GPU device bandwidth (GB/s)
    //int device_ordinal, bus_width, mem_clock_khz;
    //cudaGetDevice(&device_ordinal);
    //cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device_ordinal);
    //cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device_ordinal);
    //double bandwidth_GBs = double(bus_width) * mem_clock_khz * 2 / 8 / 1000 / 1000;
    double bandwidth_GBs = 41; //double(bus_width) * mem_clock_khz * 2 / 8 / 1000 / 1000;

    // Run test(s)
    uchar4* uchar4_pixels = NULL;
    if (!g_report)
    {
        if (!filename.empty())
        {
            // Parse targa file
            ReadTga(uchar4_pixels, width, height, filename.c_str());
            printf("File %s: width(%d) height(%d)\n\n", filename.c_str(), width, height); fflush(stdout);
        }
        else
        {
            // Generate image
            GenerateRandomImage(uchar4_pixels, width, height, entropy_reduction);
            printf("Random image: entropy-reduction(%d) width(%d) height(%d)\n\n", entropy_reduction, width, height); fflush(stdout);
        }

        TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
    }
    else
    {
        // Run test suite
        printf("Test, MIN, RLE CUB, SMEM, GMEM, , MIN, RLE_CUB, SMEM, GMEM, , MIN, RLE_CUB, SMEM, GMEM\n");

        // Entropy reduction tests
        for (entropy_reduction = 0; entropy_reduction < 5; ++entropy_reduction)
        {
            printf("entropy reduction %d, ", entropy_reduction);
            GenerateRandomImage(uchar4_pixels, width, height, entropy_reduction);
            TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        }
        printf("entropy reduction -1, ");
        GenerateRandomImage(uchar4_pixels, width, height, -1);
        TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        printf("\n");

        // File image tests
        std::vector<std::string> file_tests;
        file_tests.push_back("animals");
        file_tests.push_back("apples");
        file_tests.push_back("sunset");
        file_tests.push_back("cheetah");
        file_tests.push_back("nature");
        file_tests.push_back("operahouse");
        file_tests.push_back("austin");
        file_tests.push_back("cityscape");

        for (int i = 0; i < file_tests.size(); ++i)
        {
            printf("%s, ", file_tests[i].c_str());
            std::string filename = std::string("histogram/benchmark/") + file_tests[i] + ".tga";
            ReadTga(uchar4_pixels, width, height, filename.c_str());
            TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        }
    }

    free(uchar4_pixels);

    cudaDeviceSynchronize();
    printf("\n\n");

    return 0;
}

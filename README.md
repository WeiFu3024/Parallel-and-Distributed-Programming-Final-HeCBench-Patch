Put the folders in this repo directly under HeCBench (https://github.com/ORNL/HeCBench.git)
```
mv docs {HeCBench_Path}
mv experiment-cuda {HeCBench_Path}
```
Then follow the step-by-step instructions in `docs/operations_manual_cuda.md`

**Some notes during testing:**

1. The omp code provided is not designed for cpu, but , therefore ignore all omp related codes, we only perform experiments on cuda kernels.

    By Claude Code: 
    HeCBench 的 -omp 目錄裡，所有 parallel 實作都是 #pragma omp target（GPU offload），沒有任何一個用 #pragma omp parallel for 做 CPU 多核心。

    唯一接近的是一些 benchmark 裡的 serial CPU reference function（用來驗算 GPU 結果），例如 degrid-omp/main.cpp 裡的 degridCPU：


    void degridCPU(PRECISION2 *out, PRECISION2 *in, PRECISION2 *img, PRECISION2 *gcf) {
    gcf += GCF_DIM*(GCF_DIM+1)/2;
    for(size_t n = 0; n < NPOINTS; n++) {       // ← 可以 #pragma omp parallel for
        int sub_x = ...; int sub_y = ...;
        PRECISION sum_r = 0.0, sum_i = 0.0;
        for (int a = ...) for (int b = ...) {
        // complex multiply-accumulate
        }
        out[n].x = sum_r; out[n].y = sum_i;
    }
    }
    這是 serial 的，沒有任何 OMP 指令。可以加 #pragma omp parallel for 讓它跑多核心，但那就需要我們自己寫 baseline。

    簡單說：HeCBench 裡沒有現成的 CPU multi-core OpenMP 程式碼。 所有 -omp 版本都是 GPU target。

2. NCU is banned in the Lab Server since RmProfilingAdminOnly=1, thus forbidding GPU profiling, which is a huge problem. (The default monitoring nsight.xml failed.) We might need a server that enable profiling.

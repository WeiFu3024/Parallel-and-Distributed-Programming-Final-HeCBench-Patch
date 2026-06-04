# 實驗操作手冊（OMP Variant）
# LLM-Driven CPU OpenMP Optimization — Step-by-Step Operations Manual

---

## 總覽

本手冊對應 `experiment-omp/` 目錄，使用 `g++ -fopenmp` 編譯，`perf stat` profiling。
OMP variant 是 **CPU OpenMP**，非 GPU offload。

本實驗比較三種不同程度的資訊提供對 LLM 優化 CPU OpenMP kernel 效能的影響：

| Cell | 一句話說明 | LLM 拿到什麼 |
|------|----------|-------------|
| **Cell A** | 最無腦 baseline | 只有原始 code |
| **Cell B** | 加上靜態分析 | 原始 code + LLM 生成的 context |
| **Cell C** | 再加上迭代回饋 | 同 Cell B + perf profiling 回饋（最多 3 輪） |

比較方式：
- **A vs B** → 量化「靜態分析 context」的價值
- **B vs C** → 量化「iterative profiling feedback」的價值

---

## 前置準備

### Step 0.1：確認工具鏈 & 收集硬體資訊

```bash
# 確認 g++ 支援 OpenMP
g++ -fopenmp --version

# 確認 perf 可用
perf --version

# 收集 CPU 硬體規格
bash experiment-omp/scripts/hardware_spec.sh > experiment-omp/config/hardware_spec.txt
```

確認輸出包含 CPU 核心數、cache 層級、SIMD 能力、`omp_get_max_threads` 值、
以及 `perf list hardware` 中可用的 counter 名稱。

`--peak-bw` 值（後續 `perf_to_xml.py` 需要）：從 `hardware_spec.txt` 最末的 Note 取得，
或使用 `mbw` / 理論計算（記憶體頻道數 × data rate × bus width）。

### Step 0.2：初始化目錄結構

```bash
bash experiment-omp/scripts/init_experiment.sh <HeCBench_Path> accuracy aes fft
```

這會在 `experiment-omp/results/` 下為每個 benchmark 建立完整的目錄結構，
並從 HeCBench `src/{bench}-omp/` 複製 baseline `.cpp` 檔案。

### Step 0.3：編譯並 profile baseline

對每個 benchmark，編譯並 profile 原始 HeCBench OMP kernel：

```bash
# 1. 編譯 baseline
#    (compile.sh 會自動處理 include paths 和依賴檔案)
bash experiment-omp/scripts/compile.sh <HeCBench_Path> {benchmark} baseline 0 0
```

由於 compile.sh 是針對 LLM attempt 設計的，也可以直接手動編譯：

```bash
# accuracy
g++ -std=c++17 -O3 -fopenmp \
    -I<HeCBench>/src/accuracy-cuda \
    experiment-omp/results/accuracy/baseline/main.cpp \
    -o experiment-omp/results/accuracy/baseline/main

# aes（deps inline #include，需複製到同目錄）
cp <HeCBench>/src/aes-omp/{aes.h,kernels.cpp,reference.cpp,utils.cpp} \
   experiment-omp/results/aes/baseline/
g++ -std=c++17 -O3 -fopenmp \
    -I<HeCBench>/src/include \
    experiment-omp/results/aes/baseline/main.cpp \
    -o experiment-omp/results/aes/baseline/main

# fft
g++ -std=c++17 -O3 -fopenmp \
    -I<HeCBench>/src/fft-cuda \
    experiment-omp/results/fft/baseline/main.cpp \
    -o experiment-omp/results/fft/baseline/main
```

```bash
# 2. 用 perf stat profile（以 accuracy 為例）
BENCH=accuracy
THREADS=$(nproc)
perf stat -e \
  task-clock,cycles,instructions,branches,branch-misses,\
  L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\
  context-switches,cpu-migrations \
  -o experiment-omp/results/${BENCH}/baseline/perf_raw.txt \
  -- env OMP_NUM_THREADS=${THREADS} \
     experiment-omp/results/${BENCH}/baseline/main [arguments if any]
```

各 benchmark run arguments：

| Benchmark | arguments |
|-----------|-----------|
| `accuracy` | `8192 10000 10 100` |
| `aes` (encrypt) | `100 0 <HeCBench>/src/urng-sycl/URNG_Input.bmp` |
| `aes` (decrypt) | `100 1 <HeCBench>/src/urng-sycl/URNG_Input.bmp` |
| `fft` | `3 100` |

**aes 前置條件：** `URNG_Input.bmp` 是 DVC artifact，需先 `dvc pull`：
```bash
cd <HeCBench> && dvc pull
```

### Step 0.4：確認 perf counter 名稱

```bash
python experiment-omp/scripts/perf_to_xml.py discover \
    --input experiment-omp/results/{benchmark}/baseline/perf_raw.txt
```

確認 counter 名稱（`L1-dcache-loads` 等）在輸出中存在。
AMD CPU 的 L2 counter 名稱不同（參考 `perf list | grep -i l2`），
若不可用，對應 XML 欄位會顯示 `N/A`。

---

## 步驟一：生成 Context XML（每個 benchmark 做一次）

使用和 coding LLM **同一個模型**，開一個新的對話。

### 要貼給 LLM 的 prompt：

```
You are a CPU performance analyst.
Analyze the following source code and hardware specification,
then fill in the provided XML template.

You do NOT have access to a compiler, profiler, or runtime environment.
All analysis must be based on static code inspection and theoretical
computation. Do not guess — if you cannot determine a field from the
code, state your reasoning and best estimate.

<Hardware_Spec>
[在此貼上 experiment-omp/config/hardware_spec.txt 的完整內容]
</Hardware_Spec>

<Source_Code>
[在此貼上 baseline/main.cpp 的完整內容]
</Source_Code>

<Template>
<Context_Normalization>

  <Problem_Classification>
    <Domain>[Domain label]</Domain>
    <Algorithm>[Specific algorithm name]</Algorithm>
    <Algorithm_Purpose>[What the algorithm computes]</Algorithm_Purpose>
    <Algorithm_Steps>[Step-by-step description of the current implementation]</Algorithm_Steps>
    <Reference_Complexity>
      Time: [Time complexity]
      Space: [Space complexity]
    </Reference_Complexity>
  </Problem_Classification>

  <Bottleneck_Analysis>
    <Classification>[Compute-bound | Memory-bound | Latency-bound | Mixed]</Classification>
    <Arithmetic_Intensity>[Theoretical FLOP/Byte estimate with calculation]</Arithmetic_Intensity>
    <Ridge_Point_Comparison>[Compare against CPU ridge point:
      Peak_FLOPS / Peak_Bandwidth. e.g., 16 cores × 2 FMA × 8 floats(AVX2) × 4.5GHz
      = 1152 GFLOPS; BW = 50 GB/s → ridge = 23 FLOP/Byte]</Ridge_Point_Comparison>
    <Dominant_Limiter>[Memory bandwidth | FP throughput | Cache capacity |
      Branch misprediction | Thread synchronization overhead]</Dominant_Limiter>
  </Bottleneck_Analysis>

  <Memory_Access_Profile>
    <Pattern>[Streaming | Strided | Random | Sliding-window | Gather-scatter]</Pattern>
    <Read_Write_Ratio>[X:Y]</Read_Write_Ratio>
    <Reuse_Factor>[Theoretical reuse count per element]</Reuse_Factor>
    <Coalescing_Feasibility>[Sequential access | Strided — vectorization friendly |
      Random — not vectorizable]</Coalescing_Feasibility>
    <Optimal_Tiling_Hint>[Suggested tiling or blocking for cache reuse]</Optimal_Tiling_Hint>
  </Memory_Access_Profile>

  <Parallelism_Structure>
    <Granularity>[Work unit per thread, e.g., 1 row / 1 chunk of N/P elements]</Granularity>
    <Independence>[Fully independent | Reduction required | Has loop-carried dependency]</Independence>
    <Synchronization>[None | barrier | critical section | atomic | reduction clause]</Synchronization>
    <Recommended_Thread_Count>[Typically = physical core count for compute-bound,
      or = logical CPU count for memory-bound with SMT benefit]</Recommended_Thread_Count>
    <Vectorization_Opportunity>[Inner loop vectorizable | Requires SoA transform | Not vectorizable]</Vectorization_Opportunity>
    <NUMA_Awareness>[First-touch sufficient | Explicit binding needed | Not NUMA-sensitive]</NUMA_Awareness>
  </Parallelism_Structure>

  <Hardware_Constraints>
    <Architecture>[CPU model name]</Architecture>
    <ISA>[x86_64 | aarch64]</ISA>
    <Sockets>[Count]</Sockets>
    <Cores_Per_Socket>[Count]</Cores_Per_Socket>
    <Threads_Per_Core>[Count (SMT/HT)]</Threads_Per_Core>
    <Total_Logical_CPUs>[Count]</Total_Logical_CPUs>
    <Base_Clock_MHz>[MHz]</Base_Clock_MHz>
    <SIMD_Width>[SSE4=128bit | AVX2=256bit | AVX-512=512bit]</SIMD_Width>
    <L1d_Cache_Per_Core>[Bytes]</L1d_Cache_Per_Core>
    <L1i_Cache_Per_Core>[Bytes]</L1i_Cache_Per_Core>
    <L2_Cache_Per_Core>[Bytes]</L2_Cache_Per_Core>
    <L3_Cache_Total>[Bytes]</L3_Cache_Total>
    <Peak_Memory_Bandwidth>[GB/s]</Peak_Memory_Bandwidth>
    <NUMA_Nodes>[Count]</NUMA_Nodes>
    <NUMA_Topology>[e.g., Node 0: cores 0-15, Node 1: cores 16-31]</NUMA_Topology>
    <OpenMP_Max_Threads>[Count from omp_get_max_threads()]</OpenMP_Max_Threads>
  </Hardware_Constraints>

  <Data_Shape>
    <Input_Dimensions>[Dimensions extracted from source code]</Input_Dimensions>
    <Data_Type>[float32 | float64 | int32 | ...]</Data_Type>
    <Total_Memory_Footprint>[Estimated total bytes for all buffers]</Total_Memory_Footprint>
  </Data_Shape>

</Context_Normalization>
</Template>

Fill in every field of the template. Output only the completed XML.
```

### 儲存 context：

將 LLM 回傳的 XML 存到兩個地方（Cell B 和 Cell C 共用）：
```
experiment-omp/results/{benchmark}/cell_b/context.xml
experiment-omp/results/{benchmark}/cell_c/context.xml
```

---

## 步驟二：執行 Cell A（Pure Code, Single Round）

開一個**新的對話**。

### System Prompt（貼在 system message 或對話最開頭）：

```
You are a CPU performance engineer.
Your task is to optimize the provided OpenMP kernel for maximum CPU performance.
Output only the complete, compilable C++ source code as a single .cpp file.
Do not include explanatory text outside of code comments.

<Constraints>
  - Do NOT search the internet or reference external documentation.
  - Do NOT use external parallel libraries (Intel TBB, OpenCL, CUDA, MPI).
    Use only standard OpenMP pragmas and C/C++ standard library.
  - Do NOT modify host-side validation logic or input data generation.
  - Do NOT change data types or problem sizes from the original code.
  - You do NOT have access to a compiler, profiler, or runtime environment.
  - All compilation and profiling is handled externally.
</Constraints>
```

### User Prompt：

```
Optimize the following OpenMP kernel for maximum CPU performance.

<Source_Code>
[在此貼上 baseline/main.cpp 的完整內容]
</Source_Code>
```

### LLM 回覆後：

將 LLM 輸出的 `.cpp` 存到 `experiment-omp/results/{benchmark}/cell_a/round_1/attempt_1/main.cpp`，然後執行：

1. 編譯：
   ```bash
   bash experiment-omp/scripts/compile.sh ~/HeCBench {benchmark} cell_a 1 1
   ```
   結果寫入 `attempt_1/compile.log`，最後一行為 `COMPILE: SUCCESS` 或 `COMPILE: FAIL`。

2. **如果編譯成功** → 執行並驗證：
   ```bash
   bash experiment-omp/scripts/validate.sh ~/HeCBench {benchmark} cell_a 1 1
   ```
   結果寫入 `attempt_1/validate.log`，最後一行為 `VALIDATION: PASS` 或 `VALIDATION: FAIL`。

3. **如果編譯失敗或驗證失敗** → 進入 correctness retry（見下方）。

4. **如果通過** → profile 並記錄：
   ```bash
   # 複製 passing code
   cp experiment-omp/results/{benchmark}/cell_a/round_1/attempt_1/main.cpp \
      experiment-omp/results/{benchmark}/cell_a/round_1/final.cpp

   # Profile
   THREADS=$(nproc)
   perf stat -e \
     task-clock,cycles,instructions,branches,branch-misses,\
     L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\
     context-switches,cpu-migrations \
     -o experiment-omp/results/{benchmark}/cell_a/round_1/perf_raw.txt \
     -- env OMP_NUM_THREADS=${THREADS} \
        experiment-omp/results/{benchmark}/cell_a/round_1/attempt_1/main [args]
   ```

5. Cell A 完成。

---

## 步驟三：執行 Cell B（Context, Single Round）

開一個**新的對話**。

### System Prompt：

同 Cell A（一模一樣，直接複製 Step 2 的 system prompt）。

### User Prompt：

```
Use the following hardware and algorithmic context to guide your optimization:

[在此貼上 cell_b/context.xml 的完整內容]

Optimize the following OpenMP kernel for maximum CPU performance.

<Source_Code>
[在此貼上 baseline/main.cpp 的完整內容]
</Source_Code>
```

### LLM 回覆後：

流程同 Cell A（編譯 → 驗證 → profile）。
檔案存到 `cell_b/round_1/attempt_N/`。

---

## 步驟四：執行 Cell C（Context, Iterative）

### Round 1

開一個**新的對話**（這個對話會持續使用到 round 3）。

**System Prompt：** 同 Cell A / Cell B。

**User Prompt：** 與 Cell B 完全相同：

```
Use the following hardware and algorithmic context to guide your optimization:

[在此貼上 cell_c/context.xml 的完整內容]

Optimize the following OpenMP kernel for maximum CPU performance.

<Source_Code>
[在此貼上 baseline/main.cpp 的完整內容]
</Source_Code>
```

LLM 回覆後：編譯 → 驗證 → profile（perf stat）。存到 `cell_c/round_1/`。

如果 round 1 通過 correctness → 繼續 round 2。
如果 round 1 FAIL（3 次 retry 用完）→ 整個 benchmark 在 Cell C 標記為 FAIL，停止。

### Round 2

**在同一個對話中**（不開新對話），生成 feedback XML 後傳給 LLM。

```bash
THREADS=$(nproc)
PEAK_BW=<GB/s from hardware_spec.txt>

python experiment-omp/scripts/perf_to_xml.py feedback \
    --baseline experiment-omp/results/{benchmark}/baseline/perf_raw.txt \
    --yours    experiment-omp/results/{benchmark}/cell_c/round_1/perf_raw.txt \
    --round 1 \
    --threads ${THREADS} \
    --peak-bw ${PEAK_BW} \
    --output   experiment-omp/results/{benchmark}/cell_c/round_2/feedback.xml
```

**User Prompt（在同一對話中貼出）：**

```
Your previous kernel has been profiled. Below are the full perf stat
results for both the baseline kernel and your kernel:

[在此貼上 cell_c/round_2/feedback.xml 的完整內容]

Based on this profiling data, produce an improved version of the kernel.
Output only the complete, compilable C++ source code.
```

LLM 回覆後：編譯 → 驗證 → profile（perf stat）。存到 `cell_c/round_2/`。

如果通過 → 繼續 round 3。
如果 FAIL → 停止迭代，Cell C 最終結果取 round 1 的成績。

### Round 3

同 Round 2，但：

```bash
python experiment-omp/scripts/perf_to_xml.py feedback \
    --baseline experiment-omp/results/{benchmark}/baseline/perf_raw.txt \
    --yours    experiment-omp/results/{benchmark}/cell_c/round_2/perf_raw.txt \
    --round 2 \
    --threads ${THREADS} \
    --peak-bw ${PEAK_BW} \
    --output   experiment-omp/results/{benchmark}/cell_c/round_3/feedback.xml
```

**User Prompt（同一對話）：**

```
Your previous kernel has been profiled. Below are the full perf stat
results for both the baseline kernel and your kernel:

[在此貼上 cell_c/round_3/feedback.xml 的完整內容]

Based on this profiling data, produce an improved version of the kernel.
Output only the complete, compilable C++ source code.
```

Round 3 完成後，Cell C 最終結果 = **所有 passing round 中 wall time 最短的那個**。

---

## Correctness Retry 流程（所有 Cell 通用）

當 LLM 的 code 編譯失敗或數值驗證失敗時，**在同一個對話中**貼出以下 prompt：

```
Your submission failed with the following error:

<Error>
[在此貼上 compile.log 或 validate.log 的原始內容，不做任何修改]
</Error>

Fix the issue and resubmit the complete C++ source code.
```

LLM 回覆後，將新的 `.cpp` 存到下一個 attempt 目錄，再次執行：
```bash
bash experiment-omp/scripts/compile.sh ~/HeCBench {benchmark} {cell} {round} {attempt}
bash experiment-omp/scripts/validate.sh ~/HeCBench {benchmark} {cell} {round} {attempt}
```

### 規則：
- 每個 optimization round 最多 retry **3 次**（含第一次提交）
- Cell C 每進入新的 round，retry 計數器**歸零**
- 第 3 次還是失敗：
  - Cell A / B → 該 benchmark 標記 **FAIL**
  - Cell C → 停止迭代，保留之前 passing round 的最佳結果

### 存檔方式：
```
round_1/
├── attempt_1/
│   ├── main.cpp        ← 第 1 次提交（失敗）
│   ├── compile.log     ← 編譯錯誤
│   └── validate.log
├── attempt_2/
│   ├── main.cpp        ← retry 後的第 2 次提交
│   ├── compile.log
│   └── validate.log
├── attempt_3/
│   └── ...
├── final.cpp           ← 通過的那次 attempt 的 copy（如果有的話）
└── perf_raw.txt        ← final.cpp 的 perf stat profile
```

---

## 結果記錄

每個 benchmark 完成後，記錄以下資訊：

| 欄位 | 說明 |
|------|------|
| benchmark | benchmark 名稱 |
| cell | A / B / C |
| status | PASS / FAIL |
| best_wall_time_ms | 最佳 passing round 的 wall time（ms，從 perf_raw.txt 取） |
| baseline_wall_time_ms | HeCBench baseline 的 wall time（ms） |
| speedup | baseline_time / best_wall_time |
| total_attempts | 所有 round 的 attempt 總數 |
| passing_rounds | Cell C: 哪幾個 round 通過了 |
| context_correct | （事後標記）context 的 Bottleneck_Analysis 是否正確 |

---

## Checklist：每個 Benchmark 的完整流程

```
□ 工具確認: g++ -fopenmp 可用、perf 可用
□ Step 0: baseline 已編譯（g++ -fopenmp）、perf profile 已存檔
□ Step 0: perf counter 名稱已核對（perf_to_xml.py discover）
□ Step 1: context.xml 已生成（CPU Hardware_Constraints 格式）、存到 cell_b/ 和 cell_c/
□ Step 2: Cell A 已執行、結果已記錄
□ Step 3: Cell B 已執行、結果已記錄
□ Step 4: Cell C round 1 已執行
□ Step 4: Cell C round 2 feedback 已生成（perf_to_xml.py）、已執行
□ Step 4: Cell C round 3 feedback 已生成（perf_to_xml.py）、已執行
□ Step 5: 三個 Cell 的結果已填入結果表
□ Step 6: context 正確性已事後標記（optional）
```

---

## 附錄：完整 perf Feedback XML 範例

以下是 `perf_to_xml.py` 會產出的範例 XML，供參考：

```xml
<Iteration_Feedback round="1">

  <Baseline_Kernel>
    <Source>HeCBench original OpenMP code</Source>
    <Execution>
      <Wall_Time_ms>245.8</Wall_Time_ms>
      <Thread_Count>16</Thread_Count>
    </Execution>
    <Instruction_Profile>
      <Instructions>10,234,500,000</Instructions>
      <Cycles>5,432,100,000</Cycles>
      <IPC>1.88</IPC>
      <Branch_Miss_Rate>1.02%</Branch_Miss_Rate>
    </Instruction_Profile>
    <Cache>
      <L1d_Hit_Rate>98.7%</L1d_Hit_Rate>
      <L1d_Miss_Count>23,450,000</L1d_Miss_Count>
      <L2_Hit_Rate>92.3%</L2_Hit_Rate>
      <L2_Miss_Count>8,120,000</L2_Miss_Count>
      <LLC_Hit_Rate>78.5%</LLC_Hit_Rate>
      <LLC_Miss_Count>1,745,000</LLC_Miss_Count>
    </Cache>
    <Memory>
      <Estimated_Bandwidth_GBs>28.6</Estimated_Bandwidth_GBs>
      <Bandwidth_Utilization>57.2%</Bandwidth_Utilization>
    </Memory>
    <Threading>
      <Context_Switches>342</Context_Switches>
      <CPU_Migrations>12</CPU_Migrations>
      <CPU_Utilization>94.3%</CPU_Utilization>
    </Threading>
  </Baseline_Kernel>

  <Your_Kernel>
    <Execution>
      <Wall_Time_ms>178.2</Wall_Time_ms>
      <Thread_Count>16</Thread_Count>
    </Execution>
    <Instruction_Profile>
      <Instructions>8,912,300,000</Instructions>
      <Cycles>4,123,400,000</Cycles>
      <IPC>2.16</IPC>
      <Branch_Miss_Rate>0.87%</Branch_Miss_Rate>
    </Instruction_Profile>
    <Cache>
      <L1d_Hit_Rate>99.1%</L1d_Hit_Rate>
      <L1d_Miss_Count>18,200,000</L1d_Miss_Count>
      <L2_Hit_Rate>95.1%</L2_Hit_Rate>
      <L2_Miss_Count>4,350,000</L2_Miss_Count>
      <LLC_Hit_Rate>85.2%</LLC_Hit_Rate>
      <LLC_Miss_Count>645,000</LLC_Miss_Count>
    </Cache>
    <Memory>
      <Estimated_Bandwidth_GBs>14.1</Estimated_Bandwidth_GBs>
      <Bandwidth_Utilization>28.2%</Bandwidth_Utilization>
    </Memory>
    <Threading>
      <Context_Switches>128</Context_Switches>
      <CPU_Migrations>4</CPU_Migrations>
      <CPU_Utilization>97.8%</CPU_Utilization>
    </Threading>
  </Your_Kernel>

</Iteration_Feedback>
```

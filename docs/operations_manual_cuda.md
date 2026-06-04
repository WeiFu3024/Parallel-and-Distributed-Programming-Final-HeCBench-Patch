# 實驗操作手冊（CUDA Variant）
# LLM-Driven CUDA Optimization — Step-by-Step Operations Manual

---

## 總覽

本手冊對應 `experiment-cuda/` 目錄，使用 `nvcc` 編譯，`ncu` profiling。

本實驗比較三種不同程度的資訊提供對 LLM 優化 CUDA kernel 效能的影響：

| Cell | 一句話說明 | LLM 拿到什麼 |
|------|----------|-------------|
| **Cell A** | 最無腦 baseline | 只有原始 code |
| **Cell B** | 加上靜態分析 | 原始 code + LLM 生成的 context |
| **Cell C** | 再加上迭代回饋 | 同 Cell B + Nsight profiling 回饋（最多 3 輪） |

比較方式：
- **A vs B** → 量化「靜態分析 context」的價值
- **B vs C** → 量化「iterative profiling feedback」的價值

---

## 前置準備

### Step 0.1：收集硬體資訊

```bash
bash experiment-cuda/scripts/hardware_spec.sh > experiment-cuda/config/hardware_spec.txt
```

注：注意 GPU 只留下要拿來測試那顆的資訊。確認輸出包含 SM 數量、shared memory、bandwidth 等。

### Step 0.2：初始化目錄結構

```bash
bash experiment-cuda/scripts/init_experiment.sh <HeCBench_Path> benchmark1 benchmark2 ...
```

這會在 `experiment-cuda/results/` 下為每個 benchmark 建立完整的目錄結構，
並從 HeCBench `src/{bench}-cuda/` 複製 baseline `.cu` 檔案。

### Step 0.3：Profile baseline kernel

**所有指令均從 HeCBench 根目錄執行。**

```bash
# 1. 用 compile.sh 編譯 baseline（產出 binary 到 results/ 目錄）
bash experiment-cuda/scripts/compile.sh <HeCBench_Path> {benchmark} baseline
# 產出：experiment-cuda/results/{benchmark}/baseline/main
#       experiment-cuda/results/{benchmark}/baseline/compile.log

# 2. ⚠️  確認 nvcc 版本一致，避免版本不符導致 kernel 靜默失敗
#    compile.log 第二行顯示編譯時的 nvcc 版本，應與下方指令輸出一致
head -3 experiment-cuda/results/{benchmark}/baseline/compile.log
nvcc --version

# 3. 用 ncu profile baseline binary（路徑相對於 HeCBench 根目錄）
#    benchmark 引數請參考下方表格
ncu --set full --csv \
    --log-file experiment-cuda/results/{benchmark}/baseline/nsight_raw.csv \
    experiment-cuda/results/{benchmark}/baseline/main [arguments]

# 4. 轉成標準 XML（備用）；用 --kernel-name 指定目標 kernel 避免選錯
python experiment-cuda/scripts/profile_to_xml.py single \
    --input  experiment-cuda/results/{benchmark}/baseline/nsight_raw.csv \
    --output experiment-cuda/results/{benchmark}/baseline/nsight.xml \
    --kernel-name [target_kernel_name]
```

各 benchmark 的標準引數與目標 kernel 名稱：

| benchmark | HeCBench 目錄 | arguments | `--kernel-name` |
|-----------|-------------|-----------|-----------------|
| attention | attention-cuda | `65536 2048 0 1000` | discover 後填入 |
| attention-paged | attention-paged-cuda | `8 32 128 4096 128 100` | discover 後填入 |
| blas-gemm | blas-gemm-cuda | `4096 4096 4096 100` | `matrix_mul` |
| convolution3D | convolution3D-cuda | `32 96 256 26 26 5 100` | discover 後填入 |
| layernorm | layernorm-cuda | `8 1024 768 2000` | discover 後填入 |
| maxpool3d | maxpool3d-cuda | `2048 2048 96 100` | （單一，可省略） |
| bilateral | bilateral-cuda | `2960 1440 0.5 0.5 1000` | `bilateralFilter` |
| black-scholes | black-scholes-cuda | `100` | `getOutValOption` |
| binomial | binomial-cuda | （無引數） | `binomialOptionsKernel` |
| fft | fft-cuda | `3 100` | `fft1D_512` |
| jacobi | jacobi-cuda | （無引數） | `jacobi_step` |
| mcpr ¹ | mcpr-cuda | `<src_dir>/alphas.csv 10` | discover 後填入 |
| bh | bh-cuda | `100000 100` | discover 後填入 |
| hotspot ² | hotspot-cuda | `512 2 200 <HeCBench_Path>/data/hotspot/temp_512 <HeCBench_Path>/data/hotspot/power_512 output.out` | `calc_temp` |
| fluidSim | fluidSim-cuda | `10000` | `lbm` |
| d2q9-bgk ³ | d2q9-bgk-cuda | `<src_dir>/Inputs/input_256x256.params <src_dir>/Obstacles/obstacles_256x256.dat` | `d2q9_bgk` |
| heartwall ² | heartwall-cuda | `104` | discover 後填入 |
| bfs ² | bfs-cuda | `<HeCBench_Path>/data/bfs/graph1MW_6.txt` | discover 後填入 |
| page-rank | page-rank-cuda | `-n 20000 -i 100` | discover 後填入 |
| sssp | sssp-cuda | `-g 120 -t 1 -w 10 -r 100` | `SSSP_gpu` |
| nw | nw-cuda | `16384 10 100` | discover 後填入 |
| mis | mis-cuda | `<src_dir>/internet.egr 100` | discover 後填入 |
| scan | scan-cuda | `268435456 100` | discover 後填入 |
| bscan | bscan-cuda | `1000` | `binary_scan` |
| histogram | histogram-cuda | `--i=100` | discover 後填入 |
| segment-reduce | segment-reduce-cuda | `16384 100` | （單一，可省略） |
| sc | sc-cuda | `-a 0.1` | （單一，可省略） |
| sort | sort-cuda | `3 100` | discover 後填入 |
| transpose ⁴ | matrixT-cuda | `16384 16384 200` | discover 後填入 |
| lzss ⁵ | lzss-cuda | `-i <inputfile> -n 10` | discover 後填入 |

**備註：**

- `<src_dir>` = `<HeCBench_Path>/src/{benchmark}-cuda`
- `discover 後填入` = 該 benchmark 有多個 kernel，先用 `profile_to_xml.py discover` 確認最耗時的 kernel 名稱後填入

¹ **mcpr**：`alphas.csv.bz2` 附於 `src/mcpr-cuda/`，先解壓縮：
  `cd <HeCBench_Path>/src/mcpr-cuda && bzip2 -dk alphas.csv.bz2`

² **hotspot / heartwall / bfs**：需先從 HeCBench 根目錄執行 `dvc pull` 取得資料檔。
  - hotspot 資料：`<HeCBench_Path>/data/hotspot/`
  - heartwall 路徑硬編碼於 main.cu：`../data/heartwall/test.avi`（需從 binary 所在目錄能存取）
  - bfs 資料：`<HeCBench_Path>/data/bfs/`

³ **d2q9-bgk**：需 `dvc pull` 後解壓縮：
  `cd <HeCBench_Path>/src/d2q9-bgk-cuda && tar xzf test.tar.gz`
  產出 `Inputs/` 與 `Obstacles/` 子目錄。

⁴ **transpose**：HeCBench 目錄名為 `matrixT-cuda`（非 `transpose-cuda`）。

⁵ **lzss**：樣本資料需從外部下載：<https://github.com/hpdps-group/ICS23-GPULZ>。

### Step 0.4：確認 ncu metric 名稱

```bash
python experiment-cuda/scripts/profile_to_xml.py discover \
    --input experiment-cuda/results/{benchmark}/baseline/nsight_raw.csv
```

核對輸出的 metric 名稱是否與 `profile_to_xml.py` 頂部的 `METRIC_MAP` 一致。
如有不一致，修改 `METRIC_MAP`。

> ⚠️  **已知問題：metric 名稱格式不符（不影響）**
>
> `ncu --set full --csv` 輸出的 `Metric Name` 欄位為 **human-readable 名稱**
> （如 `Duration`、`Grid Size`、`Achieved Occupancy`、`DRAM Throughput`），
> 而非 ncu internal counter 名稱（如 `gpu__time_duration.avg`、`launch__grid_size`）。
>
> `METRIC_MAP` 的初始值為 internal 名稱，因此執行 discover 後**幾乎必定需要更新**。
> 以 CUDA 12.x 為例，正確的映射為：
>
> | METRIC_MAP key | 舊值（internal，錯誤） | 正確值（human-readable） |
> |---|---|---|
> | `kernel_time_ns` | `gpu__time_duration.avg` | `Duration` |
> | `grid_size` | `launch__grid_size` | `Grid Size` |
> | `block_size` | `launch__block_size` | `Block Size` |
> | `achieved_occupancy_pct` | `sm__warps_active.avg.pct_of_peak_sustained_active` | `Achieved Occupancy` |
> | `theoretical_occupancy_pct` | `sm__maximum_warps_per_active_cycle_pct` | `Theoretical Occupancy` |
> | `global_load_throughput` | `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second` | `Memory Throughput` |
> | `l1_hit_rate_pct` | `l1tex__t_sector_hit_rate.pct` | `L1/TEX Hit Rate` |
> | `l2_hit_rate_pct` | `lts__t_sector_hit_rate.pct` | `L2 Hit Rate` |
> | `dram_utilization_pct` | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | `DRAM Throughput` |
> | `sm_utilization_pct` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | `Compute (SM) Throughput` |
>
> 同時，`occupancy_limiting_factor()` 函式內的 key 也需要更新：
> - `"launch__registers_per_thread"` → `"Registers Per Thread"`
> - `"launch__shared_mem_per_block_dynamic"` → `"Dynamic Shared Memory Per Block"`
>
> 另注意：`ncu --set full --csv` **不輸出 per-reason 的 warp stall 分解**，
> `STALL_PREFIX` 無法匹配任何 metric，stall 分析欄位會顯示 `unknown / N/A`。

---

## 步驟一：生成 Context XML（每個 benchmark 做一次）

使用和 coding LLM **同一個模型**，開一個新的對話。

### 要貼給 LLM 的 prompt：

```
You are a GPU performance analyst.
Analyze the following source code and hardware specification,
then fill in the provided XML template.

You do NOT have access to a compiler, profiler, or runtime environment.
All analysis must be based on static code inspection and theoretical
computation. Do not guess — if you cannot determine a field from the
code, state your reasoning and best estimate.

<Hardware_Spec>
[在此貼上 experiment-cuda/config/hardware_spec.txt 的完整內容]
</Hardware_Spec>

<Source_Code>
[在此貼上 baseline/main.cu 的完整內容]
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
    <Ridge_Point_Comparison>[Above/Below theoretical ridge point]</Ridge_Point_Comparison>
    <Dominant_Limiter>[Estimated limiting factor]</Dominant_Limiter>
  </Bottleneck_Analysis>

  <Memory_Access_Profile>
    <Pattern>[Streaming | Strided | Random | Sliding-window | Gather-scatter]</Pattern>
    <Read_Write_Ratio>[X:Y]</Read_Write_Ratio>
    <Reuse_Factor>[Theoretical reuse count per element]</Reuse_Factor>
    <Coalescing_Feasibility>[Natural | Requires transformation | Not feasible]</Coalescing_Feasibility>
    <Optimal_Tiling_Hint>[Suggested tiling strategy, tile size, shared memory estimate]</Optimal_Tiling_Hint>
  </Memory_Access_Profile>

  <Parallelism_Structure>
    <Granularity>[Work unit per thread]</Granularity>
    <Independence>[Fully independent | Independent after reduction | Has RAW dependency]</Independence>
    <Synchronization>[None | __syncthreads() | Atomics | ...]</Synchronization>
    <Recommended_Block_Size>[Theoretical estimate based on hardware spec]</Recommended_Block_Size>
  </Parallelism_Structure>

  <Hardware_Constraints>
    <Architecture>[Architecture name]</Architecture>
    <Compute_Capability>[X.Y]</Compute_Capability>
    <SM_Count>[Count]</SM_Count>
    <Max_Threads_Per_SM>[Count]</Max_Threads_Per_SM>
    <Max_Threads_Per_Block>[Count]</Max_Threads_Per_Block>
    <Warp_Size>[Count]</Warp_Size>
    <Shared_Memory_Per_Block>[Bytes]</Shared_Memory_Per_Block>
    <Shared_Memory_Per_SM>[Bytes]</Shared_Memory_Per_SM>
    <Registers_Per_Block>[Count]</Registers_Per_Block>
    <Registers_Per_SM>[Count]</Registers_Per_SM>
    <L2_Cache_Size>[Bytes]</L2_Cache_Size>
    <Constant_Memory>[Bytes]</Constant_Memory>
    <Memory_Bus_Width>[Bits]</Memory_Bus_Width>
    <Peak_Memory_Bandwidth>[GB/s]</Peak_Memory_Bandwidth>
    <GPU_Clock_Rate>[MHz]</GPU_Clock_Rate>
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
experiment-cuda/results/{benchmark}/cell_b/context.xml
experiment-cuda/results/{benchmark}/cell_c/context.xml
```

---

## 步驟二：執行 Cell A（Pure Code, Single Round）

開一個**新的對話**。

### System Prompt（貼在 system message 或對話最開頭）：

```
You are a GPU performance engineer.
Your task is to optimize the provided CUDA kernel for maximum performance.
Output only the complete, compilable CUDA source code as a single .cu file.
Do not include explanatory text outside of code comments.

<Constraints>
  - Do NOT search the internet or reference external documentation.
  - Do NOT use acceleration libraries (cuBLAS, cuFFT, cuDNN, Thrust).
  - Do NOT modify host-side validation logic or input data generation.
  - Do NOT change data types or problem sizes from the original code.
  - You do NOT have access to a compiler, profiler, or runtime environment.
  - All compilation and profiling is handled externally.
</Constraints>
```

### User Prompt：

```
Optimize the following CUDA kernel for maximum GPU performance.

<Source_Code>
[在此貼上 baseline/main.cu 的完整內容]
</Source_Code>
```

### LLM 回覆後：

將 LLM 輸出的 `.cu` 存到 `experiment-cuda/results/{benchmark}/cell_a/round_1/attempt_1/main.cu`，然後執行：

1. 編譯：
   ```bash
   bash experiment-cuda/scripts/compile.sh <HeCBench_Path> {benchmark} cell_a 1 1
   ```
   結果寫入 `attempt_1/compile.log`，最後一行為 `COMPILE: SUCCESS` 或 `COMPILE: FAIL`。

2. **如果編譯成功** → 執行並驗證：
   ```bash
   bash experiment-cuda/scripts/validate.sh <HeCBench_Path> {benchmark} cell_a 1 1
   ```
   結果寫入 `attempt_1/validate.log`，最後一行為 `VALIDATION: PASS` 或 `VALIDATION: FAIL`。

3. **如果編譯失敗或驗證失敗** → 進入 correctness retry（見下方）。

4. **如果通過** → profile 並記錄：
   ```bash
   # 複製 passing code
   cp experiment-cuda/results/{benchmark}/cell_a/round_1/attempt_1/main.cu \
      experiment-cuda/results/{benchmark}/cell_a/round_1/final.cu

   # Profile；--kernel-name 為 ncu 的可選 flag，可縮小 CSV 只含目標 kernel
   ncu --set full --csv \
       --log-file experiment-cuda/results/{benchmark}/cell_a/round_1/nsight_raw.csv \
       experiment-cuda/results/{benchmark}/cell_a/round_1/attempt_1/main [arguments]
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

Optimize the following CUDA kernel for maximum GPU performance.

<Source_Code>
[在此貼上 baseline/main.cu 的完整內容]
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

Optimize the following CUDA kernel for maximum GPU performance.

<Source_Code>
[在此貼上 baseline/main.cu 的完整內容]
</Source_Code>
```

LLM 回覆後：

```bash
# 1. 編譯
bash experiment-cuda/scripts/compile.sh <HeCBench_Path> {benchmark} cell_c 1 1

# 2. 驗證
bash experiment-cuda/scripts/validate.sh <HeCBench_Path> {benchmark} cell_c 1 1

# 3. Profile；kernel 選擇由後續 profile_to_xml.py --kernel-name 控制（見 Round 2 feedback 指令）
ncu --set full --csv \
    --log-file experiment-cuda/results/{benchmark}/cell_c/round_1/nsight_raw.csv \
    experiment-cuda/results/{benchmark}/cell_c/round_1/attempt_1/main [arguments]

# 4. 複製 final.cu
cp experiment-cuda/results/{benchmark}/cell_c/round_1/attempt_1/main.cu \
   experiment-cuda/results/{benchmark}/cell_c/round_1/final.cu
```

存到 `cell_c/round_1/`。

如果 round 1 通過 correctness → 繼續 round 2。
如果 round 1 FAIL（3 次 retry 用完）→ 整個 benchmark 在 Cell C 標記為 FAIL，停止。

### Round 2

**在同一個對話中**（不開新對話），生成 feedback XML 後傳給 LLM。

```bash
python experiment-cuda/scripts/profile_to_xml.py feedback \
    --baseline experiment-cuda/results/{benchmark}/baseline/nsight_raw.csv \
    --yours    experiment-cuda/results/{benchmark}/cell_c/round_1/nsight_raw.csv \
    --round 1 \
    --kernel-name [target_kernel_name] \
    --output   experiment-cuda/results/{benchmark}/cell_c/round_2/feedback.xml
```

> **注意（多 kernel benchmark）**：`profile_to_xml.py` 預設以 `Duration` 最長的 kernel
> 作為代表，多 kernel 時可能選錯（例如 accuracy benchmark 的 `accuracy_kernel` 被選中
> 而非 `accuracy_kernel2`）。請在 `feedback` 指令加上 `--kernel-name` 指定目標 kernel。

**User Prompt（在同一對話中貼出）：**

```
Your previous kernel has been profiled. Below are the full Nsight Compute
results for both the baseline kernel and your kernel:

[在此貼上 cell_c/round_2/feedback.xml 的完整內容]

Based on this profiling data, produce an improved version of the kernel.
Output only the complete, compilable CUDA source code.
```

LLM 回覆後：

```bash
bash experiment-cuda/scripts/compile.sh <HeCBench_Path> {benchmark} cell_c 2 1
bash experiment-cuda/scripts/validate.sh <HeCBench_Path> {benchmark} cell_c 2 1

ncu --set full --csv \
    --log-file experiment-cuda/results/{benchmark}/cell_c/round_2/nsight_raw.csv \
    experiment-cuda/results/{benchmark}/cell_c/round_2/attempt_1/main [arguments]

cp experiment-cuda/results/{benchmark}/cell_c/round_2/attempt_1/main.cu \
   experiment-cuda/results/{benchmark}/cell_c/round_2/final.cu
```

存到 `cell_c/round_2/`。

如果通過 → 繼續 round 3。
如果 FAIL → 停止迭代，Cell C 最終結果取 round 1 的成績。

### Round 3

同 Round 2，但：

```bash
python experiment-cuda/scripts/profile_to_xml.py feedback \
    --baseline experiment-cuda/results/{benchmark}/baseline/nsight_raw.csv \
    --yours    experiment-cuda/results/{benchmark}/cell_c/round_2/nsight_raw.csv \
    --round 2 \
    --kernel-name [target_kernel_name] \
    --output   experiment-cuda/results/{benchmark}/cell_c/round_3/feedback.xml
```

**User Prompt（同一對話）：**

```
Your previous kernel has been profiled. Below are the full Nsight Compute
results for both the baseline kernel and your kernel:

[在此貼上 cell_c/round_3/feedback.xml 的完整內容]

Based on this profiling data, produce an improved version of the kernel.
Output only the complete, compilable CUDA source code.
```

LLM 回覆後：

```bash
bash experiment-cuda/scripts/compile.sh <HeCBench_Path> {benchmark} cell_c 3 1
bash experiment-cuda/scripts/validate.sh <HeCBench_Path> {benchmark} cell_c 3 1

ncu --set full --csv \
    --log-file experiment-cuda/results/{benchmark}/cell_c/round_3/nsight_raw.csv \
    experiment-cuda/results/{benchmark}/cell_c/round_3/attempt_1/main [arguments]

cp experiment-cuda/results/{benchmark}/cell_c/round_3/attempt_1/main.cu \
   experiment-cuda/results/{benchmark}/cell_c/round_3/final.cu
```

Round 3 完成後，Cell C 最終結果 = **所有 passing round 中 kernel time 最短的那個**。

---

## Correctness Retry 流程（所有 Cell 通用）

當 LLM 的 code 編譯失敗或數值驗證失敗時，**在同一個對話中**貼出以下 prompt：

```
Your submission failed with the following error:

<Error>
[在此貼上 compile.log 或 validate.log 的原始內容，不做任何修改]
</Error>

Fix the issue and resubmit the complete CUDA source code.
```

LLM 回覆後，將新的 `.cu` 存到下一個 attempt 目錄，再次執行：
```bash
bash experiment-cuda/scripts/compile.sh <HeCBench_Path> {benchmark} {cell} {round} {attempt}
bash experiment-cuda/scripts/validate.sh <HeCBench_Path> {benchmark} {cell} {round} {attempt}
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
│   ├── main.cu         ← 第 1 次提交（失敗）
│   ├── compile.log     ← 編譯錯誤
│   └── validate.log
├── attempt_2/
│   ├── main.cu         ← retry 後的第 2 次提交
│   ├── compile.log
│   └── validate.log
├── attempt_3/
│   └── ...
├── final.cu            ← 通過的那次 attempt 的 copy（如果有的話）
├── nsight_raw.csv      ← final.cu 的 profile
└── nsight.xml
```

---

## 結果記錄

每個 benchmark 完成後，記錄以下資訊：

| 欄位 | 說明 |
|------|------|
| benchmark | benchmark 名稱 |
| cell | A / B / C |
| status | PASS / FAIL |
| best_kernel_time_ms | 最佳 passing round 的 kernel time（ms） |
| baseline_time_ms | HeCBench baseline 的 kernel time（ms） |
| speedup | baseline_time / best_kernel_time |
| total_attempts | 所有 round 的 attempt 總數 |
| passing_rounds | Cell C: 哪幾個 round 通過了 |
| context_correct | （事後標記）context 的 Bottleneck_Analysis 是否正確 |

---

## Checklist：每個 Benchmark 的完整流程

```
□ Step 0: baseline 已編譯、ncu profile、存檔
□ Step 1: context.xml 已生成、存到 cell_b/ 和 cell_c/
□ Step 2: Cell A 已執行、結果已記錄
□ Step 3: Cell B 已執行、結果已記錄
□ Step 4: Cell C round 1 已執行
□ Step 4: Cell C round 2 feedback 已生成、已執行
□ Step 4: Cell C round 3 feedback 已生成、已執行
□ Step 5: 三個 Cell 的結果已填入結果表
□ Step 6: context 正確性已事後標記（optional）
```

---

## 附錄：完整 Nsight Feedback XML 範例

以下是 `profile_to_xml.py` 會產出的範例 XML，供參考：

```xml
<Iteration_Feedback round="1">

  <Baseline_Kernel>
    <Source>HeCBench original CUDA kernel</Source>
    <Execution>
      <Kernel_Name>fir_filter_kernel</Kernel_Name>
      <Kernel_Time_ms>12.3456</Kernel_Time_ms>
      <Grid_Size>4096, 1, 1</Grid_Size>
      <Block_Size>256, 1, 1</Block_Size>
    </Execution>
    <Occupancy>
      <Achieved>78.5%</Achieved>
      <Theoretical_Max>100.0%</Theoretical_Max>
      <Limiting_Factor>registers</Limiting_Factor>
    </Occupancy>
    <Memory>
      <Global_Load_Throughput>245.30 GB/s</Global_Load_Throughput>
      <Global_Store_Throughput>61.20 GB/s</Global_Store_Throughput>
      <Shared_Memory_Throughput>0.00 GB/s</Shared_Memory_Throughput>
      <L1_Hit_Rate>45.2%</L1_Hit_Rate>
      <L2_Hit_Rate>82.1%</L2_Hit_Rate>
      <DRAM_Utilization>67.8%</DRAM_Utilization>
    </Memory>
    <Compute>
      <SM_Utilization>35.4%</SM_Utilization>
      <FP32_Utilization>28.7%</FP32_Utilization>
      <Warp_Execution_Efficiency>0.97</Warp_Execution_Efficiency>
    </Compute>
    <Stall_Analysis>
      <Top_Stalls>
        <Stall reason="long_scoreboard" percentage="42.3%"/>
        <Stall reason="not_selected" percentage="18.9%"/>
        <Stall reason="lg_throttle" percentage="12.1%"/>
      </Top_Stalls>
    </Stall_Analysis>
  </Baseline_Kernel>

  <Your_Kernel>
    <Execution>
      <Kernel_Name>fir_filter_optimized</Kernel_Name>
      <Kernel_Time_ms>8.7210</Kernel_Time_ms>
      <Grid_Size>4096, 1, 1</Grid_Size>
      <Block_Size>256, 1, 1</Block_Size>
    </Execution>
    <Occupancy>
      <Achieved>85.2%</Achieved>
      <Theoretical_Max>100.0%</Theoretical_Max>
      <Limiting_Factor>shared_memory</Limiting_Factor>
    </Occupancy>
    <Memory>
      <Global_Load_Throughput>189.40 GB/s</Global_Load_Throughput>
      <Global_Store_Throughput>61.50 GB/s</Global_Store_Throughput>
      <Shared_Memory_Throughput>312.80 GB/s</Shared_Memory_Throughput>
      <L1_Hit_Rate>68.4%</L1_Hit_Rate>
      <L2_Hit_Rate>91.3%</L2_Hit_Rate>
      <DRAM_Utilization>48.2%</DRAM_Utilization>
    </Memory>
    <Compute>
      <SM_Utilization>52.1%</SM_Utilization>
      <FP32_Utilization>45.3%</FP32_Utilization>
      <Warp_Execution_Efficiency>0.99</Warp_Execution_Efficiency>
    </Compute>
    <Stall_Analysis>
      <Top_Stalls>
        <Stall reason="not_selected" percentage="28.5%"/>
        <Stall reason="math_pipe_throttle" percentage="22.1%"/>
        <Stall reason="short_scoreboard" percentage="15.7%"/>
      </Top_Stalls>
    </Stall_Analysis>
  </Your_Kernel>

</Iteration_Feedback>
```

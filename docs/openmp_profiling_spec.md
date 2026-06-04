# OpenMP Profiling Specification
## Supplement to CUDA Experiment Design

---

## 1. Context XML — CPU Hardware_Constraints

OpenMP benchmarks 的 Context XML 格式與 CUDA 版本相同，
只有 `<Hardware_Constraints>` 區塊替換為 CPU 版本。
其他欄位（Problem_Classification, Bottleneck_Analysis, Memory_Access_Profile,
Parallelism_Structure, Data_Shape）結構不變，但內容應反映 CPU 特性。

### 1.1 Hardware_Constraints（CPU 版本）

```xml
<Hardware_Constraints>
  <Architecture>[CPU model name, e.g., AMD Ryzen 9 7950X]</Architecture>
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
```

### 1.2 Parallelism_Structure（CPU 版本差異）

CPU 上的平行結構描述不同於 GPU，注意以下欄位的填法：

```xml
<Parallelism_Structure>
  <Granularity>[Work unit per thread, e.g., 1 row / 1 chunk of N/P elements]</Granularity>
  <Independence>[Fully independent | Reduction required | Has loop-carried dependency]</Independence>
  <Synchronization>[None | barrier | critical section | atomic | reduction clause]</Synchronization>
  <Recommended_Thread_Count>[Typically = physical core count for compute-bound,
    or = logical CPU count for memory-bound with SMT benefit]</Recommended_Thread_Count>
  <Vectorization_Opportunity>[Inner loop vectorizable | Requires SoA transform | Not vectorizable]</Vectorization_Opportunity>
  <NUMA_Awareness>[First-touch sufficient | Explicit binding needed | Not NUMA-sensitive]</NUMA_Awareness>
</Parallelism_Structure>
```

### 1.3 Bottleneck_Analysis（CPU 版本差異）

Ridge point 的計算方式不同：

```xml
<Bottleneck_Analysis>
  <Classification>[Compute-bound | Memory-bound | Latency-bound | Mixed]</Classification>
  <Arithmetic_Intensity>[Theoretical FLOP/Byte with calculation]</Arithmetic_Intensity>
  <Ridge_Point_Comparison>[Compare against CPU ridge point:
    Peak_FLOPS / Peak_Bandwidth. e.g., 16 cores × 2 FMA × 8 floats(AVX2) × 4.5GHz
    = 1152 GFLOPS; BW = 50 GB/s → ridge = 23 FLOP/Byte]</Ridge_Point_Comparison>
  <Dominant_Limiter>[Memory bandwidth | FP throughput | Cache capacity |
    Branch misprediction | Thread synchronization overhead]</Dominant_Limiter>
</Bottleneck_Analysis>
```

---

## 2. OpenMP Profiling Feedback XML

Equivalent to CUDA's `<Iteration_Feedback>`, used for Cell C round 2+.
Both baseline and LLM kernel use the same full format.

```xml
<Iteration_Feedback round="N">

  <Baseline_Kernel>
    <Source>HeCBench original OpenMP code</Source>
    <Execution>
      <Wall_Time_ms>[ms]</Wall_Time_ms>
      <Thread_Count>[OMP_NUM_THREADS used]</Thread_Count>
    </Execution>
    <Instruction_Profile>
      <Instructions>[total instruction count]</Instructions>
      <Cycles>[total cycle count]</Cycles>
      <IPC>[instructions per cycle]</IPC>
      <Branch_Miss_Rate>[%]</Branch_Miss_Rate>
    </Instruction_Profile>
    <Cache>
      <L1d_Hit_Rate>[%]</L1d_Hit_Rate>
      <L1d_Miss_Count>[count]</L1d_Miss_Count>
      <L2_Hit_Rate>[%]</L2_Hit_Rate>
      <L2_Miss_Count>[count]</L2_Miss_Count>
      <LLC_Hit_Rate>[%]</LLC_Hit_Rate>
      <LLC_Miss_Count>[count]</LLC_Miss_Count>
    </Cache>
    <Memory>
      <Estimated_Bandwidth_GBs>[GB/s — derived from LLC misses × cache line size / time]</Estimated_Bandwidth_GBs>
      <Bandwidth_Utilization>[% of peak]</Bandwidth_Utilization>
    </Memory>
    <Threading>
      <Context_Switches>[count]</Context_Switches>
      <CPU_Migrations>[count]</CPU_Migrations>
      <CPU_Utilization>[% — task-clock / wall-time / thread-count]</CPU_Utilization>
    </Threading>
  </Baseline_Kernel>

  <Your_Kernel>
    <Execution>
      <Wall_Time_ms>[ms]</Wall_Time_ms>
      <Thread_Count>[OMP_NUM_THREADS used]</Thread_Count>
    </Execution>
    <Instruction_Profile>
      <Instructions>[total instruction count]</Instructions>
      <Cycles>[total cycle count]</Cycles>
      <IPC>[instructions per cycle]</IPC>
      <Branch_Miss_Rate>[%]</Branch_Miss_Rate>
    </Instruction_Profile>
    <Cache>
      <L1d_Hit_Rate>[%]</L1d_Hit_Rate>
      <L1d_Miss_Count>[count]</L1d_Miss_Count>
      <L2_Hit_Rate>[%]</L2_Hit_Rate>
      <L2_Miss_Count>[count]</L2_Miss_Count>
      <LLC_Hit_Rate>[%]</LLC_Hit_Rate>
      <LLC_Miss_Count>[count]</LLC_Miss_Count>
    </Cache>
    <Memory>
      <Estimated_Bandwidth_GBs>[GB/s]</Estimated_Bandwidth_GBs>
      <Bandwidth_Utilization>[% of peak]</Bandwidth_Utilization>
    </Memory>
    <Threading>
      <Context_Switches>[count]</Context_Switches>
      <CPU_Migrations>[count]</CPU_Migrations>
      <CPU_Utilization>[%]</CPU_Utilization>
    </Threading>
  </Your_Kernel>

</Iteration_Feedback>
```

---

## 3. Profiling Command: perf stat

```bash
# Full profiling with hardware counters
perf stat -e \
  task-clock,\
  cycles,\
  instructions,\
  branches,\
  branch-misses,\
  L1-dcache-loads,\
  L1-dcache-load-misses,\
  l2_rqsts.references,\
  l2_rqsts.miss,\
  LLC-loads,\
  LLC-load-misses,\
  context-switches,\
  cpu-migrations \
  -o "${OUTPUT_DIR}/perf_raw.txt" \
  -- env OMP_NUM_THREADS=${THREADS} ./${BINARY} ${ARGS}
```

### Notes:
- Some counter names are Intel-specific (`l2_rqsts.references`). On AMD, use
  `l2_cache_accesses_from_dc_misses` or similar. Run `perf list` to check.
- `perf stat` can only monitor a limited number of counters simultaneously
  (typically 4-6 programmable counters). If the list above exceeds the limit,
  perf will multiplex and estimate. For precise counts, split into two runs.
- Add `--repeat 5` for statistical stability if needed.
- Wall time: parse from perf output or use `time` command.

### Alternative: likwid (more structured output)

```bash
# If likwid is installed — gives cleaner per-metric output
likwid-perfctr -C 0-${LAST_CORE} -g MEM_DP -m \
  -- env OMP_NUM_THREADS=${THREADS} ./${BINARY} ${ARGS}
```

likwid groups of interest: `MEM_DP` (memory + FP), `CACHE`, `BRANCH`, `L2`, `L3`.

---

## 4. perf Output → XML Conversion

The `perf stat -o` output looks like:

```
 Performance counter stats for '...':

         12,345.67 msec  task-clock
     5,432,100,000      cycles
    10,234,500,000      instructions     #    1.88  insn per cycle
     1,234,000,000      branches
        12,340,000      branch-misses    #    1.00% of all branches
     2,345,000,000      L1-dcache-loads
        23,450,000      L1-dcache-load-misses #    1.00% of all L1-dcache accesses
       ...
```

### Metric Derivation:

| XML Field | How to Compute |
|-----------|---------------|
| Wall_Time_ms | Parse from perf output or measure separately |
| Instructions | Direct from `instructions` counter |
| Cycles | Direct from `cycles` counter |
| IPC | `instructions / cycles` (or read from perf's `insn per cycle`) |
| Branch_Miss_Rate | `branch-misses / branches × 100` |
| L1d_Hit_Rate | `(L1-dcache-loads - L1-dcache-load-misses) / L1-dcache-loads × 100` |
| L2_Hit_Rate | `(l2_refs - l2_misses) / l2_refs × 100` |
| LLC_Hit_Rate | `(LLC-loads - LLC-load-misses) / LLC-loads × 100` |
| Estimated_Bandwidth_GBs | `LLC-load-misses × 64 (cache line) / wall_time_sec / 1e9` |
| Bandwidth_Utilization | `Estimated_Bandwidth / Peak_Bandwidth × 100` |
| CPU_Utilization | `task-clock / (wall_time × thread_count) × 100` |
| Context_Switches | Direct from `context-switches` |
| CPU_Migrations | Direct from `cpu-migrations` |

---

## 5. OpenMP-Specific Prompt Adjustments

System prompt 的 `<Constraints>` 區塊針對 OpenMP 修改：

```
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

Cell A/B/C 的 prompt 結構完全相同，只是 code 和 context 內容不同。

---

## 6. Directory Structure Additions

OpenMP benchmarks 使用與 CUDA 相同的目錄結構，
但 baseline code 來自 `{benchmark}-omp/` 而非 `{benchmark}-cuda/`：

```
experiment/results/{benchmark}-omp/
├── baseline/
│   ├── main.cpp                # from HeCBench src/{benchmark}-omp/
│   └── perf_raw.txt            # perf stat output
├── cell_a/
│   └── round_1/
│       ├── attempt_1/
│       │   ├── main.cpp
│       │   ├── compile.log
│       │   └── validate.log
│       ├── final.cpp
│       └── perf_raw.txt
├── cell_b/
│   ├── context.xml
│   └── round_1/...
└── cell_c/
    ├── context.xml
    ├── round_1/...
    ├── round_2/
    │   ├── feedback.xml        # perf-based Iteration_Feedback
    │   └── ...
    └── round_3/...
```

---

## 7. Feedback XML Example (OpenMP)

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

#!/bin/bash
# hardware_spec.sh — Collect full hardware spec for CUDA + OpenMP experiments
# Usage: bash hardware_spec.sh > hardware_spec.txt

echo "============================================"
echo "  Hardware Specification Report"
echo "  Generated: $(date -Iseconds)"
echo "============================================"

# ---- GPU (CUDA) ----
echo ""
echo "==================== GPU ===================="

if command -v nvidia-smi &>/dev/null; then
  echo ""
  echo "[nvidia-smi overview]"
  nvidia-smi --query-gpu=name,driver_version,pci.bus_id,compute_cap,memory.total,memory.free,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader,nounits | \
  awk -F', ' '{
    printf "  GPU Name:              %s\n", $1
    printf "  Driver Version:        %s\n", $2
    printf "  PCI Bus ID:            %s\n", $3
    printf "  Compute Capability:    %s\n", $4
    printf "  VRAM Total:            %s MiB\n", $5
    printf "  VRAM Free:             %s MiB\n", $6
    printf "  Max SM Clock:          %s MHz\n", $7
    printf "  Max Memory Clock:      %s MHz\n", $8
    printf "  Power Limit:           %s W\n", $9
  }'
else
  echo "  nvidia-smi not found"
fi

if command -v nvcc &>/dev/null; then
  echo ""
  echo "[CUDA Toolkit]"
  echo "  $(nvcc --version | grep 'release')"
fi

# deviceQuery — most detailed GPU spec
# Try common install paths
DEVICE_QUERY=""
for path in \
  /usr/local/cuda/extras/demo_suite/deviceQuery \
  /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery \
  "$HOME/NVIDIA_CUDA-*/extras/demo_suite/deviceQuery" \
  "$(which deviceQuery 2>/dev/null)"; do
  if [ -x "$path" ] 2>/dev/null; then
    DEVICE_QUERY="$path"
    break
  fi
done

if [ -n "$DEVICE_QUERY" ]; then
  echo ""
  echo "[deviceQuery output]"
  $DEVICE_QUERY 2>/dev/null | grep -E \
    "Device [0-9]|CUDA Capability|Total amount of global|Multiprocessors|CUDA Cores|GPU Max Clock|Memory Clock|Memory Bus Width|L2 Cache|Total amount of shared memory per block|Total number of registers|Warp size|Maximum number of threads per multiprocessor|Maximum number of threads per block|Total amount of constant memory|Total amount of shared memory per multiprocessor|Maximum memory bandwidth" \
    | sed 's/^/  /'
else
  echo ""
  echo "[deviceQuery not found — compiling inline version]"
  
  # Compile a minimal deviceQuery if nvcc is available
  if command -v nvcc &>/dev/null; then
    TMPFILE=$(mktemp /tmp/devquery_XXXXXX.cu)
    cat > "$TMPFILE" << 'CUEOF'
#include <cstdio>
int main() {
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("  Device %d: %s\n", i, p.name);
        printf("  Compute Capability:          %d.%d\n", p.major, p.minor);
        printf("  SM Count:                    %d\n", p.multiProcessorCount);
        printf("  Max Threads per SM:          %d\n", p.maxThreadsPerMultiProcessor);
        printf("  Max Threads per Block:       %d\n", p.maxThreadsPerBlock);
        printf("  Warp Size:                   %d\n", p.warpSize);
        printf("  Global Memory:               %.0f MiB\n", p.totalGlobalMem/1048576.0);
        printf("  Shared Memory per Block:     %zu bytes\n", p.sharedMemPerBlock);
        printf("  Shared Memory per SM:        %zu bytes\n", p.sharedMemPerMultiprocessor);
        printf("  Registers per Block:         %d\n", p.regsPerBlock);
        printf("  Registers per SM:            %d\n", p.regsPerMultiprocessor);
        printf("  L2 Cache Size:               %d bytes (%.1f MiB)\n", p.l2CacheSize, p.l2CacheSize/1048576.0);
        printf("  Constant Memory:             %zu bytes\n", p.totalConstMem);
        printf("  Memory Bus Width:            %d bits\n", p.memoryBusWidth);
        printf("  Memory Clock Rate:           %d MHz\n", p.memoryClockRate/1000);
        printf("  Peak Memory Bandwidth:       %.1f GB/s\n", 2.0*p.memoryClockRate*(p.memoryBusWidth/8)/1.0e6);
        printf("  GPU Clock Rate:              %d MHz\n", p.clockRate/1000);
        printf("  Concurrent Kernels:          %s\n", p.concurrentKernels ? "Yes" : "No");
        printf("  Async Engines:               %d\n", p.asyncEngineCount);
        printf("  ECC Enabled:                 %s\n", p.ECCEnabled ? "Yes" : "No");
    }
    return 0;
}
CUEOF
    TMPBIN=$(mktemp /tmp/devquery_XXXXXX)
    if nvcc -o "$TMPBIN" "$TMPFILE" 2>/dev/null; then
      $TMPBIN
    else
      echo "  Failed to compile deviceQuery. Check CUDA installation."
    fi
    rm -f "$TMPFILE" "$TMPBIN"
  else
    echo "  nvcc not found — cannot query GPU details."
  fi
fi

# ---- CPU (OpenMP) ----
echo ""
echo "==================== CPU ===================="

if [ -f /proc/cpuinfo ]; then
  echo ""
  echo "[CPU Model]"
  MODEL=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
  echo "  $MODEL"

  echo ""
  echo "[CPU Topology]"
  SOCKETS=$(grep 'physical id' /proc/cpuinfo | sort -u | wc -l)
  PHYS_CORES=$(grep 'cpu cores' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
  LOGICAL=$(nproc 2>/dev/null || grep -c '^processor' /proc/cpuinfo)
  echo "  Sockets:          $SOCKETS"
  echo "  Cores per Socket: $PHYS_CORES"
  echo "  Logical CPUs:     $LOGICAL"
  echo "  Threads per Core: $((LOGICAL / (SOCKETS * PHYS_CORES)))"
else
  echo "  /proc/cpuinfo not available"
  if command -v sysctl &>/dev/null; then
    echo "  [macOS detected]"
    sysctl -n machdep.cpu.brand_string 2>/dev/null | sed 's/^/  CPU: /'
    echo "  Physical Cores: $(sysctl -n hw.physicalcpu)"
    echo "  Logical CPUs:   $(sysctl -n hw.logicalcpu)"
  fi
fi

# CPU Cache
echo ""
echo "[CPU Cache]"
if command -v lscpu &>/dev/null; then
  lscpu | grep -i 'cache' | sed 's/^/  /'
elif [ -d /sys/devices/system/cpu/cpu0/cache ]; then
  for idx in /sys/devices/system/cpu/cpu0/cache/index*; do
    LEVEL=$(cat "$idx/level" 2>/dev/null)
    TYPE=$(cat "$idx/type" 2>/dev/null)
    SIZE=$(cat "$idx/size" 2>/dev/null)
    echo "  L${LEVEL} ${TYPE}: ${SIZE}"
  done
else
  echo "  Cache info not available"
fi

# ---- Memory ----
echo ""
echo "==================== Memory ===================="
if [ -f /proc/meminfo ]; then
  TOTAL=$(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)
  AVAIL=$(awk '/MemAvailable/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)
  echo "  Total RAM:     $TOTAL"
  echo "  Available RAM:  $AVAIL"
fi

# NUMA topology (relevant for OpenMP thread binding)
if command -v numactl &>/dev/null; then
  echo ""
  echo "[NUMA Topology]"
  numactl --hardware 2>/dev/null | grep -E 'available|node [0-9]+ size|node [0-9]+ cpus' | sed 's/^/  /'
elif [ -d /sys/devices/system/node ]; then
  echo ""
  echo "[NUMA Nodes]"
  for node in /sys/devices/system/node/node*; do
    NNAME=$(basename "$node")
    CPUS=$(cat "$node/cpulist" 2>/dev/null)
    MEM=$(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' "$node/meminfo" 2>/dev/null)
    echo "  $NNAME: CPUs [$CPUS], Memory $MEM"
  done
fi

# ---- Compiler & Runtime ----
echo ""
echo "==================== Toolchain ===================="

echo ""
echo "[Compilers]"
if command -v gcc &>/dev/null; then
  echo "  GCC:  $(gcc --version | head -1)"
fi
if command -v g++ &>/dev/null; then
  echo "  G++:  $(g++ --version | head -1)"
fi
if command -v nvcc &>/dev/null; then
  echo "  NVCC: $(nvcc --version | grep 'release' | xargs)"
fi

echo ""
echo "[OpenMP Support]"
# Check OpenMP version via a small test program
TMPFILE=$(mktemp /tmp/omp_test_XXXXXX.c)
cat > "$TMPFILE" << 'CEOF'
#include <stdio.h>
#include <omp.h>
int main() {
    printf("  OpenMP Version:      %d\n", _OPENMP);
    printf("  Max Threads:         %d\n", omp_get_max_threads());
    printf("  Num Procs:           %d\n", omp_get_num_procs());
    #pragma omp parallel
    {
        #pragma omp single
        printf("  Active Threads:      %d\n", omp_get_num_threads());
    }
    return 0;
}
CEOF
TMPBIN=$(mktemp /tmp/omp_test_XXXXXX)
if gcc -fopenmp -o "$TMPBIN" "$TMPFILE" 2>/dev/null; then
  OMP_DISPLAY_ENV=TRUE $TMPBIN 2>/dev/null
else
  echo "  OpenMP compilation failed — check gcc -fopenmp support"
fi
rm -f "$TMPFILE" "$TMPBIN"

# ---- OS ----
echo ""
echo "==================== OS ===================="
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "  OS:     $PRETTY_NAME"
fi
echo "  Kernel: $(uname -r)"
echo "  Arch:   $(uname -m)"

echo ""
echo "============================================"
echo "  Report complete"
echo "============================================"
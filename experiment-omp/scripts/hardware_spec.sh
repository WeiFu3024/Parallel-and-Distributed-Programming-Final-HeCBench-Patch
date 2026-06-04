#!/bin/bash
# hardware_spec.sh — Collect CPU hardware info for the OMP experiment
#
# Usage:
#   bash hardware_spec.sh > config/hardware_spec.txt
#
# Output: CPU topology, cache, SIMD, NUMA, OpenMP thread count, memory,
#         perf availability. Used to fill Hardware_Constraints in context XML
#         and to supply --peak-bw to perf_to_xml.py.

echo "============================================"
echo "  Hardware Specification Report (OMP variant)"
echo "  Generated: $(date -Iseconds)"
echo "============================================"

# ---- CPU ----
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
  if [ -n "$PHYS_CORES" ] && [ "$PHYS_CORES" -gt 0 ] 2>/dev/null; then
    echo "  Threads per Core: $((LOGICAL / (SOCKETS * PHYS_CORES)))"
  fi
  echo "  Base clock:"
  grep 'cpu MHz' /proc/cpuinfo | head -1 | awk '{printf "    %s MHz\n", $4}'
else
  echo "  /proc/cpuinfo not available"
fi

# CPU Cache
echo ""
echo "[CPU Cache]"
if command -v lscpu &>/dev/null; then
  lscpu | grep -iE 'L1d|L1i|L2|L3' | sed 's/^/  /'
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

# SIMD
echo ""
echo "[SIMD]"
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
  echo "  AVX-512 supported (512-bit)"
elif grep -q avx2 /proc/cpuinfo 2>/dev/null; then
  echo "  AVX2 supported (256-bit)"
elif grep -q avx /proc/cpuinfo 2>/dev/null; then
  echo "  AVX supported (256-bit)"
elif grep -q sse4_2 /proc/cpuinfo 2>/dev/null; then
  echo "  SSE4.2 supported (128-bit)"
else
  echo "  No AVX/SSE4 found in /proc/cpuinfo flags"
fi

# ---- Memory ----
echo ""
echo "==================== Memory ===================="
if [ -f /proc/meminfo ]; then
  TOTAL=$(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)
  AVAIL=$(awk '/MemAvailable/ {printf "%.1f GiB", $2/1048576}' /proc/meminfo)
  echo "  Total RAM:     $TOTAL"
  echo "  Available RAM: $AVAIL"
fi

# NUMA topology
echo ""
echo "[NUMA Topology]"
if command -v numactl &>/dev/null; then
  numactl --hardware 2>/dev/null | grep -E 'available|node [0-9]+ size|node [0-9]+ cpus' | sed 's/^/  /'
elif [ -d /sys/devices/system/node ]; then
  for node in /sys/devices/system/node/node*; do
    NNAME=$(basename "$node")
    CPUS=$(cat "$node/cpulist" 2>/dev/null)
    MEM=$(awk '/MemTotal/ {printf "%.1f GiB", $2/1048576}' "$node/meminfo" 2>/dev/null)
    echo "  $NNAME: CPUs [$CPUS], Memory $MEM"
  done
else
  echo "  numactl not found (sudo apt-get install numactl)"
fi

# ---- OpenMP ----
echo ""
echo "==================== OpenMP ===================="
echo ""
echo "[Thread Count]"
TMPFILE=$(mktemp /tmp/omp_test_XXXXXX.c)
cat > "$TMPFILE" << 'CEOF'
#include <stdio.h>
#include <omp.h>
int main() {
    printf("  omp_get_max_threads: %d\n", omp_get_max_threads());
    printf("  omp_get_num_procs:   %d\n", omp_get_num_procs());
    #pragma omp parallel
    {
        #pragma omp single
        printf("  actual parallel threads (default): %d\n", omp_get_num_threads());
    }
    return 0;
}
CEOF
TMPBIN=$(mktemp /tmp/omp_test_XXXXXX)
if gcc -fopenmp -o "$TMPBIN" "$TMPFILE" 2>/dev/null; then
  $TMPBIN 2>/dev/null
else
  echo "  OpenMP probe failed — check gcc -fopenmp support"
fi
rm -f "$TMPFILE" "$TMPBIN"

# ---- Toolchain ----
echo ""
echo "==================== Toolchain ===================="
echo ""
echo "[Compilers]"
if command -v g++ &>/dev/null; then
  echo "  G++:  $(g++ --version | head -1)"
fi
if command -v clang++ &>/dev/null; then
  echo "  clang++: $(clang++ --version | head -1)"
fi

echo ""
echo "[perf]"
if command -v perf &>/dev/null; then
  perf --version 2>/dev/null
  echo "  Available hardware events (first 15):"
  perf list hardware 2>/dev/null | head -15 | sed 's/^/    /'
  echo ""
  echo "  NOTE: If L2 events (l2_rqsts.*) are missing, find AMD/ARM equivalents"
  echo "        via: perf list | grep -i l2"
else
  echo "  WARNING: perf not found."
  echo "           Install: sudo apt-get install linux-tools-generic linux-tools-$(uname -r)"
fi

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
echo "  NOTE: Peak memory bandwidth (--peak-bw for perf_to_xml.py):"
echo "  Run: sudo apt-get install mbw"
echo "       mbw 512 | grep AVG | head -3"
echo "  Or use theoretical: (memory channels × data rate × bus width / 8) GB/s"
echo "============================================"

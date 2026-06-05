#!/bin/bash
# compile.sh — Compile an LLM-generated CUDA kernel attempt (or the baseline)
#
# Usage:
#   bash compile.sh <hecbench_path> <benchmark> <cell> <round> <attempt> [sm_arch]
#   bash compile.sh <hecbench_path> <benchmark> baseline [sm_arch]
#
# Arguments:
#   hecbench_path  Absolute path to HeCBench root (e.g. ~/HeCBench)
#   benchmark      See the case statement below for the full list of supported benchmarks.
#   cell           One of: cell_a, cell_b, cell_c  — or "baseline"
#   round          Round number: 1, 2, or 3         — omit for baseline
#   attempt        Attempt number: 1, 2, or 3       — omit for baseline
#   sm_arch        Optional CUDA arch (e.g. sm_86). Default: auto-detect.
#
# Output:
#   <target_dir>/compile.log  — nvcc stdout+stderr, ends with COMPILE: SUCCESS or COMPILE: FAIL
#   <target_dir>/main         — compiled binary (only written on success)
#   Exit 0 on success, 1 on failure.
#
# Examples:
#   bash compile.sh ~/HeCBench accuracy baseline
#   bash compile.sh ~/HeCBench accuracy cell_a 1 1

HECBENCH_PATH="${1:?Usage: $0 <hecbench_path> <benchmark> <cell> [round] [attempt] [sm_arch]}"
BENCHMARK="${2:?Missing: benchmark}"
CELL="${3:?Missing: cell (baseline | cell_a | cell_b | cell_c)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve target directory based on whether this is a baseline or LLM attempt
if [ "${CELL}" = "baseline" ]; then
  ATTEMPT_DIR="${SCRIPT_DIR}/../results/${BENCHMARK}/baseline"
  LOG_LABEL="${BENCHMARK}/baseline"
  SM_ARCH="${4:-auto}"
else
  ROUND="${4:?Missing: round (required when cell != baseline)}"
  ATTEMPT="${5:?Missing: attempt (required when cell != baseline)}"
  SM_ARCH="${6:-auto}"
  ATTEMPT_DIR="${SCRIPT_DIR}/../results/${BENCHMARK}/${CELL}/round_${ROUND}/attempt_${ATTEMPT}"
  LOG_LABEL="${BENCHMARK}/${CELL}/round_${ROUND}/attempt_${ATTEMPT}"
fi

# Map benchmark name → HeCBench source directory (most follow ${bench}-cuda)
case "${BENCHMARK}" in
  transpose) SRC_DIR="${HECBENCH_PATH}/src/matrixT-cuda" ;;
  *)         SRC_DIR="${HECBENCH_PATH}/src/${BENCHMARK}-cuda" ;;
esac

LOG="${ATTEMPT_DIR}/compile.log"
BINARY="${ATTEMPT_DIR}/main"

# --- Input validation ---
if [ ! -f "${ATTEMPT_DIR}/main.cu" ]; then
  echo "ERROR: ${ATTEMPT_DIR}/main.cu not found" >&2
  if [ "${CELL}" = "baseline" ]; then
    echo "       Run init_experiment.sh first to copy the baseline .cu file." >&2
  fi
  exit 1
fi
if [ ! -d "${SRC_DIR}" ]; then
  echo "ERROR: Source directory not found: ${SRC_DIR}" >&2
  exit 1
fi
if ! command -v nvcc &>/dev/null; then
  echo "ERROR: nvcc not found in PATH" >&2
  exit 1
fi

# --- Auto-detect SM arch ---
if [ "${SM_ARCH}" = "auto" ]; then
  _cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | head -1 | tr -d ' \r\n.')
  if [ -z "${_cap}" ]; then
    echo "WARNING: Cannot detect GPU compute capability, defaulting to sm_80" >&2
    _cap="80"
  fi
  SM_ARCH="sm_${_cap}"
fi

# --- Benchmark-specific configuration ---
# EXTRA_DEPS: individual files to copy from SRC_DIR into ATTEMPT_DIR
# EXTRA_DIRS: directories to copy recursively from SRC_DIR into ATTEMPT_DIR
# EXTRA_SRCS: extra source files passed to nvcc (relative to ATTEMPT_DIR; must also appear in EXTRA_DEPS)
# EXTRA_NVCC_FLAGS: additional nvcc flags (include paths, linker flags, etc.)
case "${BENCHMARK}" in
  accuracy)
    EXTRA_DEPS=("reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  aes)
    # kernels.cu / reference.cu / utils.cu are #include'd directly by main.cu
    EXTRA_DEPS=("aes.h" "kernels.cu" "reference.cu" "utils.cu")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=("-I${HECBENCH_PATH}/src/include")
    ;;
  attention)
    EXTRA_DEPS=("kernels.h" "reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  attention-paged)
    EXTRA_DEPS=("attention_dtypes.h" "attention_generic.cuh" "attention_kernels.cuh"
                "attention_utils.cuh" "cuda_compat.h" "dtype_bfloat16.cuh"
                "dtype_float32.cuh" "kvcache.h" "reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  bfs)
    # util.h lives in bfs-sycl/ and is referenced via -I
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=("-I${HECBENCH_PATH}/src/bfs-sycl")
    ;;
  bh)
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  bilateral)
    EXTRA_DEPS=("reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=("-Xcompiler" "-fopenmp")
    ;;
  binomial)
    # kernel.cu / reference.cu are separate translation units
    EXTRA_DEPS=("kernel.cu" "reference.cu" "binomialOptions.h" "realtype.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=("kernel.cu" "reference.cu")
    EXTRA_NVCC_FLAGS=()
    ;;
  black-scholes)
    # All extra .cu files are #include'd textually by main.cu
    EXTRA_DEPS=("blackScholesAnalyticEngineKernels.cu"
                "blackScholesAnalyticEngineKernelsCpu.cu"
                "blackScholesAnalyticEngineStructs.cuh"
                "blackScholesAnalyticEngineKernels.cuh"
                "blackScholesAnalyticEngineKernelsCpu.cuh"
                "errorFunctConsts.cuh")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  blas-gemm)
    EXTRA_DEPS=("utils.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=("-lcublas")
    ;;
  bscan)
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  convolution3D)
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  d2q9-bgk)
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  fft)
    EXTRA_DEPS=("fft1D_512.h" "ifft1D_512.h" "reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  fluidSim)
    # kernels.cu is a separate translation unit
    EXTRA_DEPS=("kernels.cu" "reference.h" "utils.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=("kernels.cu")
    EXTRA_NVCC_FLAGS=()
    ;;
  heartwall)
    # kernel/ and util/ are subdirectories with .c/.cu sources
    # main.cu uses -I./util/timer and -I./util/file (as in the original Makefile)
    EXTRA_DEPS=("main.h")
    EXTRA_DIRS=("kernel" "util")
    EXTRA_SRCS=("kernel/kernel.cu"
                "util/avi/avilib.c"
                "util/avi/avimod.c"
                "util/file/file.c"
                "util/timer/timer.c")
    EXTRA_NVCC_FLAGS=("-I${ATTEMPT_DIR}/util/timer"
                     "-I${ATTEMPT_DIR}/util/file"
                     "-I${ATTEMPT_DIR}/util/avi"
                     "-lm")
    ;;
  histogram)
    EXTRA_DEPS=("histogram_gmem_atomics.h" "histogram_smem_atomics.h"
                "mersenne.h" "test_util.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  hotspot)
    EXTRA_DEPS=("hotspot.h" "kernel.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  jacobi)
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  layernorm)
    # layernorm-cuda has its own common.h and reference.h (different from rmsnorm-cuda's).
    # reduce.cuh and utils.cuh come from rmsnorm-cuda via -I.
    EXTRA_DEPS=("common.h" "reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=("-I${HECBENCH_PATH}/src/rmsnorm-cuda")
    ;;
  lzss)
    EXTRA_DEPS=("utils.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  maxpool3d)
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  mcpr)
    EXTRA_DEPS=("kernels.h" "reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  mis)
    EXTRA_DEPS=("graph.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  nw)
    EXTRA_DEPS=("reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  page-rank)
    EXTRA_DEPS=("reference.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=("-Xcompiler" "-fopenmp")
    ;;
  sc)
    # device_sc.cu / host_sc.cpp are separate TUs; support/ contains headers
    EXTRA_DEPS=("device_sc.cu" "host_sc.cpp" "kernel.h")
    EXTRA_DIRS=("support")
    EXTRA_SRCS=("device_sc.cu" "host_sc.cpp")
    EXTRA_NVCC_FLAGS=("-lpthread")
    ;;
  scan)
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  segment-reduce)
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  sort)
    EXTRA_DEPS=("sort_bottom_scan.h" "sort_reduce.h" "sort_top_scan.h")
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  sssp)
    # kernel.cu is a separate translation unit; support/ contains headers
    EXTRA_DEPS=("kernel.cu" "kernel.h")
    EXTRA_DIRS=("support")
    EXTRA_SRCS=("kernel.cu")
    EXTRA_NVCC_FLAGS=("-lpthread")
    ;;
  transpose)
    # SRC_DIR is already remapped to matrixT-cuda above
    EXTRA_DEPS=()
    EXTRA_DIRS=()
    EXTRA_SRCS=()
    EXTRA_NVCC_FLAGS=()
    ;;
  *)
    echo "ERROR: Unsupported benchmark '${BENCHMARK}'." >&2
    echo "       Supported: accuracy, aes, attention, attention-paged, bfs, bh, bilateral," >&2
    echo "                  binomial, black-scholes, blas-gemm, bscan, convolution3D, d2q9-bgk," >&2
    echo "                  fft, fluidSim, heartwall, histogram, hotspot, jacobi, layernorm," >&2
    echo "                  lzss, maxpool3d, mcpr, mis, nw, page-rank, sc, scan, segment-reduce," >&2
    echo "                  sort, sssp, transpose" >&2
    exit 1
    ;;
esac

# --- Copy dependency files into attempt directory ---
for dep in "${EXTRA_DEPS[@]}"; do
  src="${SRC_DIR}/${dep}"
  if [ ! -f "${src}" ]; then
    echo "WARNING: Dependency not found: ${src}" >&2
  else
    cp "${src}" "${ATTEMPT_DIR}/${dep}"
  fi
done

# --- Copy dependency directories into attempt directory ---
for dir in "${EXTRA_DIRS[@]}"; do
  src="${SRC_DIR}/${dir}"
  if [ ! -d "${src}" ]; then
    echo "WARNING: Dependency directory not found: ${src}" >&2
  else
    cp -r "${src}" "${ATTEMPT_DIR}/"
  fi
done

# Remove stale binary to avoid confusion if compile fails
rm -f "${BINARY}"

# --- Write log header ---
{
  echo "=== compile.sh | ${LOG_LABEL} ==="
  echo "=== arch: ${SM_ARCH} | $(date -Iseconds) ==="
  echo "=== nvcc: $(nvcc --version 2>&1 | grep -o 'release [0-9.]*' | head -1) ==="
  echo ""
} > "${LOG}"

# --- Build extra source path list ---
EXTRA_SRC_PATHS=()
for src in "${EXTRA_SRCS[@]}"; do
  EXTRA_SRC_PATHS+=("${ATTEMPT_DIR}/${src}")
done

# --- Compile ---
tmpout=$(mktemp)
nvcc \
  -std=c++17 \
  -O3 \
  -arch="${SM_ARCH}" \
  -Xcompiler -Wall \
  "${ATTEMPT_DIR}/main.cu" \
  "${EXTRA_SRC_PATHS[@]}" \
  "${EXTRA_NVCC_FLAGS[@]}" \
  -o "${BINARY}" \
  > "${tmpout}" 2>&1
NVCC_EXIT=$?

tee -a "${LOG}" < "${tmpout}"
rm -f "${tmpout}"

# --- Report result ---
echo "" >> "${LOG}"
if [ "${NVCC_EXIT}" -eq 0 ]; then
  echo "COMPILE: SUCCESS" | tee -a "${LOG}"
  exit 0
else
  echo "COMPILE: FAIL (nvcc exit ${NVCC_EXIT})" | tee -a "${LOG}"
  echo "COMPILE: FAIL (nvcc exit ${NVCC_EXIT})" >&2
  exit 1
fi

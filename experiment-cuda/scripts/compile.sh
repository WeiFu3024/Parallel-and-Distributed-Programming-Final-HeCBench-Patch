#!/bin/bash
# compile.sh — Compile an LLM-generated CUDA kernel attempt (or the baseline)
#
# Usage:
#   bash compile.sh <hecbench_path> <benchmark> <cell> <round> <attempt> [sm_arch]
#   bash compile.sh <hecbench_path> <benchmark> baseline [sm_arch]
#
# Arguments:
#   hecbench_path  Absolute path to HeCBench root (e.g. ~/HeCBench)
#   benchmark      One of: accuracy, aes, fft
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
SRC_DIR="${HECBENCH_PATH}/src/${BENCHMARK}-cuda"
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
case "${BENCHMARK}" in
  accuracy)
    EXTRA_DEPS=("reference.h")
    EXTRA_NVCC_FLAGS=()
    ;;
  aes)
    EXTRA_DEPS=("aes.h" "kernels.cu" "reference.cu" "utils.cu")
    EXTRA_NVCC_FLAGS=("-I${HECBENCH_PATH}/src/include")
    ;;
  fft)
    EXTRA_DEPS=("fft1D_512.h" "ifft1D_512.h" "reference.h")
    EXTRA_NVCC_FLAGS=()
    ;;
  *)
    echo "ERROR: Unsupported benchmark '${BENCHMARK}'. Supported: accuracy, aes, fft" >&2
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

# Remove stale binary to avoid confusion if compile fails
rm -f "${BINARY}"

# --- Write log header ---
{
  echo "=== compile.sh | ${LOG_LABEL} ==="
  echo "=== arch: ${SM_ARCH} | $(date -Iseconds) ==="
  echo "=== nvcc: $(nvcc --version 2>&1 | grep -o 'release [0-9.]*' | head -1) ==="
  echo ""
} > "${LOG}"

# --- Compile ---
tmpout=$(mktemp)
nvcc \
  -std=c++17 \
  -O3 \
  -arch="${SM_ARCH}" \
  -Xcompiler -Wall \
  "${EXTRA_NVCC_FLAGS[@]}" \
  "${ATTEMPT_DIR}/main.cu" \
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

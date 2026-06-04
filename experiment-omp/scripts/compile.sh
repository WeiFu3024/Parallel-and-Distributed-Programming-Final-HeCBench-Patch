#!/bin/bash
# compile.sh — Compile an LLM-generated CPU OpenMP attempt (or the baseline)
#
# Compiler: g++ with -fopenmp (standard CPU OpenMP).
# Profiled with: perf stat (see perf_to_xml.py)
#
# Usage:
#   bash compile.sh <hecbench_path> <benchmark> <cell> <round> <attempt>
#   bash compile.sh <hecbench_path> <benchmark> baseline
#
# Arguments:
#   hecbench_path  Absolute path to HeCBench root (e.g. ~/HeCBench)
#   benchmark      One of: accuracy, aes, fft
#   cell           One of: cell_a, cell_b, cell_c  — or "baseline"
#   round          Round number: 1, 2, or 3         — omit for baseline
#   attempt        Attempt number: 1, 2, or 3       — omit for baseline
#
# Output:
#   <target_dir>/compile.log  — g++ stdout+stderr, ends with COMPILE: SUCCESS or COMPILE: FAIL
#   <target_dir>/main         — compiled binary (only written on success)
#   Exit 0 on success, 1 on failure.
#
# Examples:
#   bash compile.sh ~/HeCBench accuracy baseline
#   bash compile.sh ~/HeCBench accuracy cell_a 1 1

HECBENCH_PATH="${1:?Usage: $0 <hecbench_path> <benchmark> <cell> [round] [attempt]}"
BENCHMARK="${2:?Missing: benchmark}"
CELL="${3:?Missing: cell (baseline | cell_a | cell_b | cell_c)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve target directory based on whether this is a baseline or LLM attempt
if [ "${CELL}" = "baseline" ]; then
  ATTEMPT_DIR="${SCRIPT_DIR}/../results/${BENCHMARK}/baseline"
  LOG_LABEL="${BENCHMARK}/baseline"
else
  ROUND="${4:?Missing: round (required when cell != baseline)}"
  ATTEMPT="${5:?Missing: attempt (required when cell != baseline)}"
  ATTEMPT_DIR="${SCRIPT_DIR}/../results/${BENCHMARK}/${CELL}/round_${ROUND}/attempt_${ATTEMPT}"
  LOG_LABEL="${BENCHMARK}/${CELL}/round_${ROUND}/attempt_${ATTEMPT}"
fi
OMP_SRC_DIR="${HECBENCH_PATH}/src/${BENCHMARK}-omp"
LOG="${ATTEMPT_DIR}/compile.log"
BINARY="${ATTEMPT_DIR}/main"

# --- Input validation ---
if [ ! -f "${ATTEMPT_DIR}/main.cpp" ]; then
  echo "ERROR: ${ATTEMPT_DIR}/main.cpp not found" >&2
  if [ "${CELL}" = "baseline" ]; then
    echo "       Run init_experiment.sh first to copy the baseline .cpp file." >&2
  fi
  exit 1
fi
if [ ! -d "${OMP_SRC_DIR}" ]; then
  echo "ERROR: Source directory not found: ${OMP_SRC_DIR}" >&2
  exit 1
fi
if ! command -v g++ &>/dev/null; then
  echo "ERROR: g++ not found in PATH." >&2
  exit 1
fi

# --- Benchmark-specific configuration ---
# accuracy-omp: reference.h lives in accuracy-cuda (shared header)
# aes-omp:      main.cpp #includes reference.cpp, kernels.cpp, utils.cpp; needs aes.h + SDKBitMap.h
# fft-omp:      reference.h lives in fft-cuda (shared header)
case "${BENCHMARK}" in
  accuracy)
    EXTRA_DEPS=()
    EXTRA_FLAGS=("-I${HECBENCH_PATH}/src/accuracy-cuda")
    ;;
  aes)
    EXTRA_DEPS=("aes.h" "kernels.cpp" "reference.cpp" "utils.cpp")
    EXTRA_FLAGS=("-I${HECBENCH_PATH}/src/include")
    ;;
  fft)
    EXTRA_DEPS=()
    EXTRA_FLAGS=("-I${HECBENCH_PATH}/src/fft-cuda")
    ;;
  *)
    echo "ERROR: Unsupported benchmark '${BENCHMARK}'. Supported: accuracy, aes, fft" >&2
    exit 1
    ;;
esac

# --- Copy local dependency files into attempt directory ---
for dep in "${EXTRA_DEPS[@]}"; do
  src="${OMP_SRC_DIR}/${dep}"
  if [ ! -f "${src}" ]; then
    echo "WARNING: Dependency not found: ${src}" >&2
  else
    cp "${src}" "${ATTEMPT_DIR}/${dep}"
  fi
done

rm -f "${BINARY}"

# --- Write log header ---
{
  echo "=== compile.sh (OMP) | ${LOG_LABEL} ==="
  echo "=== compiler: g++ $(g++ --version | head -1) | $(date -Iseconds) ==="
  echo ""
} > "${LOG}"

# --- Compile ---
tmpout=$(mktemp)
g++ \
  -std=c++17 \
  -O3 \
  -Wall \
  -fopenmp \
  "${EXTRA_FLAGS[@]}" \
  "${ATTEMPT_DIR}/main.cpp" \
  -o "${BINARY}" \
  > "${tmpout}" 2>&1
GXX_EXIT=$?

tee -a "${LOG}" < "${tmpout}"
rm -f "${tmpout}"

echo "" >> "${LOG}"
if [ "${GXX_EXIT}" -eq 0 ]; then
  echo "COMPILE: SUCCESS" | tee -a "${LOG}"
  exit 0
else
  echo "COMPILE: FAIL (g++ exit ${GXX_EXIT})" | tee -a "${LOG}"
  echo "COMPILE: FAIL (g++ exit ${GXX_EXIT})" >&2
  exit 1
fi

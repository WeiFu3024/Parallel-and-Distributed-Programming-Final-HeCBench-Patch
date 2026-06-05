#!/bin/bash
# init_experiment.sh — Create CUDA experiment directory structure
#
# Usage:
#   bash init_experiment.sh <hecbench_path> <benchmark_name_1> [benchmark_name_2] ...
#
# Example:
#   bash init_experiment.sh ~/HeCBench accuracy aes fft
#
# This will:
#   1. Create experiment-cuda/results/{benchmark}/ with baseline/cell_a/cell_b/cell_c dirs
#   2. Copy the original .cu file from HeCBench src/{bench}-cuda/ into baseline/ as main.cu
#   3. Create round and attempt subdirectories

set -euo pipefail

HECBENCH_PATH="${1:?Usage: $0 <hecbench_path> <benchmark_1> [benchmark_2] ...}"
shift
BENCHMARKS=("$@")

if [ ${#BENCHMARKS[@]} -eq 0 ]; then
  echo "Error: provide at least one benchmark name."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/.."

mkdir -p "${EXPERIMENT_DIR}/config"

echo "Creating experiment structure at ${EXPERIMENT_DIR}/"

for bench in "${BENCHMARKS[@]}"; do
  echo "  Setting up benchmark: ${bench}"

  BASE="${EXPERIMENT_DIR}/results/${bench}"
  mkdir -p "${BASE}/baseline"

  # --- Map benchmark name → HeCBench source directory ---
  # Most benchmarks follow the pattern ${bench}-cuda; exceptions listed here.
  case "${bench}" in
    transpose) BENCH_SRC_DIR="${HECBENCH_PATH}/src/matrixT-cuda" ;;
    *)         BENCH_SRC_DIR="${HECBENCH_PATH}/src/${bench}-cuda" ;;
  esac

  # --- Map benchmark name → actual main source filename ---
  # Most use main.cu; fall back to ${bench}.cu; explicit overrides below.
  case "${bench}" in
    black-scholes) BENCH_SRC_FILE="blackScholesAnalyticEngine.cu" ;;
    histogram)     BENCH_SRC_FILE="histogram_compare.cu" ;;
    hotspot)       BENCH_SRC_FILE="hotspot.cu" ;;
    *)             BENCH_SRC_FILE="" ;;  # will try main.cu then ${bench}.cu
  esac

  # --- Find and copy baseline source (always stored as main.cu) ---
  if [ -n "${BENCH_SRC_FILE}" ]; then
    CUDA_SRC="${BENCH_SRC_DIR}/${BENCH_SRC_FILE}"
    if [ -f "${CUDA_SRC}" ]; then
      cp "${CUDA_SRC}" "${BASE}/baseline/main.cu"
      echo "    Copied baseline from ${CUDA_SRC} (renamed to main.cu)"
    else
      echo "    WARNING: ${CUDA_SRC} not found. Skipping baseline copy."
    fi
  else
    CUDA_SRC="${BENCH_SRC_DIR}/main.cu"
    CUDA_SRC_ALT="${BENCH_SRC_DIR}/${bench}.cu"
    if [ -f "${CUDA_SRC}" ]; then
      cp "${CUDA_SRC}" "${BASE}/baseline/main.cu"
      echo "    Copied baseline from ${CUDA_SRC}"
    elif [ -f "${CUDA_SRC_ALT}" ]; then
      cp "${CUDA_SRC_ALT}" "${BASE}/baseline/main.cu"
      echo "    Copied baseline from ${CUDA_SRC_ALT} (renamed to main.cu)"
    else
      echo "    WARNING: No source file found in ${BENCH_SRC_DIR}/. Skipping baseline copy."
    fi
  fi

  # --- Create cell subdirectories ---
  for attempt in 1 2 3; do
    mkdir -p "${BASE}/cell_a/round_1/attempt_${attempt}"
    mkdir -p "${BASE}/cell_b/round_1/attempt_${attempt}"
  done
  for round in 1 2 3; do
    for attempt in 1 2 3; do
      mkdir -p "${BASE}/cell_c/round_${round}/attempt_${attempt}"
    done
  done
done

echo ""
echo "Directory structure created. Next steps:"
echo "  1. bash scripts/hardware_spec.sh > ${EXPERIMENT_DIR}/config/hardware_spec.txt"
echo "  2. Copy system_prompt.txt to ${EXPERIMENT_DIR}/config/"
echo "  3. Profile each baseline: ncu --set full --csv ... > results/{bench}/baseline/nsight_raw.csv"
echo "  4. Generate context.xml for each benchmark (analysis LLM)"
echo ""
echo "Done."

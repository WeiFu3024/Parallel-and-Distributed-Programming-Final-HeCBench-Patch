#!/bin/bash
# init_experiment.sh — Create OpenMP-target experiment directory structure
#
# Usage:
#   bash init_experiment.sh <hecbench_path> <benchmark_name_1> [benchmark_name_2] ...
#
# Example:
#   bash init_experiment.sh ~/HeCBench accuracy aes fft
#
# This will:
#   1. Create experiment-omp/results/{benchmark}/ with baseline/cell_a/cell_b/cell_c dirs
#   2. Copy the original .cpp file from HeCBench src/{bench}-omp/ into baseline/
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

  # Copy the original OpenMP target source
  OMP_SRC="${HECBENCH_PATH}/src/${bench}-omp/main.cpp"
  if [ -f "${OMP_SRC}" ]; then
    cp "${OMP_SRC}" "${BASE}/baseline/main.cpp"
    echo "    Copied baseline from ${OMP_SRC}"
  else
    echo "    WARNING: ${OMP_SRC} not found. Skipping baseline copy."
  fi

  # Cell A: 1 round, up to 3 attempts
  for attempt in 1 2 3; do
    mkdir -p "${BASE}/cell_a/round_1/attempt_${attempt}"
  done

  # Cell B: 1 round, up to 3 attempts
  for attempt in 1 2 3; do
    mkdir -p "${BASE}/cell_b/round_1/attempt_${attempt}"
  done

  # Cell C: 3 rounds, each up to 3 attempts
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
echo "  3. Compile each baseline: g++ -std=c++17 -O3 -fopenmp -I... results/{bench}/baseline/main.cpp -o results/{bench}/baseline/main"
echo "  4. Profile each baseline with perf stat:"
echo "       perf stat -e task-clock,cycles,instructions,branches,branch-misses,\\"
echo "         L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\\"
echo "         context-switches,cpu-migrations \\"
echo "         -o results/{bench}/baseline/perf_raw.txt \\"
echo "         -- env OMP_NUM_THREADS=\$(nproc) results/{bench}/baseline/main [args]"
echo "  5. Generate context.xml for each benchmark (analysis LLM)"
echo ""
echo "Done."

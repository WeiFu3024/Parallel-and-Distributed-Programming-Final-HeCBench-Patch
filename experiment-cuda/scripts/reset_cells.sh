#!/bin/bash
# reset_cells.sh — Clear cell_a / cell_b / cell_c for selected benchmarks (baseline untouched)
#
# Usage:
#   bash reset_cells.sh [--yes] <benchmark_1> [benchmark_2] ...
#   bash reset_cells.sh [--yes] --benchmark-set assigned
#
# Options:
#   --yes               Skip confirmation prompt
#   --benchmark-set SET assigned = the 10 operator-assigned benchmarks
#
# Examples:
#   bash reset_cells.sh binomial bfs nw
#   bash reset_cells.sh --yes --benchmark-set assigned
#
# What is removed:
#   experiment-cuda/results/{benchmark}/cell_a/
#   experiment-cuda/results/{benchmark}/cell_b/
#   experiment-cuda/results/{benchmark}/cell_c/
#   (including main.cu, logs, binaries, nsight_*, feedback.xml, context.xml, etc.)
#
# What is kept:
#   experiment-cuda/results/{benchmark}/baseline/

set -euo pipefail

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,24p' "$0"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"

ASSIGNED_BENCHMARKS=(
  blas-gemm maxpool3d binomial mcpr fluidSim
  bfs nw bscan sc lzss
)

SKIP_CONFIRM=false
BENCHMARKS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --yes) SKIP_CONFIRM=true; shift ;;
    --benchmark-set)
      SET="${2:?Missing value for --benchmark-set}"
      if [ "${SET}" != "assigned" ]; then
        echo "ERROR: --benchmark-set must be 'assigned' (got: ${SET})" >&2
        exit 1
      fi
      BENCHMARKS=("${ASSIGNED_BENCHMARKS[@]}")
      shift 2
      ;;
    -*)
      echo "ERROR: Unknown option: $1" >&2
      exit 1
      ;;
    *)
      BENCHMARKS+=("$1")
      shift
      ;;
  esac
done

if [ ${#BENCHMARKS[@]} -eq 0 ]; then
  echo "ERROR: provide at least one benchmark name or --benchmark-set assigned" >&2
  exit 1
fi

recreate_cell_dirs() {
  local base="$1"
  local attempt

  for attempt in 1 2 3; do
    mkdir -p "${base}/cell_a/round_1/attempt_${attempt}"
    mkdir -p "${base}/cell_b/round_1/attempt_${attempt}"
  done
  for round in 1 2 3; do
    for attempt in 1 2 3; do
      mkdir -p "${base}/cell_c/round_${round}/attempt_${attempt}"
    done
  done
}

echo "The following cell directories will be REMOVED and recreated empty:"
echo "  (baseline/ is NOT touched)"
echo ""

for bench in "${BENCHMARKS[@]}"; do
  base="${RESULTS_DIR}/${bench}"
  if [ ! -d "${base}" ]; then
    echo "  WARNING: ${base} not found — will skip"
    continue
  fi
  for cell in cell_a cell_b cell_c; do
    if [ -d "${base}/${cell}" ]; then
      echo "  ${bench}/${cell}/"
    fi
  done
done

echo ""
if ! $SKIP_CONFIRM; then
  read -r -p "Type 'yes' to continue: " answer
  if [ "${answer}" != "yes" ]; then
    echo "Aborted."
    exit 1
  fi
fi

for bench in "${BENCHMARKS[@]}"; do
  base="${RESULTS_DIR}/${bench}"
  if [ ! -d "${base}" ]; then
    continue
  fi

  echo "Resetting ${bench}..."
  for cell in cell_a cell_b cell_c; do
    rm -rf "${base}/${cell}"
  done
  recreate_cell_dirs "${base}"
  echo "  Done: ${bench} (baseline preserved)"
done

echo ""
echo "Reset complete."

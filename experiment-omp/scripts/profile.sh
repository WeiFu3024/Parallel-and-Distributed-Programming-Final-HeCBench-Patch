#!/bin/bash
# profile.sh — Profile an OMP binary; uses perf stat if available, else /usr/bin/time -v
#
# Usage:
#   bash profile.sh <hecbench_path> <benchmark> baseline
#   bash profile.sh <hecbench_path> <benchmark> <cell> <round>
#
# Output files (in target directory):
#   perf_raw.txt   — written when perf_event_paranoid <= 1  (use perf_to_xml.py)
#   time_raw.txt   — written when perf is unavailable       (use time_to_xml.py)
#
# Examples:
#   bash profile.sh ~/HeCBench accuracy baseline
#   bash profile.sh ~/HeCBench accuracy cell_a 1

HECBENCH_PATH="${1:?Usage: $0 <hecbench_path> <benchmark> <cell> [round]}"
BENCHMARK="${2:?Missing: benchmark}"
CELL="${3:?Missing: cell (baseline | cell_a | cell_b | cell_c)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results/${BENCHMARK}"

# --- Resolve target directory and binary ---
if [ "${CELL}" = "baseline" ]; then
  TARGET_DIR="${RESULTS_DIR}/baseline"
  BINARY="${TARGET_DIR}/main"
  LOG_LABEL="${BENCHMARK}/baseline"
else
  ROUND="${4:?Missing: round (required when cell != baseline)}"
  TARGET_DIR="${RESULTS_DIR}/${CELL}/round_${ROUND}"
  BINARY="${TARGET_DIR}/final"
  LOG_LABEL="${BENCHMARK}/${CELL}/round_${ROUND}"
fi

# --- Pre-flight ---
if [ ! -f "${BINARY}" ]; then
  echo "ERROR: Binary not found: ${BINARY}" >&2
  if [ "${CELL}" = "baseline" ]; then
    echo "       Run: bash compile.sh ${HECBENCH_PATH} ${BENCHMARK} baseline" >&2
  else
    echo "       Binary 'final' is created after validate.sh passes." >&2
  fi
  exit 1
fi

THREADS="${OMP_NUM_THREADS:-$(nproc)}"

# --- Benchmark-specific run arguments ---
case "${BENCHMARK}" in
  accuracy)
    # CPU is ~100× slower than GPU; 5 repeats gives stable average timing
    RUN_ARGS=(8192 10000 10 5)
    ;;
  aes)
    BMP="${HECBENCH_PATH}/src/urng-sycl/URNG_Input.bmp"
    if [ ! -f "${BMP}" ]; then
      echo "ERROR: Bitmap not found: ${BMP}" >&2
      echo "       Pull it with: cd ${HECBENCH_PATH} && dvc pull" >&2
      exit 1
    fi
    # Profile encrypt only; 5 iterations for stable average
    RUN_ARGS=(5 0 "${BMP}")
    ;;
  fft)
    # CPU: 10 passes is enough for stable timing
    RUN_ARGS=(3 10)
    ;;
  *)
    echo "ERROR: Unsupported benchmark '${BENCHMARK}'. Supported: accuracy, aes, fft" >&2
    exit 1
    ;;
esac

echo "=== profile.sh (OMP) | ${LOG_LABEL} ==="
echo "=== OMP_NUM_THREADS=${THREADS} | $(date -Iseconds) ==="

# --- Decide which profiler to use ---
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)

USE_PERF=false
if command -v perf &>/dev/null && [ -n "${PARANOID}" ] && [ "${PARANOID}" -le 1 ] 2>/dev/null; then
  USE_PERF=true
fi

if $USE_PERF; then
  # ---- perf stat path ----
  echo "=== profiler: perf stat (paranoid=${PARANOID}) ==="
  PERF_OUT="${TARGET_DIR}/perf_raw.txt"

  # No spaces in event list — perf rejects whitespace between events
  EVENTS="task-clock,cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.references,l2_rqsts.miss,LLC-loads,LLC-load-misses,context-switches,cpu-migrations"

  perf stat \
    -e "${EVENTS}" \
    -o "${PERF_OUT}" \
    -- env OMP_NUM_THREADS="${THREADS}" "${BINARY}" "${RUN_ARGS[@]}"

  PROF_EXIT=$?
  if [ "${PROF_EXIT}" -ne 0 ]; then
    echo "ERROR: perf stat exited with code ${PROF_EXIT}" >&2
    exit 1
  fi

  echo ""
  echo "Output: ${PERF_OUT}"
  echo "Next:   python ${SCRIPT_DIR}/perf_to_xml.py single --input ${PERF_OUT} --threads ${THREADS} --peak-bw <GB/s> --output ${TARGET_DIR}/perf.xml"

else
  # ---- /usr/bin/time -v fallback ----
  if [ -n "${PARANOID}" ] && [ "${PARANOID}" -gt 1 ] 2>/dev/null; then
    echo "=== profiler: /usr/bin/time -v (perf blocked: paranoid=${PARANOID}) ==="
    echo "    To unlock perf: sudo sysctl -w kernel.perf_event_paranoid=1"
  else
    echo "=== profiler: /usr/bin/time -v (perf not available) ==="
  fi

  TIME_OUT="${TARGET_DIR}/time_raw.txt"
  tmpout=$(mktemp)

  # Run binary, capture stdout; capture /usr/bin/time -v output (goes to stderr)
  /usr/bin/time -v env OMP_NUM_THREADS="${THREADS}" "${BINARY}" "${RUN_ARGS[@]}" \
    > "${tmpout}" 2>"${TIME_OUT}"

  PROF_EXIT=$?

  # Prepend binary stdout to time_raw.txt so both are in one file
  {
    echo "=== BINARY OUTPUT ==="
    cat "${tmpout}"
    echo ""
    echo "=== TIME OUTPUT ==="
    cat "${TIME_OUT}"
  } > "${TIME_OUT}.combined"
  mv "${TIME_OUT}.combined" "${TIME_OUT}"
  rm -f "${tmpout}"

  if [ "${PROF_EXIT}" -ne 0 ]; then
    echo "ERROR: Binary exited with code ${PROF_EXIT}" >&2
    exit 1
  fi

  echo ""
  echo "Output: ${TIME_OUT}"
  echo "Next:   python ${SCRIPT_DIR}/time_to_xml.py single --input ${TIME_OUT} --benchmark ${BENCHMARK} --threads ${THREADS} --output ${TARGET_DIR}/time.xml"
fi

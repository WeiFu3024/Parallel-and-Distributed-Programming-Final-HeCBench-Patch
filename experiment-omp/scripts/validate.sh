#!/bin/bash
# validate.sh — Run a compiled OpenMP target attempt and verify numerical correctness
#
# Usage:
#   bash validate.sh <hecbench_path> <benchmark> <cell> <round> <attempt>
#
# Arguments:
#   hecbench_path  Absolute path to HeCBench root (e.g. ~/HeCBench)
#   benchmark      One of: accuracy, aes, fft
#   cell           One of: cell_a, cell_b, cell_c
#   round          Round number: 1, 2, or 3
#   attempt        Attempt number: 1, 2, or 3
#
# Output:
#   <attempt_dir>/validate.log  — runtime stdout+stderr, ends with VALIDATION: PASS or VALIDATION: FAIL
#   Exit 0 on pass, 1 on fail/error.
#
# Notes:
#   - The binary must already exist (run compile.sh first).
#   - Binary timeout: 120 seconds per run.
#   - For aes: requires URNG_Input.bmp (pull via `dvc pull` in HeCBench root if missing).
#
# Example:
#   bash validate.sh ~/HeCBench accuracy cell_a 1 1

HECBENCH_PATH="${1:?Usage: $0 <hecbench_path> <benchmark> <cell> <round> <attempt>}"
BENCHMARK="${2:?Missing: benchmark}"
CELL="${3:?Missing: cell}"
ROUND="${4:?Missing: round}"
ATTEMPT="${5:?Missing: attempt}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ATTEMPT_DIR="${SCRIPT_DIR}/../results/${BENCHMARK}/${CELL}/round_${ROUND}/attempt_${ATTEMPT}"
BINARY="${ATTEMPT_DIR}/main"
LOG="${ATTEMPT_DIR}/validate.log"

if [ ! -f "${BINARY}" ]; then
  echo "ERROR: Binary not found: ${BINARY}" >&2
  echo "       Run compile.sh first." >&2
  exit 1
fi

VALIDATION_PASS=true

run_binary() {
  local tmpout
  tmpout=$(mktemp)

  echo "--- $ $(basename "${BINARY}") $* ---" | tee -a "${LOG}"

  timeout 120 "${BINARY}" "$@" > "${tmpout}" 2>&1
  local exit_code=$?

  tee -a "${LOG}" < "${tmpout}"
  echo "" >> "${LOG}"
  rm -f "${tmpout}"

  if [ "${exit_code}" -eq 124 ]; then
    echo "ERROR: Binary timed out (120s)" | tee -a "${LOG}" >&2
    VALIDATION_PASS=false
  elif [ "${exit_code}" -ne 0 ]; then
    echo "ERROR: Binary exited with code ${exit_code}" | tee -a "${LOG}" >&2
    VALIDATION_PASS=false
  fi
}

{
  echo "=== validate.sh (OMP) | ${BENCHMARK}/${CELL}/round_${ROUND}/attempt_${ATTEMPT} ==="
  echo "=== $(date -Iseconds) ==="
  echo ""
} > "${LOG}"

case "${BENCHMARK}" in

  accuracy)
    # Args: <nrows> <ndims> <top_k> <repeat>
    # Output: "PASS" or "FAIL" on its own line for each grid_size iteration
    # CPU is ~100× slower than GPU; repeat=1 is enough to check correctness
    run_binary 8192 10000 10 1

    if grep -q "^FAIL$" "${LOG}"; then
      VALIDATION_PASS=false
    fi
    if ! grep -q "^PASS$" "${LOG}"; then
      VALIDATION_PASS=false
    fi
    ;;

  aes)
    # Requires URNG_Input.bmp (DVC artifact — run `dvc pull` in HeCBench root if missing)
    BMP="${HECBENCH_PATH}/src/urng-sycl/URNG_Input.bmp"
    if [ ! -f "${BMP}" ]; then
      echo "ERROR: Bitmap not found: ${BMP}" | tee -a "${LOG}" >&2
      echo "       Pull it with: cd ${HECBENCH_PATH} && dvc pull" | tee -a "${LOG}" >&2
      VALIDATION_PASS=false
    else
      # Run encrypt then decrypt; both must print "Pass"
      # CPU is ~100× slower than GPU; 1 iteration is enough for correctness
      run_binary 1 0 "${BMP}"
      run_binary 1 1 "${BMP}"

      if grep -q "^Fail$" "${LOG}"; then
        VALIDATION_PASS=false
      fi
      pass_count=$(grep -c "^Pass$" "${LOG}"; true)
      if [ "${pass_count}" -lt 2 ]; then
        VALIDATION_PASS=false
      fi
    fi
    ;;

  fft)
    # Args: <problem_size_index> <passes>  (index 3 = 256MB)
    # Output: "FFT PASS" / "FFT FAIL" and "iFFT PASS" / "iFFT FAIL"
    # CPU: 1 pass is enough to check correctness
    run_binary 3 1

    if grep -q " FAIL" "${LOG}"; then
      VALIDATION_PASS=false
    fi
    if ! grep -q "^FFT PASS$" "${LOG}" || ! grep -q "^iFFT PASS$" "${LOG}"; then
      VALIDATION_PASS=false
    fi
    ;;

  *)
    echo "ERROR: Unsupported benchmark '${BENCHMARK}'. Supported: accuracy, aes, fft" >&2
    exit 1
    ;;
esac

echo "" >> "${LOG}"
if $VALIDATION_PASS; then
  echo "VALIDATION: PASS" | tee -a "${LOG}"
  exit 0
else
  echo "VALIDATION: FAIL" | tee -a "${LOG}"
  echo "VALIDATION: FAIL" >&2
  exit 1
fi

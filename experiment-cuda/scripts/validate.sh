#!/bin/bash
# validate.sh — Run a compiled attempt and verify numerical correctness
#
# Usage:
#   bash validate.sh <hecbench_path> <benchmark> <cell> <round> <attempt>
#
# Arguments:
#   hecbench_path  Absolute path to HeCBench root (e.g. ~/HeCBench)
#   benchmark      See the case statement below for the full list of supported benchmarks.
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
#   - Binary timeout: 180 seconds per run.
#   - Data-dependent benchmarks (bfs, hotspot, heartwall, d2q9-bgk) require
#     data files from HeCBench. DVC artifacts live in src/data/{bench}/ — run
#     `dvc pull` then extract the tarball for each benchmark (see per-benchmark
#     hints printed when the data is missing).
#     bfs falls back to /tmp/graph1MW_6.txt if the DVC data is absent.
#
# Examples:
#   bash validate.sh ~/HeCBench accuracy cell_a 1 1
#   bash validate.sh ~/HeCBench blas-gemm cell_b 1 2

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

# run_binary — execute binary, stream output to stdout and append to LOG.
# Sets VALIDATION_PASS=false on timeout or non-zero exit.
run_binary() {
  local tmpout
  tmpout=$(mktemp)

  echo "--- $ $(basename "${BINARY}") $* ---" | tee -a "${LOG}"

  timeout 180 "${BINARY}" "$@" > "${tmpout}" 2>&1
  local exit_code=$?

  tee -a "${LOG}" < "${tmpout}"
  echo "" >> "${LOG}"
  rm -f "${tmpout}"

  if [ "${exit_code}" -eq 124 ]; then
    echo "ERROR: Binary timed out (180s)" | tee -a "${LOG}" >&2
    VALIDATION_PASS=false
  elif [ "${exit_code}" -ne 0 ]; then
    echo "ERROR: Binary exited with code ${exit_code}" | tee -a "${LOG}" >&2
    VALIDATION_PASS=false
  fi
}

# check_data — fail with a helpful message if a required data file/dir is missing.
# Returns 1 (and sets VALIDATION_PASS=false) when missing, 0 when present.
check_data() {
  local path="$1"
  local hint="${2:-}"
  if [ ! -e "${path}" ]; then
    echo "ERROR: Required data not found: ${path}" | tee -a "${LOG}" >&2
    [ -n "${hint}" ] && echo "       ${hint}" | tee -a "${LOG}" >&2
    VALIDATION_PASS=false
    return 1
  fi
  return 0
}

# check_pass — grep LOG for an explicit PASS token; fail if absent or if FAIL present.
# Usage: check_pass <pass_pattern> [fail_pattern]
check_pass() {
  local pass_pat="$1"
  local fail_pat="${2:-}"

  if [ -n "${fail_pat}" ] && grep -qE "${fail_pat}" "${LOG}"; then
    VALIDATION_PASS=false
    return
  fi
  if ! grep -qE "${pass_pat}" "${LOG}"; then
    VALIDATION_PASS=false
  fi
}

# --- Write log header ---
{
  echo "=== validate.sh | ${BENCHMARK}/${CELL}/round_${ROUND}/attempt_${ATTEMPT} ==="
  echo "=== $(date -Iseconds) ==="
  echo ""
} > "${LOG}"

# --- Run and check per benchmark ---
case "${BENCHMARK}" in

  accuracy)
    run_binary 8192 10000 10 10
    check_pass "^PASS$" "^FAIL$"
    ;;

  aes)
    BMP="${HECBENCH_PATH}/src/urng-sycl/URNG_Input.bmp"
    if check_data "${BMP}" "Pull it with: cd ${HECBENCH_PATH} && dvc pull"; then
      run_binary 10 0 "${BMP}"
      run_binary 10 1 "${BMP}"
      if grep -q "^Fail$" "${LOG}"; then VALIDATION_PASS=false; fi
      pass_count=$(grep -c "^Pass$" "${LOG}" || true)
      [ "${pass_count}" -lt 2 ] && VALIDATION_PASS=false
    fi
    ;;

  attention)
    run_binary 65536 2048 0 10
    check_pass "^PASS$" "^FAIL$"
    ;;

  attention-paged)
    run_binary 8 32 128 4096 128 1
    check_pass "^PASS$" "^FAIL$"
    ;;

  bh)
    # ECL-BH produces no numerical validation output — correctness check is exit-code only.
    # Adding a numerical check would require recompiling with -DDEBUG, which is out of scope.
    run_binary 100000 1
    ;;

  bilateral)
    run_binary 2960 1440 0.5 0.5 1
    check_pass "^PASS$" "^FAIL$"
    ;;

  binomial)
    run_binary
    check_pass "Test passed" "Test failed"
    ;;

  black-scholes)
    run_binary 1
    # Compare GPU vs CPU summation and per-option price.
    # Both are printed by the program; no explicit PASS/FAIL in source.
    gpu_sum=$(grep  "Summation of output prices on GPU:"  "${LOG}" | awk '{print $NF}')
    cpu_sum=$(grep  "Summation of output prices on CPU:"  "${LOG}" | awk '{print $NF}')
    gpu_price=$(grep "Output price.*on GPU:" "${LOG}" | awk '{print $NF}')
    cpu_price=$(grep "Output price.*on CPU:" "${LOG}" | awk '{print $NF}')
    if [ -z "${gpu_sum}" ] || [ -z "${cpu_sum}" ] || [ -z "${gpu_price}" ] || [ -z "${cpu_price}" ]; then
      echo "ERROR: black-scholes expected output lines not found" | tee -a "${LOG}" >&2
      VALIDATION_PASS=false
    elif ! python3 - <<PYEOF 2>/dev/null
gpu_sum   = float("${gpu_sum}")
cpu_sum   = float("${cpu_sum}")
gpu_price = float("${gpu_price}")
cpu_price = float("${cpu_price}")
rel_sum   = abs(gpu_sum   - cpu_sum)   / max(abs(cpu_sum),   1e-9)
rel_price = abs(gpu_price - cpu_price) / max(abs(cpu_price), 1e-9)
assert gpu_sum  > 0,      f"GPU sum non-positive: {gpu_sum}"
assert rel_sum  < 1e-5,   f"GPU/CPU summation mismatch rel={rel_sum:.2e}: GPU={gpu_sum} CPU={cpu_sum}"
assert rel_price < 1e-5,  f"GPU/CPU per-option price mismatch rel={rel_price:.2e}: GPU={gpu_price} CPU={cpu_price}"
PYEOF
    then
      echo "ERROR: black-scholes GPU/CPU numerical mismatch (sum=${gpu_sum} vs ${cpu_sum}, price=${gpu_price} vs ${cpu_price})" | tee -a "${LOG}" >&2
      VALIDATION_PASS=false
    fi
    ;;

  blas-gemm)
    run_binary 4096 4096 4096 1
    # Output: "Checking BLAS GEMM.. PASS/FAIL" (not a standalone line)
    check_pass "PASS" "FAIL"
    ;;

  bfs)
    # DVC data lands in src/data/bfs/ after dvc pull + extraction
    GRAPH="${HECBENCH_PATH}/src/data/bfs/graph1MW_6.txt"
    # Fall back to a copy placed in /tmp (e.g. by a prior profiling run)
    if [ ! -f "${GRAPH}" ]; then
      GRAPH="/tmp/graph1MW_6.txt"
    fi
    if check_data "${GRAPH}" \
        "Run: cd ${HECBENCH_PATH} && dvc pull && (cd src/data/bfs && tar xjf graph1MW_6.txt.tar.bz)"; then
      run_binary "${GRAPH}"
      check_pass "Passed" "Failed"
    fi
    ;;

  bscan)
    # Note: the HeCBench bscan baseline uses __ballot_sync(lanemask_lt(), p) which is
    # undefined behaviour on CUDA >= 9 and produces wrong results. Verification is
    # exit-code-only so that an LLM fix that resolves the UB still passes.
    run_binary 10
    ;;

  convolution3D)
    run_binary 32 96 256 26 26 5 1
    check_pass "^PASS$" "^FAIL$"
    ;;

  d2q9-bgk)
    INPUTS="${HECBENCH_PATH}/src/d2q9-bgk-cuda/Inputs"
    if check_data "${INPUTS}" \
        "Decompress with: cd ${HECBENCH_PATH}/src/d2q9-bgk-cuda && tar xzf test.tar.gz"; then
      run_binary \
        "${INPUTS}/input_256x256.params" \
        "${HECBENCH_PATH}/src/d2q9-bgk-cuda/Obstacles/obstacles_256x256.dat"
      # Check Reynolds number is present and within expected physical range.
      # Baseline produces ~10.0654; an incorrect kernel typically diverges far from this.
      # 1e-3 relative tolerance (±0.1%) is used rather than 1e-5 because the Reynolds
      # number accumulates 80,000 LBM iterations of FP arithmetic — valid optimisations
      # that change reduction order can shift the result by more than 1e-5 without being wrong.
      reynolds=$(grep "Reynolds number:" "${LOG}" | awk '{print $NF}')
      if [ -z "${reynolds}" ]; then
        echo "ERROR: d2q9-bgk Reynolds number line not found in output" | tee -a "${LOG}" >&2
        VALIDATION_PASS=false
      elif ! python3 -c "
r = float('${reynolds}')
expected = 10.0654268264
rel = abs(r - expected) / expected
assert rel < 1e-3, f'Reynolds {r} deviates {rel:.2e} from expected {expected} (tolerance 1e-3)'
" 2>/dev/null; then
        echo "ERROR: d2q9-bgk Reynolds number ${reynolds} deviates >0.1% from expected ~10.0654" | tee -a "${LOG}" >&2
        VALIDATION_PASS=false
      fi
    fi
    ;;

  fft)
    run_binary 3 10
    if grep -q " FAIL" "${LOG}"; then VALIDATION_PASS=false; fi
    if ! grep -q "^FFT PASS$" "${LOG}" || ! grep -q "^iFFT PASS$" "${LOG}"; then
      VALIDATION_PASS=false
    fi
    ;;

  fluidSim)
    # fluidSim produces no numerical validation output — correctness check is exit-code only.
    run_binary 100
    ;;

  heartwall)
    # Binary has hardcoded path ../data/heartwall/test.avi relative to CWD.
    # DVC data lands in src/data/heartwall/ after dvc pull + extraction.
    # Run from src/heartwall-cuda/ so that ../data/heartwall resolves correctly.
    DATA="${HECBENCH_PATH}/src/data/heartwall/test.avi"
    if check_data "${DATA}" \
        "Run: cd ${HECBENCH_PATH} && dvc pull && (cd src/data/heartwall && tar xjf heartwall.tar.bz)"; then
      (cd "${HECBENCH_PATH}/src/heartwall-cuda" && timeout 180 "${BINARY}" 104 >> "${LOG}" 2>&1)
      exit_code=$?
      if [ "${exit_code}" -eq 124 ]; then
        echo "ERROR: Binary timed out (180s)" | tee -a "${LOG}" >&2
        VALIDATION_PASS=false
      elif [ "${exit_code}" -ne 0 ]; then
        echo "ERROR: Binary exited with code ${exit_code}" | tee -a "${LOG}" >&2
        VALIDATION_PASS=false
      fi
      # Confirm the GPU kernel section actually executed (not a silent early exit).
      if ! grep -q "GPU KERNELS" "${LOG}"; then
        echo "ERROR: heartwall GPU KERNELS timing line absent — kernel did not run" | tee -a "${LOG}" >&2
        VALIDATION_PASS=false
      fi
    fi
    ;;

  histogram)
    run_binary --i=1
    # Output: "Shared memory atomics \tPASS" etc. (tab-indented, not standalone line)
    check_pass "PASS" "FAIL"
    ;;

  hotspot)
    # DVC data lands in src/data/hotspot/ after dvc pull + extraction
    TEMP="${HECBENCH_PATH}/src/data/hotspot/temp_512"
    if check_data "${TEMP}" \
        "Run: cd ${HECBENCH_PATH} && dvc pull && (cd src/data/hotspot && tar xjf hotspot.tar.bz)"; then
      run_binary 512 2 200 \
        "${HECBENCH_PATH}/src/data/hotspot/temp_512" \
        "${HECBENCH_PATH}/src/data/hotspot/power_512" \
        /tmp/hotspot_output.out
    fi
    ;;

  jacobi)
    run_binary
    check_pass "^PASS$" "^FAIL$"
    ;;

  layernorm)
    run_binary 8 1024 768 10
    check_pass "All results match"
    ;;

  lzss)
    # Use the heartwall baseline main.cu as a convenient large input file
    INPUT="${SCRIPT_DIR}/../results/heartwall/baseline/main.cu"
    if [ ! -f "${INPUT}" ]; then
      # Fall back to any .cu file we can find
      INPUT=$(find "${SCRIPT_DIR}/../results" -name "main.cu" | head -1)
    fi
    if check_data "${INPUT}" "heartwall baseline main.cu not found"; then
      run_binary -i "${INPUT}" -n 1
      if grep -q "verification failed" "${LOG}"; then
        VALIDATION_PASS=false
      fi
    fi
    ;;

  maxpool3d)
    run_binary 2048 2048 96 1
    check_pass "^PASS$" "^FAIL$"
    ;;

  mcpr)
    ALPHAS="${HECBENCH_PATH}/src/mcpr-cuda/alphas.csv"
    if check_data "${ALPHAS}" \
        "Decompress with: cd ${HECBENCH_PATH}/src/mcpr-cuda && bzip2 -dk alphas.csv.bz2"; then
      run_binary "${ALPHAS}" 1
      check_pass "^PASS$" "^FAIL$"
    fi
    ;;

  mis)
    GRAPH="${HECBENCH_PATH}/src/mis-cuda/internet.egr"
    if check_data "${GRAPH}"; then
      run_binary "${GRAPH}" 1
      # ECL-MIS prints "ERROR: found adjacent nodes in MIS" to stderr on invalid result.
      if grep -q "^ERROR:" "${LOG}"; then
        VALIDATION_PASS=false
      fi
      # Also require throughput lines to confirm the algorithm completed and produced an MIS.
      if ! grep -q "^throughput:" "${LOG}"; then
        echo "ERROR: mis throughput line absent — algorithm did not complete" | tee -a "${LOG}" >&2
        VALIDATION_PASS=false
      fi
    fi
    ;;

  nw)
    run_binary 16384 10 1
    check_pass "^PASS$" "^FAIL$"
    ;;

  page-rank)
    run_binary -n 20000 -i 10
    check_pass "^PASS$" "^FAIL$"
    ;;

  sc)
    run_binary -a 0.1
    check_pass "Test Passed" "Test Failed"
    ;;

  scan)
    # Use 1M elements for validation (original 256M is too slow for a correctness check)
    run_binary 1048576 1
    check_pass "^PASS$" "^FAIL$"
    ;;

  segment-reduce)
    # First arg is multiplier: total elements = 16384 × multiplier.
    # Using multiplier=1 (16384 elements) for fast validation.
    run_binary 1 1
    # The program prints "does not agree with the reference" on mismatch.
    if grep -q "does not agree with the reference" "${LOG}"; then
      VALIDATION_PASS=false
    fi
    # Also require all 11 Throughput lines (one per segment size) to confirm full completion.
    throughput_count=$(grep -c "^num_segments" "${LOG}" || true)
    if [ "${throughput_count}" -lt 11 ]; then
      echo "ERROR: segment-reduce expected 11 segment-size rows, got ${throughput_count}" | tee -a "${LOG}" >&2
      VALIDATION_PASS=false
    fi
    ;;

  sort)
    run_binary 3 1
    check_pass "^PASS$" "^FAIL$"
    ;;

  sssp)
    # Uses a bundled synthetic graph + pre-computed Dijkstra reference.
    # The DVC data (NYR_input.dat) is not required for correctness validation.
    VAL_INPUT="${SCRIPT_DIR}/sssp_val_input.dat"
    VAL_REF="${SCRIPT_DIR}/sssp_val_ref.out"
    run_binary -g 120 -t 1 -w 10 -r 1 -f "${VAL_INPUT}" -c "${VAL_REF}"
    check_pass "^PASS$" "^FAIL$"
    ;;

  transpose)
    run_binary 16384 16384 1
    check_pass "^PASS$" "^FAIL$"
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

# --- Final verdict ---
echo "" >> "${LOG}"
if $VALIDATION_PASS; then
  echo "VALIDATION: PASS" | tee -a "${LOG}"
  exit 0
else
  echo "VALIDATION: FAIL" | tee -a "${LOG}"
  echo "VALIDATION: FAIL" >&2
  exit 1
fi

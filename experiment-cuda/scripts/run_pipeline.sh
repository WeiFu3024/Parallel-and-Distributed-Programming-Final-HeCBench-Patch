#!/bin/bash
# run_pipeline.sh — Serial compile → validate → ncu profiling (and cell_c feedback)
#
# Skips any target directory that does not contain main.cu.
# Stops the pipeline for a target when compile or validate fails.
#
# Usage:
#   bash run_pipeline.sh <hecbench_path> [options]
#
# Options:
#   --target TARGET       baseline | cell_a | cell_b | cell_c | all   (default: all)
#   --round ROUNDS        1 | 2 | 3 | all                             (default: all)
#                         Only applies to cell_c (cell_a/cell_b always use round 1)
#   --attempt ATTEMPTS    1 | 2 | 3 | all                             (default: all)
#   --benchmark BENCH     One benchmark name or "all"                   (default: all)
#   --benchmark-set SET   all | assigned                                (default: all)
#                         assigned = blas-gemm maxpool3d binomial mcpr fluidSim
#                                    bfs nw bscan sc lzss
#   --ncu-repeat N        Timing-loop iterations for ncu (default: 10; suggest 5–10)
#   --sm-arch ARCH        CUDA arch, e.g. sm_86                        (default: auto)
#
# Examples:
#   bash run_pipeline.sh ~/HeCBench --target baseline
#   bash run_pipeline.sh ~/HeCBench --target baseline --benchmark-set assigned
#   bash run_pipeline.sh ~/HeCBench --target cell_a --benchmark bfs
#   bash run_pipeline.sh ~/HeCBench --target cell_c --round 2
#   bash run_pipeline.sh ~/HeCBench --target all --benchmark attention
#
# Notes:
#   - baseline has no validate.sh step (compile → ncu only)
#   - cell_c: after a successful round R profile, generates round_{R+1}/feedback.xml
#   - ncu uses validate-scale problem sizes with --ncu-repeat timing iterations (default 10)

set -euo pipefail

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,28p' "$0"
  exit 0
fi

HECBENCH_PATH="${1:?Usage: $0 <hecbench_path> [--target ...] [--round ...] [--attempt ...] [--benchmark ...] [--benchmark-set ...] [--ncu-repeat ...] [--sm-arch ...]}"
shift

TARGET="all"
ROUNDS="all"
ATTEMPTS="all"
BENCHMARK="all"
BENCHMARK_SET="all"
NCU_REPEAT="10"
SM_ARCH="auto"

# Operator-assigned benchmark subset (see operations_manual_cuda.md Step 0.3b)
ASSIGNED_BENCHMARKS=(
  blas-gemm maxpool3d binomial mcpr fluidSim
  bfs nw bscan sc lzss
)

while [ $# -gt 0 ]; do
  case "$1" in
    --target)        TARGET="${2:?Missing value for --target}"; shift 2 ;;
    --round)         ROUNDS="${2:?Missing value for --round}"; shift 2 ;;
    --attempt)       ATTEMPTS="${2:?Missing value for --attempt}"; shift 2 ;;
    --benchmark)     BENCHMARK="${2:?Missing value for --benchmark}"; shift 2 ;;
    --benchmark-set) BENCHMARK_SET="${2:?Missing value for --benchmark-set}"; shift 2 ;;
    --ncu-repeat)    NCU_REPEAT="${2:?Missing value for --ncu-repeat}"; shift 2 ;;
    --sm-arch)       SM_ARCH="${2:?Missing value for --sm-arch}"; shift 2 ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
COMPILE_SH="${SCRIPT_DIR}/compile.sh"
VALIDATE_SH="${SCRIPT_DIR}/validate.sh"
PROFILE_PY="${SCRIPT_DIR}/profile_to_xml.py"

COUNT_OK=0
COUNT_FAIL=0
COUNT_SKIP=0

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

round_matches() {
  local round="$1"
  [ "${ROUNDS}" = "all" ] || [ "${ROUNDS}" = "${round}" ]
}

attempt_matches() {
  local attempt="$1"
  [ "${ATTEMPTS}" = "all" ] || [ "${ATTEMPTS}" = "${attempt}" ]
}

target_selected() {
  local want="$1"
  [ "${TARGET}" = "all" ] || [ "${TARGET}" = "${want}" ]
}

get_src_dir() {
  local bench="$1"
  case "${bench}" in
    transpose) echo "${HECBENCH_PATH}/src/matrixT-cuda" ;;
    *)         echo "${HECBENCH_PATH}/src/${bench}-cuda" ;;
  esac
}

# Return benchmark-specific ncu arguments.
# Problem sizes match validate.sh; timing-loop iterations use NCU_REPEAT (default 10).
# Prints a single line of args; __NO_ARGS__ = no arguments.
get_ncu_args_line() {
  local bench="$1"
  local src_dir
  local n="${NCU_REPEAT}"
  local fluid_particles=$((100 * n))
  local bscan_size=$((10 * n))
  src_dir="$(get_src_dir "${bench}")"

  case "${bench}" in
    attention)         echo "65536 2048 0 ${n}" ;;
    attention-paged)   echo "8 32 128 4096 128 ${n}" ;;
    blas-gemm)         echo "4096 4096 4096 ${n}" ;;
    convolution3D)     echo "32 96 256 26 26 5 ${n}" ;;
    layernorm)         echo "8 1024 768 ${n}" ;;
    maxpool3d)         echo "2048 2048 96 ${n}" ;;
    bilateral)         echo "2960 1440 0.5 0.5 ${n}" ;;
    black-scholes)     echo "${n}" ;;
    binomial)          echo "__NO_ARGS__" ;;
    fft)               echo "3 ${n}" ;;
    jacobi)            echo "__NO_ARGS__" ;;
    mcpr)
      if [ -f "${src_dir}/alphas.csv" ]; then
        echo "${src_dir}/alphas.csv ${n}"
      else
        echo "__MISSING__"
      fi
      ;;
    bh)                echo "100000 ${n}" ;;
    hotspot)
      local temp="${HECBENCH_PATH}/src/data/hotspot/temp_512"
      if [ ! -f "${temp}" ]; then
        temp="${HECBENCH_PATH}/data/hotspot/temp_512"
      fi
      if [ -f "${temp}" ]; then
        local power="${temp%/*}/power_512"
        echo "512 2 200 ${temp} ${power} /tmp/hotspot_ncu_output.out"
      else
        echo "__MISSING__"
      fi
      ;;
    fluidSim)          echo "${fluid_particles}" ;;
    d2q9-bgk)
      local inputs="${src_dir}/Inputs/input_256x256.params"
      local obstacles="${src_dir}/Obstacles/obstacles_256x256.dat"
      if [ -f "${inputs}" ] && [ -f "${obstacles}" ]; then
        echo "${inputs} ${obstacles}"
      else
        echo "__MISSING__"
      fi
      ;;
    heartwall)         echo "104" ;;
    bfs)
      local graph="${HECBENCH_PATH}/src/data/bfs/graph1MW_6.txt"
      if [ ! -f "${graph}" ]; then
        graph="/tmp/graph1MW_6.txt"
      fi
      if [ -f "${graph}" ]; then
        echo "${graph}"
      else
        echo "__MISSING__"
      fi
      ;;
    page-rank)         echo "-n 20000 -i ${n}" ;;
    sssp)
      local val_input="${SCRIPT_DIR}/sssp_val_input.dat"
      local val_ref="${SCRIPT_DIR}/sssp_val_ref.out"
      if [ -f "${val_input}" ] && [ -f "${val_ref}" ]; then
        echo "-g 120 -t 1 -w 10 -r ${n} -f ${val_input} -c ${val_ref}"
      else
        echo "__MISSING__"
      fi
      ;;
    nw)                echo "16384 10 ${n}" ;;
    mis)
      if [ -f "${src_dir}/internet.egr" ]; then
        echo "${src_dir}/internet.egr ${n}"
      else
        echo "__MISSING__"
      fi
      ;;
    scan)              echo "1048576 ${n}" ;;
    bscan)             echo "${bscan_size}" ;;
    histogram)         echo "--i=${n}" ;;
    segment-reduce)    echo "1 ${n}" ;;
    sc)                echo "-a 0.1" ;;
    sort)              echo "3 ${n}" ;;
    transpose)         echo "16384 16384 ${n}" ;;
    lzss)
      local input="${RESULTS_DIR}/heartwall/baseline/main.cu"
      if [ ! -f "${input}" ]; then
        input="$(find "${RESULTS_DIR}" -name "main.cu" 2>/dev/null | head -1 || true)"
      fi
      if [ -n "${input}" ] && [ -f "${input}" ]; then
        echo "-i ${input} -n ${n}"
      else
        echo "__MISSING__"
      fi
      ;;
    accuracy)          echo "8192 10000 10 ${n}" ;;
    aes)
      local bmp="${HECBENCH_PATH}/src/urng-sycl/URNG_Input.bmp"
      if [ -f "${bmp}" ]; then
        echo "10 0 ${bmp}"
      else
        echo "__MISSING__"
      fi
      ;;
    *)
      echo "__UNKNOWN__"
      ;;
  esac
}

validate_benchmark_set() {
  case "${BENCHMARK_SET}" in
    all|assigned) ;;
    *)
      echo "ERROR: --benchmark-set must be 'all' or 'assigned' (got: ${BENCHMARK_SET})" >&2
      exit 1
      ;;
  esac
}

list_benchmarks() {
  if [ "${BENCHMARK}" != "all" ]; then
    echo "${BENCHMARK}"
    return
  fi

  validate_benchmark_set

  if [ "${BENCHMARK_SET}" = "assigned" ]; then
    local bench
    for bench in "${ASSIGNED_BENCHMARKS[@]}"; do
      echo "${bench}"
    done
    return
  fi

  if [ ! -d "${RESULTS_DIR}" ]; then
    echo "ERROR: Results directory not found: ${RESULTS_DIR}" >&2
    exit 1
  fi
  find "${RESULTS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
}

list_attempts() {
  local round_dir="$1"
  local attempt

  if [ "${ATTEMPTS}" = "all" ]; then
    for attempt_dir in "${round_dir}"/attempt_*; do
      [ -d "${attempt_dir}" ] || continue
      attempt="${attempt_dir##*/attempt_}"
      if [ -f "${attempt_dir}/main.cu" ]; then
        echo "${attempt}"
      fi
    done
    return
  fi

  if [ -f "${round_dir}/attempt_${ATTEMPTS}/main.cu" ]; then
    echo "${ATTEMPTS}"
  fi
}

run_ncu() {
  local bench="$1"
  local label="$2"
  local binary="$3"
  local raw_csv="$4"
  local xml_out="$5"

  if ! command -v ncu &>/dev/null; then
    log "FAIL ${label}: ncu not found in PATH"
    return 1
  fi

  local args_line
  args_line="$(get_ncu_args_line "${bench}")"
  if [ "${args_line}" = "__MISSING__" ]; then
    log "FAIL ${label}: required benchmark data not found for ncu (${bench})"
    return 1
  fi
  if [ "${args_line}" = "__UNKNOWN__" ]; then
    log "FAIL ${label}: no ncu argument mapping for ${bench}"
    return 1
  fi

  local ncu_args=()
  if [ "${args_line}" != "__NO_ARGS__" ]; then
    # shellcheck disable=SC2206
    ncu_args=( ${args_line} )
  fi

  mkdir -p "$(dirname "${raw_csv}")"
  log "NCU  ${label}"

  if [ "${bench}" = "heartwall" ]; then
    (cd "${HECBENCH_PATH}/src/heartwall-cuda" && \
      ncu --set full --csv \
          --log-file "${raw_csv}" \
          "${binary}" "${ncu_args[@]}")
  else
    ncu --set full --csv \
        --log-file "${raw_csv}" \
        "${binary}" "${ncu_args[@]}"
  fi

  python3 "${PROFILE_PY}" single \
      --input "${raw_csv}" \
      --output "${xml_out}"
}

generate_cell_c_feedback() {
  local bench="$1"
  local round="$2"
  local next_round=$((round + 1))
  local baseline_csv="${RESULTS_DIR}/${bench}/baseline/nsight_raw.csv"
  local yours_csv="${RESULTS_DIR}/${bench}/cell_c/round_${round}/nsight_raw.csv"
  local feedback_xml="${RESULTS_DIR}/${bench}/cell_c/round_${next_round}/feedback.xml"

  if [ ! -f "${baseline_csv}" ]; then
    log "WARN ${bench}/cell_c/round_${round}: skip feedback — baseline nsight_raw.csv missing"
    return 0
  fi
  if [ ! -f "${yours_csv}" ]; then
    log "WARN ${bench}/cell_c/round_${round}: skip feedback — yours nsight_raw.csv missing"
    return 0
  fi

  mkdir -p "$(dirname "${feedback_xml}")"
  log "FEEDBACK ${bench}/cell_c round_${round} → round_${next_round}/feedback.xml"
  python3 "${PROFILE_PY}" feedback \
      --baseline "${baseline_csv}" \
      --yours "${yours_csv}" \
      --round "${round}" \
      --output "${feedback_xml}"
}

process_baseline() {
  local bench="$1"
  local dir="${RESULTS_DIR}/${bench}/baseline"
  local label="${bench}/baseline"

  if [ ! -f "${dir}/main.cu" ]; then
    log "SKIP ${label}: main.cu not found"
    COUNT_SKIP=$((COUNT_SKIP + 1))
    return 0
  fi

  log "RUN  ${label}"
  if ! bash "${COMPILE_SH}" "${HECBENCH_PATH}" "${bench}" baseline "${SM_ARCH}"; then
    log "FAIL ${label}: compile"
    COUNT_FAIL=$((COUNT_FAIL + 1))
    return 0
  fi

  if ! run_ncu "${bench}" "${label}" \
      "${dir}/main" \
      "${dir}/nsight_raw.csv" \
      "${dir}/nsight.xml"; then
    log "FAIL ${label}: ncu"
    COUNT_FAIL=$((COUNT_FAIL + 1))
    return 0
  fi

  log "OK   ${label}"
  COUNT_OK=$((COUNT_OK + 1))
}

process_cell_attempt() {
  local bench="$1"
  local cell="$2"
  local round="$3"
  local attempt="$4"
  local dir="${RESULTS_DIR}/${bench}/${cell}/round_${round}/attempt_${attempt}"
  local round_dir="${RESULTS_DIR}/${bench}/${cell}/round_${round}"
  local label="${bench}/${cell}/round_${round}/attempt_${attempt}"

  if [ ! -f "${dir}/main.cu" ]; then
    log "SKIP ${label}: main.cu not found"
    COUNT_SKIP=$((COUNT_SKIP + 1))
    return 0
  fi

  log "RUN  ${label}"
  if ! bash "${COMPILE_SH}" "${HECBENCH_PATH}" "${bench}" "${cell}" "${round}" "${attempt}" "${SM_ARCH}"; then
    log "FAIL ${label}: compile"
    COUNT_FAIL=$((COUNT_FAIL + 1))
    return 0
  fi

  if ! bash "${VALIDATE_SH}" "${HECBENCH_PATH}" "${bench}" "${cell}" "${round}" "${attempt}"; then
    log "FAIL ${label}: validate"
    COUNT_FAIL=$((COUNT_FAIL + 1))
    return 0
  fi

  if ! run_ncu "${bench}" "${label}" \
      "${dir}/main" \
      "${round_dir}/nsight_raw.csv" \
      "${round_dir}/nsight.xml"; then
    log "FAIL ${label}: ncu"
    COUNT_FAIL=$((COUNT_FAIL + 1))
    return 0
  fi

  if [ "${cell}" = "cell_c" ] && [ "${round}" -lt 3 ]; then
    generate_cell_c_feedback "${bench}" "${round}"
  fi

  log "OK   ${label}"
  COUNT_OK=$((COUNT_OK + 1))
}

process_cell() {
  local bench="$1"
  local cell="$2"
  local round attempt attempt_list

  case "${cell}" in
    cell_a|cell_b) round=1 ;;
    cell_c)
      for round in 1 2 3; do
        round_matches "${round}" || continue
        attempt_list="$(list_attempts "${RESULTS_DIR}/${bench}/${cell}/round_${round}")"
        if [ -z "${attempt_list}" ]; then
          log "SKIP ${bench}/${cell}/round_${round}: no matching attempts with main.cu"
          COUNT_SKIP=$((COUNT_SKIP + 1))
          continue
        fi
        while IFS= read -r attempt; do
          [ -n "${attempt}" ] || continue
          attempt_matches "${attempt}" || continue
          process_cell_attempt "${bench}" "${cell}" "${round}" "${attempt}"
        done <<< "${attempt_list}"
      done
      return 0
      ;;
    *)
      echo "ERROR: Unknown cell: ${cell}" >&2
      exit 1
      ;;
  esac

  attempt_list="$(list_attempts "${RESULTS_DIR}/${bench}/${cell}/round_${round}")"
  if [ -z "${attempt_list}" ]; then
    log "SKIP ${bench}/${cell}/round_${round}: no matching attempts with main.cu"
    COUNT_SKIP=$((COUNT_SKIP + 1))
    return 0
  fi

  while IFS= read -r attempt; do
    [ -n "${attempt}" ] || continue
    attempt_matches "${attempt}" || continue
    process_cell_attempt "${bench}" "${cell}" "${round}" "${attempt}"
  done <<< "${attempt_list}"
}

main() {
  if [ ! -x "${COMPILE_SH}" ] && [ ! -f "${COMPILE_SH}" ]; then
    echo "ERROR: compile.sh not found: ${COMPILE_SH}" >&2
    exit 1
  fi
  if [ ! -f "${VALIDATE_SH}" ]; then
    echo "ERROR: validate.sh not found: ${VALIDATE_SH}" >&2
    exit 1
  fi
  if [ ! -f "${PROFILE_PY}" ]; then
    echo "ERROR: profile_to_xml.py not found: ${PROFILE_PY}" >&2
    exit 1
  fi
  if ! [[ "${NCU_REPEAT}" =~ ^[0-9]+$ ]] || [ "${NCU_REPEAT}" -lt 1 ]; then
    echo "ERROR: --ncu-repeat must be a positive integer (got: ${NCU_REPEAT})" >&2
    exit 1
  fi

  local bench bench_list
  bench_list="$(list_benchmarks)"

  log "Pipeline start | target=${TARGET} round=${ROUNDS} attempt=${ATTEMPTS} benchmark=${BENCHMARK} benchmark_set=${BENCHMARK_SET} ncu_repeat=${NCU_REPEAT}"

  while IFS= read -r bench; do
    [ -n "${bench}" ] || continue

    if target_selected "baseline"; then
      process_baseline "${bench}"
    fi
    if target_selected "cell_a"; then
      process_cell "${bench}" "cell_a"
    fi
    if target_selected "cell_b"; then
      process_cell "${bench}" "cell_b"
    fi
    if target_selected "cell_c"; then
      process_cell "${bench}" "cell_c"
    fi
  done <<< "${bench_list}"

  log "Pipeline done | OK=${COUNT_OK} FAIL=${COUNT_FAIL} SKIP=${COUNT_SKIP}"
}

main

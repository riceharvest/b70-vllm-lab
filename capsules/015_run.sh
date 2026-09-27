#!/usr/bin/env bash
# Capsule 015 driver.
#
# F-002: oneAPI must NOT be on the runtime path, but F-003 requires the L0
# devel shim in ~/.local/lib (Inductor dlopens libze_loader.so). The skill
# records this conflict: `env -u LD_LIBRARY_PATH` strips the shim back out.
# So filter oneAPI out and re-add only our shim.
set -u

CAPS="$(cd "$(dirname "$0")" && pwd)"
LAB=/mnt/ssd/b70-vllm-lab
RES="$LAB/results/015"
mkdir -p "$RES"

. "$CAPS/b70_gpu_window.sh"
b70_fix_ld

PY=/mnt/ssd/b70-venv/bin/python
export VLLM_XPU_ENABLE_XPU_GRAPH=1   # default; individual arms override

# Interleave arms (A B B A B A) - never compare against a previous run.
# graph env is set per-arm, so each arm is its own process.
run_arm () {           # $1=graph  $2=scenario  $3=tag  $4..=extra
  local g="$1" sc="$2" tag="$3"; shift 3
  echo "=== arm graph=$g scenario=$sc tag=$tag  $(date +%H:%M:%S)"
  # Contention is not a result: wait for a clean GPU window, and SKIP loudly
  # rather than record a FAIL that a sibling's exit caused.
  wait_gpu_window 20 900 || { echo "    SKIP (no stable gpu window)"; return 0; }
  VLLM_XPU_ENABLE_XPU_GRAPH="$([ "$g" = on ] && echo 1 || echo 0)" \
    timeout 600 "$PY" "$CAPS/015_graph_determinism.py" \
      --scenario "$sc" --graph "$g" \
      --out "$RES/${tag}_${sc}_${g}.json" "$@" \
      > "$RES/${tag}_${sc}_${g}.log" 2>&1
  local rc=$?
  echo "    rc=$rc  $([ $rc -eq 0 ] && echo OK || echo FAIL)"
  return 0
}

case "${1:-all}" in
determinism)
  # oracle: 8 repeats of 3 prompts, each arm, interleaved
  for i in 1 2; do
    run_arm on  determinism "det$i" --repeats 8 --keep-logprobs
    run_arm off determinism "det$i" --repeats 8 --keep-logprobs
  done
  ;;
shape)
  for i in 1 2; do
    run_arm on  shape_dependence "shp$i" --max-num-seqs 16
    run_arm off shape_dependence "shp$i" --max-num-seqs 16
  done
  ;;
prefix)
  # prefix caching ON for both arms - it is a default-on feature, so testing it
  # "off" would test a config nobody ships. The graph arm is the variable.
  for i in 1 2; do
    run_arm on  prefix_cache "pfx$i" --prefix-caching
    run_arm off prefix_cache "pfx$i" --prefix-caching
  done
  ;;
concurrent)
  for i in 1 2; do
    run_arm on  concurrent "cc$i" --concurrent-timeout 420
  done
  run_arm off concurrent "cc1" --concurrent-timeout 420
  ;;
all)
  "$0" determinism; "$0" shape; "$0" prefix; "$0" concurrent
  ;;
*)
  echo "usage: $0 {determinism|shape|prefix|concurrent|all}"; exit 2;;
esac
echo "results in $RES"

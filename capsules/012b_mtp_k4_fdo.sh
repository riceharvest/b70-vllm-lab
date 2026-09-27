#!/usr/bin/env bash
# Capsule 012b driver - the reporter's EXACT cudagraph_mode (FULL_DECODE_ONLY).
#
# 012_mtp_k4.sh lets the engine pick the mode (it chose FULL_AND_PIECEWISE).
# The reporter ran --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'.
# FULL_DECODE_ONLY = (FULL, NONE) and decode_mode() == FULL, so it DOES hit
# adjust_cudagraph_sizes_for_spec_decode (compilation.py:1499-1503), which is
# what rounds capture sizes to multiples of k+1. This arm must be run explicitly.
#
# Usage: ./012b_mtp_k4_fdo.sh <k> <on|off> <tag> [max_tokens]
set -u

K="$1"
MODE="$2"
TAG="${3:-run}"
MAXTOK="${4:-6}"

LAB=/mnt/ssd/b70-vllm-lab
OUT="$LAB/results/012b_MTP_k${K}_${MODE}"
mkdir -p "$OUT"

# F-002/F-003: keep the Level Zero devel shim on the runtime path but strip
# oneAPI entries. Never source setvars.sh.
_oneapi_ld="${LD_LIBRARY_PATH:-}"; _clean_ld=""
IFS=':' read -ra _ldparts <<< "$_oneapi_ld"
for _p in "${_ldparts[@]}"; do
  [ -n "$_p" ] || continue
  case "$_p" in /home/dario/oneapi/*) continue ;; *) _clean_ld="${_clean_ld:+$_clean_ld:}$_p" ;; esac
done
export LD_LIBRARY_PATH="$HOME/.local/lib${_clean_ld:+:$_clean_ld}"
export CPATH="$HOME/.local/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$HOME/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export HF_HOME=/mnt/ssd/huggingface
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false

if [ "$MODE" = "on" ]; then
  export VLLM_XPU_ENABLE_XPU_GRAPH=1
else
  export VLLM_XPU_ENABLE_XPU_GRAPH=0
fi

echo "[012b] k=$K graph=$MODE tag=$TAG maxtok=$MAXTOK  VLLM_XPU_ENABLE_XPU_GRAPH=$VLLM_XPU_ENABLE_XPU_GRAPH"
echo "[012b] started $(date -Is)"

timeout 1200 /mnt/ssd/b70-venv/bin/python "$LAB/capsules/012_mtp_k4.py" \
  --k "$K" --graph "$MODE" --out "$OUT/result.json" \
  --cudagraph-mode FULL_DECODE_ONLY --max-tokens "$MAXTOK" \
  > "$OUT/run.log" 2>&1
rc=$?

echo "[012b] exit=$rc  $(date -Is)"
if [ -f "$OUT/result.json" ]; then
  grep -A20 "===ARM_SUMMARY===" "$OUT/run.log" || true
else
  echo "[012b] NO RESULT JSON - tail of log:"
  tail -40 "$OUT/run.log"
fi
exit $rc

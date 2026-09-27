#!/usr/bin/env bash
# Capsule 012 driver - MTP k sweep, graph ON vs OFF, single B70.
#
# The graph decision is made at engine construction (vllm/platforms/xpu.py:301),
# so ON and OFF MUST be separate processes. Interleaving is done here, never
# inside one process (same rule as capsule 010).
#
# Usage: ./012_mtp_k4.sh <k> <on|off> <tag>
set -u

K="$1"
MODE="$2"
TAG="${3:-run}"

LAB=/mnt/ssd/b70-vllm-lab
OUT="$LAB/results/012_mTP_k${K}_${MODE}"
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

echo "[012] k=$K graph=$MODE tag=$TAG  VLLM_XPU_ENABLE_XPU_GRAPH=$VLLM_XPU_ENABLE_XPU_GRAPH"
echo "[012] started $(date -Is)"

timeout 900 /mnt/ssd/b70-venv/bin/python "$LAB/capsules/012_mtp_k4.py" \
  --k "$K" --graph "$MODE" --out "$OUT/result.json" \
  > "$OUT/run.log" 2>&1
rc=$?

echo "[012] exit=$rc  $(date -Is)"
if [ -f "$OUT/result.json" ]; then
  grep -A20 "===ARM_SUMMARY===" "$OUT/run.log" || true
else
  echo "[012] NO RESULT JSON - tail of log:"
  tail -40 "$OUT/run.log"
fi
exit $rc

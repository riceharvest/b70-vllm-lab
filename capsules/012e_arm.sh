#!/usr/bin/env bash
# Capsule 012e single arm: k, graph mode. Writes to results/012e_k<K>_<mode>/.
# Usage: ./012e_arm.sh <k> <on|off>
set -u
K="$1"
MODE="$2"
LAB=/mnt/ssd/b70-vllm-lab
OUT="$LAB/results/012e_k${K}_${MODE}"
mkdir -p "$OUT"

# F-002: never source oneapi setvars.sh. F-003: keep the L0 devel shim, but
# strip oneAPI entries and do not source libs.
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
if [ "$MODE" = "on" ]; then export VLLM_XPU_ENABLE_XPU_GRAPH=1
else export VLLM_XPU_ENABLE_XPU_GRAPH=0; fi

echo "[012e] k=$K graph=$MODE  $(date -Is)"
timeout 1500 /mnt/ssd/b70-venv/bin/python "$LAB/capsules/012_mtp_k4.py" \
  --k "$K" --graph "$MODE" --out "$OUT/result.json" \
  --cudagraph-mode FULL_DECODE_ONLY --max-tokens 8 --repeats 48 \
  > "$OUT/run.log" 2>&1
rc=$?
echo "[012e] exit=$rc $(date -Is)"
if [ -f "$OUT/result.json" ]; then
  grep -A6 "===ARM_SUMMARY===" "$OUT/run.log" || true
else
  echo "[012e] NO RESULT:"; tail -25 "$OUT/run.log"
fi
exit $rc

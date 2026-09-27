#!/usr/bin/env bash
# Capsule 011 driver - ON/OFF determinism + logprob-margin comparison.
# Same env rules as capsule 010 (no oneAPI sourcing, Level Zero shims, F-004 patch).
set -uo pipefail
LAB=/mnt/ssd/b70-vllm-lab
PY=/mnt/ssd/b70-venv/bin/python
OUT="$LAB/results/011_determinism"
mkdir -p "$OUT"

export PATH=/home/dario/.local/bin:$PATH
export HF_HOME=/mnt/ssd/huggingface
export CPATH="/home/dario/.local/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="/home/dario/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/dario/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/dario/oneapi/compiler/2025.3/bin:$PATH"
export CXX="${CXX:-icpx}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export TOKENIZERS_PARALLELISM=false

for arm in off on; do
  if [ "$arm" = "on" ]; then export VLLM_XPU_ENABLE_XPU_GRAPH=1
  else export VLLM_XPU_ENABLE_XPU_GRAPH=0; fi
  echo ">>> determinism arm=$arm graph_env=$VLLM_XPU_ENABLE_XPU_GRAPH"
  timeout 900 "$PY" "$LAB/capsules/011_determinism.py" --graph "$arm" \
      --out "$OUT/$arm.json" > "$OUT/$arm.log" 2>&1
  echo "    rc=$? -> $OUT/$arm.json"
done
echo "=== capsule 011 complete ==="

#!/usr/bin/env bash
# Capsule 010 driver - interleaved A B B A B A ON/OFF comparison of XPU CUDA graphs.
#
# CRITICAL: oneAPI must NOT be sourced (F-002). Level Zero devel shims are needed
# by torch.compile (F-003) and the F-004 triton patch is already applied.
#
# Interleaving happens HERE, not in Python: the graph decision is made in
# vllm/platforms/xpu.py:301 at engine construction, so each arm is its own
# process. Runs are ordered A B B A B A so thermal/clock drift cannot bias one
# arm. Every run is self-contained; nothing is compared against a prior run.
set -uo pipefail

LAB=/mnt/ssd/b70-vllm-lab
PY=/mnt/ssd/b70-venv/bin/python
OUT="$LAB/results/010_xpu_graph"
PROFILE="${PROFILE:-baseline}"

export PATH=/home/dario/.local/bin:$PATH
export HF_HOME=/mnt/ssd/huggingface
export CPATH="/home/dario/.local/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="/home/dario/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/dario/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/dario/oneapi/compiler/2025.3/bin:$PATH"
export CXX="${CXX:-icpx}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"

# Profile-specific knobs. "heavy" is the graph-capture-heavy config: large
# max_num_seen_tokens (max_num_batched_tokens) and a big batch, which is where
# piecewise graph capture should matter most.
case "$PROFILE" in
  baseline)
    NPROMPTS=8;  MAXTOK=96;   MAXSEQS=8;  MAXBT=2048;  MAXLEN=2048 ;;
  heavy)
    NPROMPTS=32; MAXTOK=128;  MAXSEQS=32; MAXBT=16384; MAXLEN=8192 ;;
  *)
    echo "unknown PROFILE=$PROFILE" >&2; exit 2 ;;
esac

run_one () {           # $1 = arm (off|on), $2 = seq index
  local arm="$1" idx="$2"
  local tag; tag="$(printf '%02d' "$idx")"
  local log="$OUT/${PROFILE}_${arm}_${tag}.log"
  local js="$OUT/${PROFILE}_${arm}_${tag}.json"

  if [ "$arm" = "on" ]; then
    export VLLM_XPU_ENABLE_XPU_GRAPH=1
  else
    export VLLM_XPU_ENABLE_XPU_GRAPH=0
  fi

  echo ">>> [$(date +%H:%M:%S)] profile=$PROFILE arm=$arm idx=$idx graph_env=$VLLM_XPU_ENABLE_XPU_GRAPH"
  # timeout guards against the open #54698 replay() hang; py-spy captures it if it trips.
  timeout --signal=SIGABRT "${RUN_TIMEOUT:-900}" \
    "$PY" "$LAB/capsules/010_xpu_graph_ab.py" \
      --graph "$arm" --run-index "$idx" --profile "$PROFILE" \
      --prompts "$NPROMPTS" --max-tokens "$MAXTOK" \
      --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "$MAXBT" \
      --max-model-len "$MAXLEN" --out "$js" > "$log" 2>&1
  local rc=$?

  if [ $rc -ne 0 ]; then
    echo "!!! arm=$arm idx=$idx FAILED rc=$rc  (log: $log)"
    # Capture a stack if it died or hung - this is the #54698 signature.
    PID=$(pgrep -f "010_xpu_graph_ab.py" | head -1)
    [ -n "${PID:-}" ] && py-spy dump --pid "$PID" >> "$log" 2>&1
    printf '{"ok":false,"rc":%d,"graph_requested":"%s","profile":"%s","run_index":%d}\n' \
      "$rc" "$arm" "$PROFILE" "$idx" > "$js"
  fi
}

# ---- interleaved order: A B B A B A (A=off, B=on) ----------------------
i=1
for arm in off on on off on off; do
  run_one "$arm" "$i"
  i=$((i+1))
done

echo "=== all runs complete; results in $OUT ==="
ls -1 "$OUT/${PROFILE}"_*.json 2>/dev/null

#!/usr/bin/env bash
# Capsule 017 driver: wait for a clean GPU window, then run the engine-restart
# arms. The GPU is shared, so every arm is gated; a skipped arm says so rather
# than leaving a FAIL that a sibling's memory profile caused.
set -u
CAPS="$(cd "$(dirname "$0")" && pwd)"
LAB=/mnt/ssd/b70-vllm-lab
. "$CAPS/b70_gpu_window.sh"
b70_fix_ld
PY=/mnt/ssd/b70-venv/bin/python

ARMS="${*:-A_two_engines B_three_engines C_graph_off_control}"

wait_gpu_window 22 1800 || { echo "no clean window; exiting without a verdict"; exit 0; }

# Once the window is open, run every arm back to back - re-gating between arms
# would just re-race the sibling.
exec timeout 3000 "$PY" "$CAPS/017_vllm_pool_reuse.py" \
  --arm A_two_engines --arm B_three_engines --arm C_graph_off_control \
  --gpu-mem 0.40 \
  --out "$LAB/results/017_vllm_pool_reuse.json"

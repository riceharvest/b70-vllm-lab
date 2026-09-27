#!/usr/bin/env bash
# Capsule 012 full sweep - the k cliff, interleaved, with the reporter's
# FULL_DECODE_ONLY mode.
#
# Order is interleaved ON/OFF ON/OFF rather than all-ON-then-all-OFF, because
# the GPU is shared with a live desktop and a run that comes last is the one
# most likely to be perturbed. Interleaving also means a k-arm is never
# compared only against a run from a different GPU-load epoch.
#
# The pair that matters most is (k=4 ON) vs (k=4 OFF): OFF is the reporter's
# own deterministic reference. k=1/2/3 ON are the cliff's control side.
#
# Each arm is a separate process because the graph decision is made at engine
# construction (vllm/platforms/xpu.py:301).
set -u
D=/mnt/ssd/b70-vllm-lab/capsules
LOG=/mnt/ssd/b70-vllm-lab/results/012_sweep.log
: > "$LOG"

run() {  # k mode
  echo "===== $(date -Is)  k=$1 mode=$2 =====" >> "$LOG"
  "$D/012b_mtp_k4_fdo.sh" "$1" "$2" sweep 8 >> "$LOG" 2>&1
  echo "----- exit=$? at $(date -Is) -----" >> "$LOG"
}

# Phase 1: establish the cliff ON/OFF at k=1 and k=4 (the two ends).
run 1 on
run 1 off
run 4 on
run 4 off

# Phase 2: the middle, only if phase 1 shows the sweep is worth finishing.
run 2 on
run 3 on

echo "===== SWEEP COMPLETE $(date -Is) =====" >> "$LOG"
echo "sweep done"

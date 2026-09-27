#!/usr/bin/env bash
# Capsule 012e - the DECISIVE pair for #54785, with the positive control.
#
#   arm A: k=4, graph ON  (the suspect configuration)
#   arm B: k=4, graph OFF (the reporter's own deterministic reference)
#
# Both at 48 identical greedy requests, FULL_DECODE_ONLY, bs=1, single B70.
# The verdict is a COMPARISON between the two arms, so they must run in the same
# session with the same model and the same build; that is why this is one script
# rather than two.
#
# This run also enables per_request_spec_decode_metrics="detailed", which is the
# positive control that the MTP drafter actually ran and had drafts accepted.
# Without it a clean k=4 arm cannot be distinguished from a dead drafter.
set -u
LAB=/mnt/ssd/b70-vllm-lab
LOG=$LAB/results/012e_decisive.log
: > "$LOG"

arm() {  # k mode
  echo "########## $(date -Is)  k=$1 graph=$2 ##########" >> "$LOG"
  "$LAB/capsules/012e_arm.sh" "$1" "$2" >> "$LOG" 2>&1
  echo "########## exit=$? $(date -Is) ##########" >> "$LOG"
}

# ON first: it is the arm that matters, and if it is interrupted we still have it.
arm 4 on
arm 4 off

echo "DECISIVE PAIR COMPLETE $(date -Is)"
grep -E "distinct_texts|^k=|ARM_SUMMARY|spec" "$LOG" | tail -20

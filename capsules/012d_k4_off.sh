#!/usr/bin/env bash
# Capsule 012d - the two arms that decide #54785, run one at a time.
#
# k=4 ON is already measured (012b_MTP_k4_on: 48/48 bit-identical).
# The decisive missing cell is k=4 OFF, the reporter's own deterministic
# reference: if ON and OFF agree bit-exactly at k=4, then on THIS hardware the
# cliff does not exist, and if they diverge we have a reproduction.
#
# Each arm is a separate process (graph decision is made at engine construction).
# Run one arm per invocation so a single failure cannot take the rest down.
#
# Usage: ./012d_k4_off.sh
set -u
LAB=/mnt/ssd/b70-vllm-lab
LOG=$LAB/results/012d.log
echo "[012d] start $(date -Is)  k=4 graph=off (reference arm)" >> "$LOG"
"$LAB/capsules/012b_mtp_k4_fdo.sh" 4 off ref 8 >> "$LOG" 2>&1
echo "[012d] k=4 off exit=$? $(date -Is)" >> "$LOG"
tail -12 "$LOG"

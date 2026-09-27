#!/usr/bin/env bash
# Single-lane GPU lock for the Arc Pro B70.
#
# WHY THIS EXISTS
#   The B70 is ONE scarce lane. Two capsules running at once contend for VRAM,
#   distort each other's timings, and can OOM each other mid-run -- so every
#   perf number from a contended run is junk, not merely imprecise.
#   This happened for real on 2026-09-27: two subagents ran capsules
#   concurrently and held 9.18 GiB between them.
#
# USAGE (wrap any GPU capsule)
#   with_gpu_lock bash capsules/001_first_light.sh
#   with_gpu_lock --name "54785-mtp-k4" bash capsules/012_mtp_k4.py --k 4
#
# BEHAVIOUR
#   - Blocks until the lane is free, printing who holds it and how long they
#     have held it. Waits are visible on purpose.
#   - Refuses to run if the lane is already occupied by a LIVE process, rather
#     than piling up.
#   - HARD-FAILS if VRAM is already heavily used by something outside the lock
#     (e.g. a stray engine from a previous run). This catches the exact failure
#     mode above: a process that escaped the lock still poisons the measurement.
#   - Prints a one-line receipt on release so results can be traced to a run.
set -uo pipefail

LOCKDIR="${B70_LOCKDIR:-/tmp/b70-gpu-lane}"
LOCKFILE="$LOCKDIR/lock"
STALE_SECONDS="${B70_LOCK_STALE:-1800}"   # 30 min without a heartbeat = stale
MAX_FREE_GIB="${B70_LOCK_MIN_FREE_GIB:-3}" # refuse if less than this is free

mkdir -p "$LOCKDIR"

vram_used_gib() {
  LD_LIBRARY_PATH="$HOME/.local/lib" /mnt/ssd/b70-venv/bin/python \
    /mnt/ssd/b70-vllm-lab/tools/b70_vram.py 2>/dev/null \
    | awk -F: '/"used_gib"/{gsub(/[^0-9.]/,"",$2); print $2}'
}

acquire() {
  local waited=0
  while ! mkdir "$LOCKFILE" 2>/dev/null; do
    local holder age
    holder=$(cat "$LOCKFILE/who" 2>/dev/null || echo "unknown")
    local hb; hb=$(stat -c %Y "$LOCKFILE/heartbeat" 2>/dev/null || echo 0)
    age=$(( $(date +%s) - hb ))
    if [ "$age" -gt "$STALE_SECONDS" ]; then
      echo "[lock] STALE lock from '$holder' (${age}s without heartbeat) - breaking it" >&2
      rm -rf "$LOCKFILE"
      continue
    fi
    if [ $((waited % 30)) -eq 0 ]; then
      echo "[lock] waiting ${waited}s for lane; held by '$holder' (${age}s)" >&2
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "$1" > "$LOCKFILE/who"
  date +%s > "$LOCKFILE/heartbeat"
  echo "$waited" > "$LOCKFILE/waited"
}

release() {
  local who; who=$(cat "$LOCKFILE/who" 2>/dev/null || echo "?")
  local w; w=$(cat "$LOCKFILE/waited" 2>/dev/null || echo 0)
  rm -rf "$LOCKFILE"
  echo "[lock] released '$who' (waited ${w}s to acquire)" >&2
}

NAME="unnamed"
if [ "${1:-}" = "--name" ]; then NAME="$2"; shift 2; fi

# Strip a leading `--` separator. gpu_run.sh passes one through, and without
# this it is treated as the command name -> "command not found" (exit 127).
if [ "${1:-}" = "--" ]; then shift; fi

if [ $# -eq 0 ]; then
  echo "usage: with_gpu_lock [--name LABEL] <command...>" >&2
  exit 64
fi

# Pre-flight: is the lane already dirty from outside the lock?
used=$(vram_used_gib)
if [ -n "${used:-}" ]; then
  free=$(awk -v u="$used" 'BEGIN{printf "%.2f", 31.89-u}')
  echo "[lock] pre-flight: ${used} GiB used, ${free} GiB free (desktop baseline is ~1.0 GiB)" >&2
  if awk -v f="$free" -v m="$MAX_FREE_GIB" 'BEGIN{exit !(f < m)}'; then
    echo "[lock] REFUSING: only ${free} GiB free (< ${MAX_FREE_GIB})." >&2
    echo "[lock] Another GPU process is running outside the lock. Find it with:" >&2
    echo "[lock]   ps aux | grep -E 'b70-venv/bin/python|VLLM::EngineCore' | grep -v grep" >&2
    echo "[lock] A contended run produces invalid numbers. Stop it or raise the threshold." >&2
    exit 75
  fi
fi

trap release EXIT INT TERM
acquire "$NAME"

# Heartbeat so a long run is never mistaken for a stale lock.
( while true; do sleep 20; date +%s > "$LOCKFILE/heartbeat" 2>/dev/null || break; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; release' EXIT INT TERM

echo "[lock] ACQUIRED by '$NAME'" >&2
"$@"
rc=$?
exit $rc

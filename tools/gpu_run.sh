#!/usr/bin/env bash
# Launch a B70 capsule on the single-lane GPU queue.
#
# This is the ONLY sanctioned entry point to the GPU. It wraps the call in
# tools/with_gpu_lock.sh and then VERIFIES afterwards that the lane was actually
# exclusive for the duration. A lock that is merely advisory did not prevent
# real contention on 2026-09-27, so this checks the invariant rather than
# trusting it.
#
# Usage:
#   gpu_run.sh --name 54785-mtp-k4 -- bash capsules/012_mtp_k4.py --k 4
#   gpu_run.sh --name foo --timeout 600 -- bash capsules/001_first_light.sh
set -uo pipefail

LAB=/mnt/ssd/b70-vllm-lab
LOCK="$LAB/tools/with_gpu_lock.sh"

NAME="unnamed"
TIMEOUT=900
MAX_ALLOWED_GIB="${B70_MAX_OTHER_GIB:-1.5}"   # desktop baseline is ~1.0 GiB

while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "usage: gpu_run.sh [--name LABEL] [--timeout SEC] -- <command...>" >&2
  exit 64
fi

if [ ! -x "$LOCK" ]; then
  echo "ERROR: lock script missing or not executable: $LOCK" >&2
  exit 70
fi

vram_used_gib() {
  LD_LIBRARY_PATH="$HOME/.local/lib" /mnt/ssd/b70-venv/bin/python \
    "$LAB/tools/b70_vram.py" 2>/dev/null \
    | awk -F: '/"used_gib"/{gsub(/[^0-9.]/,"",$2); print $2}'
}

start_epoch=$(date +%s)
used_before=$(vram_used_gib)
echo "[gpu_run] name='$NAME' timeout=${TIMEOUT}s pre_used=${used_before:-?}GiB"

# Refuse up front if something is already squatting on the lane.
if [ -n "${used_before:-}" ] && \
   awk -v u="$used_before" -v m="$MAX_ALLOWED_GIB" 'BEGIN{exit !(u > m)}'; then
  echo "[gpu_run] REFUSING to start: ${used_before} GiB already in use" >&2
  echo "[gpu_run] (baseline for the live desktop is ~1.0 GiB)." >&2
  echo "[gpu_run] Find the squatter with:" >&2
  echo "[gpu_run]   ps aux | grep -E 'b70-venv/bin/python|VLLM::EngineCore' | grep -v grep" >&2
  exit 75
fi

timeout "$TIMEOUT" bash "$LOCK" --name "$NAME" -- "$@"
rc=$?
elapsed=$(( $(date +%s) - start_epoch ))

# Post-check: anything left running means the lane was NOT exclusive, so this
# run's numbers must be treated as invalid even though it exited 0.
sleep 3
used_after=$(vram_used_gib)
echo "[gpu_run] name='$NAME' rc=$rc elapsed=${elapsed}s post_used=${used_after:-?}GiB"

if [ -n "${used_after:-}" ] && \
   awk -v u="$used_after" -v m="$MAX_ALLOWED_GIB" 'BEGIN{exit !(u > m)}'; then
  echo "[gpu_run] POST-CHECK FAILED: ${used_after} GiB still in use after exit." >&2
  echo "[gpu_run] Something is still holding the GPU. This run's timings are" >&2
  echo "[gpu_run] INVALID - do not report them. Find and stop the squatter:" >&2
  echo "[gpu_run]   ps aux | grep -E 'b70-venv/bin/python|VLLM::EngineCore' | grep -v grep" >&2
  pgrep -f 'b70-venv/bin/python' >/dev/null 2>&1 && \
    echo "[gpu_run] live pids: $(pgrep -f 'b70-venv/bin/python' | tr '\n' ' ')" >&2
  [ $rc -eq 0 ] && rc=75
fi

if [ $rc -eq 0 ]; then
  echo "[gpu_run] OK name='$NAME' elapsed=${elapsed}s (exclusive)"
else
  echo "[gpu_run] FAILED name='$NAME' rc=$rc elapsed=${elapsed}s"
fi
exit $rc

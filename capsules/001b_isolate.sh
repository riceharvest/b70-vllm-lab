#!/usr/bin/env bash
# Capsule 001b driver: attribute the B70 segfault to a specific mechanism.
#
# CRITICAL: oneAPI must NOT be sourced. See ENVIRONMENT.md.
set -uo pipefail

LAB=/mnt/ssd/b70-vllm-lab
PY=/mnt/ssd/b70-venv/bin/python
export PATH=/home/dario/.local/bin:$PATH
export HF_HOME=/mnt/ssd/huggingface
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# Level Zero devel shims required by torch.compile's C++ wrapper.
export CPATH="/home/dario/.local/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="/home/dario/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="/home/dario/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/dario/oneapi/compiler/2025.3/bin:$PATH"
export CXX="${CXX:-icpx}"

mkdir -p "$LAB/results"
LOG="$LAB/results/001b_isolate.log"

env -u LD_LIBRARY_PATH "$PY" "$LAB/capsules/001b_isolate.py" "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

sed -n '/===ISOLATION_JSON===/,$p' "$LOG" | tail -n +2 \
  > "$LAB/results/001b_isolate.json" 2>/dev/null || true

echo "===CAPSULE_EXIT=$rc==="
exit $rc

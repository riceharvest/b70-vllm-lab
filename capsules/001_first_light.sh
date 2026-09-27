#!/usr/bin/env bash
# Capsule 001 driver: first-light vLLM inference on the Arc Pro B70.
#
# The cheapest experiment that can produce a definitive answer. Everything else
# in the GPU queue depends on this passing.
#
# CRITICAL: oneAPI must NOT be sourced. `env -u LD_LIBRARY_PATH` below is
# deliberate and load-bearing -- see ENVIRONMENT.md.
set -uo pipefail

LAB=/mnt/ssd/b70-vllm-lab
PY=/mnt/ssd/b70-venv/bin/python
export PATH=/home/dario/.local/bin:$PATH
export HF_HOME=/mnt/ssd/huggingface
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# XPU graphs are OFF by default on the XPU platform; vLLM logs
# "XPU Graph is disabled by environment variable". Enable it so this capsule
# exercises the graph path, which is what production users actually hit.
export VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}"

# torch.compile (Inductor) compiles a C++ wrapper that includes
# <level_zero/ze_api.h> and links -lze_loader. The `oneapi-level-zero-devel`
# package is NOT installed on this host, so without these two the engine dies
# during KV-cache init with:
#   fatal error: level_zero/ze_api.h: No such file or directory
#   /usr/bin/ld: cannot find -lze_loader: No such file or directory
# The system has libze_loader.so.1 but no unversioned .so dev symlink, which is
# exactly what the -devel package ships. Both are staged rootlessly here.
export CPATH="/home/dario/.local/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="/home/dario/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# Inductor's autotuning child process dlopen()s libze_loader.so, so the runtime
# path needs our shim -- but we must NOT put oneAPI's libsycl on it (that
# shadows the system UR loader torch links against). So: unset oneAPI's entries
# and re-add only ours, rather than blanking the variable with `env -u`.
_oneapi_ld="${LD_LIBRARY_PATH:-}"
_clean_ld=""
IFS=':' read -ra _ldparts <<< "$_oneapi_ld"
for _p in "${_ldparts[@]}"; do
  [ -n "$_p" ] || continue
  case "$_p" in
    /home/dario/oneapi/*) continue ;;
    *) _clean_ld="${_clean_ld:+$_clean_ld:}$_p" ;;
  esac
done
export LD_LIBRARY_PATH="/home/dario/.local/lib${_clean_ld:+:$_clean_ld}"
unset _oneapi_ld _clean_ld _ldparts _p

# icpx needs the oneAPI *compiler* on PATH, but oneAPI's *libraries* must stay
# off LD_LIBRARY_PATH (they shadow the system UR loader torch links against).
# So: extend PATH only, and keep the `env -u LD_LIBRARY_PATH` below.
ONEAPI_BIN=/home/dario/oneapi/compiler/2025.3/bin
export PATH="$ONEAPI_BIN:$PATH"
export CXX="${CXX:-icpx}"

mkdir -p "$LAB/results"
OUT="$LAB/results/first_light.json"
LOG="$LAB/results/first_light.log"

# LD_LIBRARY_PATH is already sanitised above (oneAPI entries removed, our L0
# shim kept), so do NOT use `env -u LD_LIBRARY_PATH` here -- that would strip the
# shim the Inductor child needs to dlopen libze_loader.so.
"$PY" "$LAB/capsules/001_first_light.py" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo "===CAPSULE_EXIT=$rc==="

# Persist just the machine-readable verdict next to the log.
sed -n '/===RESULT_JSON===/,$p' "$LOG" | tail -n +2 > "$OUT" 2>/dev/null || true
[ -s "$OUT" ] && echo "wrote $OUT" || echo "no result json (capsule failed early)"
exit $rc

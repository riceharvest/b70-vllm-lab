#!/usr/bin/env bash
# Shared GPU-window gate for the B70 lab.
#
# THE GPU IS SHARED with a sibling agent. vLLM hard-fails at init under
# contention in two distinct ways, NEITHER of which is a result about graphs:
#
#   vllm/v1/worker/utils.py:544
#     "Free memory on device xpu:0 (10.26/30.3 GiB) on startup is less than
#      desired GPU memory utilization"
#   vllm/v1/worker/gpu_worker.py:600
#     "Initial free memory 20.88 GiB, current free memory 27.62 GiB. This
#      happens when other processes sharing the same container release GPU
#      memory while vLLM is profiling during initialization."
#
# The second is the nastier one: it fires when a sibling EXITS mid-profile, so
# "is there enough free VRAM right now" is not a sufficient gate. Require a
# window where no sibling engine exists AND free VRAM is stable across two
# samples 6s apart.
#
# NEVER kill another process to make room. Waiting is cheap; a wrong number is
# not. Source this, then call wait_gpu_window <min_free_gib> [budget_s].

PY=/mnt/ssd/b70-venv/bin/python

# F-002: oneAPI must not be on the runtime path; F-003 needs the L0 devel shim.
# `env -u LD_LIBRARY_PATH` strips the shim back out, so filter and re-add.
b70_fix_ld () {
  local _oneapi_ld="${LD_LIBRARY_PATH:-}" _clean_ld="" _p
  local -a _ldparts
  IFS=':' read -ra _ldparts <<< "$_oneapi_ld"
  for _p in "${_ldparts[@]}"; do
    [ -n "$_p" ] || continue
    case "$_p" in /home/dario/oneapi/*) continue ;; *) _clean_ld="${_clean_ld:+$_clean_ld:}$_p" ;; esac
  done
  export LD_LIBRARY_PATH="$HOME/.local/lib${_clean_ld:+:$_clean_ld}"
}

_b70_vram () {
  "$PY" - <<'EOF' 2>/dev/null || echo -1
import torch
print(int(torch.xpu.mem_get_info(0)[0] / 2**30))
EOF
}

# Sibling capsules that put an engine on the GPU. Extend when new ones appear.
_b70_siblings () {
  ps -eo cmd 2>/dev/null \
    | grep -E '012_mtp_k4|012_moe|011_determinism|010_xpu_graph|015_graph_determinism|017_vllm_pool' \
    | grep -v grep | wc -l
}

# wait_gpu_window <min_free_gib> [budget_s] -> 0 on success, 1 on give-up
wait_gpu_window () {
  local need="${1:-20}" budget="${2:-1500}" waited=0 n f1 f2
  while [ "$waited" -lt "$budget" ]; do
    n=$(_b70_siblings)
    f1=$(_b70_vram)
    if [ "$n" -eq 0 ] && [ "${f1:-0}" -ge "$need" ]; then
      sleep 6                      # let a dying sibling's memory settle
      f2=$(_b70_vram)
      if [ "${f2:-0}" -ge "$need" ] && [ $(( f2 > f1 ? f2 - f1 : f1 - f2 )) -le 1 ]; then
        echo "    gpu window ok: ${f2} GiB free, stable, no sibling engine (waited ${waited}s)"
        return 0
      fi
    fi
    echo "    waiting for gpu: siblings=${n} vram=${f1}GiB need=${need} (t=${waited}s)"
    sleep 20; waited=$((waited+20))
  done
  echo "    GAVE UP waiting for a stable GPU window after ${waited}s"
  return 1
}

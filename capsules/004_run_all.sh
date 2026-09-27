#!/usr/bin/env bash
# Capsule 004 — Xe2 grouped-GEMM uninitialized scheduler counter (kernels 0.1.14.1)
# Run:  ./004_run_all.sh
set -uo pipefail
PY=/mnt/ssd/b70-venv/bin/python
export LD_LIBRARY_PATH="$HOME/.local/lib"      # shim only; oneAPI NOT sourced
cd "$(dirname "$0")"

echo "### 003  smoke: op is callable and numerically correct"
$PY 003_smoke_gg.py;                     echo "   exit=$?"
echo; echo "### 004b allocator recycles a poisoned 1-elem int32 block"
$PY 004b_poison_probe.py;                 echo "   exit=$?"
echo; echo "### 004c is the eager race window actually open? (60 iters)"
$PY 004c_race_window.py;                  echo "   exit=$?"
echo; echo "### 004f binary A/B: installed 0.1.14.1 vs post-fix 0.1.15.4"
echo "    (needs the 0.1.15.4 wheel extracted under \$SCRATCH/x154)"
$PY 004f_binary_ab.py;                    echo "   exit=$?"

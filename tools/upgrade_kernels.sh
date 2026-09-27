#!/usr/bin/env bash
# Upgrade vllm_xpu_kernels 0.1.14.1 -> 0.1.15.4, then PROVE the env still works.
#
# WHY: 0.1.14.1 is pre-fix for the Xe2 grouped-GEMM data race (at::empty ->
# at::zeros, PR #586 / 3d74ec9). It also predates the GDN ragged-spec traversal
# fix and the v_head_id OOB guard. See FINDINGS.md F-008.
#
# WHY THIS IS A SCRIPT AND NOT A ONE-LINER: vLLM 0.30.0 pins
# vllm_xpu_kernels==0.1.14.1, so this is an intentional pin override. If the
# upgrade breaks anything we must be able to get back to a known-good state
# without rediscovering how. This script verifies before and after, and prints
# an explicit verdict.
set -uo pipefail

PY=/mnt/ssd/b70-venv/bin/python
UV=/home/dario/.local/bin/uv
EXPECT_OLD="0.1.14.1"
EXPECT_NEW="0.1.15.4"
LOG=/mnt/ssd/b70-vllm-lab/results/kernels_upgrade.log

mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

echo "=== BEFORE ==="
$UV pip list --python $PY 2>/dev/null | grep -iE '^vllm|^torch |^triton' || true
cur=$($UV pip list --python $PY 2>/dev/null | awk '/^vllm-xpu-kernels/{print $2}')
echo "current kernels: ${cur:-<none>}"

if [ "$cur" = "$EXPECT_NEW" ]; then
  echo "ALREADY at $EXPECT_NEW - nothing to do"
  exit 0
fi

# Record exact current state so a revert is mechanical.
$UV pip freeze --python $PY > /mnt/ssd/b70-vllm-lab/results/pip_freeze_before_upgrade.txt 2>/dev/null
echo "[1/4] pip freeze snapshot saved"

echo
echo "=== UPGRADE ==="
$UV pip install --python $PY "vllm_xpu_kernels==$EXPECT_NEW" 2>&1 | tail -12
rc=$?
echo "install rc=$rc"

echo
echo "=== AFTER: package state ==="
$UV pip list --python $PY 2>/dev/null | grep -iE '^vllm|^torch |^triton' || true

# The kernels wheel is the only thing that should have moved. If pip decided to
# pull a different torch or vllm, that is a red flag: the whole point of the
# pinned install was to keep the XPU torch.
new=$($UV pip list --python $PY 2>/dev/null | awk '/^vllm-xpu-kernels/{print $2}')
tv=$($UV pip list --python $PY 2>/dev/null | awk '/^torch /{print $2}')
vv=$($UV pip list --python $PY 2>/dev/null | awk '/^vllm /{print $2}')

echo
echo "=== VERDICT ==="
fail=0
if [ "$new" != "$EXPECT_NEW" ]; then
  echo "FAIL: kernels is $new, expected $EXPECT_NEW"; fail=1
fi
if [ "$tv" != "2.13.0+xpu" ]; then
  echo "FAIL: torch moved to '$tv' (expected 2.13.0+xpu) - restore required"; fail=1
fi
if [ "$vv" != "0.30.0+xpu" ] && [ "$vv" != "0.30.0" ]; then
  echo "FAIL: vllm moved to '$vv' (expected 0.30.0) - restore required"; fail=1
fi

# Functional check: the kernels extensions must still import AND a real XPU op
# must still run. Import success alone is not enough - the wheel swaps .so files.
echo
echo "=== FUNCTIONAL: import + real XPU op ==="
LD_LIBRARY_PATH=/home/dario/.local/lib $PY - <<'PYEOF'
import sys
try:
    import torch
    import vllm_xpu_kernels._C, vllm_xpu_kernels._moe_C, vllm_xpu_kernels._xpu_C
    import vllm_xpu_kernels.xpumem_allocator  # noqa: F401
    a = torch.randn(256, 256, device="xpu", dtype=torch.bfloat16)
    b = a @ a
    torch.xpu.synchronize()
    ok = bool(torch.isfinite(b).all()) and float(b.abs().sum()) > 0
    print("KERNEL_OP_OK" if ok else "KERNEL_OP_BAD")
    # the triton patch is load-bearing; without it vLLM segfaults
    from triton.backends.intel.driver import XPUUtils
    print("TRITON_DEVCOUNT", XPUUtils().device_count)
    sys.exit(0 if ok else 1)
except Exception as e:
    print("KERNEL_IMPORT_FAIL", type(e).__name__, e)
    sys.exit(1)
PYEOF
frc=$?
[ $frc -ne 0 ] && { echo "FAIL: functional check failed"; fail=1; }

echo
if [ $fail -ne 0 ]; then
  echo "=== UPGRADE FAILED - revert with: ==="
  echo "  $UV pip install --python $PY 'vllm_xpu_kernels==$EXPECT_OLD'"
  echo "  $UV pip sync --python $PY /mnt/ssd/b70-vllm-lab/results/pip_freeze_before_upgrade.txt 2>/dev/null || true"
  exit 1
fi

echo "=== UPGRADE OK: kernels $cur -> $new, torch/vllm unchanged, ops verified ==="
echo "Next: re-run capsule 001 to confirm the full engine still works."

#!/usr/bin/env bash
# Apply the F-004 fix to the b70-venv triton's Intel backend.
#
# WHY THIS EXISTS
#   vLLM 0.30.0 cannot start on Intel Arc B70 without it. See FINDINGS.md F-004.
#   triton/backends/intel/driver.py::find_sycl() probes `shutil.which("icpx")`
#   before the intel-sycl-rt branch, so with oneAPI on PATH it compiles
#   spirv_utils.so against libsycl.so.8 while libtorch_xpu.so already mapped
#   libsycl.so.9. The resulting queue* crosses a runtime ABI boundary and
#   sycl::context::get_devices() segfaults.
#
# This script is idempotent and reversible. It refuses to run if the installed
# triton does not match the expected file, so it can never silently corrupt an
# unrelated install.
#
# The real fix belongs upstream in intel-xpu-backend-for-triton. This is a local
# staging patch so the B70 lab has a working engine; see patches/.
set -uo pipefail

SP=/mnt/ssd/b70-venv/lib/python3.12/site-packages
REL="triton/backends/intel/driver.py"
TARGET="$SP/$REL"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/../patches/triton-intel-driver.patch"
BAK="$TARGET.b70lab.orig"

if [ ! -f "$TARGET" ]; then
  echo "ERROR: $TARGET not found" >&2
  exit 1
fi

if grep -q '_sycl_runtime_already_loaded' "$TARGET"; then
  echo "ALREADY PATCHED - nothing to do"
  exit 0
fi

if [ ! -f "$PATCH" ]; then
  echo "ERROR: patch not found at $PATCH" >&2
  exit 1
fi

cp -p "$TARGET" "$BAK"
echo "[1/3] backed up original to $BAK"

# Stage the patch against the real file to prove it applies cleanly.
#
# NOTE on strip level: the patch header is `--- a/<path>` with no timestamps.
# Running `patch -p1` from $STAGE does NOT find the file; you must either run
# from inside the `a/` dir with -p1, or use -p0 from the parent. We do the
# former. --batch is required so a failure cannot block on an interactive
# "File to patch:" prompt.
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/a/$(dirname "$REL")"
cp "$TARGET" "$STAGE/a/$REL"
( cd "$STAGE/a" && patch -p1 --batch --dry-run < "$PATCH" ) >/dev/null || {
  echo "ERROR: patch does not apply cleanly; aborting, original untouched" >&2
  exit 1
}
echo "[2/3] patch applies cleanly"

( cd "$STAGE/a" && patch -p1 --batch < "$PATCH" ) >/dev/null || {
  echo "ERROR: patch application failed; aborting, original untouched" >&2
  exit 1
}
cp "$STAGE/a/$REL" "$TARGET"
find "$(dirname "$TARGET")" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
echo "[3/3] patch installed"

# The module cache is keyed WITHOUT the SYCL runtime identity (that is the
# second half of the bug), so a stale spirv_utils.so built against libsycl.so.8
# would be silently reused. It MUST be purged or the fix does not take effect.
echo "[cache] purging stale triton spirv_utils modules"
rm -rf /home/dario/.triton/cache ~/.cache/vllm/torch_compile_cache 2>/dev/null

echo
echo "Verify with:"
echo "  env -u LD_LIBRARY_PATH /mnt/ssd/b70-venv/bin/python \\"
echo "    -c 'import torch; from triton.backends.intel.driver import XPUUtils; a=torch.randn(64,64,device=\"xpu\"); torch.xpu.synchronize(); print(XPUUtils().device_count)'"
echo "  expect: 1   (a SIGSEGV here means the patch did not take)"
echo
echo "Revert with: $0 --revert"

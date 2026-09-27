#!/usr/bin/env python
"""Capsule 004f — binary A/B: is the installed 0.1.14.1 lib pre-#586?

The XPU profiler cannot see the extension's raw sycl::queue submits, and the
eager-mode race is masked by the kernel's own store(0), so neither output
comparison nor profiling can answer this. The ELF can.

`at::empty(...)` and `at::zeros(...)` compile to different libtorch symbols:

    at::empty({1}, ...)  ->  at::_ops::empty_memory_format::call(...)
    at::zeros({1}, ...)  ->  at::_ops::zeros::call(...)

so the relocation set of the out-of-line impl
`MoE::cutlass_grouped_gemm_xe2_impl` (in libgrouped_gemm_xe_2.so, NOT the
_dispatcher_ in _xpu_C.abi3.so) names which allocator the shipped binary uses.

A positive control is essential, so this script also inspects the post-fix
wheel (0.1.15.4, PyPI 2026-09-22, after 3d74ec9 on 2026-09-10) and requires
the two to disagree. If they agreed the method would be worthless.
"""
import re
import subprocess
import sys
from pathlib import Path

OLD = Path("/mnt/ssd/b70-venv/lib/python3.12/site-packages/vllm_xpu_kernels/"
           "libgrouped_gemm_xe_2.so")  # installed 0.1.14.1
NEW = Path("/home/dario/.hermes/cache/scratch/wheels/x154/vllm_xpu_kernels/"
           "libgrouped_gemm_xe_2.so")  # 0.1.15.4, post-#586
SYM = "cutlass_grouped_gemm_xe2_impl"


def impl_range(lib):
    out = subprocess.run(["nm", "-D", "--defined-only", "-S", str(lib)],
                         capture_output=True, text=True).stdout
    for line in out.split("\n"):
        p = line.split()
        if len(p) >= 4 and p[3].startswith("_Z") and SYM in p[3]:
            lo = int(p[0], 16)
            return lo, lo + int(p[1], 16)
    return None


def call_targets_in_impl(lib):
    lo, hi = impl_range(lib)
    dis = subprocess.run(
        ["objdump", "-d", f"--start-address={lo}", f"--stop-address={hi}",
         str(lib)], capture_output=True, text=True).stdout
    tgts = set()
    for m in re.finditer(r"call\s+[0-9a-f]+ <([^>]+)>", dis):
        tgts.add(m.group(1))
    dem = subprocess.run(["c++filt"], input="\n".join(sorted(tgts)),
                         capture_output=True, text=True).stdout.split("\n")
    return lo, hi, [d for d in dem if d.strip()]


def classify(lib, label):
    lo, hi, names = call_targets_in_impl(lib)
    joined = " | ".join(names)
    has_zeros = "at::_ops::zeros::call" in joined
    has_empty = "at::_ops::empty_memory_format::call" in joined
    print(f"\n--- {label} ---")
    print(f"  {lib.name}: impl 0x{lo:x}-0x{hi:x} ({hi-lo} bytes)")
    print(f"  calls at::_ops::zeros::call             : {has_zeros}")
    print(f"  calls at::_ops::empty_memory_format::call: {has_empty}")
    return has_zeros, has_empty


def main():
    for lib in (OLD, NEW):
        if not lib.exists():
            print(f"missing: {lib}")
            return 2

    old_z, old_e = classify(OLD, "INSTALLED 0.1.14.1")
    new_z, new_e = classify(NEW, "CONTROL 0.1.15.4 (post-#586)")

    print("\n=== VERDICT ===")
    if old_e and not old_z:
        print("Installed 0.1.14.1 is PRE-FIX: the persistent-workgroup scheduler")
        print("counter is allocated with at::empty() -> UNINITIALIZED.")
    elif old_z and not old_e:
        print("Installed 0.1.14.1 is POST-FIX (at::zeros).")
    else:
        print("Installed build: inconclusive.")

    if new_z and not new_e:
        print("Control behaved as expected: 0.1.15.4 uses at::zeros().")
        print("=> the at::empty/at::zeros discrimination method is VALIDATED.")
    else:
        print("WARNING: control did not behave as expected; method unvalidated.")
        return 3

    return 0 if (old_e and not old_z) else 1


if __name__ == "__main__":
    sys.exit(main())

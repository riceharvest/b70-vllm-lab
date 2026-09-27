#!/usr/bin/env python3
"""Minimal reproducer for F-004: triton-xpu XPUUtils.__init__ segfault.

The hypothesis: triton's `spirv_utils` module links oneAPI 2025.3's
`libsycl.so.8`, while torch 2.13.0+xpu links intel-sycl-rt 2026.0.0's
`libsycl.so.9`.  Two distinct SYCL C++ runtimes in one process.  Passing
torch's `sycl::queue*` across the ABI boundary and calling
`sycl::context::get_devices()` on it dereferences vtables from the wrong
runtime -> SIGSEGV inside `init_devices`.

Run with:  env -u LD_LIBRARY_PATH /mnt/ssd/b70-venv/bin/python repro1_triton_init.py
"""
import os
import sys

print("=" * 70)
print("STEP 1: load torch.xpu, create a real sycl queue")
print("=" * 70)
import torch

print("torch:", torch.__version__)
print("xpu available:", torch.xpu.is_available())
torch.xpu.init()
torch.xpu.set_device(0)

# Force the SYCL runtime to be fully up and hand back its sycl_queue.
# This is exactly what triton's XPUUtils.get_sycl_queue() does.
q = torch.xpu.current_stream().sycl_queue
print("sycl_queue handle:", hex(q) if isinstance(q, int) else q)

# Show which libsycl torch pulled in.
with open("/proc/self/maps") as f:
    maps = f.read()
sycl_libs = sorted({ln.split()[-1] for ln in maps.splitlines() if "libsycl" in ln})
print("libsycl mapped after torch:", sycl_libs)
assert len(sycl_libs) == 1, f"expected exactly one libsycl, got {sycl_libs}"
major_torch = sycl_libs[0]

print()
print("=" * 70)
print("STEP 2: construct triton's XPUUtils (calls init_devices)")
print("=" * 70)
from triton.backends.intel.driver import XPUUtils

print("about to call XPUUtils() -> mod.init_devices(queue)", flush=True)
utils = XPUUtils()
print("UNREACHABLE?", utils.device_count)

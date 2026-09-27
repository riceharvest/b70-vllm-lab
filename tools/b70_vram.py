#!/usr/bin/env python3
"""Report Arc Pro B70 free/total device memory.

The xe kernel driver on Fedora 43 exposes no sysfs VRAM accounting:
  /sys/class/drm/card1/device/mem_info_vram_total   -> absent
  /sys/class/drm/card1/device/mem_info_vram_used     -> absent
  /sys/class/drm/card1/client*/drm-memory-*          -> empty
So the kernel gives us nothing, and the only reliable source of device-level
free memory is torch's SYCL allocator.

IMPORTANT: run with oneAPI NOT sourced.
    env -u LD_LIBRARY_PATH python3 tools/b70_vram.py     # correct
    source ~/oneapi/setvars.sh && python3 tools/...      # undefined symbol crash

A ctypes probe of the Level Zero `ext_intel_free_memory` aspect was tried first
and segfaulted: the vendor struct layout is version-sensitive and undocumented
in the installed headers. Do not repeat that approach.

Emits JSON so the experiment runner can diff VRAM headroom across a capsule.
"""

import json
import sys


def main():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"torch import failed: {exc}\n"
            "Did you source oneapi setvars.sh? That shadows the system UR "
            "loader. Use: env -u LD_LIBRARY_PATH python3 tools/b70_vram.py"
        )

    if not torch.xpu.is_available():
        raise SystemExit("torch.xpu.is_available() is False — no B70 visible")

    idx = 0
    free, total = torch.xpu.mem_get_info(idx)
    gib = 2**30

    out = {
        "device": torch.xpu.get_device_name(idx),
        "driver_version": torch.xpu.get_device_properties(idx).driver_version,
        "total_bytes": total,
        "total_gib": round(total / gib, 2),
        "free_bytes": free,
        "free_gib": round(free / gib, 2),
        "used_gib": round((total - free) / gib, 2),
        "used_pct": round(100.0 * (total - free) / total, 1),
        # Bytes the torch allocator itself holds, a subset of `used`.
        "torch_reserved_gib": round(torch.xpu.memory_reserved(idx) / gib, 3),
        "torch_allocated_gib": round(torch.xpu.memory_allocated(idx) / gib, 3),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Capsule 002: does oneDNN produce NaN on the B70 at real vLLM GEMM shapes?

Upstream claim (intel/compute-runtime#1000): B70 fp16/bf16 oneDNN GEMM returns
NaN at 512^3, 1024^3 and 128x4096x4096. The last is an actual vLLM shape
(batch 128 tokens, 4096 hidden, 4096 out), so if it reproduces it poisons every
downstream perf number -- there would be no point benchmarking anything.

This is deliberately the cheapest experiment in the lab: pure torch, no vLLM, no
model, seconds to run. Run it FIRST when the queue is empty.

Run:  LD_LIBRARY_PATH=$HOME/.local/lib /mnt/ssd/b70-venv/bin/python 002_nan_check.py
"""

import json
import sys

# (M, K, N) -- the last two are real vLLM projection shapes; 512^3/1024^3 are the
# sizes named in the upstream report.
SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (128, 4096, 4096),
    (4096, 4096, 4096),
    # Decode-shaped: batch 1-8 tokens through a 4096-wide projection.
    (1, 4096, 4096),
    (8, 4096, 4096),
    # MoE-shaped grouped sizes.
    (2048, 4096, 14336),
    (16, 4096, 11008),
]


def main() -> int:
    import torch

    if not torch.xpu.is_available():
        raise SystemExit("no XPU visible")

    print(f"torch {torch.__version__}  device {torch.xpu.get_device_name(0)}", flush=True)

    results = []
    nan_found = False

    for dtype in (torch.float16, torch.bfloat16):
        for (m, k, n) in SHAPES:
            torch.manual_seed(0)
            a = torch.randn(m, k, device="xpu", dtype=dtype)
            b = torch.randn(k, n, device="xpu", dtype=dtype)
            try:
                c = a @ b
                torch.xpu.synchronize()
                finite = bool(torch.isfinite(c).all())
                n_nan = int(torch.isnan(c).sum())
                n_inf = int(torch.isinf(c).sum())
                mag = float(c.abs().max())
            except BaseException as exc:  # noqa: BLE001 - we want the type
                finite, n_nan, n_inf, mag = False, -1, -1, float("nan")
                print(f"  EXC {dtype} {m}x{k}x{n}: {type(exc).__name__}: {exc}",
                      flush=True)

            bad = (not finite) or n_nan or n_inf
            nan_found = nan_found or bad
            results.append({
                "dtype": str(dtype).replace("torch.", ""),
                "M": m, "K": k, "N": n,
                "finite": finite,
                "nan_count": n_nan,
                "inf_count": n_inf,
                "absmax": None if mag != mag else round(mag, 3),
            })
            flag = "  <-- NON-FINITE" if bad else ""
            print(f"  {str(dtype).replace('torch.',''):9s} "
                  f"{m:>5}x{k:<5}x{n:<6} finite={finite} "
                  f"nan={n_nan} inf={n_inf} absmax={results[-1]['absmax']}{flag}",
                  flush=True)
            del a, b
            if "c" in dir():
                del c

    verdict = "FAIL_NAN" if nan_found else "PASS_ALL_FINITE"
    print(f"\nVERDICT: {verdict}")
    print("===RESULT_JSON===")
    print(json.dumps({
        "capsule": "002_nan_check",
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(0),
        "verdict": verdict,
        "results": results,
    }, indent=2))
    return 1 if nan_found else 0


if __name__ == "__main__":
    sys.exit(main())

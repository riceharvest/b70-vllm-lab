#!/usr/bin/env python3
"""Capsule 001b: isolate WHERE the B70 segfault happens.

Capsule 001 (graph path, VLLM_XPU_ENABLE_XPU_GRAPH=1) segfaults inside
torch.compile after "Dynamo bytecode transform time". This script walks the
variable space so we can attribute the crash to a specific mechanism instead of
guessing:

  A. enforce_eager=True, no XPU graph      -> pure eager, no compile
  B. enforce_eager=True, XPU graph on      -> XPU graph only, no compile
  C. enforce_eager=False, no XPU graph     -> torch.compile only
  D. enforce_eager=False, XPU graph on     -> both (the failing config)

Each case runs in a SUBPROCESS so one segfault cannot take down the others, and
so a crash is observable as a signal rather than a silent hang.

Usage:  python 001b_isolate.py A B C D
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/mnt/ssd/b70-venv/bin/python"


def run_case(case: str) -> dict:
    env = dict(os.environ)
    env["CASE"] = case
    # Build the child command explicitly rather than branching inside one
    # process: a SYCL segfault is not catchable, so isolation must be real.
    code = f"""
import os, json, time
case = os.environ["CASE"]
import torch
from vllm import LLM, SamplingParams

graph = case in ("B", "D")
eager = case in ("A", "B")
from vllm.platforms import current_platform
out = {{"case": case, "platform": current_platform.device_name,
        "enforce_eager": eager, "xpu_graph": graph}}
t0 = time.perf_counter()
llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16",
          enforce_eager=eager, gpu_memory_utilization=0.80)
out["load_seconds"] = round(time.perf_counter() - t0, 2)
r = llm.generate(["The capital of France is"],
                 SamplingParams(temperature=0.0, max_tokens=16))
out["text"] = r[0].outputs[0].text
out["ok"] = True
print("CASE_JSON:" + json.dumps(out))
"""
    proc = subprocess.run(
        [PY, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    rec = {
        "case": case,
        "returncode": proc.returncode,
        "signal": -proc.returncode if proc.returncode < 0 else None,
    }
    for line in proc.stdout.splitlines():
        if line.startswith("CASE_JSON:"):
            rec.update(json.loads(line[len("CASE_JSON:"):]))
    tail = [l for l in proc.stderr.splitlines() if l.strip()][-6:]
    rec["stderr_tail"] = tail
    return rec


def main() -> int:
    cases = sys.argv[1:] or ["A", "B", "C", "D"]
    results = []
    for c in cases:
        print(f"--- case {c} ---", flush=True)
        r = run_case(c)
        results.append(r)
        verdict = (
            "OK" if r.get("ok")
            else f"CRASH(sig {r['signal']})" if r.get("signal")
            else f"FAIL(rc {r['returncode']})"
        )
        print(f"    {verdict}", flush=True)
        for l in r.get("stderr_tail", []):
            print(f"      | {l[:150]}", flush=True)

    print("\n===ISOLATION_JSON===")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

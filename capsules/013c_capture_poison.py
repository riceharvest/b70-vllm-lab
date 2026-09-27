"""Capsule 013c - does a CAPTURED GRAPH poison subsequent EAGER calls?

013 (probe A) reported `eager_self_consistent=False` for k>=2, but 013b showed
200 bit-identical EAGER calls produce exactly 1 distinct result. Both cannot be
true, so the difference between them is the thing to explain.

The difference: 013 interleaves CAPTURE and EAGER inside one process, and the
capture happens in a `torch.xpu.stream(warm)` side stream. 013b never captures.

Hypothesis under test: once a graph is captured in this process, later EAGER
submissions on the default stream read something different (stale/poisoned
memory) - i.e. capture leaks state into eager. If true that is a serious,
independently-reportable XPU runtime defect, and it would also be a plausible
mechanism for #54785 (whose datapoint 9 shows corruption with
TORCH_COMPILE_DISABLE=1, i.e. pure eager kernels inside a captured step).

Method - for each k, in ONE process, in this order:
  A1: 20 eager calls               -> count distinct results
  A2: capture a graph for this k
  A3: 20 more eager calls          -> count distinct results
  A4: compare A1's result to A3's
  A5: 20 graph replays             -> count distinct results
  A6: compare A3's result to A5's
A clean kernel: A1 == A3 == A5, each exactly 1 distinct value.
"""

import json
import time

import torch
import vllm  # noqa: F401  registers the XPU custom ops

import sys
sys.path.insert(0, "/mnt/ssd/b70-vllm-lab/capsules")
from importlib import import_module
m013 = import_module("013_kernel_determinism")
gdn_spec_args = m013.gdn_spec_args
op = torch.ops._xpu_C.gdn_attention

DT = torch.bfloat16
RES = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "per_k": [], "ok": False}


def distinct_after_eager(k, seed, n):
    seen = {}
    for i in range(n):
        a = gdn_spec_args(k, DT, seed)
        op(**a)
        torch.xpu.synchronize()
        key = (round(a["core_attn_out"].float().sum().item(), 3),
               round(a["ssm_state"].float().sum().item(), 3))
        seen.setdefault(key, []).append(i)
    return seen


def capture(k, seed):
    warm = torch.xpu.Stream()
    warm.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(warm):
        for _ in range(3):
            op(**gdn_spec_args(k, DT, seed))
    torch.xpu.current_stream().wait_stream(warm)
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    static = gdn_spec_args(k, DT, seed)
    with torch.xpu.graph(g):
        op(**static)
    torch.xpu.synchronize()
    return g, static


def distinct_after_replay(g, static, n):
    seen = {}
    for i in range(n):
        g.replay()
        torch.xpu.synchronize()
        key = (round(static["core_attn_out"].float().sum().item(), 3),
               round(static["ssm_state"].float().sum().item(), 3))
        seen.setdefault(key, []).append(i)
    return seen


N = 20
for k in range(1, 7):
    seed = 1234 + k
    rec = {"k": k, "T": k + 1}

    a1 = distinct_after_eager(k, seed, N)
    rec["a1_distinct_eager_before_capture"] = len(a1)
    rec["a1_values"] = [list(kk) for kk in a1.keys()][:4]

    gph, static = capture(k, seed)
    rec["capture_ok"] = True

    a3 = distinct_after_eager(k, seed, N)
    rec["a3_distinct_eager_after_capture"] = len(a3)
    rec["a3_values"] = [list(kk) for kk in a3.keys()][:4]

    a1_keys = set(a1.keys())
    a3_keys = set(a3.keys())
    rec["eager_stable_across_capture"] = (len(a1_keys) == 1 and a1_keys == a3_keys)

    a5 = distinct_after_replay(gph, static, N)
    rec["a5_distinct_replays"] = len(a5)
    rec["a5_values"] = [list(kk) for kk in a5.keys()][:4]
    rec["replay_matches_eager"] = (len(a3_keys) == 1 and a3_keys == set(a5.keys()))

    print(f"k={k} T={k+1}: "
          f"eager_before={rec['a1_distinct_eager_before_capture']} "
          f"eager_after={rec['a3_distinct_eager_after_capture']} "
          f"replays={rec['a5_distinct_replays']} | "
          f"eager_stable_across_capture={rec['eager_stable_across_capture']} "
          f"replay_matches_eager={rec['replay_matches_eager']}")
    if not rec["eager_stable_across_capture"]:
        print(f"    before: {rec['a1_values'][:3]}")
        print(f"    after : {rec['a3_values'][:3]}")
    if not rec["replay_matches_eager"]:
        print(f"    eager : {rec['a3_values'][:3]}")
        print(f"    replay: {rec['a5_values'][:3]}")

    RES["per_k"].append(rec)

RES["summary"] = {
    "eager_unstable_after_capture": [
        r["k"] for r in RES["per_k"] if not r["eager_stable_across_capture"]],
    "replay_disagrees_with_eager": [
        r["k"] for r in RES["per_k"] if not r["replay_matches_eager"]],
    "eager_multiple_distinct": [
        r["k"] for r in RES["per_k"]
        if r["a1_distinct_eager_before_capture"] > 1
        or r["a3_distinct_eager_after_capture"] > 1],
}
print("\nSUMMARY:", json.dumps(RES["summary"]))
RES["ok"] = True
RES["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
with open("/mnt/ssd/b70-vllm-lab/results/013c_capture_poison.json", "w") as f:
    json.dump(RES, f, indent=2)
print("===WROTE=== /mnt/ssd/b70-vllm-lab/results/013c_capture_poison.json")

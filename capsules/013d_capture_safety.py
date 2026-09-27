"""Capsule 013d - the DECISIVE capture-safety test for the GDN spec op.

013c found that replaying a captured GDN spec graph gives a different result on
each replay. That is NOT yet a bug: the op is RECURRENT - it mutates conv_state
and ssm_state by design, so a replay that does not reset state is computing a
legitimately different (longer-history) recurrence. 013c's oracle compared
replay N against a fresh eager call, which conflates "the graph is broken" with
"the state carries over".

The decisive test separates the two:
  * reset the captured graph's static state back to the pristine eager values
    before EVERY replay, then compare to a fresh eager call.
  - If reset+replay == eager, every time -> the graph is capture-safe and
    013c's drift was the probe's fault.
  - If reset+replay still drifts -> the graph captured a STALE POINTER or stale
    scalar, and the drift is a real capture-safety defect. That is the
    #54785 failure class (silent wrong logits) and is reportable.

Also sweeps `num_accepted_tokens`, because the spec kernel reads the previous
step's state from cache_indices[num_accepted-1] - the reporter's addendum
suspects exactly this "state that is not correctly reset or re-bound between
piece replays".

For each k in 1..6:
  R1: reset+replay vs eager, num_accepted=1
  R2: reset+replay vs eager, num_accepted=k (max accept, a different state column)
  R3: no-reset replay drift count (reproduces 013c, to show the two regimes
      really differ - otherwise R1 is vacuous)
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
N = 12


def key_of(a):
    return (round(a["core_attn_out"].float().sum().item(), 3),
            round(a["ssm_state"].float().sum().item(), 3))


def eager_once(k, seed, n_acc):
    a = gdn_spec_args(k, DT, seed)
    a["num_accepted_tokens"] = torch.tensor([n_acc], dtype=torch.int32, device="xpu")
    op(**a)
    torch.xpu.synchronize()
    return key_of(a), a


def build_capture(k, seed, n_acc):
    warm = torch.xpu.Stream()
    warm.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(warm):
        for _ in range(3):
            a = gdn_spec_args(k, DT, seed)
            a["num_accepted_tokens"] = torch.tensor([n_acc], dtype=torch.int32, device="xpu")
            op(**a)
    torch.xpu.current_stream().wait_stream(warm)
    torch.xpu.synchronize()

    g = torch.xpu.XPUGraph()
    # pristine reference values, captured BEFORE the graph mutates anything
    pristine = gdn_spec_args(k, DT, seed)
    pristine["num_accepted_tokens"] = torch.tensor([n_acc], dtype=torch.int32, device="xpu")
    pristine = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in pristine.items()}
    static = gdn_spec_args(k, DT, seed)
    static["num_accepted_tokens"] = torch.tensor([n_acc], dtype=torch.int32, device="xpu")
    with torch.xpu.graph(g):
        op(**static)
    torch.xpu.synchronize()
    return g, static, pristine


def reset(static, pristine, keys):
    for kk in keys:
        static[kk].copy_(pristine[kk])


# the buffers the op mutates in place (mutation lists from the C++ signature:
# $0 core_attn_out, $1 z, $2 conv_state, $3 ssm_state)
MUTATED = ["core_attn_out", "z", "conv_state", "ssm_state"]

for k in range(1, 7):
    seed = 1234 + k
    rec = {"k": k, "T": k + 1}

    for label, n_acc in (("acc1", 1), ("accK", k)):
        ref_key, _ = eager_once(k, seed, n_acc)
        g, static, pristine = build_capture(k, seed, n_acc)

        # R1/R2: RESET then replay
        seen = {}
        for i in range(N):
            reset(static, pristine, MUTATED)
            g.replay()
            torch.xpu.synchronize()
            seen.setdefault(key_of(static), []).append(i)
        rec[f"{label}_ref"] = list(ref_key)
        rec[f"{label}_reset_replay_distinct"] = len(seen)
        rec[f"{label}_reset_replay_matches_eager"] = (set(seen.keys()) == {ref_key})

        # R3: NO reset - reproduces 013c's drift, proves the regimes differ
        seen_nr = {}
        for i in range(N):
            g.replay()
            torch.xpu.synchronize()
            seen_nr.setdefault(key_of(static), []).append(i)
        rec[f"{label}_noreset_replay_distinct"] = len(seen_nr)
        rec[f"{label}_noreset_matches_eager"] = (set(seen_nr.keys()) == {ref_key})

    bad_reset = [kk for kk in ("acc1", "accK")
                 if not rec[f"{kk}_reset_replay_matches_eager"]]
    rec["capture_safe"] = not bad_reset
    print(f"k={k} T={k+1}: "
          f"reset_replay[acc1] distinct={rec['acc1_reset_replay_distinct']} "
          f"match={rec['acc1_reset_replay_matches_eager']} | "
          f"reset_replay[accK={k}] distinct={rec['accK_reset_replay_distinct']} "
          f"match={rec['accK_reset_replay_matches_eager']} | "
          f"noreset distinct acc1={rec['acc1_noreset_replay_distinct']} "
          f"accK={rec['accK_noreset_replay_distinct']}")
    if bad_reset:
        print(f"    NOT capture-safe for {bad_reset}")
        for kk in bad_reset:
            print(f"    {kk} ref={rec[f'{kk}_ref']} "
                  f"replay_values={sorted(seen.keys()) if False else ''}")
    RES["per_k"].append(rec)

RES["summary"] = {
    "capture_safe_ks": [r["k"] for r in RES["per_k"] if r["capture_safe"]],
    "capture_UNSAFE_ks": [r["k"] for r in RES["per_k"] if not r["capture_safe"]],
    "noreset_drifts_always": all(
        r["acc1_noreset_replay_distinct"] > 1 for r in RES["per_k"]),
}
print("\nSUMMARY:", json.dumps(RES["summary"]))
RES["ok"] = True
RES["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
with open("/mnt/ssd/b70-vllm-lab/results/013d_capture_safety.json", "w") as f:
    json.dump(RES, f, indent=2)
print("===WROTE=== /mnt/ssd/b70-vllm-lab/results/013d_capture_safety.json")

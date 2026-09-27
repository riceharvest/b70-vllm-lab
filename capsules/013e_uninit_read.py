"""Capsule 013e - is the GDN spec op reading UNINITIALIZED memory?

Motivation. 013c and 013d disagree about the same no-reset replay regime
(013c: k=1 drifts, 013d: k=2..4 do not). Two runs of the "same" experiment
disagreeing means the probe is measuring something that depends on state
outside the op's arguments. The prime suspect is the allocator:

  csrc/xpu/gdn_attn/gdn_attn_interface.cpp allocates the conv-stage
  intermediates with torch::empty, i.e. UNINITIALIZED:
      torch::Tensor q = torch::empty({spec_token, num_k_heads/tp, head_k_dim}, ...);
      torch::Tensor k = torch::empty(...);
      torch::Tensor v = torch::empty(...);
      torch::Tensor b = torch::empty({spec_token, num_v_heads/tp}, ...);
      torch::Tensor a = torch::empty({spec_token, num_v_heads/tp}, ...);
  (also q/k/v/b/a are torch::zeros only on the XE2 non-spec PREFILL path)

  If the kernels do not write every element of those buffers, the unwritten
  elements are whatever the caching allocator last left there. That is the
  SAME defect class as FINDINGS.md F-008 (Xe2 grouped-GEMM at::empty
  tile counter), and it is exactly the "stale-cache/replay" hazard.

Falsification design. Vary ONE thing - the contents of the allocator's memory
pool - and see whether the op's output changes. The op's own arguments are held
bit-identical across all arms.

  ARM 1  baseline: call the op on fresh args.
  ARM 2..N  dirty the pool with a recognisable garbage pattern, FREE it (so the
            allocator will hand that memory back), then call the op with
            bit-identical args.
  ARM N+1 control: repeat ARM 1 to prove nothing drifted over time.

If the output changes when only the POOL changed, the op is reading
uninitialized memory. That is a real, silent, reportable defect. If the output
is invariant, the op writes everything it reads, and the 013c/013d
disagreement was probe-side allocator noise.
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

# A poison pattern chosen to be unmistakable and to blow up if it lands in a
# float accumulation: large, positive, and distinct per arm.
POISONS = [1.0e3, -1.0e3, 2.5e4, -7.5e4, 1.0e6, -1.0e6]


def key_of(a):
    return (round(a["core_attn_out"].float().sum().item(), 4),
            round(a["ssm_state"].float().sum().item(), 4))


def dirty_pool(value, nbytes=64 << 20):
    """Allocate, fill with `value`, and free - so the pool now holds `value`."""
    n = nbytes // 4
    t = torch.full((n,), value, dtype=torch.float32, device="xpu")
    del t


for k in range(1, 7):
    seed = 1234 + k
    rec = {"k": k, "T": k + 1, "arms": []}

    def run_arm(label, poison=None):
        if poison is not None:
            dirty_pool(poison)
        a = gdn_spec_args(k, DT, seed)
        op(**a)
        torch.xpu.synchronize()
        key = key_of(a)
        rec["arms"].append({"label": label, "poison": poison, "key": list(key)})
        return key

    base = run_arm("baseline")
    for i, p in enumerate(POISONS):
        run_arm(f"poison_{p:g}", poison=p)
    # control: repeat the baseline arm. If the op is deterministic, this must
    # equal the first baseline. If it does not, the op is order-dependent.
    ctrl = run_arm("control_repeat")

    vals = {tuple(a["key"]) for a in rec["arms"]}
    rec["distinct_outputs_across_pool_states"] = len(vals)
    rec["output_depends_on_pool_contents"] = len(vals) > 1
    rec["control_equals_baseline"] = (base == ctrl)
    print(f"k={k} T={k+1}: distinct outputs across {len(rec['arms'])} pool states = "
          f"{len(vals)}  (output_depends_on_pool={rec['output_depends_on_pool_contents']}, "
          f"control==baseline={rec['control_equals_baseline']})")
    for a in rec["arms"]:
        print(f"    {a['label']:<20} -> {a['key']}")
    RES["per_k"].append(rec)

RES["summary"] = {
    "ks_where_output_depends_on_pool": [
        r["k"] for r in RES["per_k"] if r["output_depends_on_pool_contents"]],
    "ks_where_control_differs": [
        r["k"] for r in RES["per_k"] if not r["control_equals_baseline"]],
}
print("\nSUMMARY:", json.dumps(RES["summary"]))
RES["ok"] = True
RES["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
with open("/mnt/ssd/b70-vllm-lab/results/013e_uninit_read.json", "w") as f:
    json.dump(RES, f, indent=2)
print("===WROTE=== /mnt/ssd/b70-vllm-lab/results/013e_uninit_read.json")

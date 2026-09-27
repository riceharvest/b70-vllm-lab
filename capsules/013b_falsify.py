"""Capsule 013b - FALSIFICATION test for capsule 013 probe A.

013 reported "eager not self-consistent" and max_abs ~9e10. Before that is
called a kernel bug, the probe itself must be shown capable of FAILING and the
input tensors shown to be bit-identical between calls. Otherwise a probe bug is
indistinguishable from a kernel bug, which is the exact trap FINDINGS.md F-008
warns about (an uninitialized `at::empty` read looked like corruption until a
positive control proved the method was falsifiable).

Step 1 - is the op's INPUT actually identical across two fresh() calls?
Step 2 - does the op's OUTPUT depend on anything outside its arguments?
         (compare 2 fresh calls, then 2 calls with a *shared* state buffer)
Step 3 - is the large divergence concentrated in core_attn_out or in ssm_state?
Step 4 - is it a race? run the SAME identical call 200 times and count how many
         distinct outputs appear. A race gives >1 distinct result from
         bit-identical inputs; an uninitialised read gives a stable garbage
         value; a correct kernel gives exactly 1.
Step 5 - poison test: fill ssm_state and conv_state with a recognisable pattern
         instead of zeros. If the output is unchanged, the kernel is ignoring
         its state argument entirely, which is a real defect and not a
         numerical artefact.
"""

import json
import os
import time

import torch
import vllm  # noqa: F401  registers the XPU custom ops

import sys
sys.path.insert(0, "/mnt/ssd/b70-vllm-lab/capsules")
from importlib import import_module
m013 = import_module("013_kernel_determinism")
gdn_spec_args = m013.gdn_spec_args
op = torch.ops._xpu_C.gdn_attention

RES = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "k": 3, "ok": False}


def same(a, b):
    return torch.equal(a, b)


# ---------------- Step 1: are the inputs identical between fresh() calls?
a1 = gdn_spec_args(3, torch.bfloat16, 1237)
a2 = gdn_spec_args(3, torch.bfloat16, 1237)
tensor_keys = [k for k, v in a1.items() if torch.is_tensor(v)]
identical = {k: same(a1[k], a2[k]) for k in tensor_keys}
RES["step1_inputs_identical"] = identical
RES["step1_all_identical"] = all(identical.values())
RES["step1_nonidentical_keys"] = [k for k, v in identical.items() if not v]
print("STEP1 input tensors bit-identical across fresh() calls:", RES["step1_all_identical"])
if not RES["step1_all_identical"]:
    print("  differing:", RES["step1_nonidentical_keys"])

# ---------------- Step 2/3: two independent eager calls
op(**a1)
torch.xpu.synchronize()
o1, s1 = a1["core_attn_out"].clone(), a1["ssm_state"].clone()
op(**a2)
torch.xpu.synchronize()
o2, s2 = a2["core_attn_out"].clone(), a2["ssm_state"].clone()

RES["step2_output_equal"] = same(o1, o2)
RES["step2_state_equal"] = same(s1, s2)
RES["step2_out_max_abs_diff"] = (o1.float() - o2.float()).abs().max().item()
RES["step2_state_max_abs_diff"] = (s1.float() - s2.float()).abs().max().item()
print(f"STEP2 output equal: {RES['step2_output_equal']} "
      f"max_abs={RES['step2_out_max_abs_diff']:.6g}")
print(f"STEP2 state  equal: {RES['step2_state_equal']} "
      f"max_abs={RES['step2_state_max_abs_diff']:.6g}")

# ---------------- Step 4: race check - 200 identical calls, count distinct
N = 200
seen = {}
for i in range(N):
    a = gdn_spec_args(3, torch.bfloat16, 1237)
    op(**a)
    torch.xpu.synchronize()
    key = (a["core_attn_out"].float().sum().item(),
           a["ssm_state"].float().sum().item())
    seen.setdefault(key, []).append(i)
RES["step4_calls"] = N
RES["step4_distinct_results"] = len(seen)
RES["step4_example_keys"] = [[round(x, 6) for x in k] for k in list(seen)[:6]]
print(f"STEP4 {N} bit-identical calls -> {len(seen)} distinct result(s)")
for k, idxs in list(seen.items())[:6]:
    print(f"   out_sum={k[0]:.6g} state_sum={k[1]:.6g}  first_at={idxs[0]} n={len(idxs)}")

# ---------------- Step 5: poison test - does the kernel read its state at all?
base = gdn_spec_args(3, torch.bfloat16, 1237)
op(**base)
torch.xpu.synchronize()
zero_state_out = base["core_attn_out"].clone()

pois = gdn_spec_args(3, torch.bfloat16, 1237)
# recognisable non-zero pattern in every slot the kernel could touch
pois["ssm_state"].fill_(0.5)
pois["conv_state"].fill_(0.25)
op(**pois)
torch.xpu.synchronize()
pois_out = pois["core_attn_out"].clone()
RES["step5_poison_changes_output"] = not same(zero_state_out, pois_out)
RES["step5_zero_sum"] = float(zero_state_out.float().sum())
RES["step5_poison_sum"] = float(pois_out.float().sum())
print(f"STEP5 poisoning state changes output: {RES['step5_poison_changes_output']} "
      f"(zero_sum={RES['step5_zero_sum']:.6g} poison_sum={RES['step5_poison_sum']:.6g})")

RES["ok"] = True
RES["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
with open("/mnt/ssd/b70-vllm-lab/results/013b_falsify.json", "w") as f:
    json.dump(RES, f, indent=2)
print("===WROTE=== /mnt/ssd/b70-vllm-lab/results/013b_falsify.json")

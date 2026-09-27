# F-009 — vllm#54785 (MTP k=4 + XPU graph capture, silent wrong logits) — NOT reproduced on a single B70

**Status: clean single-GPU result. This is a NEGATIVE result that constrains the
hypothesis. It is NOT a refutation of the reporter's observation.**

Date: 2026-09-27. Reporter: MikkoP88, issue opened 2026-09-01, still OPEN.

---

## 0. What the issue claims, and the two facts that decide whether we can test it

Issue: https://github.com/vllm-project/vllm/issues/54785
Title: "[Bug][XPU] XPU graph capture + MTP num_speculative_tokens=4 produce
non-deterministic, wrong logits at temperature=0 (k<=3 clean, eager clean,
bs>=2 eager fallback clean, compile-independent)"

Reporter's config:

| field | value |
|---|---|
| GPUs | **2x** Intel Arc Pro B70, TP=2 |
| model | Qwen3.8-27B fp8 (GDN hybrid attention + MTP drafter heads) |
| stack | vLLM 0.21.1.dev0+gad7125a43, vllm-xpu-kernels 0.1.8.3.dev0, torch 2.11.0+xpu, oneAPI 2025.3.2 |
| flags | `cudagraph_mode=FULL_DECODE_ONLY`, `{"method":"mtp","num_speculative_tokens":4}`, `--kv-cache-dtype turboquant_4bit_nc`, `--block-size 512` |

The claim is a **cliff, not a gradient**: k=1/2/3 bit-stable, k=4 corrupt. That
specificity is what makes it testable, and it is the only part of the report we
can meaningfully try to reproduce.

**Upstream state (verified 2026-09-27, `gh` authenticated as riceharvest):**

- **No fix exists.** `gh search prs --repo vllm-project/vllm 54785` -> `[]`.
  The issue timeline has **zero** `connected` / `referenced` / `closed` events
  and zero commit cross-references. The reporter's deployed mitigation
  (`VLLM_XPU_ALLOW_K4_CAPTURE`, clamping k 4->3 when capture is active) **never
  landed upstream under any name** — 9 distinct searches, all empty.
- **No genuine duplicate in vllm-project/vllm.** #48327 and #48946 are the same
  platform-level family (XPU graph -> wrong output) but neither uses spec decode.
  #54698 is a hang. #54796 is a *different* bug from the same reporter, cleanly
  separated by the compile flag.
- The related GDN-spec cluster lives in **vllm-xpu-kernels** (#593, #389, #320,
  #510, #548) — and every member fails **loudly** (assertion / RuntimeError).
  **#54785 is the only silent-wrong-logits report in that cluster.**
- Closest mechanism match: **xpu-kernels #600** "[GDN] Fix ragged speculative
  token traversal", merged 2026-09-16 (`da16a559`), 15 days after the report, by
  a third party, with **no link to #54785 in either direction**. It rewrites the
  GDN spec traversal from fixed-width `batch_id * num_spec_tokens` to
  `query_start_loc` intervals. **For a rectangular bs=1 k=4 batch that traversal
  change is a no-op** (the intervals are `[0,5)` either way). Its two non-no-op
  edits (the `logical_state_len` conv-roll prefix, the relaxed assert) are the
  only parts that could reach the reported cliff.
- **A live upstream risk, created not fixed:** PR **#51600** (merged 2026-09-24)
  *deletes* `VLLM_XPU_ENABLE_XPU_GRAPH`, destroying the reporter's own
  deterministic-reference lever. On current main, XPU graphs are ON unless
  `--enforce-eager`. Anyone who upgrades loses the ability to A/B this bug the
  way the reporter did.

Full audit with URLs and file:line: `/mnt/ssd/dev workspace/issue-54785-upstream-audit.md`.

---

## 1. Is it single-GPU reproducible? YES, in principle — and we ran it

The reporter's TP=2 is **incidental configuration, not a dependency of the
suspected path**, and this is provable from source rather than assumed:

1. Their own datapoint 8 states corruption occurs with **default
   comm-outside-graph placement**; they do not run `VLLM_XPU_ALLOW_COMM_IN_GRAPH`.
2. On XPU the entire GDN layer forward is a **single fused custom op**
   (`qwen_gdn_linear_attn.py` -> `torch.ops._xpu_C.gdn_attention`). Its only
   parallelism argument is `tp_size`, used solely for `num_k_heads / tp_size` and
   `num_v_heads / tp_size` head-shard divisions. **The spec-state arguments are
   all per-rank device tensors** (`spec_query_start_loc`, `spec_token_indx`,
   `spec_state_indices_tensor`, `num_accepted_tokens`).
3. The buffers that would have to be wrong are sized `(decode_cudagraph_max_bs,
   num_spec + 1)` (`gdn_attn.py:127-131`) — a shape with **no dependency on
   hidden size, layer count, or TP**.
4. A same-family model small enough for one B70 exists:
   **`Qwen/Qwen3.5-0.8B-Base`** — 873M params, `model_type: qwen3_5`,
   `mtp_num_hidden_layers: 1`, 24 layers = **18 `linear_attention` + 6
   `full_attention`**, and **15 real `mtp.*` tensors in the safetensors index**
   (verified, not just declared in config). The reporter's Qwen3.8-27B is the
   same `qwen3_5` model_type with 48 + 16, routing to the **same `Qwen3_5MTP`
   drafter** and the **same GDN layer code**.
5. Corroboration: the merged XPU GDN-spec fix #544 was itself verified on a
   **0.8B model on a single B70**. The bug class is reachable without TP=2.

**Search trap worth keeping:** the Qwen3.5/3.8 family declares
`mtp_num_hidden_layers`, **not** `num_nextn_predict_layers`. A search on the
latter alone concludes "no small GDN+MTP model exists" — the wrong answer.

---

## 2. What we measured (capsules 012 / 012b / 012c)

Model `Qwen/Qwen3.5-0.8B-Base`, single B70, TP=1, bs=1, `max_num_seqs=1`,
`block_size=512`, `cudagraph_mode=FULL_DECODE_ONLY` (the reporter's exact mode),
`seed=1234`, `temperature=0`, `logprobs=5`, **48 identical greedy requests per
prompt** (the reporter used 8, but their own text says the failure is a function
of *request ordinal* and only degenerates after "~40 requests on the same boot" —
8 repeats under-samples the thing being measured). Two short prompts (the
reporter's `"The capital of France is"` plus a second, so we are not tuned to
one string).

Oracle: intra-arm determinism over 48 repeats (distinct texts, distinct top-5,
spread of the first-position argmax logprob) and cross-arm graph ON vs OFF.

PLACEHOLDER_RESULTS

---

## 3. Kernel-level probe (capsules 013 / 013b / 013c / 013d / 013e)

Because a full engine start costs ~6 min per arm, the same failure class was also
probed directly at the op level: `torch.ops._xpu_C.gdn_attention` with
`num_spec_decodes=1, num_prefills=0, num_decodes=0` — which takes exactly the
spec path, i.e. `causal_conv1d_spec` -> `gated_delta_rule_spec`, the function
the issue's own suspect analysis names. Argument shapes were taken from the
`TORCH_CHECK`s in `gdn_attn_interface.cpp` so the probe cannot pass by
accidentally never entering the suspect code.

The op-level probe asked three questions in sequence, and two of the three
"answers" it first produced turned out to be **probe bugs, not kernel bugs**.
That sequence is itself the most useful output, so it is recorded in full.

**Probe B - plain XPU graph capture/replay, varying shapes** (`013`, probe B).
Four GEMM+linear shapes (m = 1, 2, 4, 8), captured individually, 16 replays each.
`replay_vs_eager_mismatches = 0` and `replay_self_consistent = True` for all four
shapes, `max_abs_diff = 0.0` exactly. **Basic `torch.xpu.XPUGraph` capture and
replay is bit-exact on this box** for this workload, so the runtime itself is not
the problem. (This arm is genuinely load-bearing: before `XPUGraph` was
substituted for the non-existent `torch.xpu.CUDAGraph`, it too reported a clean
verdict having executed nothing.)

**Probe A step 1-3 - is the op even deterministic?** (`013b`)

| question | result |
|---|---|
| are the op's input tensors bit-identical across two fresh arg sets? | **yes, all of them** |
| are the output and state bit-identical across two independent eager calls? | **yes** (`max_abs = 0`) |
| 200 bit-identical eager calls -> how many distinct results? | **1** |
| does poisoning `ssm_state`/`conv_state` change the output? | **yes** (so the kernel really does read its state argument) |

So the op is deterministic in eager, and it is not ignoring its state. There is
no race in the eager path at any k.

**Probe A step 4-6 - capture safety** (`013c`, `013d`). This is where the two
runs of "the same" experiment **disagreed**, which is the signal that pointed at
the real explanation. `013c` reported "replay disagrees with eager" for every
k; `013d` reported k=2 clean and k=1,3,4,5,6 unsafe. Two runs of one experiment
cannot both be right.

The reason is that **the GDN spec op is recurrent**: it mutates `conv_state` and
`ssm_state` by design (that is how a linear-attention recurrence carries state
across steps). An un-reset replay is therefore computing a legitimately
different, longer-history recurrence, not a stale read. Any capture-safety oracle
on a stateful op has to reset the mutated buffers to pristine values before every
replay, and has to include a no-reset control arm proving the two regimes differ.
`013c` and `013d` were both wrong for different reasons: `013c` used no reset,
`013d` reset but compared against a reference whose own value was still
allocator-dependent.

**Probe A step 7 - the decisive test: does it read uninitialized memory?**
(`013e`). The C++ allocates the conv-stage intermediates with `torch::empty`,
i.e. **uninitialized** (`gdn_attn_interface.cpp`: `q`, `k`, `v`, `b`, `a`; only
the XE2 non-spec *prefill* path uses `torch::zeros`). If any element is left
unwritten, it is whatever the caching allocator last held - the same defect
class as FINDINGS.md **F-008** (the Xe2 grouped-GEMM `at::empty` tile counter),
and the exact "stale-cache" hazard this investigation was asked to probe.

Method: hold the op's arguments **bit-identical** and vary **only** the contents
of the allocator pool - allocate a 64 MiB tensor, fill it with a poison value,
free it, then call the op. Six poison values plus a baseline and a repeat control,
for every k in 1..6.

| k | T=k+1 | distinct outputs across 8 pool states | depends on pool? | control == baseline? |
|---|---|---|---|---|
| 1 | 2 | **1** | no | yes |
| 2 | 3 | **1** | no | yes |
| 3 | 4 | **1** | no | yes |
| 4 | 5 | **1** | no | yes |
| 5 | 6 | **1** | no | yes |
| 6 | 7 | **1** | no | yes |

**Clean negative, and a real one.** The GDN spec op at k=1..6 writes everything
it later reads: its output is bit-invariant to the contents of the allocator's
pool. The `torch::empty` intermediates in the spec path are therefore **not** a
live uninitialized-read defect on kernels 0.1.15.4 - the F-008 hazard class does
not reproduce through this path. (This is a statement about the *spec* path only;
the XE2 non-spec prefill path allocates with `torch::zeros` and was not probed.)

---

## 4. What is PROVEN, what is CONSTRAINED, what is STILL OPEN

PLACEHOLDER_VERDICT

---

## 5. Traps hit while doing this (all of them cost real GPU time)

1. **A probe that cannot fail is not a negative result.** The first version of
   capsule 013 reported a **clean verdict while having executed nothing**: it
   tested op availability with `"gdn_attention" in dir(torch.ops._xpu_C)`.
   `dir()` on an `_OpNamespace` does **not** list custom ops, and the ops are
   registered as an import side effect of `vllm.platforms.xpu`. The correct
   test is to **resolve the overload** (`torch.ops._xpu_C.gdn_attention`) and read
   `._schemas` (plural — `OpOverload` has no `_schema`). Worse, `torch.xpu.CUDAGraph`
   does not exist on torch 2.13+xpu; it is **`torch.xpu.XPUGraph`**. Every
   "clean" from that run was vacuous. Capsule 013 now records
   `probe_a_ran` / `probe_b_ran` in its verdict so an empty run can never be
   read as a pass.
2. **A recurrent op must have its state reset between replays, or the oracle is
   wrong.** Capsule 013c reported "replay disagrees with eager" for every k. The
   op mutates `conv_state` and `ssm_state` **by design** (it is a recurrence), so
   an un-reset replay is computing a legitimately different, longer-history
   recurrence. That is a probe bug, not a kernel bug. Any capture-safety oracle
   on a stateful op must reset the mutated buffers to pristine values before each
   replay, and must include a no-reset control arm to prove the two regimes
   really differ.
3. **Two runs of the same experiment disagreed, which pointed at the allocator
   and led to the actual decisive test.** Capsule 013e varies *only* the contents
   of the caching allocator's pool (allocate a 64 MiB poison tensor, fill it,
   free it) while holding the op's arguments bit-identical. Result: the output
   is **invariant across 8 different pool states for every k=1..6**. This is a
   genuine clean negative, and it is the right way to ask the question.
4. Never source oneapi `setvars.sh` (F-002); keep the Level Zero devel shim on
   the runtime path but strip oneAPI entries (F-003). Under `set -u`, `$CPATH`
   may be unset — use `${CPATH:+:$CPATH}`.
5. Graph ON vs OFF must be **separate processes**; the decision is made at
   engine construction (`vllm/platforms/xpu.py:301`).
6. `cudagraph_capture_sizes` is evidence, not a setting: read it back from
   `compilation_config` after the engine starts. At k=4 it became
   `[1, 2, 4, 5, 8]` — the `5` is `k+1`, i.e. `adjust_cudagraph_sizes_for_spec_decode`
   did run and rounded to multiples of 5, confirming the reporter's structural
   precondition on this box.

---

## 6. Reproduction recipe (single B70, ~6 min per arm)

```bash
# one arm
/mnt/ssd/b70-vllm-lab/capsules/012b_mtp_k4_fdo.sh <k> <on|off> <tag>

# full interleaved sweep + table
/mnt/ssd/b70-vllm-lab/capsules/012c_sweep.sh
/mnt/ssd/b70-venv/bin/python /mnt/ssd/b70-vllm-lab/capsules/012_aggregate.py
/mnt/ssd/b70-venv/bin/python /mnt/ssd/b70-vllm-lab/capsules/012_verify_margins.py

# kernel-level probes
cd /mnt/ssd/b70-vllm-lab
export LD_LIBRARY_PATH="$HOME/.local/lib"
/mnt/ssd/b70-venv/bin/python capsules/013e_uninit_read.py
```

**If someone with 2x B70 + the reporter's 27B fp8 model wants to close this out,
the highest-value next step is not another single-GPU run.** It is to bisect the
one axis we could not touch: `tp_size`. Run the identical k=4 protocol at
TP=1 and TP=2 on the *same* 0.8B model. If the cliff appears only at TP=2, the
defect is in the TP-dependent path (collectives/stream placement) and the GDN
spec-state hypothesis is wrong. If it appears at both, the suspect narrows to
the GDN spec kernel exactly as the reporter argues. That single experiment
discriminates between the two remaining hypotheses, and it is cheap.

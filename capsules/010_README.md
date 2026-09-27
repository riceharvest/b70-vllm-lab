# Capsule 010 — XPU CUDA-graph default: source facts + interleaved measurement

Date: 2026-09-27. Lab: /mnt/ssd/b70-vllm-lab.
Versions at measurement time: vLLM 0.30.0 (XPU wheel), torch 2.13.0+xpu,
vllm_xpu_kernels 0.1.14.1, Python 3.12.12. GPU: Intel Arc Pro B70 (Battlemage).
Desktop session LIVE for every number below (plasmashell pid 6855); 30.92/31.89 GiB
VRAM free at start. F-004 triton patch confirmed applied.

## 1. FACTS (verified in installed source, file:line)

**Env var definition** — `vllm/envs.py:2112-2115`
```python
# Whether enable XPU graph on Intel GPU
"VLLM_XPU_ENABLE_XPU_GRAPH": lambda: bool(
    int(os.getenv("VLLM_XPU_ENABLE_XPU_GRAPH", "0"))
),
```
Default is **"0" → False**. Also declared `VLLM_XPU_ENABLE_XPU_GRAPH: bool = False`
at `envs.py:318`.

**The decision site** — `vllm/platforms/xpu.py:283-311`, `XPUPlatform.check_and_update_config`:
```python
if not supports_xpu_graph():            # :295
    cudagraph_mode = CUDAGraphMode.NONE
elif not envs.VLLM_XPU_ENABLE_XPU_GRAPH:  # :301
    cudagraph_mode = CUDAGraphMode.NONE
    logger.warning_once("XPU Graph is disabled by environment variable, ...")
else:
    logger.warning_once("XPU Graph support is experimental and currently only "
                        "supports single-GPU execution.")
```

**The torch gate** — `vllm/utils/torch_utils.py:1050-1051`
```python
def supports_xpu_graph() -> bool:
    return is_torch_equal_or_newer("2.11.0.dev")
```
torch 2.13.0 passes this, so **the env var is the only gate on this box** —
the default-off is not a torch-version fallback.

Note the "experimental / single-GPU" warning is in the `else` branch, i.e. it is
what a user sees **after** they have already opted in. It is a status note, not
the reason for the default.

## 2. UPSTREAM INTENT (verified via GitHub API)

- **Original feature PR #34482** `[XPU]Support CUDAGraph on XPU Platform`
  (xinyu-intel, merged 2026-02-25). Tracking issue **#26970** "currently pytorch-xpu
  doesn't have a graph API like CUDAGraph."
- **PR #38193** `[XPU] Disable xpu graph by default` (jikunshang, merged 2026-03-26)
  — the explicit decision, and the direct answer to "deliberate or incidental?":
  > "After torch 2.11 upgrade, xpu graph will enable by default. we found there are
  > some limitation, eg, require a specific driver version, it's not stable yet. so we
  > decide disable it by default. user can add env var `VLLM_XPU_ENABLE_XPU_GRAPH=1`."

  So the default-off was **deliberate** (driver-version instability), not an oversight.
- **PR #49419 / #50236** — experimental warning added (2026-07-25), then narrowed
  (2026-08-06) after oneAPI 2026.0 fixed the memory and FlashAttention-piecewise limits.

## 3. DECISIVE: the requested issue is ALREADY FIXED UPSTREAM

**PR #51600 `[XPU] enable XPU GRAPH by default` (zhenwei-intel) is MERGED** —
merged_at `2026-09-24T11:17:26Z`, merge commit `dcfc17e0b1cb4a4bb4eeca9a75c82b82d776f25e`.
> "Removes the `VLLM_XPU_ENABLE_XPU_GRAPH` environment variable. Removes the
> experimental single-GPU restriction warning. Enables XPU Graph by default unless
> `--enforce-eager` is specified." (depends on **#56013** `[XPU] upgrade to PyTorch 2.14`)

Verified against **current main**: `vllm/platforms/xpu.py` on main now contains only
the `supports_xpu_graph()` check — the `envs.VLLM_XPU_ENABLE_XPU_GRAPH` branch and
the experimental warning are **gone**.

Timeline: v0.30.0 was published **2026-09-22**, two days *before* #51600 merged.
This is exactly why the local 0.30.0 still has the gate. **The "feature-parity gap"
observed locally is an artifact of the installed version, not an upstream defect.**

**Consequence: filing an issue asking to enable XPU graphs by default would be a
duplicate of merged PR #51600. No such issue was filed.**

Open graphs-on bugs (the real residual risk, all verified open via API):
- **#48327** gibberish under graph on, 2x B70 — reporter: "the performance is about
  1/3 of the performance you can get with graph enabled... output eventually turns
  gibberish if the current amount of tokens being processed is high enough... only
  around 10K tokens"
- **#54698** EngineCore infinite spin in `torch.xpu.graphs.replay()` under concurrent load, B70
- **#54785** non-deterministic wrong logits with MTP k=4
- **#48946** corrupt output under piecewise capture, TP=2
- **vllm-xpu-kernels #567** `ref_fused_moe` fails during graph capture, TP=2

Gaps worth noting: #51600's **Test Plan and Test Result sections are empty**, and no
user-facing doc mentions graphs (no `docs/platforms/xpu.md`; zero "graph" mentions in
the XPU install/hardware docs at v0.30.0).

## 4. MEASUREMENT

Harness: `capsules/010_xpu_graph_ab.py` + `capsules/010_xpu_graph_ab.sh`.
Graph decision happens at engine construction (xpu.py:301), so ON/OFF are separate
processes; the shell driver interleaves `off on on off on off` (A B B A B A).
Both arms use identical prompts/seed/temperature and `enforce_eager=False`, so the
only difference is `cudagraph_mode`.

Harness bug found and fixed mid-run: `disable_log_stats=True` silently zeroed all
TTFT/ITL, because `output_processor.py:186` only builds `RequestStateStats` when
`log_stats` is on. Caught by asserting `requests_with_engine_metrics > 0` and exiting
non-zero rather than reporting a benchmark with no TTFT in it.

Results: see `results/010_xpu_graph/` and section 5 of the run report.

## 5. MEASURED RESULTS (interleaved A B B A B A, 3 runs per arm per profile)

All 6 runs in each profile succeeded. **No crash, no hang, no NaN/corruption flag in
any configuration**, including the graph-capture-heavy one. Desktop session LIVE
throughout (plasmashell pid 6855). `off` arm resolved to `cudagraph_mode=NONE`,
`on` arm to `FULL_AND_PIECEWISE`, confirmed from the engine in every run.

### baseline (8 prompts, max_tokens 96, max_num_seqs 8, max_num_batched_tokens 2048)

| metric | OFF median [min-max] | ON median [min-max] | ON/OFF |
|---|---|---|---|
| TTFT mean (ms) | 69.4 [60.7-127.0] | 46.2 [46.1-48.0] | **0.67x** |
| ITL mean (ms) | 14.8 [14.6-16.7] | 3.27 [3.26-3.28] | **0.22x** |
| decode tok/s | 494 [460-523] | 2060 [2050-2060] | **4.16x** |
| CPU ms / gen-token | 0.044 [0.041-0.056] | 0.028 [0.027-0.029] | 0.64x |
| CPU/wall ratio | 0.022 | 0.058 | 2.64x |
| wall (s) | 1.55 | 0.374 | 0.24x |
| gen tokens | 768 | 768 | 1.00x |

### heavy (32 prompts, max_tokens 128, max_num_seqs 32, max_num_batched_tokens 16384)

| metric | OFF median [min-max] | ON median [min-max] | ON/OFF |
|---|---|---|---|
| TTFT mean (ms) | 68.7 [68.4-74.0] | 45.7 [42.4-53.9] | **0.67x** |
| ITL mean (ms) | 17.0 [16.7-17.5] | 3.97 [3.95-3.98] | **0.23x** |
| decode tok/s | 1810 [1770-1850] | 7340 [7140-7360] | **4.04x** |
| CPU ms / gen-token | 0.021 [0.019-0.021] | 0.011 [0.011-0.016] | 0.52x |
| CPU/wall ratio | 0.038 | 0.081 | 2.13x |
| wall (s) | 2.26 | 0.558 | 0.25x |
| gen tokens | 4096 | 4096 | 1.00x |

Graph capture is worth **~4.1x decode throughput** on the B70 at this model size,
reproduced across 6 interleaved runs per profile. The CPU/wall ratio *rising* with
graphs is expected: wall time collapses while CPU work per step stays similar, so
CPU time stops being the bottleneck. Token counts are identical (768 / 4096), so the
speedup is not a truncation artifact.

**Caveat, and it matters:** Qwen3-0.6B is a small model where per-step launch
overhead dominates, which is the regime where graphs pay most. Do NOT read "4x" as
a general XPU claim. It happens to match #51600's own wording — "significant
performance gains for small models and MoE models under low concurrency" — so this
corroborates the PR's claim rather than extending it.

### The ON/OFF text difference is a benign tie-break, not corruption

Within each arm output was bit-reproducible (1 distinct sha256 over 3 runs), but the
two arms produced different text (2983 vs 2861 chars). `capsules/011_determinism.py`
isolated it with per-position top-5 logprobs:

- On a real (non-degenerate) prompt, first divergence at token index 18:
  - OFF arm's own top-5 there: `' is'` **-1.54230** and `' involves'` **-1.54230** —
    an **exact 0.000000-nat tie**.
  - ON picked `' involves'`, the other member of that tie.
  - Both continuations are coherent English; no gibberish, no repetition collapse.
- On the repetitive echo prompt, output was **identical token-for-token** across arms.

So the divergence is a coin-flip between numerically tied tokens, caused by graph
capture changing reduction/kernel order. It is **not** the corruption class reported
in #54785 / #48327. In particular I did **not** reproduce #48327's gibberish or
#54698's replay() hang on a single B70 — but those reports are 2x-B70 and
concurrent-load respectively, so this is not a refutation of them.


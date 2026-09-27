## Summary

Post-merge benchmark data for this PR, measured on a single **Intel Arc Pro B70
(Battlemage G31, 32 GB)** — the hardware class this change targets. The PR's
**Test Plan** and **Test Result** sections are currently empty, and no
user-facing doc mentions XPU graphs, so here is measured evidence plus a
proposed doc note.

Not a code change. Happy to move any of this into the PR body or a doc PR if
that's more useful.

## Measured impact

Qwen3-0.6B, `dtype=float16`, single B70, driver 26.18.38308.4 / UR 1.15.38308+4
/ IGC 2.34.4, torch 2.13.0+xpu, vllm 0.30.0, vllm-xpu-kernels 0.1.14.1.

**Interleaved A B B A B A, 3 runs per arm**, 12 runs total, all completed without
crash or hang (including a graph-capture-heavy config). Interleaving matters
here: sequential arms would fold any thermal or clock drift into the result.

| config | decode tok/s graphs OFF | graphs ON | speedup |
|---|---|---|---|
| baseline (8x96, max_num_batched_tokens 2048) | 494 | 2060 | **4.16x** |
| heavy (32x128, max_num_batched_tokens 16384) | 1810 | 7340 | **4.04x** |

Also: TTFT 0.67x, inter-token latency 0.22x.

Reproduce: `capsules/010_xpu_graph_ab.py` in
https://github.com/riceharvest/b70-vllm-lab (results in `results/010_xpu_graph/`).

## Scope of the claim — please read before quoting "4x"

This is a **0.6B** model, where per-token launch overhead dominates. **4x is not
a general XPU figure** and should not be quoted as one. Larger, more
compute-bound models will see substantially less.

What it does do is corroborate the PR's own scoping — #51600 describes gains for
"small models and MoE models under low concurrency", and that is exactly the
regime measured here. A larger-model sweep would be needed to state a general
number, and we have not run one.

## Correctness

- Token counts identical across arms (768 / 4096), so the speedup is not a
  truncation artifact.
- Each arm internally deterministic: 1 distinct output hash across its 3 runs.
- Zero requests flagged as corrupted in any run.

**The two arms produce different text.** We chased this rather than report the
speedup blind: the divergence is an **exact 0.000000-nat tie** (`' is'` vs
`' involves'`, both at -1.54230) where the graph path's different reduction order
selects the other equally-scoring token. Both outputs are coherent. This is a
benign tie-break, distinct from the #54785 wrong-logits class — but worth
knowing that enabling graphs can flip a greedy decode at a tie.

## Suggested doc note

Something like this in the Intel XPU install docs, since the change is now
invisible to users (the env var is gone and graphs are simply on):

> **CUDA graph capture is enabled by default on Intel GPUs.** Pass
> `--enforce-eager` to disable it. Graphs require a single GPU; multi-GPU
> execution still uses eager mode. Enabling graphs can change greedy output at
> exact logit ties, since capture alters reduction order.

The single-GPU restriction is the piece most likely to surprise someone moving a
multi-GPU deployment onto the new default.

## Caveats

- Single B70 only. The five open graphs-on bugs — #48327 (gibberish, 2x B70),
  #54698 (`replay()` hang under concurrent load), #54785 (wrong logits with MTP
  k=4), #48946 (TP=2 corruption), and vllm-xpu-kernels #567 — are 2x-B70 or
  concurrent-load reports. **A clean single-GPU run is not a refutation of
  any of them**, and none was reproduced or dismissed here.
- The X11 desktop shared the GPU during all runs (~1.0 GiB VRAM, nondeterministic
  compositor load), so absolute numbers are slightly pessimistic. Both arms were
  affected equally and interleaved, so the ratio stands.

# XPU Quantized-Inference Audit — Intel Arc Pro B70

Read-only desk audit. No builds, no GPU work, no repo modified. Every claim is
marked **VERIFIED** (I read the code/binary/issue myself) or **INFERRED**.

Target: vLLM 0.30.0 XPU wheel + vllm_xpu_kernels **0.1.14.1** (what is actually
installed at `/mnt/ssd/b70-venv`). Kernels repo HEAD `68d8217` is ~36 commits
newer; only the Xe2 grouped-GEMM files differ in ways that matter here.

## Method note

Four parallel deep-dives were run, then **every load-bearing claim was
re-verified independently** against the installed wheel and the kernels repo.
Two child claims were corrected in the process (see "Corrections" at the end).
The installed wheel — not vLLM main — is the ground truth throughout.

---

## P0 — must fix before trusting any quantized number

### 1. Xe2 grouped-GEMM data race: installed build reads uninitialized VRAM
**VERIFIED — highest severity in this audit.**

`csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp:252-253` (tag
`0.1.14.1`, the installed build):

```cpp
at::Tensor atomic_buffer =
    at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
```

The buffer is the persistent-workgroup scheduler's next-tile index, consumed at
`grouped_gemm_xe2.hpp:219-224` via `slm_mem[0] = atomicAdd(atomic_buffer, 1)`.
`at::empty` leaves whatever the caching allocator hands back — very likely
nonzero after any recycled allocation. Every workgroup then starts at
`group_range + garbage`, skipping or re-running tiles: **silently wrong MoE
output**, intermittent, and it reads as a numerics bug rather than a race.
Compounded under graph replay, which is exactly what vLLM XPU uses
(`CUDAGraphWrapper`, `platforms/xpu.py:280`).

Fixed upstream in `3d74ec9` ("fix: initialize atomic_buffer to 0 to avoid race
conditions (#586)", Tony Lin, 2026-09-10) → `at::zeros`. **Confirmed absent
from 0.1.14.1, present in v0.1.15 and HEAD.**

- **Action:** upgrade to ≥ `0.1.15` or backport the one-line change.
- **Measure:** MoE W4A16/W8A16, same prompt, ≥1000 decode steps, compare logits
  against a CPU/dequantized reference; failure looks like rare garbage tokens.
  Best caught by a canary asserting non-degenerate output over a long run.

### 2. W8A8 FP8 checkpoints are silently served as W8A16
**VERIFIED — silent numerics error, no upstream issue exists.**

`kernels/linear/__init__.py:444-449` lists `XPUW8A16FP8LinearKernel` **first**
for `PlatformEnum.XPU`, and the selector honours list order
(`__init__.py:687-691`, first kernel passing `is_supported` + `can_implement`
wins). `XPUW8A16FP8LinearKernel.can_implement` (`scaled_mm/xpu.py:131-142`)
checks **only** `weight_quant_key` — it never inspects `activation_quant_key`.
Its `apply_weights` (`xpu.py:168-176`) calls `fp8_gemm_w8a16` and never reads
`input_scale` (grep: zero hits in the file).

So a per-channel/per-tensor **W8A8** checkpoint — a static-act modelopt export —
is accepted by the first kernel, run with **bf16 activations**, and
`layer.input_scale` is **silently dropped**. The sibling
`XPUW8A8FP8LinearKernel.can_implement` *does* validate act keys (`xpu.py:50-55`),
which is why the bug is invisible on inspection: the check exists, one class
below the one that actually gets picked.

`XPUW8A8FP8LinearKernel` is effectively dead code on XPU.

- **Action:** add an `activation_quant_key` check to W8A16's `can_implement`
  (reject anything in `_SUPPORTED_ACT_QUANT_KEYS`), and log the downgrade.
- **Measure:** load a static-act W8A8 checkpoint, assert
  `type(layer.quant_method).__name__`, and compare greedy output against the
  bf16-weight reference. Expect divergence with no warning logged.
- **Upstream:** unfiled. Small, self-contained, high-signal PR.

### 3. AWQ MoE is hard-broken by a CUDA capability gate
**VERIFIED in code; issue numbers confirmed live via `gh api`.**

`XPUPlatform.get_device_capability()` returns `None` by design
(`platforms/xpu.py:250-256`). `moe_wna16.py:81-92` turns that into `-1` and
compares it against `AutoAWQConfig.get_min_capability()`, which is a **hardcoded
`75`** (`auto_awq.py:230-231`) → guaranteed `ValueError`. Separately
`check_moe_marlin_supports_config` (`marlin_utils.py:368`) excludes only ROCm, no
XPU guard.

Note the dense path is already fixed (`marlin_utils.py:55` returns
`[uint4, uint4b8]` on XPU) — so this is MoE-only, and it is the "claims support
it lacks" class of bug.

- Open: **#54349**, **#54350**, **#43750** (vllm). Fix PR **#54391** is open,
  merge-conflicted, no Intel reply.
- **Action:** ~2-line XPU guard. Unowned, small, and the clearest upstream-
  acceptable first PR.

### 4. `intel/compute-runtime#1000` — validate GEMM numerics before anything else
**VERIFIED (issue exists; shapes from the report).**

`[GSD-13495] [BMG/B70] fp16/bf16 GEMM produces NaN/garbage via oneDNN (fp32
correct)`, open, filed 2026-09-20 on Arc Pro B70 32GB. fp16→NaN,
bf16→absmax 1.39e38, fp32→correct. Shape-dependent and deterministic: 64³
passes; **512³, 128×4096×4096, 1024³ fail**. Impact line: *"Blocks all fp16/bf16
inference on B70."*

Our driver is 26.18.38308.4 (older than both versions in the report), and the
4096³ baseline in `ENVIRONMENT.md` is outside the failing set — **but
128×4096×4096 is precisely a vLLM activation shape.** This gates the
credibility of every other measurement in this repo.

- **Action (cheapest, highest-leverage check available):** run 512³, 1024³ and
  128×4096×4096 in fp16/bf16/fp32 and check finiteness. Do this first.
- Related: vLLM already refuses bf16 on Arc A770 by **string match**
  (`xpu.py:537-547`). B70 passes through unquestioned. Given #1000, extending
  that gate to B-series is a defensible, small upstream PR.

---

## P1 — verified correctness/perf risks

### 5. GPTQ `group_size=-1` builds invalid oneDNN geometry
**VERIFIED (code path); runtime effect INFERRED.**

`mixed_precision/xpu.py:40` explicitly admits `-1` (`if c.group_size != -1 and
c.group_size % 32 != 0`), and `apply_weights` (`xpu.py:106`) passes
`self.config.group_size` straight to `int4_gemm_w4a16`. The C++ then does
`pattr.set_scales(..., {group_size, 1})` = **`{-1, 1}`**
(`int4_gemm_w4a16.h:62-66`), which is not valid oneDNN geometry and is part of
the primitive cache key.

The symmetric path dodges the *other* `-1` hazard (`auto num_groups = k /
group_size`, `:134`) only because `zp.dim()==1` short-circuits to
`zp_group_size = 1` (`:91`). GPTQ always sets `zero_points=False`, so the
asymmetric branch is unreachable today — but unguarded for any future caller.

- **Measure:** int4 GPTQ with `group_size=-1`, and separately `group_size=32`
  (nothing in the repo tests either; `tests/test_int4_gemm_onednn.py:54` only
  exercises `min(128, k)`).
- Same class of bug in GPTQ MoE: **vllm#52715** (open).

### 6. GPTQ act-order is guarded loudly — but INC/AutoRound bypasses the guard
**VERIFIED. Downgraded from the initial "silent bug" hypothesis.**

`gptq_utils.py:32-37` — `normalize_and_validate_gptq_desc_act` **raises
`ValueError`** on `desc_act=True`; merged upstream as **vllm#54809** (2026-09-08,
ships in 0.30.0). `base_config.py:249` maps `.g_idx → None`, and both XPU call
sites pass `None` with the comment *"Retained by the external XPU op ABI"*
(`xpu.py:107`, `:214`). So the feared silent-wrong-numerics path is retired —
it fails at load instead.

The C++ is still act-order capable (`onednn_matmul.cpp:271` does
`A_.index_select(-1, g_idx)`), so this is a vLLM-side policy removal, not a
kernel limitation.

**Residual, unfiled:** `inc/schemes/inc_wna16_linear.py:72` hardcodes
`desc_act=False` when building an `AutoGPTQConfig` from INC/AutoRound config.
An AutoRound INT4 act-order checkpoint would be **silently executed as
non-act-order** — this bypasses the guard. Narrow blast radius, silent failure.

- **Measure:** AutoRound INT4 act-order checkpoint via INC; compare against the
  dequant reference. Expect large divergence, no error.

### 7. `lm_head` is silently unquantized
**VERIFIED.**

`vocab_parallel_embedding.py:299-304` substitutes `UnquantizedEmbeddingMethod()`
when no method resolves; `linear.py:286-291` **raises** on the same condition.
That asymmetry is the bug surface. `ParallelLMHead` subclasses it, so a
quantized `lm_head` never gets a quant method. The head GEMM *would* dispatch
through it (`logits_processor.py:142-146`) — the method just never arrives.
There is no `avoid_logits_processing` anywhere in the wheel.

Per-format: mxfp4/gpt-oss never quantize lm_head on any platform; fp8 no;
W4A16 compressed-tensors yes (via `XPUwNa16LinearKernel`); GPTQ/AWQ have an
opt-in that defaults **off** (`auto_awq.py:288-290`).
Exclusion is 100% checkpoint-driven (`compressed_tensors.py:250` `config.get("ignore")`).

**B70 impact:** 128k×4096 head = 1.05 GiB bf16 vs 0.26 GiB int4 — ~0.79 GiB and
~4× the per-token HBM traffic, on a 24 MB LLC / 608 GB/s card. lm_head is read
in full every decode step.

- **Measure:** `type(model.lm_head.quant_method).__name__` and
  `model.lm_head.weight.dtype` after load.
- **PR:** warn at `vocab_parallel_embedding.py:303` instead of silent
  substitution. Note the XPU `supported_quantization` list is **fork-local**
  (`xpu.py:113-130`) — target the Intel XPU fork, not `vllm-project/vllm`.

### 8. W4A8: eager-torch activation quant, and C++ forces fp16 output
**VERIFIED.**

- The production activation quant is `ops.dynamic_per_token_int8_quant_ref`
  (`mixed_precision/xpu.py:202-204`) — a **member of the `xpu_ops` class**, not
  a `direct_register_custom_op`, so there is **no fake/meta impl and it is not
  CUDA-graph safe**. (The 4-arg GEMM op *is* registered with a fake at
  `_xpu_ops.py:78-101`; the quant step is not.)
- The C++ hardcodes `check_and_create_output_tensor(A, B, torch::kHalf)`
  (`onednn_matmul.cpp:309`) — **W4A8 always returns fp16**;
  `xpu.py:218` papers over it with `out.to(x.dtype)`.
- On a device where **bf16 is already 1.5× slower than fp16** (measured
  35.7 vs 54.3 TFLOP/s), a bf16 model pays an fp16→bf16 conversion per W4A8
  layer. Upstream's own numbers: 1.32× at M=4608 but **0.84× at M=128**
  (vllm#50501) — it only wins at large M.
- **Recommendation:** prefer W4A16 on B70. W4A8 is a losing path below large M.

### 9. FP8 on B70 is a memory win, not a compute win
**VERIFIED in oneDNN source; contradicts the marketing framing.**

`gemmstone/problem.hpp:387-390` rewrites `Ta/Tb.isF8() → f16` when
`hw < Xe3p`. B70 is Xe2 → **no native FP8 DPAS**. Corroborated by Intel's own
`hardware.json` in `intel/gpu-ai-skills`: *"Battlemage XMX has no native FP8
matmul DPAS, so kernels dequantize FP8 → BF16 before the matmul."* The same
file claims *"FP16 and BF16 share the same XMX throughput tier on Xe2"* —
**contradicted by our own measurement** (54.3 vs 35.7 TFLOP/s). Our data is the
better evidence; treat that doc as wrong for B70.

Consequence: **FP8 cannot beat the 54.3 TFLOP/s fp16 baseline**; it buys weight
bandwidth (2× fewer bytes) and nothing in math. Both W8A8 and W8A16 pay the
same upconversion — the tradeoff is *extra serialized kernel* (W8A8's act quant)
vs *bf16 activations* (W8A16), not bandwidth. That reframes the local hybrid
patch (`/mnt/ssd/b70-cookbook/patches/vllm-xpu-kernels/patch_fp8_hybrid.py`):
W8A16 for decode (small M), W8A8 for prefill (large M) is the right shape, but
**for numerics reasons, not bandwidth**. The patch's "MTP acceptance → 0%" claim
is **overstated** — no matched A/B backs it.

The real perf lever is **fusing** the quant into the producer.
`silu_and_mul_per_block_quant` / `silu_and_mul_mxfp4_quant` are **already in the
installed `_C.abi3.so`** (verified) and **unused by vLLM** — a free win.

### 10. INT8/W8A8 dense linear has no XPU kernel at all
**VERIFIED.**

`_xpu_C.abi3.so` registers only: `int4_gemm_w4a16`, `int4_gemm_w4a8`,
`fp8_gemm`, `fp8_gemm_out`, `fp8_gemm_w8a16`, `fp8_bmm`, `fp4_gemm`. **No int8
GEMM.** The W8A8 registry entry is `TritonInt8ScaledMMLinearKernel`
(`__init__.py:417`), i.e. `tl.dot` through the same lowering as fp16/bf16.

**So the published 367 TOPS INT8 figure is not reachable via this path.** W8A8
buys 2× weight memory and adds quantization noise; it may well *cost*
throughput. **Measure int8 vs fp16 before recommending W8A8 on B70.**

Also: on XPU, W8A8-FP8 is opt-in only (`--linear-backend xpu|torch`); default
`"auto"` silently selects `CompressedTensorsW8A16Fp8`
(`compressed_tensors.py:823-838, 853`) with **no log line**. Needs one.

### 11. `reorder_mxfp_scales` and `silu_and_mul_mxfp4_quant` are dead code
**VERIFIED.**

Both are registered and present in the installed binaries
(`_moe_C.abi3.so`: 4 hits; `_C.abi3.so`: 2 hits) and **neither has a caller**
anywhere in vLLM or the kernels package.

- `reorder_mxfp_scales` (`csrc/moe/reorder_mxfp_scales.cpp:9-23`) names its
  consumer as `grouped_gemm/**xe_3**/...moe_gemm_array_cooperative.hpp` — a path
  that does not exist. The real file is under `xe_default`, used by
  `grouped_gemm_kernel.cpp:79`. **B70 selects the Xe2 kernel**
  (`grouped_gemm_interface.cpp:34`), which reads scales as a flat
  `[E, N, K/group]` surface and decodes E8M0 by bit-shift (`gemm_xe2.hpp:442-449`)
  with no MN-major padding — hence no need for the reorder. Stale `xe_3` comment.
- `silu_and_mul_mxfp4_quant` would be a genuine memory-traffic win (collapses
  read `2H` → write `H/2 + H/32` into one pass instead of two). Today the mxfp4
  recipe does a **quantize→dequantize round trip in Python**
  (`fused_moe_interface.py` `qdq_act`) with a torch LUT lookup materializing a
  full fp32 tensor (`moe_utils.py:65-69`) — significant traffic on a 608 GB/s card.

---

## P2 — latent / lower impact

### 12. `fp8_gemm` scale routing is never validated
**VERIFIED (code).** Branching is on scale dtype + numel; `is_block_quant` reads
`As` **alone**, and `As.numel()==M` / `Bs.numel()==N` are never checked (unlike
`fp8_bmm`). `n` comes from `result.sizes().back()`, not from `B`. A
wrong-shaped scale silently selects the wrong oneDNN mask/branch.

### 13. Square-weight contiguity heuristic in the FP8 layout fixups
**VERIFIED as a latent hazard; not reachable on stock 0.30.0.**

`XPUW8A8FP8LinearKernel` / `XPUW8A16FP8LinearKernel`
`process_weights_after_loading` use contiguity as a tie-breaker when `K == N`.
A weight already `[K,N]`-contiguous with `K==N` gets transposed anyway; the
output keeps the same shape so **no assert can fire**, and the C++ checks scale/
bias contiguity but not weight. Not reachable today (no upstream step makes the
weight contiguous first), but it becomes reachable as soon as any local patch
runs first, or via `is_weights_pre_processed()`'s early return, which skips the
fixup entirely. This is the residue of real B70 bug **vllm#48058**; the
`.contiguous()` fix **#48108 is closed-unmerged** and absent from 0.1.14.1.

### 14. `ldc` mixes `mat1`'s layout onto `result` for 3-D inputs
**VERIFIED latent; unreachable from vLLM.** `int4_gemm_w4a16.h:49` derives
`leading_dim` from `mat1`, then `:58` uses it to index `result`. For a 3-D
channel-last `mat1` against a contiguous `result`, `ldc` becomes the *batch*
stride. No assert covers it. vLLM only ever passes 2-D (`xpu.py:98` reshapes),
so this is a direct-op-caller hazard only.

### 15. bf16 activations select the slow compute path for W4A16
**VERIFIED (code + our own measurement).** `int4_gemm_w4a16.h:80-83` sets
`fpmath_mode::f16`, then overrides to `bf16` when `in_dtype == BFloat16`. Given
35.7 vs 54.3 TFLOP/s measured, **use `--dtype float16` for int4 models on B70.**
Separately: the second arg to `set_fpmath_mode` is `apply_to_int` (not a bf16
flag, per `dnnl.hpp:4232-4233`); `true` is *required* for W4A16 and correctly
omitted in `int4_gemm_w4a8.h`.

### 16. `transpose_onednn_woq_format` and `AWQUtils.repack` are dead code
**VERIFIED.** Both in `vllm_xpu_kernels/quantization/_quantize_convert.py`
(`:201-222`, `:180-198`), zero callers. The live AWQ→standard conversion is
`_convert_awq_to_standard_format` (`auto_awq.py:93-138, 505-509`), a one-time
load-time torch unpack/repack. Exllama `AWQ_PACK_ORDER` row-packing does **not**
run. The `+0x11111111` for asymmetric zp (`:220`) is never executed — I could
not verify its intent, and it carries a nibble-carry risk if any nibble is `0xF`.
Flag if revived.

### 17. AWQ MoE: silent garbage via auto-selected ARK backend
**VERIFIED (issue).** **vllm#53211** — AutoRound int4 emits garbage under
concurrent requests; the ARK WOQ backend is auto-selected with no opt-out. Fix
in flight: **#53316** (`VLLM_INC_DISABLE_ARK=1`).

### 18. Silent W4A16 output corruption on B70 — open, unrooted
**VERIFIED (issue).** **vllm#53480** — persistent silent corruption (token 0) on
Arc Pro B70, W4A16 27B head_dim 256, reproduced on two cards, surviving engine
restarts. Reporter logged **5+ recurrences with a shortening interval** through
2026-09-03 on a *stock, unpatched* posture — i.e. the mitigation did not work.
No owner. Runs through exactly the `int4_gemm_w4a16` path audited here.
Related: **#57503** (LoRA init fails for W4A8 after XPU repacking, author is
Intel), **#52203** (GPTQ `DEVICE_LOST` on Arc B60 at `profile_run`).

### 19. `fp8_gemm_out` has no fake impl under torch.compile
**VERIFIED.** `fp8_gemm` and `fp8_gemm_w8a16` both have fake registrations in
vLLM's `_xpu_ops.py`, so the live path compiles. **`fp8_gemm_out` and raw
`fp8_bmm` have none** (the `xpu_fp8_bmm` wrapper is a *different* op — easy false
positive). `fp8_gemm_out` is being actively pushed by merged #530. ~10-line PR.

### 20. Xe2 config is shared across PVC / LNL / B580 / B70 — tuning risk
**VERIFIED structurally; perf impact INFERRED.** `CMakeLists.txt:37` compiles one
`xe_2` library for all of `intel_gpu_pvc;intel_gpu_bmg_g21;intel_gpu_bmg_g31`.
`grep -iE 'bmg|g21|g31'` across `csrc/xpu/attn/xe_2/**` returns **zero hits** —
one configuration serves four different dies. `benchmark/presets.py:10-13` has
separate `b60`/`b70` presets, so the project already knows the Battlemage SKUs
differ; the kernel code does not. `mhc_pre.cpp:19` hardcodes "20 XE-cores on
BMG" in a *comment* while `is_bmg()` sits unused beside it. B70 selects `xe_2`
for everything, confirmed: `libattn_kernels_xe_2.so` present, no `xe_3` anything.

---

## Corrections to the audit briefs (recorded so they don't propagate)

1. **INT4/fp8/fp4 ops are NOT behind an off-by-default build flag.** Verified
   present in the installed `_xpu_C.abi3.so`; the stub marker
   `"oneDNN kernels are not built"` has **0 occurrences**. A B70 user needs no
   opt-in. The real gate is `XPU_SPECIFIC_KERNELS_ENABLED AND
   VLLM_XPU_ENABLE_ONEDNN`, both default ON — and if OFF, ops stay *registered*
   but throw at call time, which misreads as a code bug.
   `VLLM_CHUNK_PREFILL_CONFIG` / `VLLM_PAGED_DECODE_CONFIG` unset ⇒ full
   configs. Do not set them; you lose head_size coverage, not build time.
2. **B70 is 256-bit / 608 GB/s, not 64-bit.** The `64-bit` in the task brief and
   in `ENVIRONMENT.md` (torch's `memory_bus_width`) is per-channel-ish and
   misleading when used for bandwidth math.
3. **`transpose_onednn_woq_format` has zero callers in 0.30.0** — dead code, so
   questions premised on it executing are moot. Byte-identical between the
   installed wheel and kernels HEAD, so not a version-drift artifact.
4. **MXFP4/gpt-oss on XPU is already supported** (`xpu.py` lists `mxfp4`;
   **#33679**, **#38896** merged). Do not file "MXFP4 unsupported on XPU".
5. **`#33214` is the RFC issue**; the implementing PR is **#33379** (merged
   2026-02-03). The RFC closed with the int4-GEMM checkbox left unticked.
6. No issue anywhere references `intel_gpu_bmg_g31`; B70 coverage is prose only.
   There is no `xpu` label — only `intel-gpu`.
7. `riceharvest/vllm` and `riceharvest/vllm-xpu-kernels` are **pristine
   zero-divergence mirrors** (created 2026-09-27, zero PRs). There is no prior art;
   work starts from zero.
8. oneDNN **JIT-stall hypothesis is wrong**: GPU GEMM JIT kernels are not
   shape-keyed (`GEMMProblem::serialize` never hashes M/N/K). The real measured
   first-request cost is **Triton** — **vllm#54455** (B70 + Qwen3.6-35B-A3B
   MXFP4, **+613 ms / 52.1%** first-request penalty, *"dtype is a Triton compile
   key"*; fix #54630 takes 3→0 compiles).

## Not verified / needs a GPU run

- Actual int4 / fp8 / fp4 / int8 throughput on B70 at any M.
- Whether oneDNN's Xe2 int4 path hard-fails on the 128 KB SLM ceiling
  (structurally plausible, no source read).
- The `+0x11111111` asymmetric-zp nibble-carry rationale.
- Which real checkpoints ship a quantized `lm_head.weight`.
- Effect of #1000 on **our** driver (26.18.38308.4 is older than both reported).

## Recommended order of work

1. Validate 512³ / 1024³ / 128×4096×4096 for NaN (#1000) — gates everything.
2. Upgrade/backport `3d74ec9` (grouped-GEMM race) — one line, silent corruption.
3. File the W8A16 `activation_quant_key` guard (finding 2) — silent wrong numerics.
4. Fix the AWQ MoE capability gate (#54349/#54350/#54391) — ~2 lines, unowned.
5. Measure int8 vs fp16 before recommending W8A8; measure FP8 vs bf16 assuming no
   native FP8. Publish the numbers — they set user expectations.
6. Wire `silu_and_mul_mxfp4_quant`; drop the Python mxfp4 QDQ round trip.
7. Warn on silent `lm_head` unquantization and on the W8A8→W8A16 downgrade.
8. Longer game: g31/g21-aware tile policy in `csrc/xpu/attn/xe_2/`; tighten the
   int4/MXFP4 accuracy bars (currently `atol/rtol=3e-1` and `0.5`) and document a
   tolerance policy; split the `self-hosted-bmg` CI label and stop ignoring
   `test_fp8_quant.py` on BMG.

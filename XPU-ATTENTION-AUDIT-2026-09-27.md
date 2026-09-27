# vLLM XPU attention path — P0/P1 audit

Date: 2026-09-27
vLLM main: `924707f1bf94ff583d89bff7522ee12ff032c286` (2026-09-27)
vllm-xpu-kernels main: `68d82174393df4f47b895cbd2fbeceb11ef53f59` (2026-09-24)
vLLM pins `vllm_xpu_kernels==0.1.15.4` (= latest on PyPI, released 2026-09-22 — **pin is current, no lag**).

All findings below are **source-derived**. No B70 hardware run was performed. Each is
labelled with what was read directly versus what remains unverified.

Checkouts used for this audit:
- `/mnt/ssd/vllm-audit/vllm-main` (sparse, vllm/ tests/ requirements/ docs/ .buildkite/)
- `/mnt/ssd/vllm-audit/xpu-kernels` (full)

---

## P0 — correctness: silently wrong numbers

### P0-1. `logits_soft_cap` is accepted and then silently discarded on the XPU FlashAttention path

**Confidence: HIGH on the code path. UNVERIFIED on numeric magnitude.**

`vllm/_xpu_ops.py:980-1056` is the XPU shim that stands in for
`flash_attn_varlen_func` (`fa_utils.is_flash_attn_varlen_func_available()` returns
`True` unconditionally for XPU, `fa_utils.py:366`). It accepts `softcap` and
`alibi_slopes` as parameters (`:991`, `:993`) and then does not forward them:

```python
# vllm/_xpu_ops.py:1048-1052
            s_aux=s_aux,
            window_size=real_window_size,
            # alibi_slopes = alibi_slopes,
            # softcap=softcap,
            return_softmax_lse=return_softmax_lse,
```

The kernel side does not implement either. In
`vllm-xpu-kernels/csrc/flash_attn/flash_api.cpp`, `alibi_slopes_` (`:110`) and
`softcap` (`:123`) appear **only** in the function signature and the schema string
(`:454`, `:460`). Grepping the whole of `csrc/xpu/attn/` for `alibi` and `softcap`
returns zero hits. `window_size` *is* plumbed through
(`attn_interface.cpp:29-30,56-57,88-89`) and `s_aux`/sinks *is* plumbed through
(`attn_interface.cpp:28,35`; `flash_api.cpp:118,200,245`), so those two are clean.

Nothing rejects the combination. `get_flash_attn_version()` short-circuits to `2` on
XPU at `fa_utils.py:83-84`, **before** any of the `requires_softcap` /
`requires_alibi` fallback logic at `fa_utils.py:127-137` runs, and
`XpuPlatform.get_attn_backend_cls` (`platforms/xpu.py:142-219`) has no softcap or
ALiBi branch — a softcap model with dtype != float32 falls straight through to
`return AttentionBackendEnum.FLASH_ATTN.get_path()` at `:219`.

Upstream caller passes the real value: `flash_attn.py:1581,1670,1701,1777` all pass
`softcap=self.logits_soft_cap`, which is read from the HF config at
`flash_attn.py:1115-1137` and originates in `gemma2.py:202`,
`gemma4.py:592`, `gemma4_dspark.py:63`, `gemma4_mtp.py:281`.

**Effect:** Gemma-2 / Gemma-4 family on XPU FA runs with **no attention logit soft
cap at all**. Unbounded logits, no warning, no error, HTTP 200. ALiBi models
(`bloom.py`, `falcon.py`, `step1.py`, `whisper_causal.py` — the only four that pass
`alibi_slopes=`) silently get uniform attention.

**B70 relevance:** direct. The lab's target class is a 27B W4A16 at head_dim 256,
and the head_dim-256 + sliding-window class of model is exactly the Gemma-4 /
Gemma-2 neighbourhood. This is the single most likely source of "model loads, runs,
generates plausible-but-wrong tokens" on the B70.

**Duplicate check:** `gh search issues "xpu softcap" / "xpu logits soft cap" / "xpu
alibi"` on both repos returned nothing relevant. Nearest existing item is
vllm-project/vllm#57614 (ROCm gfx1151, same class of bug — "paged attention silently
accepts unsupported ALiBi"), still open, so the class is not yet fixed upstream.

**Suggested fix (either direction is acceptable upstream):**
- fail closed — in `platforms/xpu.py::get_attn_backend_cls`, route to
  `TRITON_ATTN` when `getattr(hf_text_config, "attn_logit_softcapping", None)` or
  `alibi_slopes` is set, matching what the platform already does for `dtype ==
  float32` (`:203-208`) and `use_mm_prefix` (`:184-202`); or
- implement it in `vllm-xpu-kernels/csrc/xpu/attn/` and uncomment `_xpu_ops.py:1050-1051`.

Note the Triton path *does* implement softcap correctly
(`triton_attn.py:678` → `triton_unified_attention.py`, normalized + gated), so the
fallback is cheap and already tested.

---

## P0 — correctness: structurally dead validation

### P0-2. `XPUPlatform.get_attn_backend_cls` never calls `backend_class.validate_configuration`, so every capability gate is dead on XPU

**Confidence: HIGH (direct grep absence + positive read of both peers).**

```
$ grep -rn 'validate_configuration' vllm/ --include=*.py
vllm/platforms/cuda.py:399
vllm/platforms/cuda.py:451
vllm/platforms/rocm.py:546
vllm/v1/attention/backends/mla/prefill/selector.py:154,200
(+ a handful of model-local overrides)
```

`vllm/platforms/xpu.py` does not appear. The selector
(`vllm/v1/attention/selector.py:215`) only calls `get_attn_backend_cls` and resolves
the class — it never validates either.

`AttentionBackend.validate_configuration` (`vllm/v1/attention/backend.py:274-350`) is
where these are enforced: `supports_head_size`, `supports_dtype`,
`supports_kv_cache_dtype`, `supports_block_size`, `supports_mm_prefix`, `has_sink`,
`supports_attn_type`, `supports_sliding_window`, `supports_rswa`,
`supports_non_causal`, `supports_batch_invariance`, `use_pcp`, `use_dcp`,
`supports_kv_connector`, `use_adaptive_verification` +
`supports_device_cpu_query_lens_mismatch`.

**Effect:** on XPU, a request that CUDA/ROCm would reject at config time instead
reaches the kernel. This is the structural root cause that converts P0-1, P1-1 and
P1-3 from loud failures into silent ones. It is also why the capability-flag
contradictions below have no second line of defence.

**B70 relevance:** high — B70 is enumerated by no capability gate, so it takes
whatever the model config asks for.

---

## P0 — correctness: capability flags that lie

### P0-3. `FlashAttentionBackend` advertises batch invariance and non-causal support on XPU, contradicting `platforms/xpu.py`'s own warnings

**Confidence: HIGH on the contradiction. MEDIUM on live impact (the warnings partly
mitigate it).**

`vllm/v1/attention/backends/flash_attn.py`:
- `:389-390` `supports_batch_invariance() -> True` (unconditional)
- `:394` `supports_non_causal() -> True`
- `:397-405` `supports_attn_type()` accepts ENCODER / ENCODER_ONLY / ENCODER_DECODER
- `:441-442` `supports_mm_prefix() -> is_fa_version_supported(4)`
- `:445-448` `supports_sink()` → `flash_attn_supports_sinks()`, which returns `True`
  unconditionally on XPU (`fa_utils.py:319-322`)

`vllm/platforms/xpu.py` says the opposite in the same repo:
- `:165-178` — *"Flash Attention on XPU has not been validated for batch invariance
  on XPU and may produce non-deterministic results across batch sizes."*
- `:184-197` — *"Flash Attention on XPU has no FA4 kernel, so it cannot apply the
  multimodal prefix-LM bidirectional mask … image/video inputs will produce
  incorrect results."*

Both branches **honour an explicit request and return anyway**, behind only a
`logger.warning_once`. So `VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN`
produces a server that logs a warning and then serves non-deterministic results.

`supports_non_causal` is the sharper one and has **no** platform-level mitigation at
all. It is consumed at `backend.py:334-335`. Because of P0-2 that consumption never
happens on XPU — and the kernel side supports it only partially: the compiled
chunk-prefill table (`chunk_prefill_default.conf`) has `causal=false` entries for
head 64/96/128/192/256 but the paged-decode path is
`is_causal: always false for decode` by construction
(`flash_api.cpp:336,428`), so bidirectional *encoder* attention over a paged KV cache
has no compiled variant at all.

**B70 relevance:** direct — FA is the default backend for every non-MLA model on XPU
(`platforms/xpu.py:219`).

---

## P1 — feature gaps

### P1-1. `XPUPlatform` omits the `disable_chunked_mm_input` force that CUDA and ROCm both apply for prefix-LM models

**Confidence: MEDIUM. The guard's absence and the consumer's existence are both
verified; whether the Triton kernel corrupts or merely drops the mask for a split
prefix was NOT traced.**

```
cuda.py:337-343   → forces scheduler_config.disable_chunked_mm_input = True
rocm.py:1025-1031 → same
platforms/xpu.py  → absent (grep)
```

Applies to `MM_PREFIX_LM_MODELS` (`model_arch_config_convertor.py:375-381`: bagel,
gemma3, molmo2, moondream3, paligemma). Chunking a multimodal item splits the
bidirectional prefix range that the Triton `mm_prefix` range tensor is built from
(`triton_attn.py:252-253`).

**B70 relevance:** moderate — 32 GB makes a 4-bit Gemma3 / PaliGemma the plausible
local VLM, and that is exactly the path.

### P1-2. `num_splits` is accepted by the XPU FA shim and silently dropped

**Confidence: HIGH on the drop. LOW on impact — perf, not correctness.**

`vllm/_xpu_ops.py:1005` declares `num_splits=0` and the forwarding call at
`:1035-1056` never passes it. The kernel side would reject it anyway
(`vllm_xpu_kernels/flash_attn_interface.py`: `if num_splits > 1: raise
NotImplementedError("FA2 does not support num_splits > 1")`).

Meanwhile `FlashAttentionMetadataBuilder` sets `max_num_splits =
flash_attn_max_num_splits_for_cuda_graph` (default **32**,
`vllm/config/attention.py:61`) for full cudagraphs
(`flash_attn.py:771-772`), and that value is passed at every FA call site
(`:1587,1677,…`) into a parameter that is thrown away. The cudagraph-mode contract
the user thinks they enabled is not in effect.

### P1-3. `head_size=256` attention tuning in Triton is gated on CUDA compute-capability family 10.x, which is unreachable on XPU

**Confidence: HIGH. Perf only, not correctness.**

`vllm/v1/attention/ops/triton_unified_attention.py:950-955`:

```python
    tuned_large_head = (
        head_size == 256
        and max_seqlen_q > 1
        and num_queries_per_kv <= 16
        and current_platform.is_device_capability_family(100)
    )
```

`XpuPlatform.get_device_capability()` returns `None` by design
(`platforms/xpu.py:267-273`, comment: *"capacity format differs from cuda's and will
cause unexpected failure, so use None directly"*), and
`is_device_capability_family` returns `False` on `None`. So the lab's head_dim 256
decode/prefill path gets the generic defaults (BLOCK_Q=8, TILE=32, 4 warps) instead
of the config the comment says is **~2× faster**.

The same `None` return makes every `has_device_capability()` gate in the attention
stack `False`. The codebase already patches around this in places —
`triton_reshape_and_cache_flash.py:27,29` explicitly add
`or current_platform.is_xpu()` — which is direct evidence the `None` return is a
known footgun. A platform-level `is_device_capability_family` override, or an
explicit XPU opt-in for the head-256 tuning, is the natural fix.

### P1-4. The XPU attention kernel is a precompiled variant table; head 256 is missing the sliding-window and sink combinations

**Confidence: HIGH on the table contents and the miss behaviour. MEDIUM on which
model configs land on a miss.**

`csrc/xpu/attn/kernel_configs/chunk_prefill_default.conf` ships **7** head-256
entries out of 240 possible tuples; `paged_decode_default.conf` ships 3 head-256
entries. A miss is a hard `TORCH_CHECK`, not a fallback
(`xe_2/chunk_prefill_utils.hpp:62-105`), with a good error message pointing at the
config file and issue #364.

Head-256 prefill tuples actually compiled:
```
256,true,false,true,false,false     paged, NON-causal, local
256,true,true,false,false,false     paged, causal
256,true,true,false,false,true     paged, causal, +softmax_lse
256,false,true,false,false,false    non-paged, causal
256,false,true,false,false,true     non-paged, causal, +softmax_lse
256,false,false,false,false,true
256,false,false,false,false,false
```
Compare head 128, which has 13 including `128,true,true,true,true,false`
(causal + sliding window + sink). **Missing for head 256: causal+sliding-window,
and any sink=true combination.** Paged-decode head 256 has only
`8,256,32,false,true,false`, `8,256,64,false,true,false`, `8,256,64,false,false,false`
— page size 16 and sink are absent.

So: take a head_dim-256 model and give it a sliding window, or attention sinks, and
the run dies with a rebuild-required error. This is loud, therefore lower severity
than P0-1, but it is a real coverage gap for a hardware class whose flagship
use-case is head_dim 256.

**B70 relevance:** direct.

### P1-5. No attention-kernel coverage in the XPU batch-invariance test suite

**Confidence: HIGH.**

`tests/v1/determinism/test_xpu_batch_invariant_ut.py` has 6 tests
(`test_rejects_weight_quantization`, `test_rejects_quantized_kv_cache`,
`test_accepts_unquantized_kv_cache`, `test_quantized_kv_cache_allowed_without_batch_invariance`,
`test_residual_norm_preserves_batch_invariance`, `test_collectives_...`,
`test_seeded_sampler_...`). All are config/norm/collective/sampler. **There is no
attention kernel test**, despite `TRITON_ATTN` being the backend
`platforms/xpu.py:179-183` advertises as batch-invariant.

Related: `is_batch_invariant` is bound at **module import**
(`triton_unified_attention.py:34`) rather than per-call. Latent today (nothing sets
the env post-import) but fragile.

### P1-6. `use_td` (Intel tensor-descriptor path) auto-enables on XPU with no XPU marker and no XPU CI job

**Confidence: MEDIUM.** Logic is unit-tested for head 128/256/96 but nothing marks
it XPU-specific or exercises Intel TD lowering.

`triton_attn.py:535-543` auto-enables tensor descriptors on XPU, with correct
guards (TILE clamped to block_size to satisfy the kernel `static_assert`,
`use_td_qo` falls back for non-pow2 head/query counts, explicit stride asserts fail
fast). The risk is coverage, not logic.

---

## P0/P1 — MLA sparse backend (narrower blast radius)

`XPU_MLA_SPARSE` is only reached when `attn_selector_config.use_sparse` is set
(`platforms/xpu.py:156-158`), i.e. DeepSeek-V3.2 / GLM-DSA class models. The lab's
head_dim-256 27B case does **not** reach it. Reporting at lower rank for that reason.

### MLA-1. `req_id_per_token` is built from the **CPU** query lengths, unlike every sibling builder

**Confidence: HIGH on the divergence. MEDIUM on reachability (needs adaptive
verification).**

`vllm/v1/attention/backends/mla/xpu_mla_sparse.py:156-160`:

```python
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
```

The base helper it bypasses, `CommonAttentionMetadata.token_to_req_indices`
(`vllm/v1/attention/backend.py:492-504`), deliberately uses the **device** tensor and
documents exactly why:

> *"Built from the device query_start_loc: adaptive verification decides the
> per-request draft split on device, so the CPU copy carries the right total but not
> the right per-request boundaries."*

Under adaptive verification the two disagree → tokens map to the **wrong request** →
wrong `block_table` row → wrong KV rows. Silently wrong attention, no crash. The
gate that should catch this is `supports_device_cpu_query_lens_mismatch`
(`backend.py:213-227`), which `XPUMLASparseBackend` does not override (so it
inherits `not cls.is_ssm()` = `True` = "I can handle it") — and even that gate is
never consulted on XPU because of P0-2.

Secondary: `np_to_pinned_tensor` at `:164` allocates a fresh pinned buffer each
forward and copies `non_blocking=True` into it; the temporary is dropped while the
DMA may still be reading. UNVERIFIED without a live run, and a per-step allocation
cost on exactly the bandwidth-limited device where it hurts.

### MLA-2. Prefill is routed through a kernel with no causal term

**Confidence: HIGH on routing and the absent mask. MEDIUM overall — an upstream
invariant appears to cover it, so this is "fragile", not "broken".**

`xpu_mla_sparse.py:180-184` sets `num_prefills=0` and
`num_decode_tokens=num_actual_tokens`, so **every** token including prefill goes
through `forward_mqa`. `vllm/v1/attention/ops/xpu_mla_sparse.py` contains no
`causal`/`q_pos` term; the only masks are on the index axis (`mask_indice:85`,
`mask_kv:97`), and `query_start_loc` is stored in the metadata and never read by
`forward_mqa`.

Mitigating: the indexer applies a per-token causal bound when building prefill
top-k — `vllm/v1/attention/backends/mla/indexer.py:457` (`global_ctx = start_pos + 1
+ offset`) flows into `cu_seqlen_ke` and then
`ops.top_k_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, …)`
(`sparse_attn_indexer.py:616-625`). Future positions are excluded before the kernel
sees them. The indexer's DCP/PCP chunk paths were **not** audited, so whether the
invariant always holds is unconfirmed. There is no local assertion guarding it.

### MLA-3. Missing metadata fields + `lse = None` vs a DCP assert

**Confidence: HIGH on the field absence. Currently unreachable on XPU.**

`mla_attention.py:989-1002` reads `attn_metadata.prefill_max_seq_len` (`:995`) and
`attn_metadata.prefill` (`:999`); neither exists on `XPUMLASparseMetadata`
(fields at `xpu_mla_sparse.py:80-109`). It is short-circuited only because
`fuse_attn_quant` is force-disabled on XPU (`platforms/xpu.py:357`) — i.e. one
fusion pass away from an `AttributeError`.

Separately, `forward_mqa` returns `lse = None` unconditionally
(`xpu_mla_sparse.py:277`) while `mla_attention.py:1130-1131` asserts
`lse is not None` when `dcp_world_size > 1` → runtime crash instead of config-time
rejection, again because of P0-2.

---

## Checked and found CLEAN

Recorded so the next audit does not re-derive them.

- **`s_aux` / attention sinks** — correctly forwarded end to end
  (`_xpu_ops.py:1048` → `attn_interface.cpp:28,35` → `flash_api.cpp:118,200,245`).
- **`window_size` / sliding window on the FA path** — plumbed
  (`_xpu_ops.py:1028-1033,1049` → `attn_interface.cpp:29-30,56-57,88-89`).
- **Triton path: sliding window, RSWA, softcap, sinks, mm_prefix** — all fully
  plumbed (`triton_attn.py:690-691` → `triton_unified_attention.py:1140-1142` →
  `triton_attention_helpers.py:383-387`); softcap normalized/gated/applied; sinks
  doubly validated (`triton_attn.py:517`, `triton_unified_attention.py:902`);
  mm_prefix OR-ed after SW in FlexAttention order.
- **Speculative decoding on the Triton path** — no uniform-query-length or MQA
  assumption. Ragged per-sequence lengths are resolved by binary search in
  `resolve_seq_and_query_len` (`triton_attention_helpers.py:49-74`) and masked with
  `query_pos < cur_batch_query_len`.
- **`head_size=256` on the Triton path** — genuinely supported, not a silent
  fallback (`HEAD_SIZE_PADDED = 256` with an all-true `dim_mask`).
- **`triton_reshape_and_cache_flash.py` XPU branches** (`:27,29,323,418,576`) — all
  launch-config heuristics; no scale dropped, no dtype cast skipped.
- **GDN fp8 MQA logits** — scale is per-KV-token, applied before per-head weights,
  with hard dtype asserts in the interface.
- **XPU MLA sparse kernel math** — index gather against `flat_kv_row_view` +
  `triton_convert_req_index_to_global_index` is correct including interleaved pages;
  `index_topk % BLOCK_N == 0` asserted; `sm_scale *= LOG2E` consistent with
  `max_logits = e_max * LOGE2` returning natural-log LSE; the `-1e30` sentinel
  genuinely fixes `exp2(-inf - -inf) = NaN`. Re-implemented the online softmax in
  Python: 7.4e-16 max relative error.
- **vLLM#49924 (GDN memory corruption) pin-lag class is closed** — main pins
  `0.1.15.4`, the latest on PyPI, and the `v_head_id` OOB-write guard (kernels
  `18b78776a`, issues #438/#439) is present in that wheel.
- **vllm-project/vllm-xpu-kernels is public and healthy** — 54 issues (19 open),
  559 PRs, last push 2026-09-24. No upstream-integration blocker.

## Open GDN/linear-attention risks in vllm-xpu-kernels (context, not new findings)

- **#389 / PR #391** — GDN spec-metadata shape checks reject graph-padded DFlash
  decode batches (`spec_query_start_loc must have size [num_spec_decodes+1]`). The
  fix PR has been open **116 days**. Mentions B70 / Xe2.
- **#593** — reduced active speculative width rejected when the state-index cache
  retains the configured width. No dedicated fix PR.
- **#552 (PR)** — *"Fix XE2 delta epilogue OOB write with non-contiguous token_indx"*
  — a live OOB-write bug, still unmerged.

The strict `TORCH_CHECK(spec_token == num_spec_decodes * (num_speculative_tokens+1))`
behind vLLM#54740 is **already fixed** in kernels `da16a55` (#600, merged
2026-09-16) and is a real traversal change, not a relaxed assert
(`gated_delta_rule.hpp:620-626`, `causal_conv1d.hpp:620-625` now iterate off
`spec_query_start_loc`). Caveat: the `vllm_xpu_kernels` build in
`/mnt/ssd/b70-venv` is **0.1.14.1** and still contains the strict `==` — that
environment will still crash. Upgrade it.

Also note: the Python assert in `gdn_attn.py:452-456` is unrelated and dead — lines
320-324 unconditionally zero `num_decodes` when `num_spec_decodes > 0`. Anyone
chasing #54740 via that line is chasing the wrong assert.

## Unverified — flagged, not claimed

1. **P0-1's numeric magnitude.** Whether dropped softcap produces garbage or merely
   degraded quality on B70 is untested. The code path is certain; the severity on
   hardware is not.
2. **MLA-2's upstream invariant.** The indexer's DCP/PCP chunk paths were not
   audited, so "causal masking is always applied upstream" is not established.
3. **MLA-1's pinned-temporary race.** Plausible from the code; needs a live run.
4. **GDN padded-batch `non_spec_query_start_loc` size mismatch.**
   `gdn_attn.py:527-528` pads to `batch_size + 1` where `batch_size = m.num_reqs`
   (= `num_reqs_padded`, `gpu_model_runner.py:2432`), while `num_decodes` is counted
   from the *unpadded* per-request lens (`gdn_attn.py:304,306`). The kernel demands
   exact equality (`gdn_attn_interface.cpp:298-301`). GDN advertises
   `AttentionCGSupport.UNIFORM_BATCH` (`gdn_attn.py:85`) so FULL graphs are
   permitted. GDN's own capture assert (`:567-576`) checks only totals, unlike
   `mamba_attn.py:238-246` which asserts `max_query_len == 1 + num_spec_tokens`. If
   reachable this is a hard crash, not a numerics bug. **Not confirmed — no padded
   capture was executed.**

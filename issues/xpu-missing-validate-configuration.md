## Summary

`XPUPlatform.get_attn_backend_cls` never calls `validate_configuration()` on the
resolved attention backend, while the CUDA and ROCm platforms both do.

```python
# vllm/platforms/cuda.py   -> 2 occurrences
# vllm/platforms/rocm.py   -> 2 occurrences
# vllm/platforms/xpu.py    -> 0 occurrences
```

Because the validation is skipped, every capability gate the attention backend
defines is **dead code on XPU**. An unsupported configuration is not rejected —
it is accepted and run, and the failure mode depends on what the gate would have
caught.

## Why this is the structural issue

The validation function in `vllm/attention/backends/registry.py` (roughly lines
274-350) is where backends declare and enforce their constraints: head size,
KV-cache dtype, block size, `mm_prefix` support, attention sinks, batch
invariance, and PCP/DCP constraints. On XPU, none of it runs.

This is why several other XPU attention bugs present as *silent* wrongness
instead of a clean error. It is the difference between "unsupported head size
256 raises a clear error" and "the kernel runs anyway and produces garbage".

## Concrete example

`FlashAttentionBackend` reports capability flags that contradict the XPU
platform's own documented limitations:

- `supports_batch_invariance() -> True`
- `supports_non_causal() -> True`

while `vllm/platforms/xpu.py:165-178` and `:184-197` state that FlashAttention
on XPU is **not** batch-invariant and **cannot** service the `mm_prefix`
bidirectional mask. Both branches honour an explicit request and return anyway.
With `validate_configuration` wired up, those requests would be rejected up
front instead of silently mis-executed.

## Suggested fix

Mirror what CUDA and ROCm do in `get_attn_backend_cls`: after resolving the
backend class, call its `validate_configuration` and let a mismatch raise.

```python
# in XPUPlatform.get_attn_backend_cls, after resolving the backend class
backend_cls.validate_configuration(...)
```

Worth pairing with: make the unsupported cases fail loudly. A user asking for
batch-invariant attention on XPU should get a clear error, not a config that
silently is not batch-invariant.

## Environment

- GPU: Intel Arc Pro B70 (Battlemage G31, `0xe223`), 32 GB
- Driver / compute-runtime 26.18.38308.4, UR 1.15.38308+4, IGC 2.34.4
- Fedora 43, kernel 7.1.8, `xe` driver
- vLLM 0.30.0+XPU, torch 2.13.0+xpu, vllm-xpu-kernels 0.1.14.1

## Verification

Confirmed by reading the installed vLLM source: occurrence counts of
`validate_configuration` per platform file, and the
`FlashAttentionBackend.supports_batch_invariance` / `supports_non_causal`
implementations versus the warnings in `platforms/xpu.py`. No GPU execution was
needed — this is a control-flow gap, verifiable statically.

Related: the DeepSeek-V4.1, `minimax_m3`, `inkling`, `dots3_note` and `kimi_k3`
model packages also branch only on `is_rocm()` and silently fall through to
NVIDIA paths on XPU. That is a related family of missing-platform-check issues
and may be worth tracking together.

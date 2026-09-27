## Summary

`logits_soft_cap` is silently dropped on the XPU FlashAttention path. Models that
require attention logit soft-capping — notably **Gemma-2 and Gemma-4** — run on
Intel GPUs with **no soft cap applied at all**, and nothing warns the user.

This is a silent numerical-correctness issue: the model loads, generates fluent
text, and produces plausible-but-wrong logits. There is no error and no log line.

## Where it is dropped

`vllm/_xpu_ops.py`, `XPUOps.flash_attn_varlen_func`:

The parameters are **accepted into the signature** (`:969`, `:971`):

```python
        alibi_slopes: torch.Tensor | None = None,
        window_size: list[int] | None = None,
        softcap: float | None = 0.0,
```

and then **commented out at the forwarding call** (`:1028-1029`):

```python
            window_size=real_window_size,
            # alibi_slopes = alibi_slopes,
            # softcap=softcap,
            return_softmax_lse=return_softmax_lse,
```

So a caller that correctly sets `logits_soft_cap` in its config has it silently
discarded one layer down.

## Why no fallback catches it

`get_flash_attn_version()` short-circuits to version `2` on XPU
(`vllm/attention/fa_utils.py:83-84`) **before** reaching the softcap / ALiBi
fallback logic. `vllm/platforms/xpu.py` has no softcap branch either. So the
usual "fall back to a backend that supports this" safety net never engages.

## The kernel side already plumbs it

Worth noting for whoever fixes this: the plumbing is **not** missing in
`vllm-xpu-kernels`. `vllm_xpu_kernels/flash_attn_interface.py` already accepts
and forwards `softcap` through to the kernel (`:72`, `:417`, `:536`, `:589`,
`:620`), and the compiled kernel exposes it. So this looks like a
vLLM-side-only change, not a kernels change.

## Impact

- **Gemma-2 / Gemma-3 / Gemma-4 on XPU**: attention logits are uncapped. For
  Gemma-2 this is not a minor deviation — soft-capping is what keeps the
  attention distribution inside the range the model was trained for.
- Any model or config setting `logits_soft_cap` (or `alibi_slopes`) on Intel
  hardware.
- Silent: no warning is emitted, so a user has no signal that the cap is off.

## Suggested fix

Uncomment and forward both parameters at `vllm/_xpu_ops.py:1028-1029`, and add a
guard so this class of drop fails loudly rather than silently. Something like
raising (or at minimum `logger.warning_once`) when `softcap not in (0.0, None)`
or `alibi_slopes is not None` reaches a backend that cannot honour it — the same
pattern `qwen4_exp/__init__.py:30` already uses by raising instead of silently
importing an NVIDIA path.

If some XPU attention kernel genuinely cannot support softcap yet, then the
correct interim behaviour is to reject the configuration with a clear error, not
to run the model uncapped.

## Environment

- GPU: Intel Arc Pro B70 (Battlemage G31, `0xe223`), 32 GB
- Driver / compute-runtime 26.18.38308.4, UR 1.15.38308+4, IGC 2.34.4
- Fedora 43, kernel 7.1.8, `xe` driver
- vLLM 0.30.0+XPU, torch 2.13.0+xpu, vllm-xpu-kernels 0.1.14.1

## Verification

Confirmed by reading the installed vLLM source at the line numbers above
(`vllm/_xpu_ops.py`) and the installed kernels wrapper
(`vllm_xpu_kernels/flash_attn_interface.py`). Searched existing open and closed
issues in both `vllm-project/vllm` and `vllm-project/vllm-xpu-kernels` for
`softcap` / `logits_soft_cap` / `alibi`; no existing issue covers this.

The *numerical magnitude* of the divergence is not measured here — this report
establishes that the parameter is dropped, which is verifiable from source
alone. A follow-up measuring Gemma-2 output with and without the cap on XPU
would quantify the impact.

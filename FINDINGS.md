# B70 Lab — Findings

Negative and positive results, recorded so they are not rediscovered.
Each entry: what was tried, what happened, and what it means.

---

## F-001 — `pip install vllm` silently installs the CUDA build

**Status:** root-caused, worked around, upstream-doc gap.

`uv pip install vllm==0.30.0` into the XPU venv **exits 0** and produces a
broken environment. The PyPI wheel is the CUDA build.

Observed:

- ~2 GB of `nvidia-*`, `nvidia-cutlass-dsl*`, `flashinfer-python` pulled in
- CUDA `triton==3.8.0` installed alongside `triton-xpu==3.7.2`, shadowing it
- vLLM logs `triton not found; flop counting will not work for triton kernels`
- `vllm.platforms.current_platform` resolves to **`None`** — see below

The trap: `torch.xpu.is_available()` still returns `True` afterwards, so a
naive health check passes. **Check `current_platform`, not torch.**

Why `current_platform` is `None` with the CUDA wheel: the XPU platform gate
(`vllm/platforms/__init__.py::xpu_platform_plugin`) is purely

```python
if hasattr(torch, "xpu") and torch.xpu.is_available():
    is_xpu = True
```

A CUDA torch still *has* a `torch.xpu` attribute and may report available, but
without the XPU platform plugin resolving, `current_platform` stays `None` and
nothing selects the B70.

### Probe gotcha when checking this

Do **not** verify with `current_platform.device_name`. Accessing it can return
`None`/warn (`does not have 'device_name' attribute`) even on a perfectly good
XPU install, because `device_name` lives on the resolved subclass, not the
`Platform` base. That produced one false "still broken" reading for me.

Verify with either of these instead:

```bash
# best
uv pip list --python /mnt/ssd/b70-venv/bin/python | grep -iE '^(vllm|torch|triton|vllm-xpu-kernels)'
# expect: vllm 0.30.0+xpu, torch 2.13.0+xpu, triton 3.7.2+xpu, vllm-xpu-kernels 0.1.14.1

# runtime
python -c "import vllm; from vllm.platforms import current_platform as c; print(c.device_type, c.is_xpu())"
# expect: xpu True
```

Confirmed-good state of the current venv: `vllm 0.30.0+xpu`, `torch 2.13.0+xpu`,
`triton 3.7.2+xpu` + `triton-xpu 3.7.2`, `vllm-xpu-kernels 0.1.14.1`, zero
nvidia/cutlass/flashinfer packages, `current_platform.device_type == 'xpu'`.

Fix: install the XPU wheel from the vLLM GitHub release with both extra indexes.
See ENVIRONMENT.md. Verified clean: zero nvidia packages, `platform: xpu`.

**Upstream angle:** the vLLM docs' XPU install page does not warn that the
default PyPI install is CUDA. That is a P7 install-maturity issue.

---

## F-002 — Sourcing oneAPI `setvars.sh` breaks torch XPU

**Status:** root-caused, worked around.

```
ImportError: libsycl.so.9: undefined symbol: urDeviceWaitExp, version LIBUR_LOADER_0.12
```

`setvars.sh` prepends oneAPI 2025.3's bundled `libsycl.so.9` to
`LD_LIBRARY_PATH`, shadowing the newer system UR loader
(`/usr/lib64/libze_loader.so.1`, from `oneapi-level-zero` 1.28.6) that
`torch 2.13.0+xpu` links against.

This is the **opposite** of the intuitive fix. I initially recorded "you must
source setvars.sh" in the lab manifest because that is why `sycl-ls` appeared
missing — then the first torch XPU run failed and proved it backwards.

Rule: `env -u LD_LIBRARY_PATH python ...`. oneAPI is only for offline
compilation.

---

## F-003 — torch.compile needs Level Zero **devel** files

**Status:** root-caused, worked around, upstream-doc gap.

`oneapi-level-zero-devel` is not installed. vLLM's `torch.compile` path needs
both of its outputs, and fails *after* weights have loaded, so the log looks
healthy right up to the crash:

```
fatal error: level_zero/ze_api.h: No such file or directory
/usr/bin/ld: cannot find -lze_loader: No such file or directory
```

The system has `libze_loader.so.1` but no unversioned `libze_loader.so` dev
symlink. Fixing only `CPATH` (headers) just moves the error to the linker —
`LIBRARY_PATH` is the one that is easy to miss.

Workaround: stage both rootlessly under `~/.local` (see ENVIRONMENT.md).

**Upstream angle:** a clean machine following the documented XPU install hits
this wall. P7.

---

## F-004 — vLLM v1 engine cannot start on the B70: SYCL segfault (P0)

**Status:** OPEN. Reproduced deterministically. Specialist assigned.

Every vLLM 0.30.0 XPU engine start on the B70 segfaults. Startup proceeds
normally — platform `xpu` selected, Flash Attention backend chosen, weights
loaded (1.12 GiB), `Using LBNHC KV cache layout` — then:

```
!!!!!!! Segfault encountered !!!!!
  sycl::_V1::detail::context_impl::get_info<sycl::info::context::devices>()
  sycl::context::get_devices()
  init_devices
  ffi_call
  _call_function_pointer
  _ctypes_callproc
  PyCFuncPtr_call
  slot_tp_init          <- during object __init__
  type_call
```

### What it is NOT (each hypothesis tested and eliminated)

| Hypothesis | Test | Result |
|---|---|---|
| XPU CUDA-graph path | `VLLM_XPU_ENABLE_XPU_GRAPH=0` | still segfaults |
| torch.compile / Inductor | `enforce_eager=True` | still segfaults |
| fork vs spawn | traced `get_mp_context()` | **already correctly `spawn`**; `xpu_is_initialized()=True` |
| `vllm_xpu_kernels` import | import each of `_C`/`_moe_C`/`_xpu_C`/`xpumem_allocator` | all clean |
| `XPUPluggableAllocator` construction | `get_pluggable_allocator(...)` | constructs fine; crash is later |

Note the trap: a *manual* `os.fork()` test produces torch's
`Cannot re-initialize XPU in forked subprocess` error, which looks like the
answer but is a **different code path** — vLLM already uses spawn.

### Where it points

The frame chain `slot_tp_init → ffi_call → _ctypes_callproc` means a **ctypes
callback from C++ into Python during object construction**, ending in a SYCL
`context::get_devices()` on a context that is not valid in this process.
`vllm/device_allocator/xpumem.py` is the prime suspect: it passes raw Python
callbacks into the SYCL allocator via
`xpumem_allocator.init_module(python_malloc_fn, python_free_func)` and then
`XPUPluggableAllocator(lib_name, "my_malloc", "my_free")`, which **dlopens**
`vllm_xpu_kernels/xpumem_allocator.abi3.so` and resolves SYCL symbols by name —
a classic way to pick up a second, mismatched SYCL runtime.

Last log line before death: `utils.py:320 Using LBNHC KV cache layout`, inside
`core.py:308 _initialize_kv_caches → determine_available_memory`.

### Why this is P0

Every vLLM XPU user on Battlemage hits this. It is not a niche config: it is
the default single-GPU path, eager or compiled, graph on or off. Until it is
fixed, no other B70 correctness or performance work can be validated, because
there is no working engine to validate it with.

---

## F-005 — XPU CUDA-graphs are off by default (P1)

```
WARNING [xpu.py:303] XPU Graph is disabled by environment variable,
        please set VLLM_XPU_ENABLE_XPU_GRAPH=1 to enable it.
WARNING [xpu.py:308] XPU Graph support is experimental and currently only
        supports single-GPU execution.
```

CUDA users get graphs enabled by default. XPU users must opt in via env var.
Feature-parity gap worth tracking separately from F-004.

---

## F-006 — `xpu-smi` does not exist on Fedora 43

Not packaged in any enabled repo. `oneapi-level-zero` ships only
`libze_loader`, not the tool. The `vllm/vllm-openai-xpu` Docker image *does*
bundle xpu-smi, which is another argument for the container path.

Use `gputop` instead. It requires a TTY: piped or redirected it prints nothing
and exits 0, which falsely reads as "GPU idle".

---

## F-007 — ctypes probe of `ext_intel_free_memory` segfaults

The Level Zero device aspect looks like the clean way to query free VRAM, but
its vendor struct is version-sensitive and undocumented in the installed
headers; the probe segfaults. Use `torch.xpu.mem_get_info(0)` via
`tools/b70_vram.py` instead.

---

## Measured baselines

GEMM, 4096³, 30 iters, desktop session live, all outputs finite:

| dtype | time | throughput |
|---|---|---|
| float16 | 75.9 ms | **54.3 TFLOP/s** |
| bfloat16 | 115.6 ms | **35.7 TFLOP/s** |

bf16 is 1.5x slower than fp16 on Battlemage. Unexplained; a candidate P3
workstream (oneDNN bf16 vs fp16 path).

Device: 256 EUs, 32 subslices, sub_group_sizes [16, 32], 2800 MHz mem,
64-bit bus, 24 MB LLC, **`local_mem_size` only 128 KB**. That tiny SLM is a hard
constraint on kernel design: no SLM-heavy tiling will fit, so XPU kernels must
lean on the 24 MB LLC and plain USM.

VRAM: 30.88 GiB free of 31.89 GiB (desktop holds ~1.0 GiB).

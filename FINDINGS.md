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
XPU install, because `device_name` lives on the resolved subclass, not on the
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
`torch 2.13.0+xpu` is linked against.

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
healthy right up until it isn't:

```
fatal error: level_zero/ze_api.h: No such file or directory
/usr/bin/ld: cannot find -lze_loader: No such file or directory
```

The system has `libze_loader.so.1` but no unversioned `libze_loader.so` dev
symlink. Fixing only `CPATH` (headers) just moves the error to the linker —
`LIBRARY_PATH` is the one that is easy to miss.

Workaround: stage both rootlessly under `~/.local` (see ENVIRONMENT.md).

**Runtime half, found while fixing F-004:** Inductor's *generated* kernels
`dlopen("libze_loader.so")` at run time, so `LD_LIBRARY_PATH=$HOME/.local/lib`
is required at run time too, not just at compile time. This is safe and does
**not** violate the F-002 rule: `~/.local/lib/libze_loader.so` is a symlink to
the *system* `/usr/lib64/libze_loader.so.1` that `libtorch_xpu.so` is already
linked against, so no oneAPI shadowing occurs.

**Upstream angle:** a clean machine following the documented XPU install hits
this wall. P7.

---

## F-004 — vLLM v1 engine cannot start on the B70: SYCL segfault (P0)

**Status:** ROOT-CAUSED and FIX VERIFIED. The bug is in **triton
(intel-xpu-backend-for-triton)** — not vLLM, not vllm-xpu-kernels.

### Crash site

`triton/backends/intel/driver.py:364`, inside `XPUUtils.__init__`
(driver.py:344):

```python
self.device_count = mod.init_devices(self.get_sycl_queue())
```

`init_devices` is a C symbol in the triton-compiled `spirv_utils*.so`, called
over ctypes, and it dies on its first SYCL call. gdb pins the faulting library
exactly:

```
#0  sycl::detail::context_impl::get_info<sycl::info::context::devices>()
        from /home/dario/oneapi/compiler/2025.3/lib/libsycl.so.8   <-- dies here
#1  sycl::context::get_devices()       from libsycl.so.8
#2  init_devices()                     from spirv_utils.cpython-312-...so
#3  ffi_call ... #7 _ctypes_callproc  #8 PyCFuncPtr_call
```

Note the `!!!!!!! Segfault encountered !!!!!!!` banner is **tvm_ffi's** signal
handler (`tvm_ffi/src/ffi/backtrace.cc:156`), not vLLM's. That is why the
trace looks like it starts mid-SYCL-call.

### Root cause: two SYCL C++ runtimes in one process

| Component | SYCL runtime it links |
|---|---|
| `libtorch_xpu.so` (torch 2.13.0+xpu) | **libsycl.so.9** — intel-sycl-rt 2026.0.0, `/mnt/ssd/b70-venv/lib` |
| triton `spirv_utils.so` | **libsycl.so.8** — oneAPI 2025.3, `/home/dario/oneapi/compiler/2025.3/lib` |

`XPUUtils.get_sycl_queue()` hands a `sycl::queue*` created by the .so.9
runtime into code compiled against the .so.8 runtime. The object layouts and
vtables differ, so `sycl::context::get_devices()` dereferences the wrong
runtime's vtable and the process dies. Proven directly via `/proc/self/maps`:
after `dlopen`ing the cached `spirv_utils.so`, **both** are mapped at once.

Why triton picks the wrong one: `find_sycl()` (driver.py:50-56) probes
`shutil.which("icpx")` **first**, so with oneAPI 2025.3 on `$PATH` it resolves
SYCL headers+libs to that toolchain — even though the process has already
loaded a different SYCL runtime via torch. It never reaches its
`intel-sycl-rt` wheel branch (driver.py:67-91), which is the branch that
matches torch.

A second, independent bug makes it sticky: `compile_module_from_src()`
(driver.py:288-294) keys the module cache on `__CACHE_VERSION + src +
platform_key()` only. The SYCL runtime identity is **not** in the key, so a
`spirv_utils.so` built against libsycl.so.8 is silently reused after the
runtime changes. Any fix must also purge that cached entry.

### Minimal standalone reproducer (no vLLM)

`repro1_triton_init.py` — `torch.xpu` init, then `XPUUtils()`. Exits **139
(SIGSEGV)** in seconds. `repro2_two_runtimes.py` additionally prints
`/proc/self/maps` before/after to show both runtimes coexisting.

### Verified fix

Make `find_sycl()` prefer the SYCL runtime already loaded in the process, and
fold that runtime into the module cache key. Patch lives at
`/mnt/ssd/dev workspace/_f004_segfault/triton-intel-driver.patch` (upstream
paths, targets `triton/backends/intel/driver.py`).

A/B under an identical environment, icpx on `$PATH`, no env hack:

| Run | `libsycl.so` linked | Result |
|---|---|---|
| baseline | `libsycl.so.8` | **SIGSEGV**, exit 139 |
| patched | `libsycl.so.9` | `device_count = (1,)`, exit 0 |

End-to-end, `LLM(model="Qwen/Qwen3-0.6B", enforce_eager=False)`:

- **baseline** → segfaults with the exact reported trace
- **patched** → engine starts in 157.8 s and generates
  `" Paris. The capital of France is also the capital of the French Republic."`

F-003 does not mask F-004: with `LD_LIBRARY_PATH=$HOME/.local/lib` applied, the
segfault is gone in the patched run and F-003 is simply the next wall.

### What it is NOT (each hypothesis tested and eliminated)

| Hypothesis | Test | Result |
|---|---|---|
| XPU CUDA-graph path | `VLLM_XPU_ENABLE_XPU_GRAPH=0` | still segfaults |
| torch.compile / Inductor | `enforce_eager=True` | still segfaults |
| fork vs spawn | traced `get_mp_context()` | **already correctly `spawn`**; `xpu_is_initialized()=True` |
| `vllm_xpu_kernels` import | import each of `_C`/`_moe_C`/`_xpu_C`/`xpumem_allocator` | all clean |
| `XPUPluggableAllocator` construction | `get_pluggable_allocator(...)` | constructs fine; crash is later |
| `xpumem.py` ctypes callbacks (original suspect) | gdb shows the ctypes frame belongs to **triton**, not the allocator | **ruled out** |

Note the trap: a *manual* `os.fork()` test produces torch's
`Cannot re-initialize XPU in forked subprocess` error, which looks like the
answer but is a **different code path** — vLLM already uses spawn.

The original suspicion that `vllm/device_allocator/xpumem.py` was at fault was
wrong. The `ffi_call` / `_ctypes_callproc` frames are real, but they belong to
triton's `SpirvUtils` (`driver.py:198`), not to the XPU pluggable allocator.

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

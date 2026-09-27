## Summary

On a torch XPU install, `triton/backends/intel/driver.py` compiles its SYCL
helper module (`spirv_utils*.so`) against a **different SYCL C++ runtime** than
the one already loaded in the process. Passing a `sycl::queue*` across that
runtime boundary dereferences the wrong vtable and the process dies with
SIGSEGV.

This makes vLLM (and anything else using triton's Intel XPU backend) completely
unusable on the GPU — the engine cannot start at all.

## Environment (reproduced and verified on real hardware)

| Component | Version |
|---|---|
| GPU | Intel Arc Pro B70 (Battlemage G31, `0xe223`), 32 GB |
| Driver / compute-runtime | 26.18.38308.4 |
| UR | 1.15.38308+4 |
| IGC | 2.34.4 |
| OS | Fedora 43, kernel 7.1.8, `xe` driver |
| torch | 2.13.0+xpu (links `libsycl.so.9`, intel-sycl-rt 2026.0.0) |
| oneAPI on host | 2025.3 (ships `libsycl.so.8`) |
| vLLM | 0.30.0+XPU wheel |
| vllm-xpu-kernels | 0.1.14.1 |

## Root cause

Two distinct SYCL runtimes end up mapped into one process:

```
$ ldd .../torch/lib/libtorch_xpu.so | grep sycl
    libsycl.so.9 => /mnt/ssd/b70-venv/lib/libsycl.so.9
```

```
$ ls ~/oneapi/compiler/2025.3/lib/libsycl.so*
    libsycl.so -> libsycl.so.8
```

`XPUUtils.get_sycl_queue()` obtains a `sycl::queue*` created by the **.so.9**
runtime (via torch) and hands it to `init_devices()` inside a module compiled
against **.so.8**. Object layouts and vtables differ between the two, so:

```cpp
sycl::context::get_devices()   // reads a vtable owned by the other runtime
```

segfaults. Confirmed with gdb — both runtimes are mapped simultaneously after
`dlopen` of the cached `spirv_utils.so` (visible in `/proc/self/maps`).

### Why triton picks the wrong runtime

`find_sycl()` (driver.py:50-56) probes `shutil.which("icpx")` **first**:

```python
icpx_path = shutil.which("icpx")
if icpx_path:
    # only `icpx` compiler knows where sycl runtime binaries and header files are
    ...
```

So when a oneAPI toolchain is on `$PATH`, triton resolves SYCL headers and libs
to that toolchain — even though the process has *already loaded* a different
SYCL runtime through torch. It never reaches its `intel-sycl-rt` wheel branch
(driver.py:67-91), which is the branch that actually matches torch.

A runtime that is already loaded owns the `sycl::queue` objects in play, so
shadowing it is an ABI mismatch, not a preference.

### Second, independent bug: stale module cache

`compile_module_from_src()` (driver.py:288-294) keys the compiled-module cache
on `__CACHE_VERSION + src + platform_key()` only. **The SYCL runtime identity is
not part of the key.**

Consequence: after switching runtimes (oneAPI toolchain -> `intel-sycl-rt`
wheel), a `spirv_utils.so` built against the previous `libsycl` is silently
reused from cache. Any fix must also fold the runtime into the key, and users
must purge the existing cache entry once when upgrading.

## Minimal reproducer (no vLLM required)

```python
import torch
a = torch.randn(64, 64, device="xpu")
torch.xpu.synchronize()

from triton.backends.intel.driver import XPUUtils
u = XPUUtils()          # <-- SIGSEGV here
print(u.device_count)
```

Exits **139 (SIGSEGV)** in a few seconds.

```
[1] torch xpu ok
[2] constructing XPUUtils (this is the crash site)...
Segmentation fault
```

## Impact

vLLM 0.30.0 on XPU never reaches a working engine. Startup proceeds normally —
platform `xpu` selected, attention backend chosen, weights loaded, KV-cache
layout selected — and then the process dies during `torch.compile` /
`determine_available_memory`:

```
!!!!!!! Segfault encountered !!!!!
  sycl::_V1::detail::context_impl::get_info<sycl::info::context::devices>()
  sycl::context::get_devices()
  init_devices
  ffi_call ... _ctypes_callproc ... slot_tp_init
```

(That banner is tvm_ffi's signal handler, not vLLM's — worth knowing when
triaging, since it makes the trace look like it starts mid-SYCL-call.)

Affects both `enforce_eager=True` and compiled paths, and both
`VLLM_XPU_ENABLE_XPU_GRAPH=0` and `=1` — it is not graph- or compile-specific.

## Proposed fix

1. **Prefer the already-loaded SYCL runtime.** Before probing `icpx`, inspect
   `/proc/self/maps` for a mapped `libsycl.so*` and derive include+lib dirs from
   it (wheel layout: `<root>/include` beside `<root>/lib`). Headers and libraries
   must come from the same runtime, so derive includes from the lib dir. If no
   matching header is found, fall through to the existing search order unchanged.

2. **Fold the runtime identity into the module cache key**, hashing the resolved
   `libsycl` path plus a content hash of the library, so switching runtimes can
   never silently reuse a stale `spirv_utils.so`.

A patch implementing both is attached. Verified A/B on the B70, identical
environment, no env-var workarounds:

| Run | links | minimal repro | full vLLM engine |
|---|---|---|---|
| baseline | `libsycl.so.8` | **SIGSEGV (139)** | segfaults |
| patched | `libsycl.so.9` | `device_count=(1,)`, exit 0 | starts, generates correct text |

Patched engine output for `"The capital of France is"` (greedy):

```
" Paris. The capital of France is also the capital of the French Republic. The"
```

## Workaround for users (until fixed)

Do not put a oneAPI toolchain on `PATH` when running torch XPU + triton. If
`icpx` is not discoverable, `find_sycl()` falls through to the
`intel-sycl-rt` wheel branch that matches torch:

```bash
# ensure icpx is NOT on PATH
env -u LD_LIBRARY_PATH python your_script.py
# and purge any stale compiled module once
rm -rf ~/.triton/cache
```

Note this conflicts with needing `icpx` for other compilation steps, which is
part of why the runtime should be detected from what is already loaded rather
than from `$PATH`.

## Checklist

- [x] Reproduced on real hardware (Arc Pro B70)
- [x] Minimal standalone reproducer (no vLLM)
- [x] gdb backtrace pinning the faulting library
- [x] Root cause identified (two SYCL runtimes, wrong vtable)
- [x] Fix implemented and verified A/B on the GPU
- [x] Full vLLM engine verified generating correct output with the fix

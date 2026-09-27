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

## F-005 — XPU CUDA-graphs are off by default (P1) — RESOLVED UPSTREAM

```
WARNING [xpu.py:303] XPU Graph is disabled by environment variable,
        please set VLLM_XPU_ENABLE_XPU_GRAPH=1 to enable it.
WARNING [xpu.py:308] XPU Graph support is experimental and currently only
        supports single-GPU execution.
```

**Status: CLOSED as a local-version artifact. The upstream gap no longer exists.**

Root cause of the local observation: vLLM **0.30.0 was published 2026-09-22**, and
upstream **PR #51600 `[XPU] enable XPU GRAPH by default` merged 2026-09-24**, two days
later. It removes the `VLLM_XPU_ENABLE_XPU_GRAPH` env var and the experimental
warning entirely, and enables graphs by default unless `--enforce-eager`. Verified
against current main: the `envs.VLLM_XPU_ENABLE_XPU_GRAPH` branch in
`vllm/platforms/xpu.py` no longer exists.

The default-off was **deliberate**, not an oversight — PR #38193 (merged
2026-03-26): *"require a specific driver version, it's not stable yet. so we decide
disable it by default."* The stated reason (driver-version instability) was later
addressed by torch XPU 2.14 switching SYCL Graph -> Level Zero Graph (#56013).

**Do not file an "enable by default" issue — it is a duplicate of merged #51600.**

### Measured impact on the B70 (interleaved A B B A B A, 3 runs/arm)

Capsule `010_xpu_graph_ab.py`. Qwen3-0.6B, desktop session live, 12 runs, all OK.

| config | decode tok/s OFF | ON | speedup |
|---|---|---|---|
| baseline (8x96, mbt 2048) | 494 | 2060 | **4.16x** |
| heavy (32x128, mbt 16384) | 1810 | 7340 | **4.04x** |

TTFT 0.67x, ITL 0.22x. Token counts identical across arms (768/4096), so the
speedup is not a truncation artifact. Each arm is internally deterministic
(1 distinct sha256 per arm over 3 runs), and zero requests flagged corrupted.

**Caveat, stated plainly:** this is a 0.6B model, where launch overhead
dominates. "4x" is **not** a general XPU claim. It does match #51600's own
wording ("small models and MoE models under low concurrency"), so it
corroborates the PR rather than extending it. Larger models will see less.

The two arms produce *different text*. Traced to an **exact 0.000000-nat tie**
(`' is'` vs `' involves'`, both -1.54230) where the graph path's different
reduction order picks the other member. Both outputs are coherent — a benign
tie-break, not the #54785 corruption class. Recorded because "graphs change your
output" is the kind of claim that must not be reported without evidence.

### Still open, and the real risk

Five graphs-on bugs remain unaddressed: #48327, #54698, #54785, #48946, and
vllm-xpu-kernels #567. Not reproduced on this single B70, but they are 2x-B70
and concurrent-load reports, so a clean single-GPU run is not a refutation.

**Actionable gap:** PR #51600's **Test Plan and Test Result sections are empty**,
and no user-facing doc mentions XPU graphs. The benchmark above can backfill
both — that is a concrete, high-value contribution to offer upstream.

---

## F-008 — Xe2 grouped-GEMM data race: fixed upstream, but vLLM 0.30.0 can never get it

**Status: FIXED upstream (PR #586, commit `3d74ec9`, 2026-09-10), but our env is
vulnerable and there is no release path to 0.30.0.**

`csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp:252` at tag `v0.1.14`:

```cpp
at::Tensor atomic_buffer =
    at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
```

`3d74ec9` changes `at::empty` -> `at::zeros` and deletes the in-kernel
`atm.store(0)`. The race: the grid is persistent
(`global(1, sm_count*512/wg_size, 1)`), lane 0 of every workgroup steals tiles
via `cutlass::atomicAdd(atomic_buffer, 1)` (`grouped_gemm_xe2.hpp:229`) behind
only a workgroup-local barrier, while the reset was workgroup 0 / lane 0 with no
device-wide barrier. A steal landing first yields a garbage ticket and every
later ticket is garbage+1, +2, ... so tiles in
`[group_range, group_range+garbage)` are never computed, keeping whatever
`torch.empty` left. **Silently wrong output, not a crash.**

**Why it could never reach vLLM 0.30.0:** the fix is in `v0.1.15` and PyPI
`0.1.15.1/.3/.4`, but 0.30.0 pins `vllm_xpu_kernels==0.1.14.1` and **there is no
`release/0.30` branch** in the kernels repo. Tag `v0.1.14.1` does not exist
(PyPI-only patch), so there is nothing to pin to a fixed build.

Our venv runs **0.1.14.1** and is therefore pre-fix.

### Reachability is broad — no quantization required

`oracle/unquantized.py` sets `_AVAILABLE_BACKENDS = [XPU, TRITON]` on XPU, so XPU
is **first**, i.e. the default. Smallest trigger: `axolotl-ai-co/tiny-mixtral-30m`
(8 experts, hidden 256) — the same model upstream uses for its XPU MoE coverage.

### Honest reproduction result

- Op callable and correct (max err 0.5 bf16, no NaN).
- **Uninitialized read is real: 60/60.** Recycling a poisoned 1-element int32
  block makes `at::empty` return `100000` intact.
- **But the eager race never fired: 0/60 wrong outputs.** Workgroup 0's
  `store(0)` is the kernel's first instruction and the first `atomicAdd` only
  happens after a full GEMM tile, so it always wins an eager launch.

**Latent in eager, real under graph replay.** No wrong output was observed and
none is claimed. The value here is the binary proof plus the version analysis,
not a reproduced corruption.

Residual structural hazard tracked by **PR #457** (open): the work-steal design
is not cudagraph-capture-safe. #586 fixes the garbage value; #457 removes the
class.

### Two traps worth keeping

- `cutlass_grouped_gemm_interface` in `_xpu_C.abi3.so` is a 1.9 KB dispatcher;
  the `at::empty` lives in the out-of-line `MoE::cutlass_grouped_gemm_xe2_impl`
  in `libgrouped_gemm_xe_2.so`. A first ELF pass that only inspects the
  dispatcher reads "inconclusive" and is wrong.
- **`torch.profiler` sees 0 kernels** for these calls (raw `sycl::queue`
  submits). Trusting it would have "confirmed" the pre-fix state from a broken
  instrument.

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

## F-008 — Xe2 grouped-GEMM tile counter is allocated uninitialized (P2, FIXED upstream)

**Status: CONFIRMED in source and in the shipped binary. Already fixed upstream
(#586) and shipped to PyPI. Not filed — would have been a duplicate.**

Claim: in `vllm_xpu_kernels` 0.1.14.1, the Xe2 grouped-GEMM persistent-workgroup
scheduler counter is allocated with `at::empty()` (uninitialized).

### Root cause (verified by diff, not by the report)

`csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp:252` at tag `v0.1.14`:

```cpp
at::Tensor atomic_buffer =
    at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));
```

`MoEGEMMLauncher` launches a **persistent** grid
(`global(1, sm_count*512/wg_size, 1)`). Every workgroup's lane 0 steals the next
tile with `cutlass::atomicAdd(atomic_buffer, 1)` (`grouped_gemm_xe2.hpp:229`)
guarded only by a **workgroup-local** barrier. Pre-fix, the counter was reset by
workgroup 0 / lane 0 (`grouped_gemm_xe2.hpp:105-112`) with **no device-wide
barrier**. If any steal lands before that store, that workgroup's ticket is
garbage and every later ticket is garbage+1, +2, ... Tiles in
`[group_range, group_range+garbage)` are never computed and keep whatever
`torch.empty` left in `ptr_D` — **silently wrong MoE output, not a crash**.

### Upstream status: fixed, merged, released

| | |
|---|---|
| PR | **#586** `fix: initialize atomic_buffer to 0 to avoid race conditions` |
| commit | `3d74ec9` (2026-09-10), ancestor of `v0.1.15` and `origin/main` |
| change | `at::empty` → `at::zeros` **and** deletes the in-kernel `store(0)` |
| in tag `v0.1.14`? | **NO** (`git merge-base --is-ancestor 3d74ec9 v0.1.14` → false) |
| on PyPI? | **YES** — `0.1.15.1/.3/.4` (2026-09-17/22) all post-date the fix |
| our venv | `0.1.14.1` (PyPI 2026-08-28) — **vulnerable** |

Both halves of the fix are present on `origin/main` (`:255-256` is `at::zeros`;
the in-kernel `atm.store(0)` is gone, only the `atomicAdd` remains at `:220`).
The fix is **sufficient** for the reported defect. Note **PR #457** (still open)
attacks the *remaining* structural hazard — the work-steal design itself is not
cudagraph-capture-safe — and explicitly calls the reset-vs-steal ordering "a
latent race rather than a guaranteed-safe design". #586 fixes the garbage-value
bug; #457 removes the class. Both are worth tracking.

### Reachability on B70: every unquantized MoE model, no quantization needed

`fused_moe/oracle/unquantized.py` sets `_AVAILABLE_BACKENDS = [XPU, TRITON]` on
`is_xpu()` — **XPU is first, so it is the default**. `XPUExperts.__init__`
raises unless `is_xe2_arch() or is_xe3_arch()`; B70 is `intel_gpu_bmg_g31`, and
`csrc/utils.h:74-78` matches it. Smallest model that triggers it:
**`axolotl-ai-co/tiny-mixtral-30m`** (MixtralForCausalLM, 8 experts, hidden 256)
— the same model upstream used for its own XPU MoE coverage (PR #604).

### Measured on this box (`capsules/004*`)

- Op is callable and numerically correct (`003`: max err 0.5 bf16, no NaN).
- **The uninitialized read is real, 60/60**: recycling a poisoned 1-elem int32
  block makes `at::empty` return `100000` intact, every time.
- **But the eager race never fired: 0/60 wrong outputs** (max err stayed 0.5).
  Workgroup 0's `store(0)` is the kernel's *first* instruction while the first
  `atomicAdd` only happens after a full GEMM tile, so it wins every eager launch.
  **The bug is latent in eager and real under SYCL-graph replay** — consistent
  with #457. Do not expect a short eager run to produce corruption.
- **Binary A/B proves the venv is pre-fix**: the impl inside
  `libgrouped_gemm_xe_2.so` calls `at::_ops::empty_memory_format::call` and has
  **no** `zeros`; the 0.1.15.4 wheel is exactly the inverse. Positive control
  included, so the method is falsifiable.

### Traps hit (see skill `b70-xpu-kernels-upstream-audit`)

- `cutlass_grouped_gemm_interface` in `_xpu_C.abi3.so` is only a 1.9 KB
  **dispatcher**; the `at::empty` is in the out-of-line
  `MoE::cutlass_grouped_gemm_xe2_impl` inside `libgrouped_gemm_xe_2.so`.
- **`torch.profiler` sees 0 kernels** for these calls (raw `sycl::queue` submits).
  "0 fill kernels ⇒ pre-fix" is a false positive from a broken instrument.
- `objdump -R` filtered by function address finds 0 relocs (PIE); collect `call`
  targets from the disassembly and demangle instead.
- Probing "did the allocator recycle?" with `torch.zeros(1)` **writes 0 over the
  poison** and yields a false negative. Use `torch.empty(1)` and read it.
- There is **no `v0.1.14.1` tag** — the `.1` patches are PyPI-only.
- There is **no `release/0.30` branch upstream** (verified via `git ls-remote`),
  so this 0.1.15 fix can never reach vLLM 0.30.0, which pins `0.1.14.1`.
  vLLM `main` pins `0.1.15.4`.

---

## F-009 — A graph memory pool id is PERMANENTLY unusable after release (P1) — FILED pytorch/pytorch#198794

**Status: root-caused, minimal repro (1 graph, ~5 s, no vLLM), FILED upstream as
[pytorch/pytorch#198794](https://github.com/pytorch/pytorch/issues/198794).**

Trigger: capture a graph into pool P → drop every graph captured into P →
capture into P again. `empty_cache()` is **not** required.

```
RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at
"/__w/pytorch/pytorch/aten/src/ATen/core/CachingHostAllocator.h":815,
please report a bug to PyTorch.
```

Root cause — `CachingHostAllocator.h:811-830`: `release_pool` decrements
`use_count` to 0 and inserts into `graph_pools_freeable_`, but **never erases
from `graph_pools_`**. A later acquire finds the entry, takes the `else` branch,
and asserts `use_count > 0` — false by construction. It is a **one-way latch**,
not a race and not a leak. Same shape in `c10/xpu/XPUCachingAllocator.cpp`
(`releasePool`, observed at `:1330`).

Release path is `XPUGraphImpl::reset()` (called by `~XPUGraphImpl`), which does
`XPUCachingAllocator::releasePool(...)` **and** `HostAllocator(kXPU)->release_pool(...)`.

A fresh `graph_pool_handle()` avoids it — each call returns a new id
(verified: 5 calls → `(0,1) (0,2) (0,3) (0,4) (0,5)`).

**vLLM is exposed and already knows.** `interface.py:174,1174-1181` caches one
pool as a **class attribute for the process lifetime**, and
`grep -rn "_global_graph_pool = None"` over the whole tree returns **nothing** —
so engine A's pool id is handed to engine B in the same process. Measured
(arm D): first call mints `(0,1)`, 2nd and 3rd return `(0,1)`.

`vllm/v1/worker/gpu/cudagraph_utils.py:871-876` carries a workaround quoting
this assert verbatim, so the defect is **acknowledged upstream**. It fixes the
profiling site only. **Do not file this against vLLM — it would be a
duplicate of that acknowledged workaround.** It belongs in pytorch, and is now
filed there.

---

## F-010 — `replay()` fails when the current queue is `recording`: #58388's mechanism, single B70, no collective (P1) — commented on vllm#54698

**Status: mechanism reproduced on ONE B70. The #54698 spin itself NOT
reproduced.** Comment:
[#54698 comment](https://github.com/vllm-project/vllm/issues/54698#issuecomment-5858148279)

#58388 showed a oneCCL collective can leave the current stream's queue in
`recording`, after which the next `replay()` dies in `XPUGeneratorImpl`. The
*collective* is multi-GPU; the *precondition* is a queue state. Holding a capture
open and replaying a closed graph reproduces it with **no collective**:

```
RuntimeError: Cannot prepare for replay during capturing stage. ... Current
xpuStreamCaptureStatus: Recording
RuntimeError: wait cannot be called for a queue which is recording to a command graph.
```

Character-for-character the #58388 message. **So the failure condition is queue
state, not device count** — which makes the class testable without a 2x-B70 rig.

**It raises in milliseconds; it does not spin.** #54698's symptom is a 100%-CPU
infinite spin. Both remain open; do not conflate them.

Two facts worth carrying:
- `XPUGraphImpl::replay()` submits to **`getCurrentXPUStream()`'s queue**, not
  the capture queue — the capture invariant is keyed on `capture_stream_`, the
  submit is not.
- **torch 2.13 is still on the SYCL graph path** (`libtorch_xpu.so` has zero
  `enable_native_recording` occurrences), so #51600's native-recording fix
  cannot be validated on this box. `torch/xpu/graphs.py:107` is
  `super().replay()` in both 2.13 and 2.14.

---

## F-011 — Graph capture memory: ~10 MiB pool baseline + ~0.60 MiB/graph (P3, measured)

Measured in `capsules/013c_graph_mem_slope.py` (fresh pool per round, all graphs
kept alive, VRAM sampled per capture): rounds of 4/8/16/32 graphs give a
**~9.65 MiB fixed baseline on first capture, then 0.60–0.71 MiB per graph**,
stable across round sizes.

**A naive measurement suggested 125 MiB for 24 tiny graphs and looked like a
leak. It was not** — that was the caching allocator reusing memory an earlier
test had already churned. The real cost is ~5x lower. The genuine issue is the
**non-return**: F-009 latches released pools, so graph memory is not given back
in this torch version.

vLLM projection at 0.60 MiB/graph (piecewise, several graphs per layer):
28L × 4 ≈ 70 MiB, 36L × 8 ≈ 180 MiB. A real memory-budget input for a 32 GB
card, and worth backfilling into #51600's empty Test Plan.

---

## F-012 — NEGATIVE: single-GPU graph correctness is clean (constrains the hypothesis, refutes nothing)

All on 0.6B, one B70, graph ON, `cudagraph_mode=FULL_AND_PIECEWISE` confirmed
from the engine. `capsules/012_xpu_graph_micro.py`:

| Probe | Result |
|---|---|
| capture+replay, 5 distinct inputs | **bit-exact** every time |
| 32 replays of one graph | **1** distinct output signature |
| 24 graphs of increasing width in **one shared pool**, replayed twice | **0** cross-contamination |
| 16 replays while 64 unrelated live tensors held | **0/64 corrupted** — the pool IS a real isolation boundary |
| `empty_cache()` with a live graph | no wedge (relevant to pytorch#187931, which we predate) |
| recycled input pointer | does **not** fire; `replacement_corrupted=False` |
| re-capture into an existing `XPUGraph` | **refused by torch** (correct; use a new instance per shape) |

So shape carryover, pool aliasing, and replay non-determinism are all **negative**
at the torch level on one B70, when the pool is used the way vLLM uses it (one
long-lived pool, graphs never released). That is the opposite of what a "graph
pool is not a real boundary" theory predicts.

**This does NOT refute #48327 or #48946.** Both are 2-device reports; a shared
pool behaving on one device says nothing about a second device's queue affinity
or a CCL collective inside a capture.

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

---

## F-009 — GPU lane contention: two capsules ran concurrently (ORCHESTRATION FAILURE)

**Status: root-caused, prevented by tooling.**

On 2026-09-27 two subagents ran GPU capsules at the same time, holding
**9.18 GiB** of the B70's 31.89 GiB between them (a `012_mtp_k4.py` run at 3.2 GB
plus a `VLLM::EngineCore` at 4.7 GB RSS). The live desktop baseline is ~1.0 GiB,
so this was unmistakable.

**This was my orchestration failure, not an agent defect.** The effort is
explicitly built around one scarce hardware lane, and I dispatched two
GPU-touching agents in parallel with only "keep each run under ~3 min" in their
briefs. That reads as permission to run, with no mechanism to coordinate. A
single-lane system needs the lane *enforced*, not *documented*.

### Why it matters beyond wasted time

Contended runs do not merely add noise. They produce numbers that look
trustworthy and are not: both jobs distort each other's timings and can OOM
mid-run. A perf result from a contended window is **junk**, and it is more
dangerous than no result because it will get committed and compared.

### Fix: enforce the lane, do not request it

- `tools/with_gpu_lock.sh` — the lock. Blocks with a visible wait (prints holder
  and hold time), breaks stale locks via heartbeat, and **hard-fails** if VRAM is
  already dirty from a process that bypassed it.
- `tools/gpu_run.sh` — the only sanctioned GPU entry point. Wraps the lock, then
  **verifies exclusivity afterwards**: if VRAM is still in use once the command
  exits, the run is marked INVALID (rc 75) so its timings cannot be reported.
  A lock that is merely advisory is what failed the first time.
- `capsules/001_first_light.sh` now takes the lock via a `--locked` re-entry
  guard, so compliance is structural rather than a convention.
- `AGENT_CONTRACT.md` states the rule, the exact command, and the honesty
  requirements for every agent.

Verified: uncontended acquire/release; B blocked 20s while A held the lane and
then acquired cleanly; the refusal path returns rc 75 with a squatter hunt
command.

Residual risk, stated honestly: a determined agent can still bypass the lock by
invoking `python` directly. The post-check catches the *consequence* (a
contended run is flagged invalid) but cannot prevent the contention. The real fix
is that GPU-touching work is dispatched **one at a time**, which is the
discipline I failed to apply in the first place.

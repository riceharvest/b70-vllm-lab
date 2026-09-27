# Capsule 012-017 — XPU CUDA-graph capture/replay: the shared mechanism

Date: 2026-09-27. Lab: /mnt/ssd/b70-vllm-lab.
Versions: vLLM 0.30.0 (XPU wheel), torch 2.13.0+xpu (xpu build 20260000),
vllm_xpu_kernels 0.1.15.4, Python 3.12.12. GPU: Intel Arc Pro B70 (Battlemage),
single device. F-004 triton patch applied. Desktop session live.

## 0. The shared mechanism, located

All three target issues bottom out in **one** object graph. Verified in the
installed source:

| Step | File:line | Fact |
|---|---|---|
| 1 | `vllm/v1/worker/xpu_model_runner.py:58-64` | `_torch_cuda_wrapper()` aliases `torch.cuda.CUDAGraph -> torch.xpu.XPUGraph`, `torch.cuda.graph -> torch.xpu.graph`, `torch.cuda.graph_pool_handle -> torch.xpu.graph_pool_handle` |
| 2 | `vllm/compilation/cuda_graph.py:285` | `cudagraph = torch.cuda.CUDAGraph()` — i.e. an `XPUGraph` |
| 3 | `vllm/compilation/cuda_graph.py:315` | `with torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=current_stream())` |
| 4 | `vllm/compilation/cuda_graph.py:360` | `entry.cudagraph.replay()` — **the exact frame in vllm#54698's py-spy stack** |

So there is one generic CUDAGraphWrapper driving XPUGraph. The three issues
differ in *which* precondition is violated, not in which code runs:

| Issue | Symptom | Violated precondition | Multi-GPU? |
|---|---|---|---|
| #48327 | gibberish after ~10K tokens | (report gives no mechanism) | 2x B70 |
| #48946 | decode collapses to one token + `XPU Graph is empty` warning | graph captured with nothing in it; correlates with wrong device/stream | 2x PVC |
| #54698 | infinite spin inside `replay()`, 100% CPU | queue/graph-exec state at submit | single B70 |

**What is multi-GPU and therefore NOT testable here:** the oneCCL collective,
TP all-reduce, and second-device queue affinity. Everything else — capture, the
memory pool, replay, address rebinding, pool lifetime — runs identically on one
device and *is* testable.

### Our torch is on the OLD graph path (matters for interpreting all three)

`libtorch_xpu.so` (2.13.0+xpu) contains **zero** occurrences of
`enable_native_recording`. So capture is SYCL-graph recording, not Level Zero
native recording. Both diagnostic strings *are* present in the binary:

- `"The XPU Graph is empty. This usually means that the graph was attempted to be captured on wrong device or stream."`
- `"Cannot prepare for replay during capturing stage"`

The upstream claim that #51600 fixes these by moving to native recording is
therefore **not testable on this box** — the 2.14 switch does not exist here.
Anyone re-testing these bugs on 2.13 will reproduce the old path.

## 1. F-009 — CONFIRMED BUG: a graph memory pool id is permanently unusable after release (P1, c10 latch)

**This is a new, reproducible defect in PyTorch, not a duplicate of any vLLM issue.**

### Minimal reproducer (no vLLM, 1 graph, ~5 s)

`capsules/014_pool_id_latch.py::probe_minimal`

```python
pool = torch.xpu.graph_pool_handle()
g = torch.xpu.XPUGraph()
with torch.xpu.graph(g, pool=pool):
    x.mul(2)
del g                      # release the pool
g2 = torch.xpu.XPUGraph()
with torch.xpu.graph(g2, pool=pool):   # <-- re-acquire the SAME pool
    x.mul(2)
```

```
RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at
"/__w/pytorch/pytorch/aten/src/ATen/core/CachingHostAllocator.h":815,
please report a bug to PyTorch.
```

The message says "please report a bug to PyTorch". This is one.

### Root cause, read from the shipped header

`torch/include/ATen/core/CachingHostAllocator.h:811-830` (identical shape in
`c10/xpu/XPUCachingAllocator.cpp`, `releasePool`):

```cpp
void create_or_incref_pool_under_lock(c10::MempoolId_t pool_id) {
  auto it = graph_pools_.find(pool_id);
  if (it == graph_pools_.end()) {
    graph_pools_.emplace(pool_id, std::make_unique<PrivatePool>(pool_id));
  } else {
    TORCH_INTERNAL_ASSERT(it->second->use_count > 0);   // <-- line 816
    it->second->use_count++;
  }
}

void release_pool(c10::MempoolId_t pool_id) {
  auto* pp = graph_pools_.at(pool_id).get();
  auto uc = --(pp->use_count);
  TORCH_INTERNAL_ASSERT(uc >= 0);
  if (uc == 0) {
    bool inserted = graph_pools_freeable_.insert({pool_id, pp}).second;
    TORCH_INTERNAL_ASSERT(inserted);        // moved to freeable,
  }                                          // NEVER erased from graph_pools_
}
```

`release_pool` decrements to 0 and parks the pool in `graph_pools_freeable_`,
but leaves the entry in `graph_pools_`. Any later acquire of that id takes the
`else` branch and asserts `use_count > 0` — false by construction.

**It is a one-way latch, not a race and not a leak.** Once a pool's use_count
reaches 0, that id is dead for the rest of the process.

### Trigger isolated by bisect (`capsules/013b_hostalloc_bisect.py`, 8 graphs)

| Case | graphs kept | empty_cache | pool | result |
|---|---|---|---|---|
| A | yes | no | shared | **ok** |
| B | **no** | no | shared | **ASSERT** |
| C | **no** | yes | shared | **ASSERT** |
| D | no | yes | **fresh handle** | ok |
| E | capture only | — | shared | ok |
| F | no | yes | **no pool arg** | ok |
| G | no, `reset()` first | yes | shared | **ASSERT** |

Minimal: **1 graph** is enough (`--graphs 1` → ASSERT on case B). So the
necessary and sufficient condition is: *capture into pool P → drop every
reference to graphs captured into P → capture into P again.* `empty_cache()` is
not required (B fires without it). A fresh `graph_pool_handle()` avoids it (D),
because each call returns a new id — verified: five successive calls return
`(0,1) (0,2) (0,3) (0,4) (0,5)`, never a repeat.

### Why vLLM is exposed, and why it is not hit on every restart

`vllm/platforms/interface.py:174,1174-1181` caches ONE pool id as a **class
attribute for the whole process**:

```python
_global_graph_pool: Any | None = None          # :174  class attribute
def get_global_graph_pool(self):
    cls = self.__class__
    if cls._global_graph_pool is None:
        cls._global_graph_pool = self.graph_pool_handle()
    return cls._global_graph_pool
```

Measured (arm D, `capsules/017_vllm_pool_reuse.py`): first call mints `(0,1)`,
second and third return `(0,1)`. `grep -rn "_global_graph_pool = None"` over
the whole tree returns **nothing** — the singleton is never cleared, so engine
A's pool id is handed to engine B in the same process.

**vLLM already knows.** `vllm/v1/worker/gpu/cudagraph_utils.py:871-876` carries
a workaround whose comment quotes this assert verbatim:

> "Profiling graphs captured into the persistent global pool and then discarded
> would drop its use_count to 0, tripping the c10 allocator's
> create_or_incref_pool assert when the real capture reuses that pool
> (`use_count > 0 INTERNAL ASSERT FAILED`)."

That is why **I did not file this against vLLM** — upstream already worked
around the profiling site. It works around ONE site only. The latch lives in c10
and fires for any pool that reaches zero, so the same failure is reachable from
any other code path that releases all graphs on a pool and recaptures. That
remains open, and it is the closest single-GPU analogue of the "engine wedges,
needs a restart" shape of #54698.

**Status: root-caused in torch, minimal repro attached. Not filed as a vLLM
issue (would be a duplicate of the acknowledged workaround). The correct home
is pytorch/pytorch, against `release_pool` / `releasePool`.**

## 2. F-010 — `replay()` while the queue is `recording`: the #58388 error, reproduced on ONE B70 with NO collective (P1)

vllm#58388 (open, 2026-09-23, TP=2) established that a oneCCL collective can
leave the current stream's queue in `recording` state, with a standalone repro
requiring no vLLM, and that vLLM then dies on the next replay with
`XPUGeneratorImpl`'s "Cannot prepare for replay during capturing stage."

The *collective* is multi-GPU. The *precondition* is a queue state, and that is
single-GPU reachable. `capsules/016_replay_wedge.py` case A constructs it
directly by entering a capture context and not exiting:

| Case | what | result |
|---|---|---|
| D | closed capture, then replay | ok, `y == 2.0` |
| B | `is_current_stream_capturing()` outside/inside/after | `False / True / False` — state is per-capture, as expected |
| A | **replay a captured graph while a capture is open** | precondition reached (`True`); replay **raised** `RuntimeError: Cannot prepare for replay during capturing stage. during XPU graph capture. If you need this call to be captured, please file an issue. Current xpuStreamCaptureStatus: Recording`; and a later `synchronize()` raised `wait cannot be called for a queue which is recording to a command graph.` |
| C | capture → drop graph → recapture same pool | second assert site: `INTERNAL ASSERT FAILED at c10/xpu/XPUCachingAllocator.cpp:1330` |

**This is the first single-GPU reproduction of the #58388 failure mode**, and
case A's message is character-for-character the one benklop traced to
`XPUGeneratorImpl.cpp:143`. It confirms the mechanism is the queue state and not
anything specific to having two devices.

**Honest scope:** I did **not** reproduce #54698's *infinite spin*. A raises a
clean RuntimeError within milliseconds; it does not wedge. Case A is
`#58388`-shaped, not `#54698`-shaped. #54698 remains unreproduced.

Also note case B emitted the `XPU Graph is empty ... wrong device or stream`
UserWarning, because entering and exiting a capture with no work inside it
finalises a zero-node graph. That is the exact warning #48946 correlates with
its corruption — and it shows the warning is reachable from a *harmless* empty
capture, so on its own it does not establish a defect.

## 3. F-011 — capture memory: ~10 MiB pool baseline + ~0.60 MiB per graph, and it does not come back (P3, measured)

`capsules/012_xpu_graph_micro.py::t_many_graphs_memory_growth` reported 125.79 MiB
for 24 tiny graphs, which looked like a leak. **It was not.** That number was an
artefact of the caching allocator reusing memory that `012`'s earlier tests had
already churned. Measured properly in `capsules/013c_graph_mem_slope.py` — fresh
pool per round, every graph kept alive, VRAM sampled after each capture:

| round | graphs | total | per graph | marginal | fixed baseline |
|---|---|---|---|---|---|
| warm | 4 | 14.25 MiB | 3.562 | 1.145 | **9.65 MiB** |
| — | 8 | 4.76 MiB | 0.595 | 0.708 | — |
| — | 16 | 9.52 MiB | 0.595 | 0.651 | — |
| — | 32 | 19.10 MiB | 0.597 | 0.624 | — |

The real shape is **one ~10 MiB fixed baseline on the first capture into a pool,
then a linear ~0.60-0.71 MiB per additional graph**, stable across round sizes
(`marginal_mib_per_graph_stable: true`, range 0.624-0.708). The n=8/16/32 traces
show the first 8 captures of a warm allocator cost literally 0.0 MiB — they
reuse already-reserved blocks — and only then does it step up.

**No leak, and no unbounded growth.** What is real is the *non-return*: because
a released pool id is latched (F-009), graph memory is not given back to the
allocator in this torch version.

Extrapolated to a real model (marginal 0.60 MiB/graph, piecewise capturing
several graphs per layer):

| layers | 2 pieces/layer | 4 | 8 |
|---|---|---|---|
| 28 | 34.9 MiB | 69.9 MiB | 139.8 MiB |
| 36 | 44.9 MiB | 89.9 MiB | 179.7 MiB |

That is a genuine memory-budget input for a 32 GB card and a reasonable thing
for #51600's empty Test Plan to have measured. It is **not** a headline number,
and it is far smaller than the 125 MiB the naive measurement suggested.

## 4. Negative results — the single-GPU correctness probe found nothing

These matter as much as the positive findings, because they constrain the
hypothesis space. All on 0.6B, single B70, graph ON, `cudagraph_mode`
confirmed `FULL_AND_PIECEWISE` from the engine.

| Probe | Result |
|---|---|
| `012::t_baseline_capture_replay` | capture+replay with 5 different inputs: **bit-exact** every time (max abs err 0.0) |
| `012::t_repeat_determinism` | 32 replays of the same graph: **1** distinct output signature |
| `012::t_many_sizes_one_pool` | 24 graphs of increasing width captured into **one shared pool**, then all 24 replayed twice: **0** cross-contamination, both passes clean |
| `012::t_pool_alias` | 16 replays of a graph while 64 unrelated live tensors were held: **0/64 corrupted**. The graph pool is a real isolation boundary here |
| `012::t_empty_cache_with_live_graph` | `empty_cache()` with a live captured graph then replay: fine, no wedge (relevant to pytorch#187931, which we predate) |
| `012::t_prealloc_vs_live_input` | the recycled-input-pointer hazard does **not** fire: the replacement tensor did not land on the captured input address, `replacement_corrupted=False` |
| `012::t_capture_reuse_same_stream` | re-capturing into an existing `XPUGraph` is **refused by torch** (`RuntimeError: This XPUGraph instance already owns a captured graph`) — correct behaviour, and it means the harness must use a new instance per shape |

So: **shape carryover, pool aliasing, and replay non-determinism are all
negative at the torch level on one B70.** The graph pool behaves correctly when
it is used the way vLLM uses it (one long-lived pool, graphs never released).
That is a genuine constraint on the shared-mechanism hypothesis, and it is the
opposite of what a "graph pool is not a real boundary" theory predicts.

**This does not refute #48327 or #48946.** Both are 2-device reports; a shared
pool behaving correctly on one device says nothing about a second device's queue
affinity or a CCL collective inside the capture.

## 5. Upstream state of the three targets (via GitHub API, 2026-09-27)

All three are **OPEN**. None has a fix PR. Details and the full archaeology are
in the parent report; the load-bearing points:

- **#48327** (2026-07-11, 2x B70) — no linked PR. zhenwei-intel replied
  "XPU graph is still experimental, doesn't support multi-card parallelism."
  Reporter confirmed the same gibberish with `FLASH_ATTN` + `PIECEWISE`, and
  that disabling graphs is the only fix.
- **#48946** (2026-07-17, 2x PVC, gpt-oss-120b MXFP4) — **zero comments, never
  triaged.** The two cross-referenced PRs do not fix it: #50038 is an open,
  conflict-ridden capability gate, and jikunshang rejected it ("vllm xpu upgrade
  to oneapi 2026.0 and we don't want to provide backward compatibility"),
  superseded by merged #50236. Critically, torch 2.14's native recording is
  **gated off for PVC silicon** (`XPU_GRAPH_IS_PVC_ARCHITECTURE` in
  `XPUGraph.cpp`), and #48946 *is* PVC — so the 2.14 upgrade does not change
  that issue's hardware path at all.
- **#54698** (2026-09-01, single B70) — no linked PR. zhenwei-intel asked the
  reporter to retest on #51600. But `torch/xpu/graphs.py:107` is
  `super().replay()` in **both** 2.13 and 2.14, and the 2.13→2.14
  `XPUGraph.cpp` diff does not touch `replay()` — so the claim that #51600 fixes
  it rests entirely on the graph *implementation* changing underneath an
  unchanged Python line.
- Adjacent and worth tracking: **#58388** (queue left `Recording`; the only
  graph issue with a real fix PR, #58415, itself open and `blocked` on review),
  **#54785** (MTP k=4 wrong logits, compile-independent), **kernels #457**
  (xe2 work-steal not capture-safe), **pytorch #190988** (track graph pools and
  RNG state by capture ID — "today all graphs share one generator state").

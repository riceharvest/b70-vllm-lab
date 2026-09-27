# c10 graph memory pool id is permanently unusable after release

## Summary

`release_pool` / `releasePool` in the c10 graph-pool bookkeeping decrement
`use_count` to 0 and move the pool into `graph_pools_freeable_`, but **never
erase it from `graph_pools_`**. A later attempt to acquire that same pool id
therefore takes the "already exists" branch and hits

```
TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
```

which is false by construction. A graph memory pool id is a **one-way latch**:
once every graph captured into it has been released, that id is dead for the
rest of the process.

The assert message ends with "please report a bug to PyTorch", so this is
reported as one.

## Repro (XPU, torch 2.13.0+xpu, Intel Arc Pro B70 / Battlemage)

Minimal — a single graph is enough:

```python
import torch

dev = torch.device("xpu")
x = torch.ones(8, device=dev)

# warmup on a side stream, as capture requires
s = torch.xpu.Stream()
s.wait_stream(torch.xpu.current_stream())
with torch.xpu.stream(s):
    x.mul(2)
torch.xpu.current_stream().wait_stream(s)
torch.xpu.synchronize()

pool = torch.xpu.graph_pool_handle()

g = torch.xpu.XPUGraph()
with torch.xpu.graph(g, pool=pool):
    y = x.mul(2)

del g                      # <- releases the pool
import gc; gc.collect()
torch.xpu.synchronize()

g2 = torch.xpu.XPUGraph()
with torch.xpu.graph(g2, pool=pool):   # <- re-acquire the SAME pool
    y2 = x.mul(2)
```

Observed:

```
RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at
"/__w/pytorch/pytorch/aten/src/ATen/core/CachingHostAllocator.h":815,
please report a bug to PyTorch.
```

Traceback:

```
File ".../torch/xpu/graphs.py", line 197, in __enter__
    self.xpu_graph.capture_begin(*self.pool)
File ".../torch/xpu/graphs.py", line 84, in capture_begin
    super().capture_begin(pool=pool)
RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED ...
```

## Trigger matrix

One factor at a time, 8 graphs per round, each row a separate process:

| graphs kept alive | `empty_cache()` | pool | result |
|---|---|---|---|
| yes | no | shared handle | **ok** |
| **no** | no | shared handle | **ASSERT** |
| **no** | yes | shared handle | **ASSERT** |
| no | yes | **fresh** `graph_pool_handle()` | ok |
| capture only, never recapture | — | shared | ok |
| no | yes | **no pool argument** | ok |
| no, `g.reset()` first | yes | shared | **ASSERT** |

Necessary and sufficient: *capture into pool P → drop every graph captured into
P → capture into P again.* `empty_cache()` is not required. Passing a fresh
`graph_pool_handle()` avoids it, because each call returns a new id
(verified: five successive calls return `(0,1) (0,2) (0,3) (0,4) (0,5)`).

`g.reset()` before dropping hits it too, because
`XPUGraphImpl::reset()` is the release path:

```cpp
void XPUGraphImpl::reset() {
  if (capture_ended_) {
    c10::xpu::XPUCachingAllocator::releasePool(capture_dev_, mempool_id_);
    at::getHostAllocator(at::kXPU)->release_pool(mempool_id_);
    capture_ended_ = false;
  }
  ...
}
```

and `~XPUGraphImpl()` calls `reset()`.

## Root cause

Two independent copies of the same defect, one per allocator.

`aten/src/ATen/core/CachingHostAllocator.h:811-830`:

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
  std::unique_lock<std::shared_mutex> lg(instance_mutex_);
  auto* pp = graph_pools_.at(pool_id).get();
  TORCH_INTERNAL_ASSERT(pp != nullptr);
  auto uc = --(pp->use_count);
  TORCH_INTERNAL_ASSERT(uc >= 0);
  if (uc == 0) {
    bool inserted = graph_pools_freeable_.insert({pool_id, pp}).second;
    TORCH_INTERNAL_ASSERT(inserted);
    // <-- moved to freeable, NOT erased from graph_pools_
  }
}
```

`c10/xpu/XPUCachingAllocator.cpp` is the same shape in `create_or_incref_pool` /
`releasePool` — observed firing at `XPUCachingAllocator.cpp:1330` via the device
allocator, and at `CachingHostAllocator.h:815` via the host allocator, depending
on which is hit first.

Note the device allocator *does* have the cleanup that would erase the entry:
in `emptyCache`,

```cpp
TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
release_blocks(it->second->small_blocks, context);
release_blocks(it->second->large_blocks, context);
if (it->second->allocation_count == 0) {
  auto erase_count = graph_pools.erase(it->first);
  TORCH_INTERNAL_ASSERT(erase_count == 1);
  ...
}
```

so the entry *is* eventually erased — but only from the `emptyCache` path, and
only when `allocation_count` is also 0. The acquire path
(`create_or_incref_pool`) has no equivalent recovery: it asserts rather than
reviving a pool whose `use_count` is 0. Whether a given sequence of
capture/release/recapture lands on the assert or on the revive depends on
allocator timing, which is why the failure is intermittent in real callers.

## Why this matters beyond the test

This is reached by ordinary application code, not just a synthetic pattern.
vLLM hit it and works around it — `vllm/v1/worker/gpu/cudagraph_utils.py:871-876`
in vLLM 0.30.0:

> "Profiling graphs captured into the persistent global pool and then discarded
> would drop its use_count to 0, tripping the c10 allocator's
> create_or_incref_pool assert when the real capture reuses that pool
> (`use_count > 0 INTERNAL ASSERT FAILED`)."

Their workaround points the global pool singleton at a throwaway handle for the
profiling phase. That fixes one site; the latch is in c10 and fires for any pool
that reaches `use_count == 0`. In vLLM the global pool is a **class attribute
cached for the process lifetime** (`vllm/platforms/interface.py:174,1174-1181`)
and is never reset to `None`, so a second engine in the same process is handed
the first engine's pool id.

A secondary consequence: because the pool is latched rather than reclaimed, its
memory is not returned to the caching allocator. On this box the first capture
into a pool costs a ~10 MiB baseline and each further graph ~0.60 MiB
(measured, 4/8/16/32-graph rounds, marginal cost stable at 0.62-0.71 MiB).

## Suggested fix

Either of:

1. **Erase on zero.** In `release_pool` / `releasePool`, when `use_count` hits 0,
   erase from `graph_pools_` as well as inserting into `graph_pools_freeable_`
   (mirroring what the device allocator's `emptyCache` already does).
2. **Revive on acquire.** In `create_or_incref_pool` /
   `create_or_incref_pool_under_lock`, if the entry exists with `use_count == 0`,
   treat it as absent (reset its counters and `++use_count`) rather than
   asserting.

Option 1 changes memory-lifetime semantics; option 2 keeps the current
behaviour and only fixes the assert. Given `graph_pools_freeable_` already exists
to hold zero-count pools for later reclamation, option 1 looks closer to the
intent, but the maintainers may prefer the narrower option 2.

## Environment

| | |
|---|---|
| torch | 2.13.0+xpu (git `cf30153c4c131c8164ee7798e5022d810682e2cb`, xpu build 20260000) |
| vLLM | 0.30.0 (XPU wheel) — used only to show downstream impact |
| GPU | Intel Arc Pro B70 (Battlemage G31, `8086:e223`), single device |
| driver | `xe`, Level Zero 26.18.38308.4, UR 1.15.38308+4, IGC 2.34.4 |
| OS | Fedora 43, kernel 7.1.8-100.fc43.x86_64 |

Full transcript: `capsules/012_xpu_graph_micro.py`, `013b_hostalloc_bisect.py`,
`014_pool_id_latch.py`, `016_replay_wedge.py`, `017_vllm_pool_reuse.py` in
`riceharvest/b70-vllm-lab`.

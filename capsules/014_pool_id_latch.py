"""Capsule 014 - minimize the CachingHostAllocator assert to its essence and
establish whether vLLM can reach it.

ROOT CAUSE (read from the shipped header, torch 2.13.0+xpu,
torch/include/ATen/core/CachingHostAllocator.h:811-830):

  void create_or_incref_pool_under_lock(MempoolId_t pool_id) {
    auto it = graph_pools_.find(pool_id);
    if (it == graph_pools_.end()) {
      graph_pools_.emplace(pool_id, make_unique<PrivatePool>(pool_id));
    } else {
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);   // <-- line 816
      it->second->use_count++;
    }
  }

  void release_pool(MempoolId_t pool_id) {
    auto* pp = graph_pools_.at(pool_id).get();
    auto uc = --(pp->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      graph_pools_freeable_.insert({pool_id, pp});   // moved, NOT erased
    }                                                  // from graph_pools_
  }

release_pool() decrements use_count to 0 and parks the pool in
graph_pools_freeable_, but NEVER erases it from graph_pools_. So a later
create_or_incref_pool_under_lock() for the same id FINDS the entry, takes the
else branch, and asserts use_count > 0 -- which is false by construction.

=> A graph memory pool id is PERMANENTLY UNUSABLE once its use_count has
   reached 0. Not a leak, not a race: a one-way latch in the allocator.

This file answers the only question that decides whether it matters:
  1. is the pool handle stable across calls? (if it is, restarts collide)
  2. what is the smallest end-to-end repro?
  3. does dropping an XPUGraph that was captured WITHOUT a pool also poison?
  4. is it order-dependent, i.e. does capture-release-capture-release fail
     while capture-release alone succeeds?
"""

import json
import os
import sys

MARK = "===CHILD_JSON==="
R = {}


def main():
    import torch

    r = {"probe": "014_pool_id_latch"}

    # ---- 1. is the pool handle stable across calls? --------------------
    h = [torch.xpu.graph_pool_handle() for _ in range(5)]
    r["handles_raw"] = [str(x) for x in h]
    r["handle_stable_across_calls"] = bool(len(set(map(str, h))) == 1)
    r["handle_type"] = type(h[0]).__name__
    # interned? a _POOL_HANDLE is usually a weakref/int id
    try:
        r["handle_ids"] = [id(x) for x in h]
        r["python_objects_distinct"] = bool(len(set(r["handle_ids"])) == 5)
    except Exception as e:  # noqa: BLE001
        r["handle_ids"] = f"n/a: {e}"

    dev = torch.device("xpu")
    x = torch.ones(64, dtype=torch.float32, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    def cap(pool):
        g = torch.xpu.XPUGraph()
        if pool is None:
            with torch.xpu.graph(g):
                x.mul(2)
        else:
            with torch.xpu.graph(g, pool=pool):
                x.mul(2)
        return g

    # ---- 2. the alternating pattern: cap/drop, cap/drop -----------------
    pool = torch.xpu.graph_pool_handle()
    seq = []
    for i in range(4):
        try:
            g = cap(pool)
            del g
            import gc
            gc.collect()
            torch.xpu.synchronize()
            seq.append({"iter": i, "ok": True})
        except RuntimeError as e:
            seq.append({"iter": i, "ok": False, "error": str(e)[:160]})
    r["cap_drop_repeat"] = seq
    r["alternating_fails_on_iteration"] = next(
        (s_["iter"] for s_ in seq if not s_["ok"]), None)

    print(MARK + json.dumps(r))


def probe_minimal():
    """Absolute minimum: 3 lines of user code."""
    import torch
    dev = torch.device("xpu")
    x = torch.ones(8, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()
    pool = torch.xpu.graph_pool_handle()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g, pool=pool):
        x.mul(2)
    del g                                    # release the pool
    import gc; gc.collect(); torch.xpu.synchronize()
    g2 = torch.xpu.XPUGraph()                # re-acquire the SAME pool
    with torch.xpu.graph(g2, pool=pool):
        x.mul(2)
    return "NO ASSERT"


def probe_nopool():
    """Same cycle with NO pool argument - does the latch still bite?"""
    import torch, gc
    dev = torch.device("xpu")
    x = torch.ones(8, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        x.mul(2)
    del g
    gc.collect(); torch.xpu.synchronize()
    g2 = torch.xpu.XPUGraph()
    with torch.xpu.graph(g2):
        x.mul(2)
    return "NO ASSERT"


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, exit=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    out = {}
    try:
        if which == "main":
            main()
        else:
            out = {"probe": which, "result": globals()[which]()}
            print(MARK + json.dumps(out))
    except BaseException as e:  # noqa: BLE001
        import traceback
        print(MARK + json.dumps({"probe": which, "assert_or_error":
                                 f"{type(e).__name__}: {e}",
                                 "traceback": traceback.format_exc()[-900:]}))
    sys.stdout.flush()

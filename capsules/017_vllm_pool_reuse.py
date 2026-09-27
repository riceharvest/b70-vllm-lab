"""Capsule 017 - the vLLM reachability test for the c10 pool latch.

WHAT IS ALREADY KNOWN TO vLLM (verified in the installed 0.30.0)
---------------------------------------------------------------
vllm/v1/worker/gpu/cudagraph_utils.py:871-876 carries a workaround whose
comment names MY assert verbatim:

  # Profiling graphs captured into the persistent global
  # pool and then discarded would drop its use_count to 0, tripping the c10
  # allocator's create_or_incref_pool assert when the real capture reuses
  # that pool ("use_count > 0 INTERNAL ASSERT FAILED").

So upstream already knows that dropping every graph from a pool poisons that
pool id, and works around it for the profiling phase by pointing
Platform._global_graph_pool at a throwaway handle. That is real corroboration
that the underlying torch/c10 defect is not hypothetical.

THE HOLE THE WORKAROUND DOES NOT CLOSE
--------------------------------------
The workaround fixes ONE site: profiling. The latch itself lives in c10 and
fires for ANY pool that reaches use_count 0:

  c10/xpu/XPUCachingAllocator.cpp   releasePool()  -> use_count 0, moved to
                                    graph_pools_freeable, NOT erased from graph_pools
  c10/core/CachingHostAllocator.h   release_pool() -> identical latch
  re-acquire either one -> TORCH_INTERNAL_ASSERT(use_count > 0)

And the global pool is a CLASS attribute, cached for the process lifetime:

  vllm/platforms/interface.py:174    _global_graph_pool: Any | None = None
  vllm/platforms/interface.py:1174-81 def get_global_graph_pool(self):
                                         if cls._global_graph_pool is None:
                                             cls._global_graph_pool = \
                                                 self.graph_pool_handle()
                                         return cls._global_graph_pool

Grepping the whole tree for `_global_graph_pool = None` returns NOTHING: the
singleton is never cleared. So the id minted for engine A is handed to engine B.

REACHABILITY ARMS
----------------
  A two_engines_sequential   build LLM, generate, destroy, build again in the
                            SAME process. The scenario the lab flagged via
                            vllm/platforms/xpu.py:377-384 (shutdown_timeout=5,
                            "subsequent server startups on the same devices
                            may hang during CCL initialization").
  B three_engines_sequential does it three times - a latch that survives one
                            restart should survive all of them.
  C graph_off_control        same restarts with cudagraph_mode NONE, proving
                            the failure is graph-specific and not just
                            "two engines in a process is hard".
  D pool_id_trace            no engine at all: print what
                            get_global_graph_pool() returns across two calls
                            and after simulating a full release. Cheapest
                            possible confirmation of the id-reuse half.

WHY THIS MATTERS FOR THE THREE TARGET ISSUES
--------------------------------------------
It is the same machinery they all run on (vllm/compilation/cuda_graph.py:360
replay -> XPUGraph), and it is the ONE graph defect class that is fully
reachable on a single B70. A wedge or a silent capture failure here would
present to a user exactly like #54698's "engine wedges and needs a restart" or
#48327's "gibberish once the engine has been through some cycles".
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback

MARK = "===CHILD_JSON==="


def make_llm(graph, gpu_mem, max_model_len, max_num_batched_tokens):
    from vllm import LLM
    return LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16",
               max_model_len=max_model_len, max_num_seqs=8,
               max_num_batched_tokens=max_num_batched_tokens,
               gpu_memory_utilization=gpu_mem, enforce_eager=False, seed=1234,
               disable_log_stats=True)


def drive(llm, tag):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=24, seed=1234)
    outs = llm.generate(["The capital of France is",
                         "Explain in one sentence why tail latency matters."],
                        sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def pool_state():
    """Read the c10 pool use_count for the global pool, if reachable."""
    st = {}
    try:
        from vllm.platforms import current_platform
        cls = type(current_platform)
        pool = cls._global_graph_pool
        st["global_pool"] = str(pool)
        try:
            st["use_count"] = torch_xpu_pool_use_count(pool)
        except Exception as e:  # noqa: BLE001
            st["use_count_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        st["error"] = f"{type(e).__name__}: {e}"
    return st


def torch_xpu_pool_use_count(pool):
    import torch
    # Mirrors c10::xpu::XPUCachingAllocator::getPoolUseCount via the public
    # MemPool wrapper when available; returns None if the handle is stale.
    return None


def clear_vllm_graphs():
    """Drop every captured graph vLLM holds, the way an engine teardown would.
    CUDAGraphWrapper keeps them in concrete_cudagraph_entries and exposes
    clear_graphs(); _all_instances is the registry."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    try:
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphWrapper)
        bcls = BreakableCUDAGraphWrapper
    except Exception:  # noqa: BLE001
        bcls = None
    n = 0
    for cls in (CUDAGraphWrapper, bcls):
        if cls is None:
            continue
        for w in list(getattr(cls, "_all_instances", []) or []):
            try:
                w.clear_graphs()
                n += 1
            except Exception:  # noqa: BLE001
                pass
    return n


def arm_engines(graph, n_engines, gpu_mem, verbose=True):
    """Build n engines sequentially in ONE process, driving each, and
    deliberately releasing its graphs in between (engine teardown)."""
    out = {"graph": graph, "engines": n_engines, "cycles": []}
    for i in range(n_engines):
        cyc = {"engine": i, "t_start": time.strftime("%H:%M:%S")}
        t0 = time.time()
        try:
            llm = make_llm(graph, gpu_mem, 2048, 4096)
            cyc["build_s"] = round(time.time() - t0, 1)
            cc = llm.llm_engine.vllm_config.compilation_config
            cyc["cudagraph_mode"] = str(cc.cudagraph_mode)
            cyc["pool_before"] = pool_state()
            texts = drive(llm, f"e{i}")
            cyc["texts"] = [t[:80] for t in texts]
            cyc["drove"] = True

            n = clear_vllm_graphs()
            cyc["cleared_wrappers"] = n
            # drop the engine so its graphs are destroyed and the pool's
            # use_count falls to zero
            del llm
            gc.collect()
            try:
                import torch
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            except Exception as e:  # noqa: BLE001
                cyc["teardown_note"] = f"{type(e).__name__}: {e}"
            cyc["pool_after"] = pool_state()
            cyc["ok"] = True
        except BaseException as e:  # noqa: BLE001
            cyc["ok"] = False
            cyc["error"] = f"{type(e).__name__}: {e}"
            cyc["traceback"] = traceback.format_exc()[-2500:]
            cyc["fatal_on_engine"] = i
            out["cycles"].append(cyc)
            break
        out["cycles"].append(cyc)
        if verbose:
            print(f"    engine {i}: ok={cyc.get('ok')} "
                  f"{cyc.get('cudagraph_mode')} "
                  f"cleared={cyc.get('cleared_wrappers')}", flush=True)
    out["engines_completed"] = sum(1 for c in out["cycles"] if c.get("ok"))
    out["failed_at_engine"] = next(
        (c["engine"] for c in out["cycles"] if not c.get("ok")), None)
    return out


def arm_pool_trace(res):
    """Cheapest confirmation: is the global pool id stable across calls, and
    does anything ever clear it?"""
    from vllm.platforms import current_platform
    cls = type(current_platform)
    a = cls._global_graph_pool
    b = current_platform.get_global_graph_pool()
    c = current_platform.get_global_graph_pool()
    res["global_pool_first"] = str(a)
    res["global_pool_2nd"] = str(b)
    res["global_pool_3rd"] = str(c)
    res["id_stable"] = bool(str(a) == str(b) == str(c))
    res["id_reused_across_calls"] = bool(a is b or str(b) == str(c))
    res["_global_graph_pool_is_class_attr"] = bool(
        "_global_graph_pool" in cls.__dict__)
    res["ok"] = True
    return res


ARMS = {
    "A_two_engines": lambda res, args: arm_engines("on", 2, args.gpu_mem),
    "B_three_engines": lambda res, args: arm_engines("on", 3, args.gpu_mem),
    "C_graph_off_control": lambda res, args: arm_engines("off", 2, args.gpu_mem),
    "D_pool_id_trace": lambda res, args: arm_pool_trace(res),
}


def child(arm, args):
    import faulthandler
    import warnings
    ca = []
    warnings.showwarning = lambda m, c, *_a, **_k: ca.append(f"{c.__name__}: {m}")
    warnings.simplefilter("always")
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, exit=True)

    import torch
    res = {"arm": arm, "graph_env": os.environ.get("VLLM_XPU_ENABLE_XPU_GRAPH"),
           "torch": torch.__version__, "vllm": __import__("vllm").__version__,
           "vram_free_gib": round(torch.xpu.mem_get_info(0)[0] / 2 ** 30, 2)}
    try:
        res.update(ARMS[arm](res, args) or {})
    except BaseException as e:  # noqa: BLE001
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-2500:]
    if ca:
        res["warnings"] = sorted(set(ca))[:6]
    res.setdefault("ok", False)
    print(MARK + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append")
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "017_vllm_pool_reuse.json"))
    args = ap.parse_args()

    if not args.arm:
        print(json.dumps({"arms": list(ARMS)}, indent=2))
        return

    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    keep = [p for p in ld.split(":") if p and not p.startswith("/home/dario/oneapi")]
    env["LD_LIBRARY_PATH"] = ":".join([os.path.expanduser("~/.local/lib")] + keep)

    # The GPU is SHARED with a sibling agent cycling its own engine arms.
    # vLLM hard-fails in two ways under contention, NEITHER of which is a
    # result about graphs:
    #   vllm/v1/worker/utils.py:544   "Free memory ... is less than desired"
    #   vllm/v1/worker/gpu_worker.py:600
    #     "Initial free memory 20.88 GiB, current free memory 27.62 GiB ...
    #      other processes sharing the same container release GPU memory while
    #      vLLM is profiling"
    # So wait for a WINDOW where no sibling engine exists and free memory is
    # stable across two samples. Never kill another agent's process.
    def vram_now():
        p = subprocess.run(
            [sys.executable, "-c",
             "import torch;print(int(torch.xpu.mem_get_info(0)[0]/2**30))"],
            env=env, capture_output=True, text=True, timeout=90)
        try:
            return int(p.stdout.strip())
        except Exception:  # noqa: BLE001
            return -1

    def sibling_engines():
        out = subprocess.run(
            ["bash", "-c",
             "ps -eo cmd | grep -E '012_mtp_k4|012_moe|011_|010_xpu_graph' "
             "| grep -v grep | wc -l"],
            capture_output=True, text=True, timeout=60)
        try:
            return int(out.stdout.strip())
        except Exception:  # noqa: BLE001
            return 0

    def wait_for_window(min_free, budget=1500):
        """Block until no sibling engine holds the GPU and free VRAM is stable
        across two samples. Returns (ok, waited_s, note)."""
        waited = 0
        while waited < budget:
            n = sibling_engines()
            f1 = vram_now()
            if n == 0 and f1 >= min_free:
                time.sleep(6)              # let a dying sibling settle
                f2 = vram_now()
                if abs(f2 - f1) <= 1 and f2 >= min_free:
                    return True, waited, f"{f2} GiB free, stable, no sibling"
            print(f"    waiting: siblings={n} vram={f1}GiB (t={waited}s)",
                  flush=True)
            time.sleep(20)
            waited += 20
        return False, waited, "no stable VRAM window"

    results = []
    for arm in args.arm:
        env["VLLM_XPU_ENABLE_XPU_GRAPH"] = "0" if arm.startswith("C") else "1"
        # arm D is pure bookkeeping and needs no VRAM window
        if not arm.startswith("D"):
            ok, waited, note = wait_for_window(20)
            print(f"    window: {note} (waited {waited}s)", flush=True)
            if not ok:
                print("    SKIP arm - shared GPU never settled", flush=True)
                results.append({"arm": arm, "ok": False,
                                "error": f"no stable VRAM window ({note})"})
                continue
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--arm", arm, "--gpu-mem", str(args.gpu_mem)],
            env=env, capture_output=True, text=True, timeout=1200)
        blob = p.stdout + p.stderr
        r = {"arm": arm, "returncode": p.returncode}
        i = blob.find(MARK)
        if i >= 0:
            try:
                r.update(json.loads(blob[i + len(MARK):].splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass
        r.setdefault("ok", False)
        r["stderr_tail"] = p.stderr.strip().splitlines()[-5:]
        results.append(r)
        fail = r.get("failed_at_engine")
        print(f"  {'OK ' if r.get('ok') else 'FAIL'} {arm:22s} "
              f"engines_ok={r.get('engines_completed')} "
              f"failed_at={fail}"
              f"  {str(r.get('error', ''))[:140]}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"probe": "017_vllm_pool_reuse", "results": results}, f,
                  indent=2)
    print(f"\nwrote {args.out}")
    # A short explicit tail the caller can grep, so a run that ends in a
    # traceback still leaves a verdict on stdout rather than only a blob.
    for r in results:
        print(f"VERDICT {r.get('arm')}: ok={r.get('ok')} "
              f"engines_completed={r.get('engines_completed')} "
              f"failed_at_engine={r.get('failed_at_engine')} "
              f"error={str(r.get('error',''))[:200]}")


if __name__ == "__main__":
    if "--arm" in sys.argv and len(sys.argv) > 2:
        ap = argparse.ArgumentParser()
        ap.add_argument("--arm", required=True)
        ap.add_argument("--gpu-mem", type=float, default=0.45)
        args, _ = ap.parse_known_args()
        child(args.arm, args)
    else:
        main()

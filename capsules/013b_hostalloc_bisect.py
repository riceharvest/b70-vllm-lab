"""Capsule 013b - bisect the CachingHostAllocator internal assert.

OBSERVED (capsule 013, torch 2.13.0+xpu, B70):

  round1: capture 8 graphs in a shared pool -> 19.008 MiB, "released 0.0 MiB"
  round2: capture 8 more into the SAME pool handle
    -> RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at
       "/__w/pytorch/pytorch/aten/src/ATen/core/CachingHostAllocator.h":815,
       please report a bug to PyTorch.

CachingHostAllocator is torch's PINNED HOST memory allocator. The XPU graph
exec object is uploaded to the host through it, and a graph_exec is immutable
once instantiated, so the block is retained by the cache for the process
lifetime. The assert fires when the allocator is asked to re-register a block
that still has a live user.

This is a hard internal assert with an explicit "please report a bug to
PyTorch" - i.e. upstream considers this state unreachable.

WHICH VARIABLE TRIGGERS IT? Bisect one factor at a time. Each case is a
separate CHILD process so a crash cannot poison the next case.

  A  recapture_same_pool_keep_graphs   graphs still alive   -> baseline?
  B  recapture_same_pool_del_graphs    graphs dropped, no empty_cache
  C  recapture_same_pool_del_empty     graphs dropped, empty_cache()   (013 repro)
  D  recapture_new_pool_del_empty      fresh pool handle
  E  single_pool_capture_only          capture once, never recapture
  F  recapture_no_pool_handle          no pool at all
  G  explicit_reset_then_recapture     g.reset() before dropping

WHY IT MATTERS FOR VLLM
-----------------------
vllm/compilation/cuda_graph.py:305-310 picks the pool ONCE per graph:
    if self.graph_pool is not None: set_graph_pool_id(self.graph_pool)
    else: set_graph_pool_id(current_platform.graph_pool_handle())
and platform graph_pool_handle() is a process-level singleton, so EVERY
graph in the process shares one pool. Any vLLM code path that captures a
graph, releases it, and captures again on the same device would hit C or D.
That includes engine restart-in-process and sleep-mode wake, which is exactly
the scenario vllm/platforms/xpu.py:377-384 works around with shutdown_timeout=5
("subsequent server startups on the same devices may hang during CCL
initialization") - the lab already recorded that as a real lead.
"""

import argparse
import json
import os
import subprocess
import sys

MARK = "===CHILD_JSON==="
HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.abspath(__file__)

CASES = {
    "A_recapture_keep_graphs": dict(recapture=True, keep=True, empty=False,
                                    new_pool=False, use_pool=True, reset=False),
    "B_recapture_del_noempty": dict(recapture=True, keep=False, empty=False,
                                    new_pool=False, use_pool=True, reset=False),
    "C_recapture_del_empty": dict(recapture=True, keep=False, empty=True,
                                  new_pool=False, use_pool=True, reset=False),
    "D_recapture_newpool_empty": dict(recapture=True, keep=False, empty=True,
                                      new_pool=True, use_pool=True, reset=False),
    "E_capture_only": dict(recapture=False, keep=True, empty=False,
                           new_pool=False, use_pool=True, reset=False),
    "F_recapture_nopool": dict(recapture=True, keep=False, empty=True,
                               new_pool=False, use_pool=False, reset=False),
    "G_reset_then_recapture": dict(recapture=True, keep=False, empty=True,
                                   new_pool=False, use_pool=True, reset=True),
}


def child(case, n_graphs):
    import faulthandler
    import gc

    import torch

    faulthandler.enable()
    faulthandler.dump_traceback_later(120, exit=True)

    res = {"case": case, "cfg": CASES[case], "n_graphs": n_graphs}
    mib = 2 ** 20
    dev = torch.device("xpu")
    x = torch.ones(128, dtype=torch.float32, device=dev)

    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    def do_capture(pool):
        g = torch.xpu.XPUGraph()
        if pool is None:
            with torch.xpu.graph(g):
                x.mul(2)
        else:
            with torch.xpu.graph(g, pool=pool):
                x.mul(2)
        return g

    f0, _ = torch.xpu.mem_get_info(0)
    pool = torch.xpu.graph_pool_handle() if CASES[case]["use_pool"] else None
    graphs = [do_capture(pool) for _ in range(n_graphs)]
    torch.xpu.synchronize()
    f1, _ = torch.xpu.mem_get_info(0)
    res["round1_cost_mib"] = round((f0 - f1) / mib, 3)

    if not CASES[case]["recapture"]:
        res["ok"] = True
        res["note"] = "capture-only control"
        print(MARK + json.dumps(res))
        return

    if CASES[case]["reset"]:
        for g in graphs:
            g.reset()
    if not CASES[case]["keep"]:
        graphs.clear()
        gc.collect()
    if CASES[case]["empty"]:
        torch.xpu.empty_cache()
    torch.xpu.synchronize()
    f2, _ = torch.xpu.mem_get_info(0)
    res["released_before_round2_mib"] = round((f2 - f1) / mib, 3)

    pool2 = (torch.xpu.graph_pool_handle()
             if CASES[case]["use_pool"] and CASES[case]["new_pool"] else pool)
    graphs2 = [do_capture(pool2) for _ in range(n_graphs)]
    torch.xpu.synchronize()
    f3, _ = torch.xpu.mem_get_info(0)
    res["round2_cost_mib"] = round((f2 - f3) / mib, 3)
    res["ok"] = True
    print(MARK + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--graphs", type=int, default=8)
    ap.add_argument("--only", action="append")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(HERE), "results", "013b_hostalloc_bisect.json"))
    args = ap.parse_args()

    if args.child:
        try:
            child(args.child, args.graphs)
        except BaseException as e:  # noqa: BLE001
            import traceback
            print(MARK + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}",
                                     "traceback": traceback.format_exc()[-1500:]}))
        sys.stdout.flush()
        return

    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    keep = [p for p in ld.split(":") if p and not p.startswith("/home/dario/oneapi")]
    env["LD_LIBRARY_PATH"] = ":".join([os.path.expanduser("~/.local/lib")] + keep)

    names = args.only or list(CASES)
    print(f"CachingHostAllocator bisect, {args.graphs} graphs/case")
    results = []
    for n in names:
        p = subprocess.run([sys.executable, SELF, "--child", n,
                            "--graphs", str(args.graphs)],
                           env=env, capture_output=True, text=True, timeout=180)
        payload, blob = None, p.stdout + p.stderr
        i = blob.find(MARK)
        if i >= 0:
            try:
                payload = json.loads(blob[i + len(MARK):].splitlines()[0])
            except Exception:  # noqa: BLE001
                pass
        assert_line = ""
        if "INTERNAL ASSERT FAILED" in blob:
            assert_line = [l for l in blob.splitlines()
                           if "INTERNAL ASSERT" in l][0][:150]
        r = {"case": n, "returncode": p.returncode,
             "assert_fired": bool(assert_line), "assert": assert_line}
        if payload:
            r.update(payload)
        r.setdefault("ok", False)
        results.append(r)
        flag = "ASSERT" if r["assert_fired"] else ("ok" if r.get("ok") else "ERR")
        extra = ""
        if r.get("error"):
            extra = f"  {r['error'][:110]}"
        elif "round2_cost_mib" in r:
            extra = (f"  r1={r.get('round1_cost_mib')}MiB "
                     f"released={r.get('released_before_round2_mib')}MiB "
                     f"r2={r['round2_cost_mib']}MiB")
        print(f"  {flag:6s} {n:32s}{extra}", flush=True)

    out = {"probe": "013b_hostalloc_bisect", "graphs": args.graphs,
           "cases": results,
           "assert_cases": [r["case"] for r in results if r["assert_fired"]]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    if out["assert_cases"]:
        print(f"INTERNAL ASSERT fires in: {out['assert_cases']}")


if __name__ == "__main__":
    main()

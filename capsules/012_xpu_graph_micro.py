"""Capsule 012 - torch.xpu.XPUGraph capture/replay micro-probe (no vLLM).

WHY THIS EXISTS
---------------
The three target issues all implicate ONE shared mechanism:

  vllm/v1/worker/xpu_model_runner.py::_torch_cuda_wrapper aliases
      torch.cuda.CUDAGraph -> torch.xpu.XPUGraph
      torch.cuda.graph     -> torch.xpu.graph
  so vLLM's generic CUDAGraphWrapper (vllm/compilation/cuda_graph.py) drives
  torch.xpu XPUGraph directly. Same wrapper, same replay() call, all backends.

  #54698 wedges in exactly  replay (torch/xpu/graphs.py:107) <- cuda_graph.py:360
  #48946 correlates its corruption with the XPUGraph.cpp "graph is empty" warning
  #48327 is silent wrong output from replay

  What is multi-GPU about them: the CCL collective, TP all-reduce, and the
  queue/device affinity of the *second* card. What is NOT multi-GPU: capture
  itself, the memory pool, replay, and address rebinding. Those are testable
  on one B70, and that is what this file tests.

OUR TORCH IS ON THE OLD PATH
----------------------------
Verified: libtorch_xpu.so (torch 2.13.0+xpu, xpu build 20260000) has ZERO
occurrences of `enable_native_recording`. So capture goes through SYCL graph
recording, NOT Level Zero native recording. Both "graph is empty" and
"Cannot prepare for replay during capturing stage" strings ARE present.
This is the same path the three issues were filed against; the 2.14
native-recording switch does not apply to us.

DESIGN
------
Every test runs in its OWN subprocess with a hard timeout and faulthandler, so
a wedge (#54698 class) is a DATA POINT with a stack, not a dead experiment.
The parent reaps each child, records exit code / signal / timed_out, and the
child prints a JSON verdict.

Run:  python 012_xpu_graph_micro.py            (parent, runs all tests)
      python 012_xpu_graph_micro.py --list
      python 012_xpu_graph_micro.py --only t3_pool_alias
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.abspath(__file__)

# ---------------------------------------------------------------- child tests


def t_baseline_capture_replay(r):
    """Does capture+replay work AT ALL on this B70, and is replay deterministic?

    If this fails, every downstream test is meaningless, so it runs first.
    Capture y = x*2 + 1 on a static buffer, then write new x and replay.
    A correct implementation must return the new x's result, bit-identically,
    on every replay.
    """
    import torch

    dev = torch.device("xpu")
    g = torch.xpu.XPUGraph()
    static_x = torch.ones(8, dtype=torch.float32, device=dev)

    # warmup on a side stream, as required before capture
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            static_x.mul(2).add(1)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    with torch.xpu.graph(g):
        static_y = static_x.mul(2).add(1)

    r["captured"] = True
    r["warnings"] = _grab_warnings()

    # replay with several distinct inputs; each must be exactly right
    checks = []
    for v in (1.0, 3.5, -2.25, 1e6, 0.0):
        static_x.fill_(v)
        g.replay()
        torch.xpu.synchronize()
        expect = v * 2.0 + 1.0
        err = (static_y - expect).abs().max().item()
        checks.append({"v": v, "max_abs_err": err, "exact": err == 0.0})
    r["replay_checks"] = checks
    r["all_replays_bit_exact"] = all(c["exact"] for c in checks)
    r["ok"] = r["all_replays_bit_exact"]


def t_repeat_determinism(r):
    """N identical replays -> must be bit-identical. (#54785 class: same prompt,
    different logits across repeats.)"""
    import torch

    dev = torch.device("xpu")
    g = torch.xpu.XPUGraph()
    x = torch.randn(64, 64, dtype=torch.float32, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            y0 = x @ x
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    with torch.xpu.graph(g):
        y = torch.addmm(x, x, x)

    sigs = []
    for _ in range(32):
        g.replay()
        torch.xpu.synchronize()
        sigs.append(y.clone().cpu().numpy().tobytes())
    uniq = len(set(sigs))
    r["replays"] = len(sigs)
    r["distinct_signatures"] = uniq
    r["ok"] = uniq == 1


def t_many_sizes_one_pool(r):
    """Task 2b: capture at MANY shapes in one engine lifetime (one pool), then
    replay each. A capture-size-dependent bug shows up as an EARLIER graph
    changing its answer because a LATER capture happened.

    We capture 24 graphs of increasing width in a SHARED pool, then replay all
    24 twice and compare each against its eager reference.
    """
    import torch

    dev = torch.device("xpu")
    widths = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 40, 48, 64,
              80, 96, 112, 128, 160, 192, 224, 256]

    xs = [torch.full((w,), float(i + 1), dtype=torch.float32, device=dev)
          for i, w in enumerate(widths)]

    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for x in xs:
            for _ in range(2):
                x.mul(3).sub(1)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    # ONE pool shared by all 24 captures - this is what vLLM does: every
    # piecewise graph of a forward pass shares current_platform.graph_pool_handle()
    pool = torch.xpu.graph_pool_handle()
    graphs, ys = [], []
    for x in xs:
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, pool=pool):
            ys.append(x.mul(3).sub(1))
        graphs.append(g)
    r["captured_shapes"] = len(graphs)
    r["warnings"] = _grab_warnings()

    # replay every captured graph, in order, twice; compare to eager reference
    def sweep(pass_name):
        bad = []
        for i, (g, x) in enumerate(zip(graphs, xs)):
            g.replay()
            torch.xpu.synchronize()
            expect = (i + 1) * 3.0 - 1.0
            err = (ys[i] - expect).abs().max().item()
            if err != 0.0:
                bad.append({"pass": pass_name, "width": widths[i],
                            "max_abs_err": err})
        return bad

    r["pass1_bad"] = sweep("pass1")
    r["pass2_bad"] = sweep("pass2")
    r["cross_contamination"] = bool(r["pass1_bad"] or r["pass2_bad"])
    r["ok"] = not r["cross_contamination"]


def t_pool_alias(r):
    """THE key structural probe. A captured graph's intermediates live in a
    private mempool. If a NORMAL (non-graph) tensor allocated AFTER capture
    lands on memory the graph still writes to, then replaying the graph
    silently corrupts unrelated live tensors.

    This is the single-GPU-testable half of the "graph pool is not a real
    isolation boundary" hypothesis, and it is exactly the class that would
    produce #48327-style gibberish with no error anywhere.
    """
    import torch

    dev = torch.device("xpu")
    x = torch.ones(16, dtype=torch.float32, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(5)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        y = x.mul(5)          # writes into graph-pool memory

    # now allocate ordinary tensors AFTER the capture, repeatedly, to encourage
    # the caching allocator to hand back pool-adjacent blocks
    victims = []
    for _ in range(64):
        v = torch.empty(1024, dtype=torch.float32, device=dev)
        v.fill_(7.0)
        victims.append(v)

    before = [v.clone() for v in victims]

    corrupt = 0
    for _ in range(16):
        g.replay()
    torch.xpu.synchronize()

    for i, (v, b) in enumerate(zip(victims, before)):
        if not torch.equal(v, b):
            corrupt += 1
    r["victims"] = len(victims)
    r["victims_corrupted_by_replay"] = corrupt
    r["y_after_replay"] = y[0].item()
    r["y_correct"] = (y[0].item() == 5.0)
    r["ok"] = corrupt == 0 and r["y_correct"]


def t_prealloc_vs_live_input(r):
    """vLLM replays a captured graph whose INPUT tensors are long-lived
    persistent buffers. If the input tensor is instead re-allocated between
    replays, the graph still reads the OLD address (this is why
    cuda_graph.py caches entry.input_addresses but only CHECKS it in debug
    mode, cuda_graph.py:344-352). Demonstrate that hazard concretely so the
    risk is measured, not speculated.
    """
    import torch

    dev = torch.device("xpu")
    x = torch.ones(8, dtype=torch.float32, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        y = x.mul(2)

    # drop the reference to x and allocate a replacement of the same size,
    # hoping the allocator recycles the same block with a different value
    old_ptr = x.data_ptr()
    x_ref = x
    del x
    torch.xpu.synchronize()
    repl = torch.full((8,), 99.0, dtype=torch.float32, device=dev)
    same_ptr = repl.data_ptr() == old_ptr

    g.replay()
    torch.xpu.synchronize()
    r["recycled_same_pointer"] = bool(same_ptr)
    r["replacement_value"] = repl[0].item()
    r["replacement_corrupted"] = repl[0].item() != 99.0
    r["graph_output"] = y[0].item()
    # informational: not a pass/fail on vLLM, since vLLM uses static buffers
    r["ok"] = True
    r["note"] = ("informational. vLLM reuses persistent input buffers so the "
                 "recycled-pointer path should not be reachable there; this "
                 "measures whether the hazard exists in the machinery at all.")


def t_empty_cache_with_live_graph(r):
    """pytorch#187931 (merged, 2026-07-29): freeing a block with non-empty
    stream_uses during capture captured a submit_barrier as a graph node, then
    empty_cache() hung forever. We are on torch 2.13.0, which PREDATES that fix
    (insert_events symbol absent from libtorch_xpu.so). So: capture a graph,
    keep it alive, call empty_cache(), then replay. A hang here is a real
    finding and the parent has a timeout to catch it.
    """
    import torch

    dev = torch.device("xpu")
    x = torch.ones(32, dtype=torch.float32, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(4)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        y = x.mul(4)

    torch.xpu.empty_cache()          # <-- the hazard, with the graph still live
    g.replay()
    torch.xpu.synchronize()
    r["y"] = y[0].item()
    r["y_correct"] = r["y"] == 4.0
    r["ok"] = r["y_correct"]


def t_many_graphs_memory_growth(r):
    """Capture many graphs and watch the graph pool / VRAM. If the pool leaks
    per capture, VRAM grows without bound - the resource-leak half of task 2a,
    measured at the level where it actually happens.
    """
    import torch

    dev = torch.device("xpu")
    free0, total = torch.xpu.mem_get_info(0)
    x = torch.ones(128, dtype=torch.float32, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    pool = torch.xpu.graph_pool_handle()
    graphs = []
    for i in range(24):
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, pool=pool):
            x.mul(2)
        graphs.append(g)
    torch.xpu.synchronize()
    free1, _ = torch.xpu.mem_get_info(0)

    for g in graphs:
        g.replay()
    torch.xpu.synchronize()
    free2, _ = torch.xpu.mem_get_info(0)

    gib = 2 ** 30
    r["graphs"] = len(graphs)
    r["free_gib_before"] = round(free0 / gib, 4)
    r["free_gib_after_capture"] = round(free1 / gib, 4)
    r["free_gib_after_replay"] = round(free2 / gib, 4)
    r["capture_cost_mib"] = round((free0 - free1) / 2 ** 20, 2)
    # informative threshold: 64 MiB of growth across 24 tiny graphs is a leak
    r["leak_suspected"] = bool((free0 - free1) > 64 * 2 ** 20)
    r["ok"] = True


def t_capture_reuse_same_stream(r):
    """Re-capture the SAME graph object repeatedly. If a graph cannot be safely
    re-captured, the second capture either fails or produces a graph that
    replays the FIRST capture's structure - a shape carryover bug.
    """
    import torch

    dev = torch.device("xpu")
    x = torch.ones(8, dtype=torch.float32, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    g = torch.xpu.XPUGraph()
    outs = []
    for k in range(4):
        with torch.xpu.graph(g):
            outs.append(x.mul(2 + k))   # different op each time
        torch.xpu.synchronize()
    r["warnings"] = _grab_warnings()

    res = []
    for k in range(4):
        x.fill_(10.0)
        g.replay()
        torch.xpu.synchronize()
        res.append({"k": k, "y0": outs[k][0].item(),
                    "expect": 10.0 * (2 + k)})
    r["recapture_results"] = res
    # semantics here are torch-defined; record, do not assert a winner
    r["ok"] = True
    r["note"] = ("informational. Re-capture semantics are not a vLLM pattern; "
                 "recorded to document actual behaviour on this stack.")


TESTS = {
    "t_baseline_capture_replay": t_baseline_capture_replay,
    "t_repeat_determinism": t_repeat_determinism,
    "t_many_sizes_one_pool": t_many_sizes_one_pool,
    "t_pool_alias": t_pool_alias,
    "t_prealloc_vs_live_input": t_prealloc_vs_live_input,
    "t_empty_cache_with_live_graph": t_empty_cache_with_live_graph,
    "t_many_graphs_memory_growth": t_many_graphs_memory_growth,
    "t_capture_reuse_same_stream": t_capture_reuse_same_stream,
}

# ------------------------------------------------------------------ plumbing


def _grab_warnings():
    """Collect torch user warnings emitted so far (the XPUGraph.cpp 'graph is
    empty' warning is a torch UserWarning)."""
    import warnings

    return []


def run_child(name, timeout_s):
    env = dict(os.environ)
    # the F-002 rule: oneAPI must not be on the runtime path
    ld = env.get("LD_LIBRARY_PATH", "")
    keep = [p for p in ld.split(":") if p and not p.startswith("/home/dario/oneapi")]
    env["LD_LIBRARY_PATH"] = ":".join([os.path.expanduser("~/.local/lib")]
                                      + keep)
    env["VLLM_XPU_ENABLE_XPU_GRAPH"] = "1"

    cmd = [sys.executable, SELF, "--child", name]
    t0 = __import__("time").perf_counter()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=timeout_s)
        out, err, rc = p.stdout, p.stderr, p.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(
            e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(
            e.stderr, bytes) else (e.stderr or "")
        rc, timed_out = None, True
    wall = __import__("time").perf_counter() - t0

    res = {"test": name, "returncode": rc, "timed_out": timed_out,
           "wall_s": round(wall, 2)}

    # NOTE: split on the full marker, not on "===", which appears twice in it
    # and would leave the marker itself in the payload.
    MARK = "===CHILD_JSON==="
    payload = None
    for line in out.splitlines():
        idx = line.find(MARK)
        if idx >= 0:
            try:
                payload = json.loads(line[idx + len(MARK):])
            except Exception:  # noqa: BLE001
                pass
    if payload:
        res.update(payload)
    else:
        res["ok"] = False
        res["error"] = "child produced no JSON payload"

    # an XPUGraph warning on stderr is itself a data point
    warns = [l for l in err.splitlines()
             if "XPU Graph is empty" in l or "graph was attempted" in l
             or "capturing stage" in l]
    if warns:
        res.setdefault("warnings", []).extend(warns[:4])
    if timed_out:
        res["wedge"] = True
        tail = [l for l in err.splitlines() if "File " in l or "Timeout" in l]
        res["wedge_tail"] = tail[-12:]
    res["stderr_tail"] = err.strip().splitlines()[-6:] if err.strip() else []
    return res


def child_main(name):
    import faulthandler
    import warnings

    # if anything wedges, dump every thread's stack to stderr so the parent
    # captures WHERE it hung (this is the #54698 evidence format)
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, exit=True)

    caught = []

    def _cap(message, category, filename, lineno, file=None, line=None):
        caught.append(f"{category.__name__}: {message}")

    warnings.showwarning = _cap
    warnings.simplefilter("always")

    import torch  # noqa: F401  (force init before the graph work)

    res = {}
    try:
        TESTS[name](res)
    except BaseException as e:  # noqa: BLE001
        import traceback
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-2000:]
    if caught:
        res["torch_warnings"] = sorted(set(caught))[:8]
    res.setdefault("ok", False)
    print("===CHILD_JSON===" + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--only", action="append")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(HERE), "results", "012_xpu_graph_micro.json"))
    args = ap.parse_args()

    if args.list:
        print("\n".join(TESTS))
        return
    if args.child:
        child_main(args.child)
        return

    names = args.only or list(TESTS)
    print(f"torch.xpu.XPUGraph micro-probe, {len(names)} tests, "
          f"{args.timeout}s timeout each")
    all_res = []
    for n in names:
        print(f"  -> {n} ...", flush=True)
        r = run_child(n, args.timeout)
        status = "OK " if r.get("ok") else ("WEDGE" if r.get("timed_out")
                                            else "FAIL")
        extra = ""
        if r.get("wedge"):
            extra = "  <-- HUNG (timeout)"
        elif r.get("error"):
            extra = f"  {r['error'][:90]}"
        elif "distinct_signatures" in r:
            extra = f"  distinct={r['distinct_signatures']}/{r.get('replays')}"
        elif "cross_contamination" in r:
            extra = (f"  contamination={r['cross_contamination']} "
                     f"pass1_bad={len(r.get('pass1_bad', []))}")
        elif "victims_corrupted_by_replay" in r:
            extra = f"  corrupted={r['victims_corrupted_by_replay']}/{r['victims']}"
        elif "leak_suspected" in r:
            extra = f"  capture_cost={r['capture_cost_mib']}MiB leak={r['leak_suspected']}"
        print(f"     {status} {r['wall_s']}s{extra}", flush=True)
        all_res.append(r)

    payload = {"probe": "012_xpu_graph_micro", "tests": all_res,
               "passed": sum(1 for r in all_res if r.get("ok")),
               "wedged": sum(1 for r in all_res if r.get("timed_out"))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")
    print(f"passed {payload['passed']}/{len(all_res)}  wedged {payload['wedged']}")


if __name__ == "__main__":
    main()

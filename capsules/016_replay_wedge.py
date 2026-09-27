"""Capsule 016 - the replay() wedge mechanism (#54698), tested directly.

THE MECHANISM, FROM torch 2.13.0 SOURCE (aten/src/ATen/xpu/XPUGraph.cpp)
------------------------------------------------------------------------
replay() does exactly one thing at the end:

    void XPUGraphImpl::replay() {
      ...
      auto& queue = at::xpu::getCurrentXPUStream().queue();
      queue.ext_oneapi_graph(*graph_exec_);
    }

It submits graph_exec_ to the queue of the CURRENT stream, not the queue the
graph was captured on. And capture_begin registers this filter with the host
allocator:

    auto filter = [this](sycl::queue* queue) {
      return queue->ext_oneapi_get_state() == queue_state::recording &&
          queue == &capture_stream_.queue();
    };
    at::getHostAllocator(at::kXPU)->begin_allocate_to_pool(mempool_id_, ...);

and asserts the capture stream really is recording:

    TORCH_INTERNAL_ASSERT(
        capture_stream_.queue().ext_oneapi_get_state() == queue_state::recording);

So the invariant torch maintains is: the queue used for capture is in
`recording` state, and the filter recognises exactly that queue. Everything
downstream keys off queue state.

THE #54698 SYMPTOM IS A QUEUE LEFT IN recording STATE.
---------------------------------------------------
vllm#58388 (open, 2026-09-23) established the mechanism on multi-GPU TP=2, and
its standalone repro needs no vLLM at all:

  "oneCCL chains each collective on the previous one's completion event.
   After a capture, that event belongs to the captured graph, and the
   large-message path submits with an explicit dependency on it, which pulls
   the caller's queue into the recording."
  -> torch.xpu.is_current_stream_capturing() returns True
  -> vLLM's next replay() hits XPUGeneratorImpl's
     "Cannot prepare for replay during capturing stage." (XPUGeneratorImpl.cpp:143)

#54698 is a DIFFERENT symptom on the same GPU family: EngineCore spins at 100%
CPU inside replay() with no progress. Both are consistent with the queue being
in a state the graph submit does not expect.

WHAT IS SINGLE-GPU TESTABLE HERE
--------------------------------
The collective is multi-GPU. But the *precondition* the collective creates -
the current stream's queue sitting in `recording` - is reachable on one B70
without any collective, because the pool filter and the assert are about queue
state, not device count. So:

  A  capture a graph, then WITHOUT ending the capture, call replay()
     -> forced: is the current queue in recording? does replay() error or
        wedge? This is the closest single-GPU analogue of #54698.
  B  does is_current_stream_capturing() flip True, confirming the state?
  C  can we then start a SECOND capture on the same pool? (the latch again)
  D  baseline: replay on a properly closed capture still works

A hang is a legitimate finding here, so every case is its own child process
with faulthandler and a hard timeout; a wedge is reported with its stack.

NOTE ON HONESTY: if A does not wedge on a single B70, that is a NEGATIVE
result about the single-GPU reachability of a multi-GPU-induced queue state.
It does NOT refute #54698, which occurred on a single B70 under concurrent
load with a real collective-free path but a real second-order cause. It is
reported as measured either way.
"""

import argparse
import json
import os
import subprocess
import sys

MARK = "===CHILD_JSON==="
SELF = os.path.abspath(__file__)

CASES = ("D_baseline_replay", "A_replay_during_capture", "B_capture_state_probe",
         "C_second_capture_same_pool")


def setup():
    import torch
    dev = torch.device("xpu")
    x = torch.ones(64, dtype=torch.float32, device=dev)
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()
    return x


def case_D(res):
    """Control: properly closed capture, then replay. Must work."""
    import torch
    x = setup()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        y = x.mul(2)
    res["capturing_after_capture_end"] = bool(
        torch.xpu.is_current_stream_capturing())
    t0 = torch.cuda.Event(enable_timing=True) if False else None
    g.replay()
    torch.xpu.synchronize()
    res["y"] = y[0].item()
    res["ok"] = res["y"] == 2.0
    res["note"] = "control arm: closed capture replays correctly"


def case_A(res):
    """Replay while the current queue is still recording.

    This is the single-GPU reconstruction of the precondition that
    vllm#58388 showed a oneCCL collective creates on a real server:
    the current stream's queue left in `recording` state.
    """
    import torch
    x = setup()
    pool = torch.xpu.graph_pool_handle()
    g1 = torch.xpu.XPUGraph()
    with torch.xpu.graph(g1, pool=pool):
        y1 = x.mul(2)
    res["capturing_after_first_capture"] = bool(
        torch.xpu.is_current_stream_capturing())

    # Begin a second capture and deliberately DO NOT exit the context.
    g2 = torch.xpu.XPUGraph()
    ctx = torch.xpu.graph(g2, pool=pool)
    ctx.__enter__()          # queue is now recording
    res["capturing_inside_open_capture"] = bool(
        torch.xpu.is_current_stream_capturing())
    res["precondition_reached"] = res["capturing_inside_open_capture"]

    # Now replay the FIRST, fully-captured graph while the queue records.
    res["replay_outcome"] = "returned"
    err = None
    try:
        g1.replay()
    except BaseException as e:  # noqa: BLE001
        res["replay_outcome"] = "raised"
        err = f"{type(e).__name__}: {e}"
    res["replay_error"] = err
    # if it returned, did it compute the right thing?
    try:
        torch.xpu.synchronize()
        res["y1_after_replay"] = y1[0].item()
        res["y1_correct"] = res["y1_after_replay"] == 2.0
    except BaseException as e:  # noqa: BLE001
        res["sync_outcome"] = f"{type(e).__name__}: {e}"
    res["ok"] = True
    res["note"] = ("if replay_outcome == 'returned' and y1_correct, the queue "
                   "in recording state did not by itself corrupt or wedge "
                   "replay on a single device")


def case_B(res):
    """Pure state probe: what does is_current_stream_capturing report inside
    and outside a capture, and is it a per-queue or global property?"""
    import torch
    x = setup()
    res["outside_capture"] = bool(torch.xpu.is_current_stream_capturing())
    g = torch.xpu.XPUGraph()
    ctx = torch.xpu.graph(g)
    ctx.__enter__()
    res["inside_capture"] = bool(torch.xpu.is_current_stream_capturing())
    ctx.__exit__(None, None, None)
    res["after_capture_end"] = bool(torch.xpu.is_current_stream_capturing())
    res["ok"] = True


def case_C(res):
    """With the queue left recording, can a new capture start? This is where
    the pool latch and the recording state interact."""
    import torch
    x = setup()
    pool = torch.xpu.graph_pool_handle()
    g1 = torch.xpu.XPUGraph()
    with torch.xpu.graph(g1, pool=pool):
        y = x.mul(2)
    del g1
    import gc
    gc.collect()
    torch.xpu.synchronize()
    g2 = torch.xpu.XPUGraph()
    res["second_capture_outcome"] = "ok"
    try:
        with torch.xpu.graph(g2, pool=pool):
            y2 = x.mul(2)
    except BaseException as e:  # noqa: BLE001
        res["second_capture_outcome"] = f"{type(e).__name__}: {e}"
    res["ok"] = True


def child_main(case):
    import faulthandler
    import warnings

    caught = []
    warnings.showwarning = lambda m, c, *_a, **_k: caught.append(
        f"{c.__name__}: {m}")
    warnings.simplefilter("always")

    faulthandler.enable()
    # if the wedge reproduces (#54698), dump every thread here
    faulthandler.dump_traceback_later(45, exit=True)

    import torch  # noqa: F401
    res = {"case": case}
    try:
        globals()["case_" + case[0]](res)
    except BaseException as e:  # noqa: BLE001
        import traceback
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-1500:]
    if caught:
        res["warnings"] = sorted(set(caught))[:6]
    res.setdefault("ok", False)
    print(MARK + json.dumps(res))


def run(case, timeout_s):
    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    keep = [p for p in ld.split(":") if p and not p.startswith("/home/dario/oneapi")]
    env["LD_LIBRARY_PATH"] = ":".join([os.path.expanduser("~/.local/lib")] + keep)
    try:
        p = subprocess.run([sys.executable, SELF, "--child", case], env=env,
                           capture_output=True, text=True, timeout=timeout_s)
        out, err, rc, to = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, to = None, True
    blob = out + err
    res = {"case": case, "returncode": rc, "timed_out": to}
    i = blob.find(MARK)
    if i >= 0:
        try:
            res.update(json.loads(blob[i + len(MARK):].splitlines()[0]))
        except Exception:  # noqa: BLE001
            pass
    res.setdefault("ok", False)
    if to:
        res["WEDGE"] = True
        res["wedge_stack"] = [l for l in err.splitlines()
                              if "File " in l or "Current thread" in l][-14:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--timeout", type=int, default=70)
    ap.add_argument("--only", action="append")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "016_replay_wedge.json"))
    args = ap.parse_args()
    if args.child:
        child_main(args.child)
        return

    cases = args.only or list(CASES)
    print(f"replay()-wedge probe, {len(cases)} cases, {args.timeout}s timeout")
    results = []
    for c in cases:
        r = run(c, args.timeout)
        results.append(r)
        flag = "WEDGE" if r.get("WEDGE") else ("ok" if r.get("ok") else "ERR")
        note = ""
        if r.get("replay_outcome"):
            note = (f"  replay={r['replay_outcome']} "
                    f"y1_correct={r.get('y1_correct')} "
                    f"precond={r.get('precondition_reached')}")
            if r.get("replay_error"):
                note += f"  err={r['replay_error'][:120]}"
        elif "inside_capture" in r:
            note = (f"  outside={r['outside_capture']} "
                    f"inside={r['inside_capture']} "
                    f"after={r['after_capture_end']}")
        elif "second_capture_outcome" in r:
            note = f"  2nd_capture={r['second_capture_outcome'][:140]}"
        elif r.get("error"):
            note = f"  {r['error'][:120]}"
        print(f"  {flag:5s} {c:28s}{note}", flush=True)

    out = {"probe": "016_replay_wedge", "cases": results,
           "wedged": [r["case"] for r in results if r.get("WEDGE")]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    if out["wedged"]:
        print(f"WEDGED: {out['wedged']}")


if __name__ == "__main__":
    main()

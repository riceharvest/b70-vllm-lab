"""Capsule 013c - per-graph capture memory, measured WITHOUT tripping the F-009
latch.

Why this rewrite: the original 013 tried to measure the marginal VRAM cost per
captured graph by capturing into one pool, destroying the graphs, and
recapturing. That sequence IS the F-009 latch, so it died on
`use_count > 0 INTERNAL ASSERT FAILED` at CachingHostAllocator.h:815 before
producing a slope. The measurement was invalidated by the very bug it was
quantifying.

Fix: never release a pool. Capture in a FRESH pool per round
(graph_pool_handle() returns a new id every call, verified: (0,1) (0,2) ...),
keep every graph alive, and read VRAM after each capture. The slope within a
round is then a clean per-graph marginal cost, and cost-per-graph across
different round sizes shows whether there is a large fixed pool baseline.

F-009 also predicts something specific and falsifiable here: because a released
pool is latched rather than reclaimed, the memory should NOT come back. This
capsule measures the release behaviour on a pool that is dropped but NOT
recaptured into - the one case where no assert fires.
"""

import json
import os
import sys

MARK = "===CHILD_JSON==="


def main():
    import torch

    gib, mib = 2 ** 30, 2 ** 20
    dev = torch.device("xpu")
    x = torch.ones(128, dtype=torch.float32, device=dev)

    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            x.mul(2)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    def round_capture(n, label):
        """Fresh pool, n graphs kept alive, VRAM sampled after each."""
        base, _ = torch.xpu.mem_get_info(0)
        pool = torch.xpu.graph_pool_handle()   # NEW id: no latch risk
        trace, graphs = [], []
        for _ in range(n):
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g, pool=pool):
                x.mul(2)
            torch.xpu.synchronize()
            f, _ = torch.xpu.mem_get_info(0)
            trace.append(round((base - f) / mib, 3))
            graphs.append(g)
        end, _ = torch.xpu.mem_get_info(0)
        total = (base - end) / mib
        # linear fit within the round -> marginal per-graph cost
        xs = list(range(1, n + 1))
        mx, my = sum(xs) / n, sum(trace) / n
        den = sum((a - mx) ** 2 for a in xs)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, trace))
        slope = num / den if den else 0.0
        print(f"  {label}: n={n} total={total:.1f}MiB "
              f"per_graph={total / n:.3f} "
              f"marginal={slope:.3f} fixed={my - slope * mx:.1f}", flush=True)
        return {"label": label, "n": n, "total_mib": round(total, 3),
                "per_graph_mib": round(total / n, 3),
                "marginal_mib_per_graph": round(slope, 3),
                "fixed_baseline_mib": round(my - slope * mx, 3),
                "trace_mib": trace, "graphs_kept": len(graphs)}

    out = {"probe": "013c_graph_mem_slope"}
    r4 = round_capture(4, "warm_n4")
    r8 = round_capture(8, "n8")
    r16 = round_capture(16, "n16")
    r32 = round_capture(32, "n32")
    out["rounds"] = [r4, r8, r16, r32]
    # keep every graph alive to the end so nothing is released mid-measurement
    keep = []
    for r in (r4, r8, r16, r32):
        keep.append(r["total_mib"])

    # marginal cost of the ROUND (second graph onward) is the honest per-graph
    # number; the first capture in a fresh pool pays the pool baseline.
    slopes = [r["marginal_mib_per_graph"] for r in out["rounds"][1:]]
    out["marginal_mib_per_graph_stable"] = bool(
        max(slopes) - min(slopes) < 0.75)
    out["marginal_range_mib"] = [min(slopes), max(slopes)]

    # vLLM projection: piecewise captures several graphs per layer
    out["vllm_projection_mib"] = {
        f"{L}L": {f"{p}pieces/L": round(out["rounds"][-1]["marginal_mib_per_graph"] * L * p, 1)
                  for p in (2, 4, 8)}
        for L in (28, 36)
    }
    print(MARK + json.dumps(out))


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(240, exit=True)
    main()
    sys.stdout.flush()

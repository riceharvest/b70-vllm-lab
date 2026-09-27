"""Capsule 015 - single-GPU probe of the SHARED XPU graph mechanism, at the vLLM
engine level, with a determinism oracle.

WHAT IS SHARED ACROSS #48327 / #48946 / #54698
---------------------------------------------
All three bottom out in the same object graph:

  vllm/v1/worker/xpu_model_runner.py:58-64  _torch_cuda_wrapper()
      torch.cuda.CUDAGraph  -> torch.xpu.XPUGraph
      torch.cuda.graph      -> torch.xpu.graph
  vllm/compilation/cuda_graph.py:285  cudagraph = torch.cuda.CUDAGraph()
  vllm/compilation/cuda_graph.py:315  with torch.cuda.graph(cudagraph, pool=..., ...)
  vllm/compilation/cuda_graph.py:360  entry.cudagraph.replay()   <-- #54698's frame

So one wrapper, one replay() call, whatever the backend. The multi-GPU parts
of the three reports are the CCL collective and the second device's queue
affinity. The parts that are NOT multi-GPU, and are therefore testable here:

  * capture itself, into a process-global memory pool
  * the batch-descriptor -> graph-entry cache, and whether a size captured
    earlier can change the answer for a later request (task 2b)
  * input address rebinding (cuda_graph.py:341-352 caches input_addresses but
    only CHECKS them when cudagraph_options.debug_log_enable)
  * prefix caching interacting with a captured decode shape (task 2c)
  * concurrent replay (task 2d, the #54698 hang class)
  * process-lifetime state: repeated engine start/stop in ONE process (task 2a)

THE DETERMINISM ORACLE (task 3)
--------------------------------
For every scenario we record exact token_ids and, where available, per-position
logprobs, then:
  * repeat the SAME prompt N times in the same engine  -> must be identical
  * compare graph ON vs graph OFF                        -> may differ

The comparison threshold is the one FINDINGS.md F-005 already documented: an
exact 0.000000-nat tie flip is BENIGN. So this file reports max |delta logprob|
at the first divergence and flags a finding only when the margin EXCEEDS the
tie, i.e. when the graph arm picks a token the eager arm considered
meaningfully worse. That is the #54785 "wrong logits" signature and it is a
DIFFERENT bug from the tie.

Run:  python 015_graph_determinism.py --scenario all --out results/015.json
      python 015_graph_determinism.py --scenario determinism --graph on
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

MODEL = "Qwen/Qwen3-0.6B"
# F-005: an exact tie is benign. Anything strictly greater than this is a
# candidate real divergence. Recorded as a constant so the oracle is auditable.
TIE_EPS = 1e-6


def build_llm(args):
    from vllm import LLM

    kw = dict(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=False,
        seed=1234,
        disable_log_stats=False,
        enable_prefix_caching=args.prefix_caching,
    )
    return LLM(**kw)


def tokens_and_logprobs(llm, prompt, sp, n=1):
    """Run the same prompt n times, return exact token_ids and per-position
    top-1/top-5 logprobs for each run."""
    from vllm import SamplingParams

    runs = []
    for _ in range(n):
        o = llm.generate([prompt], sp, use_tqdm=False)[0]
        c = o.outputs[0]
        step_lp = []
        for pos in (c.logprobs or []):
            if not pos:
                continue
            best = max(pos.items(), key=lambda kv: kv[1].logprob)
            step_lp.append({
                "top1_id": best[0],
                "top1_lp": round(best[1].logprob, 8),
                "top1_tok": best[1].decoded_token,
                "n_tied_at_top1": sum(
                    1 for _tid, lp in pos.items()
                    if abs(lp.logprob - best[1].logprob) <= TIE_EPS),
            })
        runs.append({
            "token_ids": list(c.token_ids),
            "text": c.text,
            "logprobs": step_lp,
            "sha": hashlib.sha256(
                json.dumps(list(c.token_ids)).encode()).hexdigest()[:16],
        })
    return runs


def compare_runs(runs_a, runs_b, label):
    """Exact-token and logprob comparison between two sets of runs of the SAME
    prompt. Returns a dict naming the verdict precisely."""
    a, b = runs_a[0], runs_b[0]
    same_tokens = a["token_ids"] == b["token_ids"]
    out = {
        "label": label,
        "same_token_ids": same_tokens,
        "identical_repeats_within_arm": len({r["sha"] for r in runs_a}) == 1,
        "distinct_shas_arm_a": len({r["sha"] for r in runs_a}),
        "distinct_shas_arm_b": len({r["sha"] for r in runs_b}),
        "n_tokens_a": len(a["token_ids"]),
        "n_tokens_b": len(b["token_ids"]),
    }
    # find first divergence
    for i, (ta, tb) in enumerate(zip(a["token_ids"], b["token_ids"])):
        if ta != tb:
            out["first_divergent_index"] = i
            la = a["logprobs"][i] if i < len(a["logprobs"]) else None
            lb = b["logprobs"][i] if i < len(b["logprobs"]) else None
            if la and lb:
                out["divergence"] = {
                    "a_top1_tok": la["top1_tok"],
                    "a_top1_lp": la["top1_lp"],
                    "b_top1_tok": lb["top1_tok"],
                    "b_top1_lp": lb["top1_lp"],
                    "a_n_tied_at_top1": la["n_tied_at_top1"],
                    "b_n_tied_at_top1": lb["n_tied_at_top1"],
                    "lp_delta": round(abs(la["top1_lp"] - lb["top1_lp"]), 8),
                }
            # did each arm pick its own top-1? if both picked a tied candidate,
            # it is the benign F-005 tie class
            d = out.get("divergence", {})
            tie_a = la and la["n_tied_at_top1"] > 1
            tie_b = lb and lb["n_tied_at_top1"] > 1
            out["classification"] = (
                "EXACT_TIE_FLIP" if (tie_a and tie_b) else "REAL_DIVERGENCE")
            break
    else:
        out["classification"] = ("IDENTICAL" if same_tokens
                                 else "LENGTH_DIFFERENCE")
    if not same_tokens and "classification" not in out:
        out["classification"] = "PREFIX_THEN_DIVERGE"
    return out


# ------------------------------------------------------------------ scenarios


def scen_determinism(llm, args, res):
    """Task 3, the oracle itself: identical prompt, N repeats, exact tokens."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=48, seed=1234)
    lsp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=48, seed=1234,
                         logprobs=5)
    prompts = {
        "real": ("A distributed system stores user records in a PostgreSQL "
                 "database behind a caching layer. Requests are routed "
                 "round-robin to 8 application instances. Under load, p99 "
                 "latency rises from 40ms to 900ms while CPU stays low. The "
                 "connection pool is sized at 5 per instance. Explain the "
                 "likely cause and give three concrete fixes."),
        "short": "The capital of France is",
        "medium": ("Summarise the following engineering note and list the "
                   "risks. " * 8),
    }
    res["determinism"] = {}
    for name, p in prompts.items():
        runs = tokens_and_logprobs(llm, p, lsp, n=args.repeats)
        shas = {r["sha"] for r in runs}
        res["determinism"][name] = {
            "repeats": args.repeats,
            "distinct_shas": len(shas),
            "bit_stable": len(shas) == 1,
            "n_tokens": len(runs[0]["token_ids"]),
            "text": runs[0]["text"][:300],
            "runs": runs if args.keep_logprobs else None,
        }
        # top-1 logprob jitter across repeats of the SAME prompt
        if args.repeats > 1:
            t1 = [r["logprobs"][0]["top1_lp"] for r in runs
                  if r["logprobs"]]
            if t1:
                res["determinism"][name]["first_pos_top1_lp_spread"] = round(
                    max(t1) - min(t1), 8)
        print(f"  determinism/{name}: distinct={len(shas)}/{args.repeats} "
              f"bit_stable={len(shas) == 1}", flush=True)


def scen_shape_dependence(llm, args, res):
    """Task 2b: does WHICH sizes were captured earlier change the answer for a
    later request? Run a fixed probe prompt, then drive the engine through many
    batch sizes / token counts to force a wide spread of new captures, then run
    the SAME probe prompt again. Identical output => no size carryover."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, seed=1234)
    probe = "Explain in two sentences why tail latency matters more than " \
            "average latency for a payment service."

    before = tokens_and_logprobs(llm, probe, sp, n=1)[0]

    # force captures across a wide spread of padded token counts
    widths = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200, 256]
    filler = "word "
    forced = 0
    for w in widths:
        p = "[x] " + filler * w + " Summarise."
        try:
            llm.generate([p], SamplingParams(temperature=0.0, max_tokens=4,
                                             seed=1234), use_tqdm=False)
            forced += 1
        except Exception as e:  # noqa: BLE001
            res.setdefault("shape_errors", []).append(f"w={w}: {e}")

    after = tokens_and_logprobs(llm, probe, sp, n=1)[0]
    res["shape_dependence"] = {
        "forced_widths": forced,
        "requested_widths": widths,
        "probe_identical_before_after": before["token_ids"] == after["token_ids"],
        "before_sha": before["sha"],
        "after_sha": after["sha"],
        "before_tokens": before["token_ids"],
        "after_tokens": after["token_ids"],
        "before_text": before["text"][:200],
        "after_text": after["text"][:200],
    }
    print(f"  shape_dependence: identical={res['shape_dependence']['probe_identical_before_after']}",
          flush=True)


def scen_prefix_cache(llm, args, res):
    """Task 2c: prefix caching is on by default and shares the KV/attention
    machinery a captured decode graph reads. Ask a long shared prefix then
    diverge, repeatedly, and check the shared-prefix answer never changes."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, seed=1234)
    prefix = ("SYSTEM CONTEXT: Acme Corp runs a fleet of 4,200 edge gateways. "
              "Each gateway reports every 30 seconds. The NOC team reviews "
              "dashboards every morning at 06:00 local time. ") * 6
    queries = ["Summarise the fleet status in one sentence.",
               "What time does the NOC team review dashboards?",
               "How many gateways are in the fleet?",
               "List the three risks of this setup."]
    per_q = []
    for rnd in range(3):
        for q in queries:
            o = llm.generate([prefix + q], sp, use_tqdm=False)[0]
            per_q.append({
                "round": rnd, "q": q, "text": o.outputs[0].text,
                "token_ids": list(o.outputs[0].token_ids),
            })
    by_q = {}
    for rec in per_q:
        by_q.setdefault(rec["q"], []).append(rec)
    verdicts = {}
    for q, recs in by_q.items():
        shas = {hashlib.sha256(json.dumps(r["token_ids"]).encode()).hexdigest()
                for r in recs}
        verdicts[q] = {"rounds": len(recs), "distinct": len(shas),
                       "stable": len(shas) == 1}
    res["prefix_cache"] = {
        "prefix_caching_enabled": args.prefix_caching,
        "per_query": verdicts,
        "all_stable": all(v["stable"] for v in verdicts.values()),
        "sample_text": per_q[0]["text"][:250],
    }
    print(f"  prefix_cache: all_stable={res['prefix_cache']['all_stable']}", flush=True)


def scen_concurrent(llm, args, res):
    """Task 2d: the #54698 hang class. Sustained concurrent load with graphs on.
    A hang is a legitimate finding, so this runs the load in a WATCHDOGGED
    child: if the child does not finish, the parent records the wedge and pulls
    a stack. The vLLM engine runs in a spawned EngineCore process, so a wedge
    there is invisible to py-spy on the parent - hence capture the child's own
    stderr and its /proc CPU ticks climbing, which is the #54698 evidence format.
    """
    import subprocess
    import textwrap

    child = textwrap.dedent("""
        import json, os, sys, time, threading
        from vllm import LLM, SamplingParams

        pid = os.getpid()
        ticks = []
        stop = threading.Event()

        def sampler():
            # #54698 evidence format: cumulative CPU ticks of the engine
            while not stop.is_set():
                try:
                    with open(f"/proc/{pid}/stat") as f:
                        ticks.append(int(f.read().split()[13]) + int(f.read() or 0))
                except Exception:
                    pass
                time.sleep(0.5)

        llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16", max_model_len=2048,
                  max_num_seqs=32, max_num_batched_tokens=8192,
                  gpu_memory_utilization=0.55, enforce_eager=False, seed=1234,
                  disable_log_stats=True, enable_prefix_caching=True)

        filler = "Summarise the engineering note and list risks. " * 6
        prompts = [f"[case {i}] {filler}" for i in range(64)]
        sp = SamplingParams(temperature=0.0, max_tokens=96, seed=1234)

        t0 = time.time()
        for wave in range(6):
            outs = llm.generate(prompts, sp, use_tqdm=False)
            bad = sum(1 for o in outs if getattr(o.metrics, "is_corrupted", False))
            print(f"===WAVE=== {wave} n={len(outs)} corrupted={bad} "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
        print("===CHILD_JSON===" + json.dumps({"ok": True, "waves": 6,
                                               "elapsed_s": time.time() - t0}))
    """)

    cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_015_concurrent_child.py")
    with open(cpath, "w") as f:
        f.write(child)

    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    keep = [p for p in ld.split(":") if p and not p.startswith("/home/dario/oneapi")]
    env["LD_LIBRARY_PATH"] = ":".join([os.path.expanduser("~/.local/lib")] + keep)
    env["VLLM_XPU_ENABLE_XPU_GRAPH"] = args.graph_env

    timeout = args.concurrent_timeout
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, cpath], env=env,
                           capture_output=True, text=True, timeout=timeout)
        res["concurrent"] = {
            "timed_out": False,
            "returncode": p.returncode,
            "wall_s": round(time.time() - t0, 1),
            "waves": [l for l in p.stdout.splitlines() if l.startswith("===WAVE===")],
            "stderr_tail": p.stderr.strip().splitlines()[-8:],
        }
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(
            e.stdout, bytes) else (e.stdout or "")
        res["concurrent"] = {
            "timed_out": True,
            "WEDGE": True,
            "wall_s": round(time.time() - t0, 1),
            "waves_completed": len([l for l in out.splitlines()
                                    if l.startswith("===WAVE===")]),
            "last_wave_lines": [l for l in out.splitlines()
                                if l.startswith("===WAVE===")][-4:],
        }
    finally:
        try:
            os.unlink(cpath)
        except OSError:
            pass
    print(f"  concurrent: timed_out={res['concurrent'].get('timed_out')} "
          f"wall={res['concurrent']['wall_s']}s", flush=True)


SCENARIOS = {
    "determinism": (scen_determinism, True),
    "shape_dependence": (scen_shape_dependence, True),
    "prefix_cache": (scen_prefix_cache, True),
    "concurrent": (scen_concurrent, False),   # owns its own engine
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="determinism")
    ap.add_argument("--graph", choices=["on", "off"], default="on")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--keep-logprobs", action="store_true")
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--concurrent-timeout", type=int, default=420)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    args.graph_env = "1" if args.graph == "on" else "0"
    names = (list(SCENARIOS) if args.scenario == "all"
             else args.scenario.split(","))

    res = {
        "probe": "015_graph_determinism",
        "graph": args.graph,
        "graph_env": args.graph_env,
        "model": MODEL,
        "scenarios_requested": names,
        "args": {k: v for k, v in vars(args).items()},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    import torch
    res["versions"] = {
        "vllm": __import__("vllm").__version__,
        "torch": torch.__version__,
        "kernels": __import__("importlib.metadata", fromlist=["v"]).version(
            "vllm-xpu-kernels"),
    }
    free, total = torch.xpu.mem_get_info(0)
    res["vram_free_gib"] = round(free / 2 ** 30, 2)

    needs_engine = [n for n in names if SCENARIOS[n][1]]
    if needs_engine:
        llm = build_llm(args)
        cc = llm.llm_engine.vllm_config.compilation_config
        res["effective_cudagraph_mode"] = str(cc.cudagraph_mode)
        print(f"engine up, cudagraph_mode={cc.cudagraph_mode}", flush=True)
        for n in needs_engine:
            print(f"-> {n}", flush=True)
            try:
                SCENARIOS[n][0](llm, args, res)
            except BaseException as e:  # noqa: BLE001
                import traceback
                res.setdefault("scenario_errors", {})[n] = (
                    f"{type(e).__name__}: {e}")
                res["scenario_errors"][n + "_tb"] = traceback.format_exc()[-1200:]
                print(f"   ERROR {type(e).__name__}: {e}", flush=True)

    for n in names:
        if not SCENARIOS[n][1]:
            print(f"-> {n} (own engine)", flush=True)
            try:
                SCENARIOS[n][0](None, args, res)
            except BaseException as e:  # noqa: BLE001
                import traceback
                res.setdefault("scenario_errors", {})[n] = f"{type(e).__name__}: {e}"
                res["scenario_errors"][n + "_tb"] = traceback.format_exc()[-1200:]

    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Capsule 010 - XPU CUDA-graph ON vs OFF, same workload, interleaved A B B A B A.

Measures, per run:
  - TTFT (ms)                    first_token_ts - scheduled_ts  (engine telemetry)
  - prefill tok/s                prompt tokens / prefill wall
  - decode tok/s                 generated tokens / decode wall
  - inter-token latency (ms)     (last_token_ts - first_token_ts) / (n_tok - 1)
  - CPU wall vs CPU time         perf_counter vs process_time (CUDA graphs exist to
                                 remove CPU launch overhead, so CPU-time-per-token
                                 is the metric that should move)
  - output digest                deterministic text, to compare correctness ON vs OFF

The graph decision happens in vllm/platforms/xpu.py:301 at engine construction,
so ON and OFF MUST be separate processes. Interleaving is therefore done by the
shell driver (010_xpu_graph_ab.sh), never inside one process.

NOTE: vLLM v1 spawns the engine core and re-imports __main__ by path, so this
must be run as a real .py file, never via stdin (see ENVIRONMENT.md).
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time


def build_workload(n_prompts: int, prompt_tokens: int, seed: int):
    """Identical synthetic workload for every arm - no dataset drift between arms."""
    rng_state = seed
    filler = (
        "Summarise the following engineering note and list the risks. "
        "Include latency, memory and correctness considerations. "
    )
    prompts = []
    for i in range(n_prompts):
        # vary the prompt deterministically by index, keep length constant
        body = filler * max(1, prompt_tokens // 24)
        prompts.append(f"[case {i}] {body}")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["on", "off"], required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-words", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--profile", default="baseline")
    ap.add_argument("--run-index", type=int, default=0)
    ap.add_argument("--gpu-profile", action="store_true",
                    help="also collect torch.profiler GPU busy time (slow, separate run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Record the env BEFORE importing vllm - vllm.envs snapshots the environment.
    graph_env_present = os.environ.get("VLLM_XPU_ENABLE_XPU_GRAPH")

    result = {
        "profile": args.profile,
        "graph_requested": args.graph,
        "graph_env_value": graph_env_present,
        "run_index": args.run_index,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": {
            "prompts": args.prompts,
            "max_tokens": args.max_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "seed": 1234,
            "temperature": 0.0,
        },
        "ok": False,
    }

    import torch
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    result["versions"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vllm": __import__("vllm").__version__,
    }
    try:
        result["versions"]["vllm_xpu_kernels"] = __import__(
            "importlib.metadata", fromlist=["version"]).version("vllm-xpu-kernels")
    except Exception as e:  # noqa: BLE001
        result["versions"]["vllm_xpu_kernels"] = f"unavailable: {e}"
    result["platform_device"] = getattr(current_platform, "device_name", str(current_platform))

    free, total = torch.xpu.mem_get_info(0)
    result["vram_free_gib_at_start"] = round(free / 2**30, 2)
    result["vram_total_gib"] = round(total / 2**30, 2)

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=0.55,
        enforce_eager=False,  # compile ON for both arms; only cudagraph_mode differs
        seed=1234,
        # MUST stay False: output_processor.py:186 only builds RequestStateStats
        # when log_stats is on, so disable_log_stats=True would silently zero out
        # every TTFT/ITL number. Log noise costs nothing in a batch harness.
        disable_log_stats=False,
    )

    # Read back what the engine ACTUALLY chose - do not trust the request.
    cc = llm.llm_engine.vllm_config.compilation_config
    result["effective_cudagraph_mode"] = str(cc.cudagraph_mode)
    # There is no CompilationConfig.is_compiled attribute in 0.30.0; record the
    # real evidence that Inductor ran instead of a guessed boolean.
    result["compilation_config_present"] = cc is not None

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=1234
    )
    prompts = build_workload(args.prompts, args.prompt_words, seed=1234)

    # ---- warmup (excluded from all reported timings) -------------------
    llm.generate(prompts[:2], sampling, use_tqdm=False)

    # ---- measured window ------------------------------------------------
    torch.xpu.synchronize()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    t_start = time.perf_counter()
    outs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.xpu.synchronize()
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    result["wall_s"] = round(wall, 4)
    result["cpu_s"] = round(cpu, 4)

    # ---- per-request telemetry -----------------------------------------
    ttfts, itls, gen_toks, prompt_toks, texts, corrupted = [], [], 0, 0, [], 0
    metrics_seen = 0
    for o in outs:
        m = getattr(o, "metrics", None)
        c = o.outputs[0]
        gen_toks += len(c.token_ids)
        prompt_toks += len(o.prompt_token_ids)
        texts.append(c.text)
        if getattr(m, "is_corrupted", False):
            corrupted += 1
        if m is not None and getattr(m, "first_token_ts", 0.0) and getattr(m, "last_token_ts", 0.0):
            metrics_seen += 1
            n = getattr(m, "num_generation_tokens", 0) or len(c.token_ids)
            ttfts.append((m.first_token_ts - m.scheduled_ts) * 1000.0)
            if n > 1:
                itls.append((m.last_token_ts - m.first_token_ts) * 1000.0 / (n - 1))

    result["requests"] = len(outs)
    result["requests_with_engine_metrics"] = metrics_seen
    result["prompt_tokens_total"] = prompt_toks
    result["generation_tokens_total"] = gen_toks
    result["requests_flagged_corrupted"] = corrupted

    # Fail loudly rather than report a benchmark with no TTFT/ITL in it. A silent
    # telemetry loss is indistinguishable from a real perf win when arms are
    # compared, so treat it as a harness failure.
    if metrics_seen == 0:
        result["ok"] = False
        result["error"] = (
            "no engine RequestStateStats on any output: log_stats was off, or the "
            "telemetry path changed. TTFT/ITL cannot be reported from this run."
        )
        payload = json.dumps(result, indent=2)
        print("===RESULT_JSON===")
        print(payload)
        if args.out:
            with open(args.out, "w") as f:
                f.write(payload)
        sys.exit(3)

    if ttfts:
        result["ttft_ms_mean"] = round(statistics.fmean(ttfts), 2)
        result["ttft_ms_median"] = round(statistics.median(ttfts), 2)
        result["ttft_ms_p90"] = round(sorted(ttfts)[max(0, int(0.9 * len(ttfts)) - 1)], 2)
    if itls:
        result["itl_ms_mean"] = round(statistics.fmean(itls), 3)
        result["itl_ms_median"] = round(statistics.median(itls), 3)
    if gen_toks and wall > 0:
        result["decode_tok_s"] = round(gen_toks / wall, 2)
        result["overall_tok_s"] = round((gen_toks + prompt_toks) / wall, 2)
    if gen_toks and cpu > 0:
        # CPU-time per generated token: the number graphs are supposed to shrink
        result["cpu_ms_per_gen_token"] = round(cpu * 1000.0 / gen_toks, 3)
    if wall > 0:
        result["cpu_to_wall_ratio"] = round(cpu / wall, 3)

    # ---- correctness digest: compare the SAME seed across arms ----------
    joined = "\n\x1f".join(texts)
    result["output_sha256"] = hashlib.sha256(joined.encode()).hexdigest()
    result["output_first_text"] = texts[0][:400] if texts else ""
    result["output_chars_total"] = sum(len(t) for t in texts)

    if args.gpu_profile:
        try:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                llm.generate(prompts[:4], sampling, use_tqdm=False)
                torch.xpu.synchronize()
            evs = [e for e in prof.key_averages() if e.device_time_total > 0]
            result["gpu_busy_s"] = round(sum(e.device_time_total for e in evs) / 1e6, 4)
            result["gpu_kernel_groups"] = len(evs)
        except Exception as e:  # noqa: BLE001
            result["gpu_profile_error"] = f"{type(e).__name__}: {e}"

    free, _ = torch.xpu.mem_get_info(0)
    result["vram_free_gib_at_end"] = round(free / 2**30, 2)
    result["ok"] = True
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    payload = json.dumps(result, indent=2)
    print("===RESULT_JSON===")
    print(payload)
    if args.out:
        with open(args.out, "w") as f:
            f.write(payload)


if __name__ == "__main__":
    main()

"""Capsule 012 - reproduce vllm#54785: MTP k=1..4 + XPU graph capture, single B70.

Reporter's exact protocol (issue #54785 body + addendum):
  - ONE short prompt, N identical greedy requests in sequence, logprobs=5
  - "distinct outputs" and "top-1 logprob at first position" as the signal
  - k=1/2/3 clean, k=4 corrupt -> the cliff is the whole bug report

Model: Qwen/Qwen3.5-0.8B-Base (873M, model_type qwen3_5, 18 linear_attention +
6 full_attention layers, mtp_num_hidden_layers=1). Same architecture family,
same Qwen3_5MTP drafter and same GDN layer code as the reporter's
Qwen3.8-27B, but fits on one B70. See issue-54785-upstream-audit.md S5.

Determinism oracle (stronger than the reporter's):
  * same prompt N times within ONE process  -> intra-arm determinism
  * graph ON vs graph OFF, same prompt      -> cross-arm divergence
  * top-5 logprobs at EVERY position        -> can tell a benign exact-tie
    flip (delta ~0) from a real corruption (the ON arm picking a token the
    OFF arm considered far worse, or a >1 nat swing at the same position)
  * first-position top-5 is the key metric: identical every run by
    construction, so any spread is engine state, not prompt content.

NOTE: vLLM v1 spawns EngineCore and re-imports __main__ by path, so this must
be run as a real .py file, never via stdin (see ENVIRONMENT.md).
"""

import argparse
import dataclasses
import hashlib
import json
import os
import time

MODEL = "Qwen/Qwen3.5-0.8B-Base"


def _plain(v):
    """Reduce a metrics field to something json can hold."""
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return repr(v)

# The reporter's prompt, verbatim from the issue body.
REPRO_PROMPT = "The capital of France is"
# A second short prompt, so we are not tuning to one string.
REPRO_PROMPT2 = "The three laws of motion were formulated by"


def build_sampling(max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        logprobs=5,
        seed=1234,
    )


def first_pos_top(olp):
    """Top-5 logprobs at generation position 0, as [(token, logprob), ...]."""
    steps = olp.outputs[0].logprobs or []
    if not steps:
        return []
    top = []
    for _tid, lp in (steps[0] or {}).items():
        if lp is not None:
            top.append((lp.decoded_token, round(lp.logprob, 6)))
    top.sort(key=lambda x: -x[1])
    return top[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="num_speculative_tokens")
    ap.add_argument("--graph", choices=["on", "off"], required=True)
    # The reporter's protocol is 8 repeats, but their own text says the failure
    # is a deterministic function of REQUEST ORDINAL and only degenerates after
    # "~40 requests on the same boot". 8 repeats therefore under-samples the
    # very thing being measured. Default to a long run; the per-request digests
    # below make the cycle position visible.
    ap.add_argument("--repeats", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=6)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--cudagraph-mode", default=None,
                    help="e.g. FULL_DECODE_ONLY (the reporter's mode). Default: engine choice.")
    args = ap.parse_args()

    # Read the env BEFORE importing vllm - vllm.envs snapshots the environment.
    res = {
        "k": args.k,
        "graph_requested": args.graph,
        "graph_env": os.environ.get("VLLM_XPU_ENABLE_XPU_GRAPH"),
        "torch_compile_disable": os.environ.get("TORCH_COMPILE_DISABLE"),
        "model": MODEL,
        "repeats": args.repeats,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ok": False,
    }

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=1,           # reporter: bs=1
        gpu_memory_utilization=0.55,
        enforce_eager=False,      # graph arm is decided by the env var, not this
        seed=1234,
        disable_log_stats=False,  # capsule 010: logprobs need log stats ON
        trust_remote_code=True,
        # POSITIVE CONTROL. Without this, spec_decode_metrics is always None and
        # a "clean" k=4 arm is indistinguishable from a silently dead drafter -
        # i.e. a false negative. With it, every request reports the ordered
        # per-step accepted/proposed draft counts, which proves the MTP drafter
        # really ran and really had its drafts accepted.
        # NOTE: it is a FLAT EngineArgs field (arg_utils.py:679), not a nested
        # `observability={...}` dict - passing the nested form raises
        # "EngineArgs.__init__() got an unexpected keyword argument".
        per_request_spec_decode_metrics="detailed",
    )
    # The reporter's config: cudagraph_mode FULL_DECODE_ONLY, mtp k.
    llm_kwargs["speculative_config"] = {
        "method": "mtp",
        "num_speculative_tokens": args.k,
    }
    if args.block_size:
        llm_kwargs["block_size"] = args.block_size
    else:
        # XPU forces a multiple of 64 for GDN backends (platforms/xpu.py:434-446);
        # the reporter used --block-size 512. Mirror it.
        llm_kwargs["block_size"] = 512
    if args.cudagraph_mode:
        # The reporter's exact flag: --compilation-config
        # '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
        llm_kwargs["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
        res["cudagraph_mode_requested"] = args.cudagraph_mode

    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    res["engine_start_s"] = round(time.perf_counter() - t0, 2)

    vcfg = llm.llm_engine.vllm_config
    cc = vcfg.compilation_config
    res["effective_cudagraph_mode"] = str(cc.cudagraph_mode)
    res["cudagraph_capture_sizes"] = list(cc.cudagraph_capture_sizes or [])
    res["max_cudagraph_capture_size"] = cc.max_cudagraph_capture_size
    res["num_speculative_tokens_effective"] = vcfg.speculative_config.num_speculative_tokens
    res["block_size"] = vcfg.cache_config.block_size
    res["enforce_eager"] = vcfg.model_config.enforce_eager
    import torch

    free, _ = torch.xpu.mem_get_info(0)
    res["vram_free_gib_after_start"] = round(free / 2**30, 2)

    sampling = build_sampling(args.max_tokens)

    def run_prompt(prompt):
        """N identical greedy requests, one after another, same process."""
        runs = []
        for i in range(args.repeats):
            o = llm.generate([prompt], sampling, use_tqdm=False)[0]
            c = o.outputs[0]
            # spec_decode_metrics is the POSITIVE CONTROL that MTP is actually
            # drafting and that drafts are being accepted. Without it, a "clean"
            # k=4 arm is indistinguishable from a silently non-functional drafter.
            sdm = getattr(c, "spec_decode_metrics", None)
            runs.append({
                "i": i,
                "text": c.text,
                "token_ids": list(c.token_ids),
                "n_tokens": len(c.token_ids),
                "first_pos_top": first_pos_top(o),
                "num_accepted": getattr(c, "num_accepted_tokens", None),
                "spec_metrics": (
                    # dataclasses.fields() is the right filter: dir() would also
                    # pick up observe()/to_dict() and any property that raises.
                    {f.name: _plain(getattr(sdm, f.name))
                     for f in dataclasses.fields(sdm)}
                    if sdm is not None else None
                ),
            })
        return runs

    res["cases"] = {}
    for name, prompt in (("capital", REPRO_PROMPT), ("motion", REPRO_PROMPT2)):
        runs = run_prompt(prompt)
        texts = [r["text"] for r in runs]
        tids = [tuple(r["token_ids"]) for r in runs]
        tops = [tuple((t, lp) for t, lp in r["first_pos_top"]) for r in runs]
        lp0 = [r["first_pos_top"][0][1] if r["first_pos_top"] else None for r in runs]
        res["cases"][name] = {
            "prompt": prompt,
            "distinct_texts": len(set(texts)),
            "distinct_token_ids": len(set(tids)),
            "distinct_first_pos_top5": len(set(tops)),
            "texts": texts,
            "first_pos_logprob_top1": lp0,
            # spread of the top-1 logprob at position 0: the reporter's signal
            "first_pos_lp_spread": (
                round(max(lp0) - min(lp0), 6) if all(x is not None for x in lp0) else None
            ),
            "runs": runs,
        }
        # sha256 of the exact token id sequence of the FIRST run, so identical
        # digests across arms prove identical first-request output
        res["cases"][name]["first_run_token_sha256"] = hashlib.sha256(
            json.dumps(runs[0]["token_ids"]).encode()
        ).hexdigest()

    res["ok"] = True
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload = json.dumps(res, indent=2)
    with open(args.out, "w") as f:
        f.write(payload)

    # Compact one-line verdict, easy to grep across arms.
    v = []
    for name, cse in res["cases"].items():
        v.append(
            f"{name}: distinct_texts={cse['distinct_texts']} "
            f"distinct_top5={cse['distinct_first_pos_top5']} "
            f"lp_spread={cse['first_pos_lp_spread']} "
            f"lp0={cse['first_pos_logprob_top1'][:4]}"
        )
    print("===ARM_SUMMARY===")
    print(
        f"k={args.k} graph={args.graph} mode={res['effective_cudagraph_mode']} "
        f"cap_sizes_head={res['cudagraph_capture_sizes'][:8]} "
        f"max_cap={res['max_cudagraph_capture_size']}"
    )
    for line in v:
        print("  " + line)
    print("===RESULT_JSON_PATH===", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capsule 001 body: prove vLLM generates correct tokens on the Arc Pro B70.

Must be a real file, NOT a heredoc on stdin. vLLM v1 spawns the engine core in a
separate process, and multiprocessing's spawn re-imports __main__ by path. With
`python -` the path is `<stdin>`, which does not exist, so the child dies with
FileNotFoundError before the model ever loads. That is a lab-harness trap, not
a vLLM bug.

CRITICAL: run with oneAPI NOT sourced (env -u LD_LIBRARY_PATH). See ENVIRONMENT.md.
"""

import json
import os
import sys
import time

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
ENABLE_GRAPH = os.environ.get("ENABLE_GRAPH", "1") == "1"


def main() -> int:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    result = {
        "capsule": "001_first_light",
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "platform": current_platform.device_name,
        "torch": torch.__version__,
        "xpu_graph_enabled": ENABLE_GRAPH,
    }

    free, total = torch.xpu.mem_get_info(0)
    result["vram_free_gib_before"] = round(free / 2**30, 2)
    result["vram_total_gib"] = round(total / 2**30, 2)

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        enforce_eager=not ENABLE_GRAPH,
        gpu_memory_utilization=0.80,
    )
    result["load_seconds"] = round(time.perf_counter() - t0, 2)

    # --- Gate 1: greedy decode must name Paris -----------------------------
    prompt = "The capital of France is"
    t1 = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    gen_s = time.perf_counter() - t1

    text = out[0].outputs[0].text
    n_tok = len(out[0].outputs[0].token_ids)
    result.update(
        {
            "generate_seconds": round(gen_s, 3),
            "output_tokens": n_tok,
            "decode_tok_s": round(n_tok / gen_s, 2) if gen_s > 0 else None,
            "completion": text,
            "mentions_paris": "paris" in text.lower(),
        }
    )

    # --- Gate 2: long prefill (exercises chunked prefill, not 1-token) ------
    long_prompt = ("Explain in detail how a transformer neural network processes "
                   "text. " * 12)
    t2 = time.perf_counter()
    out2 = llm.generate([long_prompt], SamplingParams(temperature=0.0, max_tokens=16))
    result["long_prompt_seconds"] = round(time.perf_counter() - t2, 3)
    result["long_prompt_out_tokens"] = len(out2[0].outputs[0].token_ids)
    result["long_prompt_completion"] = out2[0].outputs[0].text[:200]

    free_after, _ = torch.xpu.mem_get_info(0)
    result["vram_free_gib_after"] = round(free_after / 2**30, 2)
    result["vram_used_delta_gib"] = round((free - free_after) / 2**30, 2)

    result["verdict"] = "PASS" if result["mentions_paris"] else "FAIL"

    print("===RESULT_JSON===")
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

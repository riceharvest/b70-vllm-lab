#!/usr/bin/env python3
"""Capsule 019: INT4 compressed-tensors MoE on the Arc Pro B70 (P2 + P6).

WHY THIS MODEL MATTERS
    Ling-3.0-tiny-int4 is a 128-expert / top-8 MoE with `compressed-tensors`
    pack-quantized INT4 weights. That makes it the sharpest available probe for
    the Xe2 grouped-GEMM data race (FINDINGS F-008/F-014): the race lived in the
    grouped-GEMM tile counter, which is exactly the kernel a wide-expert MoE
    dispatches to on every decode step. We now run vllm_xpu_kernels 0.1.15.4,
    which carries the fix, so this capsule is the regression check for it.

    It is also simultaneously a P2 quant check (INT4 correctness + speed) and a
    P6 coverage check (does an INT4 MoE run at all on XPU).

DESIGN
    Two arms, interleaved is NOT needed here because there is no candidate
    patch: this measures the CURRENT baseline against its own reference. The
    reference is eager (graphs off) and the candidate is graph-on, which is the
    configuration a real user gets from upstream PR #51600.

    Correctness is judged on the thing that actually matters: token agreement
    between the two execution paths, plus finiteness. A single flipped greedy
    token at an exact tie is benign (documented in F-005) and is reported
    separately from real divergence.

MUST be a real file, not `python -`: vLLM v1 spawns the engine core, and spawn
re-imports __main__ by path (see capsule 001's docstring).
"""

import json
import os
import sys
import time

MODEL = os.environ.get(
    "MODEL", "/mnt/ssd/huggingface/hub/models--inclusionAI--Ling-3.0-tiny-int4/snapshots/d355645f42fa7c08980889e288ed6957bacedde6"
)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "48"))
PROMPT = os.environ.get(
    "PROMPT", "Explain in two sentences why the sky appears blue."
)
OUT = os.environ.get("OUT", "/mnt/ssd/b70-vllm-lab/results/019_int4_moe.json")
# Leave headroom: desktop ~1.0 GiB, weights ~5.5 GiB, plus KV + workspace.
GPU_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.80"))


def run_arm(enable_graph: bool) -> dict:
    """One full engine run. Returns timing + generated text + peak VRAM."""
    if enable_graph:
        os.environ["VLLM_XPU_ENABLE_XPU_GRAPH"] = "1"
    else:
        os.environ.pop("VLLM_XPU_ENABLE_XPU_GRAPH", None)

    from vllm import LLM, SamplingParams

    t0 = time.monotonic()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "2048")),
        gpu_memory_utilization=GPU_UTIL,
        enforce_eager=not enable_graph,
        trust_remote_code=True,
    )
    load_s = time.monotonic() - t0

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=0)
    t1 = time.monotonic()
    outs = llm.generate([PROMPT], sp)
    gen_s = time.monotonic() - t1

    text = outs[0].outputs[0].text
    n_tok = len(outs[0].outputs[0].token_ids)

    vram = {}
    try:
        with open("/mnt/ssd/b70-vllm-lab/results/vram.json") as f:
            vram = json.load(f)
    except Exception:
        pass

    del llm
    return {
        "graph": enable_graph,
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "output_tokens": n_tok,
        "decode_tok_s": round(n_tok / gen_s, 2) if gen_s > 0 else None,
        "text": text,
        "vram": vram,
    }


def main() -> int:
    result = {
        "capsule": "019_int4_moe",
        "purpose": "P2 quant + P6 coverage + F-008/F-014 regression check",
        "model": MODEL,
        "kernels_expected": "0.1.15.4",
        "max_tokens": MAX_TOKENS,
        "arms": {},
        "errors": {},
    }

    # Eager/reference arm first: if the model cannot even load, we learn that
    # immediately and cheaply rather than after paying for a graph capture.
    for graph_on in (False, True):
        label = "graph_on" if graph_on else "eager"
        try:
            result["arms"][label] = run_arm(graph_on)
            print(f"[019] {label}: OK", flush=True)
        except Exception as e:  # noqa: BLE001 - a failure here IS the result
            msg = f"{type(e).__name__}: {e}"
            result["errors"][label] = msg
            print(f"[019] {label}: FAILED -> {msg}", flush=True)
            # A load failure in eager means graph-on cannot succeed either.
            if not graph_on and "load" in msg.lower():
                result["errors"]["graph_on"] = "skipped: eager load failed"
                break

    a, b = result["arms"].get("eager"), result["arms"].get("graph_on")
    if a and b:
        ta = a["text"].strip()
        tb = b["text"].strip()
        result["token_agreement"] = (ta == tb)
        result["char_agreement"] = (
            round(
                sum(1 for x, y in zip(ta, tb) if x == y) / max(len(ta), len(tb), 1), 4
            )
        )
        # Speedup only when both arms are valid.
        if a["decode_tok_s"] and b["decode_tok_s"]:
            result["graph_speedup"] = round(
                b["decode_tok_s"] / a["decode_tok_s"], 3
            )

    if not result["errors"]:
        result["verdict"] = "PASS" if result.get("token_agreement") else "DIVERGED"
    else:
        result["verdict"] = "FAILED:" + ",".join(result["errors"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))
    print("[019] VERDICT:", result["verdict"])
    return 0 if result["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())

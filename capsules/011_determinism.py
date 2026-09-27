"""Capsule 011 - is the ON/OFF text divergence benign near-tie or wrong logits?

Capsule 010 showed: within each arm output is bit-reproducible, but ON and OFF
produce different text at temperature=0. Two very different explanations:

  (a) BENIGN - graph capture changes reduction/kernel order, shifting logits in
      the last bits, so a near-tie argmax flips. Text diverges but neither is wrong.
  (b) REAL BUG - graphs corrupt state (cf. vllm#54785 "non-deterministic, wrong
      logits at temperature=0", vllm#48327 gibberish).

Distinguishing test: find the first divergent token index, then ask for the logprobs
at that position under BOTH arms. If the two candidates were within a hair of each
other in the OFF run, it is (a). If the ON arm picked a token the OFF arm considered
far worse, it is (b).

Also runs a real (non-degenerate) prompt, because capsule 010's prompt was repetitive
filler that the model echoes, which makes near-ties far more likely than in practice.
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["on", "off"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    res = {"graph_requested": args.graph,
           "graph_env": os.environ.get("VLLM_XPU_ENABLE_XPU_GRAPH"),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16", max_model_len=2048,
              max_num_seqs=1, max_num_batched_tokens=2048,
              gpu_memory_utilization=0.55, enforce_eager=False, seed=1234,
              disable_log_stats=False)
    cc = llm.llm_engine.vllm_config.compilation_config
    res["effective_cudagraph_mode"] = str(cc.cudagraph_mode)

    # (1) a real, non-repetitive prompt
    real_p = ("A distributed system stores user records in a PostgreSQL database "
              "behind a caching layer. Requests are routed round-robin to 8 "
              "application instances. Under load, p99 latency rises from 40ms to "
              "900ms while CPU stays low. The connection pool is sized at 5 per "
              "instance. Explain the likely cause and give three concrete fixes.")
    # (2) the degenerate echo prompt from capsule 010
    echo_p = "[case 1] " + ("Summarise the following engineering note and list "
                            "the risks. " * 8)

    greedy = SamplingParams(temperature=0.0, top_p=1.0,
                            max_tokens=args.max_tokens, seed=1234)
    # logprobs at the top-5 so we can measure how close the decision was
    with_lp = SamplingParams(temperature=0.0, top_p=1.0,
                             max_tokens=args.max_tokens, seed=1234, logprobs=5)

    res["cases"] = {}
    for name, prompt in (("real", real_p), ("echo", echo_p)):
        o = llm.generate([prompt], greedy, use_tqdm=False)[0]
        text = o.outputs[0].text
        toks = list(o.outputs[0].token_ids)
        olp = llm.generate([prompt], with_lp, use_tqdm=False)[0]
        # per-position top-5 logprobs
        steps = []
        for i, pos in enumerate(olp.outputs[0].logprobs or []):
            top = []
            for _tid, lp in (pos or {}).items():
                if lp is not None:
                    top.append([lp.decoded_token, round(lp.logprob, 6)])
            top.sort(key=lambda x: -x[1])
            steps.append({"i": i, "top": top[:5]})
        res["cases"][name] = {
            "token_ids": toks,
            "n_tokens": len(toks),
            "text": text,
            "logprob_steps": steps,
        }

    res["ok"] = True
    payload = json.dumps(res, indent=2)
    print("===RESULT_JSON===")
    print(payload)
    with open(args.out, "w") as f:
        f.write(payload)


if __name__ == "__main__":
    main()

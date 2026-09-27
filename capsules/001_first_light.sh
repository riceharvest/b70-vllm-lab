#!/usr/bin/env bash
# First-light capsule: prove vLLM generates correct tokens on the Arc Pro B70.
#
# This is capsule #1 of the B70 lab. Everything else in the queue depends on
# this passing. It is deliberately tiny and fast (~1-2 min) because it is the
# cheapest experiment that can produce a definitive answer.
#
# CRITICAL: oneAPI must NOT be sourced. See ENVIRONMENT.md.
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_TOKENS="${MAX_TOKENS:-32}"
PY=/mnt/ssd/b70-venv/bin/python
OUT="${OUT:-/mnt/ssd/b70-vllm-lab/results/first_light.json}"

mkdir -p "$(dirname "$OUT")"

env -u LD_LIBRARY_PATH "$PY" - "$MODEL" "$MAX_TOKENS" <<'PYEOF'
import json, sys, time

model, max_tokens = sys.argv[1], int(sys.argv[2])

import torch
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

result = {
    "model": model,
    "max_tokens": max_tokens,
    "platform": current_platform.device_name,
    "torch": torch.__version__,
}

free, total = torch.xpu.mem_get_info(0)
result["vram_free_gib_before"] = round(free / 2**30, 2)
result["vram_total_gib"] = round(total / 2**30, 2)

t0 = time.perf_counter()
llm = LLM(
    model=model,
    dtype="bfloat16",
    enforce_eager=False,   # exercise the graph path, not just eager
    gpu_memory_utilization=0.80,
)
result["load_seconds"] = round(time.perf_counter() - t0, 2)

prompt = "The capital of France is"
sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

t1 = time.perf_counter()
out = llm.generate([prompt], sp)
gen_s = time.perf_counter() - t1

text = out[0].outputs[0].text
n_tok = len(out[0].outputs[0].token_ids)

result.update({
    "generate_seconds": round(gen_s, 3),
    "output_tokens": n_tok,
    "decode_tok_s": round(n_tok / gen_s, 2) if gen_s > 0 else None,
    "completion": text,
    # Correctness gate: greedy decode of this prompt must name Paris.
    "mentions_paris": "paris" in text.lower(),
})

free_after, _ = torch.xpu.mem_get_info(0)
result["vram_free_gib_after"] = round(free_after / 2**30, 2)
result["peak_vram_used_gib"] = round(
    (free - free_after) / 2**30, 2
)

# A second, longer prompt exercises prefill + chunked prefill rather than a
# single-token prefill, which is where XPU attention kernels tend to differ.
long_prompt = ("Explain in detail how a transformer neural network processes "
               "text. " * 12)
t2 = time.perf_counter()
out2 = llm.generate([long_prompt], SamplingParams(temperature=0.0, max_tokens=16))
pf_s = time.perf_counter() - t2
result["long_prompt_chars"] = len(long_prompt)
result["long_prompt_seconds"] = round(pf_s, 3)
result["long_prompt_tokens"] = len(out2[0].outputs[0].token_ids)

print("===RESULT_JSON===")
print(json.dumps(result, indent=2))
PYEOF
rc=$?
echo "===CAPSULE_EXIT=$rc==="
exit $rc

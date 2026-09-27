"""Aggregate capsule 010 interleaved ON/OFF runs. Reports median + spread per arm.

Interleaving (A B B A B A) means any arm-to-arm difference that is a real effect
survives; drift shows up as variance WITHIN an arm. Both are printed.
"""
import glob
import json
import statistics
import sys

prof = sys.argv[1] if len(sys.argv) > 1 else "baseline"
rows = []
for f in sorted(glob.glob(f"/mnt/ssd/b70-vllm-lab/results/010_xpu_graph/{prof}_*.json")):
    d = json.load(open(f))
    d["_file"] = f.split("/")[-1]
    rows.append(d)

if not rows:
    print("no results")
    sys.exit(1)

bad = [r for r in rows if not r.get("ok")]
good = [r for r in rows if r.get("ok")]
print(f"profile={prof}  runs={len(rows)}  ok={len(good)}  failed={len(bad)}")
for r in bad:
    print("  FAILED:", r["_file"], r.get("rc"), r.get("error", "")[:120])
if not good:
    sys.exit(1)

print("\nmodes seen:")
for r in good:
    print(f"  {r['_file']:28s} arm={r['graph_requested']:3s} "
          f"cudagraph={r['effective_cudagraph_mode']}")

FIELDS = [
    ("ttft_ms_mean", "TTFT mean (ms)"),
    ("ttft_ms_median", "TTFT median (ms)"),
    ("itl_ms_mean", "ITL mean (ms)"),
    ("decode_tok_s", "decode tok/s"),
    ("cpu_ms_per_gen_token", "CPU ms/gen-token"),
    ("cpu_to_wall_ratio", "CPU/wall ratio"),
    ("wall_s", "wall (s)"),
    ("generation_tokens_total", "gen tokens"),
]

print(f"\n{'metric':22s} {'OFF (A)':>26s} {'ON (B)':>26s} {'ratio':>8s}")
print("-" * 88)
for key, label in FIELDS:
    a = [r[key] for r in good if r["graph_requested"] == "off" and key in r]
    b = [r[key] for r in good if r["graph_requested"] == "on" and key in r]
    if not a or not b:
        print(f"{label:22s} {'n/a':>26s} {'n/a':>26s} {'':>8s}")
        continue

    def fmt(v):
        if len(v) > 1:
            return f"{statistics.median(v):>10.3g} [{min(v):.3g}-{max(v):.3g}]"
        return f"{v[0]:>10.3g} {'(n=1)':>14s}"
    ma, mb = statistics.median(a), statistics.median(b)
    ratio = f"{mb/ma:.2f}x" if ma else "n/a"
    print(f"{label:22s} {fmt(a)} {fmt(b)} {ratio:>8s}")

print("\nper-run detail (interleaved order):")
print(f"  {'idx':>3s} {'arm':>4s} {'mode':>20s} {'ttft':>8s} {'itl':>8s} "
      f"{'dec tok/s':>10s} {'cpu ms/tok':>10s} {'wall':>7s} {'sha256':>10s}")
for r in sorted(good, key=lambda x: x["run_index"]):
    print(f"  {r['run_index']:>3d} {r['graph_requested']:>4s} "
          f"{r['effective_cudagraph_mode']:>20s} "
          f"{r.get('ttft_ms_mean', 0):>8.2f} {r.get('itl_ms_mean', 0):>8.3f} "
          f"{r.get('decode_tok_s', 0):>10.1f} {r.get('cpu_ms_per_gen_token', 0):>10.4f} "
          f"{r.get('wall_s', 0):>7.3f} {r['output_sha256'][:10]}")

print("\ncorrectness:")
for arm in ("off", "on"):
    shas = {r["output_sha256"] for r in good if r["graph_requested"] == arm}
    print(f"  {arm:3s}: {len(shas)} distinct sha256 over "
          f"{len([r for r in good if r['graph_requested']==arm])} runs")
    corr = sum(r.get("requests_flagged_corrupted", 0) for r in good
               if r["graph_requested"] == arm)
    print(f"       requests_flagged_corrupted total={corr}")
    for r in good:
        if r["graph_requested"] == arm:
            print(f"       {r['_file']:28s} chars={r.get('output_chars_total')} "
                  f"gen_tok={r.get('generation_tokens_total')}")

ver = good[0].get("versions", {})
dev = good[0].get("platform_device", "?")
print(f"\nenv: {ver} device={dev}")

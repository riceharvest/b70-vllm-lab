"""Capsule 012 aggregator - read every 012/012b result.json and print the table.

The comparison that matters is a CLIFF, not a single number: the reporter says
k=1/2/3 are bit-stable and k=4 is corrupt. So print k in order, with graph ON and
OFF side by side, and mark the k=4 row explicitly.

A clean single-GPU k=4 row is a NEGATIVE result that constrains the hypothesis.
It is NOT a refutation of the reporter's 2x-B70 observation, and must never be
reported as one. See FINDINGS.md.
"""

import glob
import json
import os
import sys

LAB = "/mnt/ssd/b70-vllm-lab"
PATTERNS = [
    f"{LAB}/results/012_mTP_k*_*/result.json",
    f"{LAB}/results/012b_MTP_k*_*/result.json",
]


def load_all():
    rows = []
    for pat in PATTERNS:
        for path in sorted(glob.glob(pat)):
            try:
                with open(path) as f:
                    d = json.load(f)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {path}: {e}")
                continue
            if not d.get("ok"):
                print(f"SKIP {path}: ok=false")
                continue
            arm = os.path.basename(os.path.dirname(path))
            for case, c in d.get("cases", {}).items():
                rows.append({
                    "arm": arm,
                    "k": d["k"],
                    "graph": d["graph_requested"],
                    "case": case,
                    "mode": d.get("effective_cudagraph_mode"),
                    "cap_sizes": d.get("cudagraph_capture_sizes"),
                    "max_cap": d.get("max_cudagraph_capture_size"),
                    "repeats": d.get("repeats"),
                    "distinct_texts": c["distinct_texts"],
                    "distinct_top5": c["distinct_first_pos_top5"],
                    "lp_spread": c["first_pos_lp_spread"],
                    "lp0": c["first_pos_logprob_top1"],
                    "texts": c["texts"][:4],
                    "first_sha": c.get("first_run_token_sha256"),
                })
    return rows


def main():
    rows = load_all()
    if not rows:
        print("no result.json files found")
        return
    rows.sort(key=lambda r: (r["k"], r["graph"], r["case"], r["arm"]))

    print(f"{'arm':<26} {'k':>2} {'gr':<4} {'case':<8} {'mode':<20} "
          f"{'maxcap':>6} {'dText':>5} {'dTop5':>5} {'lpSpread':>9}  n")
    print("-" * 108)
    for r in rows:
        ls = r["lp_spread"]
        ls_s = f"{ls:.6f}" if isinstance(ls, (int, float)) else str(ls)
        flag = ""
        # A real corruption looks like dText>1 with a large lp spread.
        if r["distinct_texts"] > 1 and isinstance(ls, (int, float)) and ls > 0.05:
            flag = "  <== CORRUPT"
        elif r["distinct_texts"] > 1:
            flag = "  <== text varies (check margin: benign exact-tie?)"
        print(f"{r['arm']:<26} {r['k']:>2} {r['graph']:<4} {r['case']:<8} "
              f"{str(r['mode']):<20} {str(r['max_cap']):>6} "
              f"{r['distinct_texts']:>5} {r['distinct_top5']:>5} {ls_s:>9}  "
              f"{r['repeats']}{flag}")

    print()
    print("per-arm first-position top-1 logprobs (first 6 requests):")
    for r in rows:
        print(f"  k={r['k']} graph={r['graph']:<3} {r['case']:<8} "
              f"{[round(x, 4) if isinstance(x, (int, float)) else x for x in r['lp0'][:6]]}")

    print()
    print("sample texts per arm:")
    for r in rows:
        uniq = sorted({t for t in r["texts"]})
        print(f"  k={r['k']} graph={r['graph']:<3} {r['case']:<8} {uniq}")


if __name__ == "__main__":
    main()

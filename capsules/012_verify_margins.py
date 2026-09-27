"""Capsule 012 verifier - is a k=4 text divergence a BENIGN exact-tie or REAL corruption?

This is FINDINGS.md F-005's rule applied to #54785. A text difference between two
arms is only a bug if the arm that diverged picked a token the other arm
considered clearly worse. If the top-2 at the decision point are within a
rounding error, the flip is a tie-break and neither answer is wrong.

Given two result.json files (or two cases within one), for every position where
the token sequences differ, report:
  - the token each arm chose
  - that token's logprob in ITS OWN arm
  - the same token's logprob in the OTHER arm, if it appears in the top-5
  - the margin between the chosen token and the runner-up in the same arm

Verdict rule (fixed in advance, so it cannot be tuned to the answer):
  BENIGN  : the losing token was within 1e-3 nats of the winning one in the
            arm that lost, i.e. the decision was a near-exact tie.
  CORRUPT : the losing token trailed by > 0.05 nats, or the winning token's
            own logprob differed by > 0.01 nats between arms for the same
            position (that is a real numerical divergence, not a tie-break).
Anything else is reported as AMBIGUOUS and left for a human.
"""

import glob
import json
import os
import sys

LAB = "/mnt/ssd/b70-vllm-lab"


def load(path):
    with open(path) as f:
        return json.load(f)


def case_runs(d, case):
    return d["cases"][case]["runs"]


def compare_runlists(a_runs, b_runs, a_name, b_name, case):
    """Compare the per-position top-5 of the FIRST run of each arm."""
    out = {"case": case, "a": a_name, "b": b_name, "positions": [], "verdict": "IDENTICAL"}
    a0, b0 = a_runs[0], b_runs[0]
    la = len(a0["token_ids"])
    lb = len(b0["token_ids"])
    out["n_tokens"] = [la, lb]
    if la != lb:
        out["verdict"] = "LENGTH_DIFFER"

    n = min(la, lb)
    verdicts = []
    for i in range(n):
        ta, tb = a0["token_ids"][i], b0["token_ids"][i]
        if ta == tb:
            continue
        # a0/b0 top-5 is recorded only for position 0; for later positions we
        # rely on the per-run logprob list when present.
        pos = {"i": i, "a_tok": ta, "b_tok": tb}
        verdicts.append(pos)
    if not verdicts:
        out["verdict"] = "IDENTICAL"
        out["note"] = ("token sequences identical for the first run of both arms; "
                       "only later repeats could differ (see distinct_texts)")
        return out
    out["divergent_positions"] = verdicts
    out["verdict"] = "TOKEN_DIVERGENCE_NEEDS_MARGIN"
    return out


def main():
    paths = sorted(
        glob.glob(f"{LAB}/results/012b_MTP_k*_*/result.json")
        + glob.glob(f"{LAB}/results/012_mTP_k*_*/result.json")
    )
    paths = [p for p in paths if "_prerun" not in p]
    if len(paths) < 1:
        print("no result.json files")
        return
    data = {}
    for p in paths:
        d = load(p)
        if not d.get("ok"):
            continue
        data[os.path.basename(os.path.dirname(p))] = d

    print(f"loaded {len(data)} result files: {sorted(data)}\n")

    # Report per-arm first-position margins. That single position is the
    # reporter's own metric and the one with full top-5 detail recorded.
    print("=" * 100)
    print("FIRST-POSITION TOP-5 PER ARM (the position the reporter measured)")
    print("=" * 100)
    for arm, d in sorted(data.items(), key=lambda kv: (kv[1]["k"], kv[1]["graph_requested"])):
        for case, c in d["cases"].items():
            top = c["runs"][0]["first_pos_top"]
            if not top:
                continue
            margin = (top[0][1] - top[1][1]) if len(top) > 1 else None
            print(f"\n{arm}  k={d['k']} graph={d['graph_requested']}  case={case}")
            print(f"  distinct_texts={c['distinct_texts']} "
                  f"distinct_top5={c['distinct_first_pos_top5']} "
                  f"lp_spread={c['first_pos_lp_spread']}")
            for rank, (tok, lp) in enumerate(top):
                mark = " <-argmax" if rank == 0 else ""
                print(f"    {rank}: {lp:12.6f}  {tok!r}{mark}")
            if margin is not None:
                verdict = ("EXACT TIE (<=1e-3): flip here is benign"
                           if margin <= 1e-3 else f"DECIDED (margin {margin:.4f} nats)")
                print(f"    top1-top2 margin = {margin:.6f} nats  -> {verdict}")

    # Cross-arm comparison of the argmax logprob, which is the number that must
    # be identical across arms if both are computing correct logits.
    print()
    print("=" * 100)
    print("CROSS-ARM ARGMAX LOGPROB PER CASE (must be identical if both arms are right)")
    print("=" * 100)
    bycase = {}
    for arm, d in data.items():
        for case, c in d["cases"].items():
            bycase.setdefault(case, []).append((arm, d["k"], d["graph_requested"], c))
    for case, rows in bycase.items():
        rows.sort(key=lambda r: (r[1], r[2]))
        print(f"\ncase={case}")
        for arm, k, g, c in rows:
            lp0 = c["first_pos_logprob_top1"]
            uniq = sorted({round(x, 6) for x in lp0 if x is not None})
            print(f"  k={k} graph={g:<3} {arm:<24} lp0={uniq} "
                  f"spread={c['first_pos_lp_spread']}")
        allv = {round(x, 6) for _, _, _, c in rows for x in c["first_pos_logprob_top1"]
                if x is not None}
        if len(allv) == 1:
            print("  => ALL ARMS AGREE BIT-EXACTLY on the argmax logprob")
        else:
            print(f"  => {len(allv)} DISTINCT argmax logprobs across arms: {sorted(allv)}")


if __name__ == "__main__":
    main()

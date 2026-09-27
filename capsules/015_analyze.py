"""Capsule 015 analyzer - turn the per-arm JSON into a verdict table.

The decision rules, stated up front so the analysis cannot be quietly relaxed:

  BIT_STABLE      identical token_ids across all N repeats in one engine
  EXACT_TIE_FLIP  arms differ, and BOTH arms' divergent position had >1 token
                  tied at the top. This is the F-005 benign class: a
                  numerically exact tie, a coin flip, both continuations
                  coherent. NOT a finding.
  REAL_DIVERGENCE arms differ and the divergence is not a tie. This is the
                  #54785 "wrong logits" signature and IS a finding.
  NON_DETERMINISTIC distinct outputs within ONE arm at temperature 0. The
                  single most serious signature: greedy decoding must be a
                  function of the input, and graph replay must not perturb it.

A length difference is reported separately: it means one arm stopped early,
which for a correct implementation is itself a divergence.

Usage: python 015_analyze.py results/015/*.json
"""

import glob
import json
import os
import sys


def load(paths):
    docs = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
            d["_path"] = p
            docs.append(d)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(p)}: {e}")
    return docs


def arm_key(d):
    return (d.get("graph"), d.get("_path", ""))


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob("/mnt/ssd/b70-vllm-lab/results/015/*.json"))
    docs = load(paths)
    if not docs:
        print("no result JSON found")
        return 1

    print(f"loaded {len(docs)} run(s)\n")

    # ---- 1. determinism: within-arm bit stability ---------------------
    det_docs = [d for d in docs if "determinism" in d]
    if det_docs:
        print("== TASK 3 DETERMINISM ORACLE: same prompt, N repeats, temp=0 ==")
        print(f"{'run':34s} {'prompt':8s} {'distinct/N':>11s}  {'bit_stable':>10s}  lp_spread")
        verdicts = {}
        for d in det_docs:
            tag = os.path.basename(d["_path"]).replace(".json", "")
            for pname, rec in (d.get("determinism") or {}).items():
                line = (f"{tag:34s} {pname:8s} "
                        f"{rec['distinct_shas']}/{rec['repeats']:>8}  "
                        f"{str(rec['bit_stable']):>10s}  "
                        f"{rec.get('first_pos_top1_lp_spread', '-')}")
                print("  " + line)
                verdicts.setdefault(pname, []).append(
                    (d.get("graph"), rec["bit_stable"], rec["distinct_shas"]))
        print()
        for pname, vs in verdicts.items():
            unstable = [v for v in vs if not v[1]]
            status = "BIT-STABLE" if not unstable else "*** NON-DETERMINISTIC ***"
            print(f"  {pname:10s} {status}"
                  f"  (unstable in {len(unstable)}/{len(vs)} runs)")
        print()

    # ---- 2. shape dependence ------------------------------------------
    shp = [d for d in docs if "shape_dependence" in d]
    if shp:
        print("== TASK 2b SHAPE / CAPTURE-SIZE CARRYOVER ==")
        for d in shp:
            s = d["shape_dependence"]
            tag = os.path.basename(d["_path"]).replace(".json", "")
            print(f"  {tag:38s} graph={str(d.get('graph')):3s} "
                  f"forced={s['forced_widths']:2d} widths  "
                  f"probe_identical={s['probe_identical_before_after']}")
            if not s["probe_identical_before_after"]:
                print(f"      before: {s['before_text'][:150]!r}")
                print(f"      after : {s['after_text'][:150]!r}")
                print("      *** SAME PROMPT, DIFFERENT OUTPUT AFTER NEW "
                      "CAPTURES ***")
        print()

    # ---- 3. prefix caching --------------------------------------------
    pfx = [d for d in docs if "prefix_cache" in d]
    if pfx:
        print("== TASK 2c PREFIX CACHING + GRAPHS ==")
        for d in pfx:
            s = d["prefix_cache"]
            tag = os.path.basename(d["_path"]).replace(".json", "")
            print(f"  {tag:38s} graph={str(d.get('graph')):3s} "
                  f"all_stable={s['all_stable']}")
            for q, v in s["per_query"].items():
                if not v["stable"]:
                    print(f"      *** UNSTABLE: {q!r} distinct={v['distinct']}"
                          f"/{v['rounds']}")
        print()

    # ---- 4. concurrent (the #54698 hang class) ------------------------
    cc = [d for d in docs if "concurrent" in d]
    if cc:
        print("== TASK 2d CONCURRENT LOAD, HANG CLASS (#54698) ==")
        for d in cc:
            c = d["concurrent"]
            tag = os.path.basename(d["_path"]).replace(".json", "")
            if c.get("WEDGE"):
                print(f"  {tag:38s} graph={str(d.get('graph')):3s} "
                      f"*** WEDGED after {c['wall_s']}s ***")
                print(f"      waves completed: {c.get('waves_completed')}")
                for w in c.get("last_wave_lines", []):
                    print(f"      {w}")
            else:
                waves = c.get("waves", [])
                print(f"  {tag:38s} graph={str(d.get('graph')):3s} "
                      f"ok rc={c.get('returncode')} {c.get('wall_s')}s "
                      f"waves={len(waves)}")
                for w in waves:
                    print(f"      {w}")
                for e in c.get("stderr_tail", []):
                    print(f"      | {e[:150]}")
        print()

    # ---- 5. scenario errors ------------------------------------------
    errs = {d["_path"]: d["scenario_errors"] for d in docs if "scenario_errors" in d}
    if errs:
        print("== SCENARIO ERRORS (not silent) ==")
        for p, e in errs.items():
            for k, v in e.items():
                if k.endswith("_tb"):
                    continue
                print(f"  {os.path.basename(p):38s} {k}: {v[:160]}")
        print()

    # ---- 6. ON vs OFF token comparison --------------------------------
    det_on = [d for d in det_docs if d.get("graph") == "on"]
    det_off = [d for d in det_docs if d.get("graph") == "off"]
    if det_on and det_off and det_on[0].get("determinism"):
        print("== ON vs OFF: exact token comparison on identical prompts ==")
        for pname in det_on[0]["determinism"]:
            on_runs = det_on[0]["determinism"][pname].get("runs") or []
            off_runs = det_off[0]["determinism"][pname].get("runs") or []
            if not on_runs or not off_runs:
                print(f"  {pname:10s} (no logprobs kept - rerun with "
                      "--keep-logprobs)")
                continue
            a, b = on_runs[0], off_runs[0]
            same = a["token_ids"] == b["token_ids"]
            print(f"  {pname:10s} same_token_ids={same} "
                  f"(on {a['n_tokens'] if 'n_tokens' in a else len(a['token_ids'])} tok, "
                  f"off {len(b['token_ids'])} tok)")
            if not same:
                for i, (ta, tb) in enumerate(zip(a["token_ids"], b["token_ids"])):
                    if ta != tb:
                        la = a["logprobs"][i] if i < len(a["logprobs"]) else None
                        lb = b["logprobs"][i] if i < len(b["logprobs"]) else None
                        print(f"      first divergence at index {i}")
                        if la and lb:
                            tie_a = la["n_tied_at_top1"] > 1
                            tie_b = lb["n_tied_at_top1"] > 1
                            cls = ("EXACT_TIE_FLIP (benign, F-005)" if
                                   (tie_a and tie_b) else "*** REAL DIVERGENCE ***")
                            print(f"      on : {la['top1_tok']!r} "
                                  f"{la['top1_lp']:.6f} tied={la['n_tied_at_top1']}")
                            print(f"      off: {lb['top1_tok']!r} "
                                  f"{lb['top1_lp']:.6f} tied={lb['n_tied_at_top1']}")
                            print(f"      classification: {cls}")
                        break
                else:
                    print(f"      LENGTH DIFFERS: on={len(a['token_ids'])} "
                          f"off={len(b['token_ids'])} tokens")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

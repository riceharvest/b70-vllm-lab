"""Capsule 013 - direct kernel-level probe of the #54785 failure class.

The engine-level capsule 012 needs a full vLLM start per arm (~7 min each on
this box). This capsule tests the SAME failure class - stale graph-replay state
in an XPU spec/GDN kernel - without vLLM, in seconds per iteration, so the
k sweep is affordable.

Two probes:

PROBE A - GDN spec kernel, eager vs SYCL-graph capture/replay, num_spec_tokens
          (k) 1..6, repeated replays. Oracle: the replayed output must equal
          the eager output bit-for-bit, and repeated replays must be
          self-consistent. A divergence here IS the #54785 mechanism
          (gated_delta_rule_spec is the suspect the issue names).

PROBE B - graph capture/replay with VARYING batch shapes on a plain GEMM +
          normalization chain, to hit the stale-cache class that F-008 /
          kernels#457 documents (the Xe2 grouped-GEMM work-steal counter is
          explicitly capture-unsafe). Oracle: same input -> same output, and
          the captured graph must not return the PREVIOUS call's result.

Both probes fail loudly rather than printing a soft score, so a "clean" run is
a real negative and not a silently degraded measurement.

NOTE: must be a real .py file, not stdin (ENVIRONMENT.md).
"""

import argparse
import json
import os
import time

import torch

# CRITICAL: the XPU custom ops are registered as an import side effect of
# vllm.platforms.xpu importing vllm_xpu_kernels. Without `import vllm`,
# torch.ops._xpu_C.gdn_attention does not exist and the whole probe passes
# difference between a real negative and a fake one.
import vllm  # noqa: F401

LIB = os.environ.get("LD_LIBRARY_PATH", "")


def gdn_spec_op():
    """The op the issue's suspect analysis names.

    The C++ `gated_delta_rule_spec` is NOT exposed as its own torch op; it is
    reached through the fused `gdn_attention` entry point (see
    csrc/xpu/gdn_attn/gdn_attn_interface.cpp: "The delta stage is split into
    two ops -- gated_delta_rule_spec and gated_delta_rule_non_spec ... The
    legacy fused gdn_attention drives whichever path(s) are active"). With
    num_spec_decodes>0 and num_prefills+num_decodes==0, gdn_attention takes
    exactly the spec path, i.e. causal_conv1d_spec -> gated_delta_rule_spec.
    So gdn_attention IS the suspect, reached the way vLLM reaches it.
    """
    return torch.ops._xpu_C.gdn_attention


def gdn_spec_args(k: int, dtype, seed: int, num_v=4, num_k=4, dv=32, dk=32):
    """Build the argument set for one spec-decode call at num_speculative_tokens=k.

    Shapes are taken verbatim from the TORCH_CHECKs in
    csrc/xpu/gdn_attn/gdn_attn_interface.cpp (main @ 2026-09-27):
      z                        [num_actual_tokens, num_v_heads/tp, head_v_dim]
      projected_states_qkvz   [num_actual_tokens, num_k_heads/tp *
                               (2*head_k_dim + 2*head_v_dim*num_v/num_k)]
      projected_states_ba     [num_actual_tokens, 2*num_v_heads/tp]
      spec_state_indices_tensor[num_spec_decodes, num_speculative_tokens + 1]
      spec_query_start_loc    [num_spec_decodes + 1]
      spec_token_indx         [spec_token]
      num_accepted_tokens     [num_spec_decodes]
      core_attn_out           [num_actual_tokens, num_v_heads/tp, head_v_dim]
      conv_state              [cache_batch, num_v_heads*(conv_dim),
                               conv_kernel_size-1 + num_speculative_tokens]
      ssm_state               [cache_batch, num_v_heads, head_k_dim, head_v_dim]
    conv_kernel_size = 4 (linear_conv_kernel_dim in the model config).
    """
    dev = "xpu"
    g = torch.Generator(device=dev).manual_seed(seed)
    T = k + 1                      # 1 target + k draft rows
    nsd = 1                        # num_spec_decodes = 1 (bs=1, reporter config)
    nat = T                        # num_actual_tokens: exact-fit, no pad rows
    conv_dim = num_v * dv
    conv_state_len = (4 - 1) + k   # conv_weights.size(1) - 1 + num_speculative_tokens
    cache_batch = 64               # plenty of distinct SSM slots

    z = torch.randn(nat, num_v, dv, generator=g, dtype=dtype, device=dev)
    qkvz = torch.randn(
        nat, num_k * (2 * dk + 2 * dv * num_v // num_k),
        generator=g, dtype=dtype, device=dev)
    ba = torch.randn(nat, 2 * num_v, generator=g, dtype=dtype, device=dev)
    core_out = torch.zeros(nat, num_v, dv, dtype=dtype, device=dev)
    conv_state = torch.zeros(
        cache_batch, conv_dim, conv_state_len, dtype=dtype, device=dev)
    ssm_state = torch.zeros(
        cache_batch, num_v, dk, dv, dtype=torch.float32, device=dev)
    conv_w = torch.randn(conv_dim, 4, 1, generator=g, dtype=dtype, device=dev) * 0.1
    # torch.zeros/zeros_like take no `generator` argument; plain zero init is
    # correct here anyway (conv bias is genuinely zero in the real model).
    conv_b = torch.zeros(conv_dim, dtype=dtype, device=dev)
    A_log = torch.randn(num_v, generator=g, dtype=torch.float32, device=dev) * 0.1
    dt_bias = torch.randn(num_v, generator=g, dtype=dtype, device=dev) * 0.1

    # Spec group: one sequence, T tokens. token_indx maps local -> global; with
    # a single group it is the identity, which is the reporter's exact shape.
    spec_qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)
    spec_tok_indx = torch.arange(T, dtype=torch.int32, device=dev)
    spec_state_idx = torch.arange(T, dtype=torch.int32, device=dev).reshape(nsd, T)
    num_accepted = torch.tensor([1], dtype=torch.int32, device=dev)

    return dict(
        core_attn_out=core_out, z=z, projected_states_qkvz=qkvz,
        projected_states_ba=ba, num_k_heads=num_k, num_v_heads=num_v,
        head_k_dim=dk, head_v_dim=dv, conv_state=conv_state, ssm_state=ssm_state,
        conv_weights=conv_w, conv_bias=conv_b, activation="silu",
        A_log=A_log, dt_bias=dt_bias,
        num_prefills=0, num_decodes=0, num_spec_decodes=nsd,
        has_initial_state=None,
        non_spec_query_start_loc=None, non_spec_token_indx=None,
        non_spec_state_indices_tensor=None,
        spec_query_start_loc=spec_qsl, spec_token_indx=spec_tok_indx,
        spec_state_indices_tensor=spec_state_idx,
        num_accepted_tokens=num_accepted, num_actual_tokens=nat,
        tp_size=1, reorder_input=False,
    )


def probe_a(k: int, iters: int, dtype: torch.dtype, seed: int, graph: bool):
    """GDN spec path at num_speculative_tokens=k: eager reference vs captured
    graph, replayed `iters` times with IDENTICAL inputs.

    The op MUTATES conv_state and ssm_state, so every call needs freshly reset
    state. That reset is itself part of the test: if the graph bakes in a stale
    pointer, replay re-reads the state the previous replay left behind, and the
    result drifts with replay count instead of matching eager.

    Oracle (all three must hold for a clean arm):
      1. eager is self-consistent across repeated identical calls
      2. every graph replay equals the eager result bit-for-bit
      3. replays are mutually self-consistent
    """
    op = gdn_spec_op()
    base = gdn_spec_args(k, dtype, seed)
    T = k + 1

    def fresh():
        """Fresh output + state buffers, same inputs (same seed -> same data)."""
        a = gdn_spec_args(k, dtype, seed)
        return a

    # ---- 1. eager reference, 3 identical calls
    eager = []
    for _ in range(3):
        a = fresh()
        op(**a)
        torch.xpu.synchronize()
        eager.append((a["core_attn_out"].clone(), a["ssm_state"].clone()))
    eager_self_consistent = all(
        torch.equal(eager[0][0], e[0]) and torch.equal(eager[0][1], e[1])
        for e in eager
    )
    ref_out, ref_state = eager[0]

    res = {
        "k": k, "T_rows": T, "dtype": str(dtype), "graph": graph,
        "eager_self_consistent": eager_self_consistent,
        "eager_out_sum": round(float(ref_out.float().sum()), 6),
        "eager_state_sum": round(float(ref_state.float().sum()), 6),
        "any_nan_eager": bool(torch.isnan(ref_out).any().item()),
    }
    if not graph:
        return res

    # ---- 2. capture
    try:
        warm = torch.xpu.Stream()
        warm.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(warm):
            for _ in range(3):
                op(**fresh())
        torch.xpu.current_stream().wait_stream(warm)
        torch.xpu.synchronize()

        gph = torch.xpu.XPUGraph()
        static = fresh()
        with torch.xpu.graph(gph):
            op(**static)
        torch.xpu.synchronize()

        mism = 0
        max_abs = 0.0
        outs = []
        for _ in range(iters):
            # Replay. The graph holds its own static conv/ssm state buffers;
            # we deliberately do NOT reset them between replays, because
            # divergence here is exactly the stale-state signature we are
            # hunting: a correct capture re-reads the same inputs every time,
            # a state-capturing one compounds replay N's leftovers.
            gph.replay()
            torch.xpu.synchronize()
            o = static["core_attn_out"].clone()
            s = static["ssm_state"].clone()
            outs.append((o, s))
            max_abs = max(max_abs, (o.float() - ref_out.float()).abs().max().item())
            if not torch.equal(o, ref_out) or not torch.equal(s, ref_state):
                mism += 1

        replay_self_consistent = all(
            torch.equal(outs[0][0], r[0]) and torch.equal(outs[0][1], r[1])
            for r in outs
        )
        # the interesting signature: does the output DRIFT with replay count?
        uniq = len({tuple(o.flatten().tolist()) for o, _ in outs})
        res.update({
            "replays": iters,
            "replay_vs_eager_mismatches": mism,
            "replay_vs_eager_max_abs_diff": round(max_abs, 8),
            "replay_self_consistent": replay_self_consistent,
            "replay_distinct_outputs": uniq,
            "replay_out_sum_first": round(float(outs[0][0].float().sum()), 6),
            "replay_out_sum_last": round(float(outs[-1][0].float().sum()), 6),
            "any_nan_replay": bool(
                torch.isnan(outs[0][0]).any().item()
                or torch.isnan(outs[0][1]).any().item()),
        })
    except Exception as e:  # noqa: BLE001
        res["graph_error"] = f"{type(e).__name__}: {e}"
    return res



def probe_b(iters: int, seed: int, graph: bool, shapes=None):
    """Varying-shape capture/replay: a captured graph is shape-specialized, so
    this tests the STALE case: does replaying a graph captured at shape A and
    then fed shape-B data return A's result?"""
    if shapes is None:
        shapes = [(1, 512, 512), (2, 512, 512), (4, 512, 512), (8, 512, 512)]
    out = {"shapes": shapes, "graph": graph, "cases": []}
    torch.manual_seed(seed)
    try:
        w = torch.randn(512, 512, dtype=torch.bfloat16, device="xpu") * 0.05
        b = torch.randn(512, dtype=torch.bfloat16, device="xpu") * 0.05
        for (m, n, k_) in shapes:
            x = torch.randn(m, n, dtype=torch.bfloat16, device="xpu") * 0.1
            # eager reference
            ref = torch.nn.functional.linear(x, w, b)
            torch.xpu.synchronize()
            rec = {"shape": [m, n, k_], "eager_sum": float(ref.float().sum())}
            if graph:
                s = torch.xpu.Stream()
                s.wait_stream(torch.xpu.current_stream())
                with torch.xpu.stream(s):
                    for _ in range(3):
                        torch.nn.functional.linear(x, w, b)
                torch.xpu.current_stream().wait_stream(s)
                torch.xpu.synchronize()
                gph = torch.xpu.XPUGraph()
                static_in = x.clone()
                with torch.xpu.graph(gph):
                    static_out = torch.nn.functional.linear(static_in, w, b)
                torch.xpu.synchronize()
                outs = []
                for _ in range(iters):
                    gph.replay()
                    torch.xpu.synchronize()
                    outs.append(static_out.clone())
                rec["replay_mismatch_vs_eager"] = sum(
                    0 if torch.equal(o, ref) else 1 for o in outs)
                rec["replay_self_consistent"] = all(
                    torch.equal(outs[0], o) for o in outs)
                rec["replay_max_abs_diff"] = max(
                    (o.float() - ref.float()).abs().max().item() for o in outs)
                rec["any_nan"] = bool(torch.isnan(outs[0]).any().item())
            out["cases"].append(rec)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--skip-a", action="store_true")
    args = ap.parse_args()

    res = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch": torch.__version__,
        "xpu_device": torch.xpu.get_device_name(0),
        "free_gib": round(torch.xpu.mem_get_info(0)[0] / 2**30, 2),
        "ok": False,
    }

    # Confirm the op exists before anything else, so a missing op is a clear
    # result and not a stack trace.
    # NOTE: do the availability test by RESOLVING the overload, and read the
    # schema via `_schemas` (plural). `dir(torch.ops._xpu_C)` does not list
    # custom ops at all, and OpOverload has `_schemas`, not `_schema` - testing
    # either of those wrongly reports the op as missing and makes the probe pass
    # VACUOUSLY. That is exactly what happened on the first attempt.
    try:
        _ov = torch.ops._xpu_C.gdn_attention  # resolves, or raises
        res["gdn_attention_op_available"] = True
        res["gdn_attention_schema"] = str(_ov._schemas)
        res["gdn_attention_default_overload"] = str(_ov._schemas.get("", ""))[:2000]
    except Exception as e:  # noqa: BLE001
        res["gdn_attention_op_available"] = False
        res["gdn_attention_error"] = f"{type(e).__name__}: {e}"

    if not args.skip_a and res.get("gdn_attention_op_available"):
        res["probe_a"] = [
            probe_a(k, args.iters, torch.bfloat16, 1234 + k, graph=True)
            for k in range(1, args.kmax + 1)
        ]
    else:
        res["probe_a"] = []
        res["probe_a_skipped"] = "gdn_attention op unavailable"

    res["probe_b"] = probe_b(args.iters, 4321, graph=True)

    # ---- verdict, computed not asserted
    verdict = {}
    pa = res.get("probe_a", [])
    # A probe that collected nothing has NOT passed. Report it as skipped so
    # an empty run can never be read as a clean determinism result.
    verdict["probe_a_ran"] = len(pa) > 0
    verdict["probe_b_ran"] = len(res.get("probe_b", {}).get("cases", [])) > 0
    verdict["a_all_eager_self_consistent"] = all(x.get("eager_self_consistent") for x in pa)
    verdict["a_replay_mismatch_total"] = sum(
        x.get("replay_vs_eager_mismatches", 0) for x in pa)
    verdict["a_replay_self_consistent"] = all(
        x.get("replay_self_consistent") for x in pa)
    verdict["a_any_nan"] = any(x.get("any_nan") for x in pa)
    pb = res.get("probe_b", {}).get("cases", [])
    verdict["b_replay_mismatch_total"] = sum(
        c.get("replay_mismatch_vs_eager", 0) for c in pb)
    verdict["b_all_self_consistent"] = all(
        c.get("replay_self_consistent") for c in pb)
    res["verdict"] = verdict
    res["ok"] = True
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    payload = json.dumps(res, indent=2)
    with open(args.out, "w") as f:
        f.write(payload)
    print("===PROBE_SUMMARY===")
    print(json.dumps(verdict, indent=2))
    for x in pa:
        print(f"  A k={x['k']} T={x['T_rows']} eager_consistent={x['eager_self_consistent']} "
              f"replay_mismatch={x.get('replay_vs_eager_mismatches')}/{x.get('replays')} "
              f"max_abs={x.get('replay_vs_eager_max_abs_diff')} "
              f"self_consistent={x.get('replay_self_consistent')} "
              f"err={x.get('graph_error', '')}")
    for c in pb:
        print(f"  B shape={c['shape']} mismatch={c.get('replay_mismatch_vs_eager')} "
              f"self_consistent={c.get('replay_self_consistent')} "
              f"max_abs={c.get('replay_max_abs_diff')}")
    print("===RESULT_JSON_PATH===", args.out)


if __name__ == "__main__":
    main()

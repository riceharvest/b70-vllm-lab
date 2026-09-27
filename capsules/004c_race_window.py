#!/usr/bin/env python
"""Capsule 004c — is the eager-mode race window actually closed on B70?

004 proved the poisoned 1-elem int32 block reaches the kernel's at::empty
intact (5/5), i.e. the scheduler counter really is uninitialized at 0.1.14.1.
But no wrong output appeared. Hypothesis (matches upstream PR #457's own
analysis): in eager submits, workgroup 0 / lane 0 runs `atm.store(0)` as the
kernel's FIRST instruction, while the first `atomicAdd` only happens after a
full GEMM tile has been computed. The store therefore wins the race essentially
always, and the uninitialized value is masked.

This script runs the poisoned path many times at a size that heavily exercises
the persistent steal loop, and reports how often output is wrong.
"""
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401

DEV = "xpu"
POISON = 100_000
ITERS = 60


def main():
    torch.manual_seed(7)
    E, K, N = 8, 1024, 1024
    TOTAL_M = 32768  # 4096 rows/expert -> many tiles -> steal loop runs a lot
    A = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEV).contiguous()
    B = torch.randn(E, K, N, dtype=torch.bfloat16, device=DEV).contiguous()
    rows = torch.full((E,), TOTAL_M // E, dtype=torch.int32, device=DEV)

    ref = torch.empty(TOTAL_M, N, dtype=torch.bfloat16, device=DEV)
    p = 0
    for i in range(E):
        r = TOTAL_M // E
        ref[p:p + r] = (A[p:p + r].float() @ B[i].float()).to(torch.bfloat16)
        p += r
    torch.xpu.synchronize()

    poison_intact = 0
    wrong = 0
    maxerr = 0.0
    for it in range(ITERS):
        # Recycle a poisoned 1-elem int32 block, then call the op with no
        # intervening same-size allocation.
        t = torch.full((1,), POISON, dtype=torch.int32, device=DEV)
        del t
        chk = torch.empty(1, dtype=torch.int32, device=DEV)
        if chk.item() == POISON:
            poison_intact += 1
        del chk
        t = torch.full((1,), POISON, dtype=torch.int32, device=DEV)
        del t

        D = torch.empty(TOTAL_M, N, dtype=torch.bfloat16, device=DEV)
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=A, ptr_A_scale=None, ptr_B=B, ptr_B_scale=None, ptr_bias=None,
            ptr_D=D, rows_per_expert=rows, N=N, K=K, num_experts=E)
        torch.xpu.synchronize()
        e = (D.float() - ref.float()).abs().max().item()
        maxerr = max(maxerr, e)
        if e > 1.0:  # bf16 GEMM noise is <=0.5 in the clean smoke test
            wrong += 1

    print(f"iterations                  : {ITERS}")
    print(f"poison intact at at::empty  : {poison_intact}/{ITERS}")
    print(f"iterations with wrong output: {wrong}/{ITERS}")
    print(f"worst max|err| observed     : {maxerr:.4f}")
    print()
    if wrong == 0:
        print("CONCLUSION: counter IS uninitialized, but the in-kernel store(0)")
        print("            wins the race on every eager launch -> latent, not")
        print("            observable in eager mode on this device.")
    else:
        print("CONCLUSION: race manifested in eager mode.")


if __name__ == "__main__":
    main()

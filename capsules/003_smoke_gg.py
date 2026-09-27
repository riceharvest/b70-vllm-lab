# Smoke test: can we call the Xe2 grouped GEMM directly and get a correct result?
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers the op)

DEV = "xpu"
torch.manual_seed(7)

E, K, N = 8, 1024, 1024
TOTAL_M = 4096

A = torch.randn(TOTAL_M, K, dtype=torch.bfloat16, device=DEV).contiguous()
B = torch.randn(E, K, N, dtype=torch.bfloat16, device=DEV).contiguous()
rows = torch.full((E,), TOTAL_M // E, dtype=torch.int32, device=DEV)
D = torch.empty(TOTAL_M, N, dtype=torch.bfloat16, device=DEV)

torch.ops._xpu_C.cutlass_grouped_gemm_interface(
    ptr_A=A, ptr_A_scale=None, ptr_B=B, ptr_B_scale=None, ptr_bias=None,
    ptr_D=D, rows_per_expert=rows, N=N, K=K, num_experts=E)
torch.xpu.synchronize()

# reference: per-expert matmul
ref = torch.empty_like(D)
p = 0
r_cpu = rows.cpu().tolist()
for i, r in enumerate(r_cpu):
    ref[p:p + r] = (A[p:p + r].float() @ B[i].float()).to(torch.bfloat16)
    p += r
torch.xpu.synchronize()

err = (D.float() - ref.float()).abs()
print("rows_per_expert:", r_cpu)
print("max abs err :", err.max().item())
print("mean abs err:", err.mean().item())
print("nan in out  :", bool(torch.isnan(D).any().item()))
print("allclose    :", torch.allclose(D.float(), ref.float(), rtol=2e-2, atol=1e-2))

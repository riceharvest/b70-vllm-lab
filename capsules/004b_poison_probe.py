import torch

DEV = "xpu"
POISON = 100_000

# 1) Does the caching allocator hand the SAME block back after free?
t = torch.full((1,), POISON, dtype=torch.int32, device=DEV)
addr = t.data_ptr()
del t

# torch.empty == exactly what the kernel's at::empty does. Read it WITHOUT writing.
u = torch.empty(1, dtype=torch.int32, device=DEV)
print("freed addr   :", hex(addr))
print("realloc addr :", hex(u.data_ptr()))
print("same block   :", u.data_ptr() == addr)
print("value seen   :", u.item(), "  (== POISON means at::empty returns "
      "uninitialized memory)")
del u

# 2) Repeat: is it stable across several cycles?
ok = 0
for i in range(5):
    a = torch.full((1,), POISON, dtype=torch.int32, device=DEV)
    pa = a.data_ptr()
    del a
    b = torch.empty(1, dtype=torch.int32, device=DEV)
    if b.data_ptr() == pa and b.item() == POISON:
        ok += 1
    del b
print(f"recycled-with-poison-intact: {ok}/5")

# Agent Contract — B70 Lab

**Read this before running ANY GPU work.** It exists because the single-lane rule
was violated in practice, not hypothetically.

## The one hard rule

**The B70 is ONE lane. Never run two GPU jobs at once.**

Two concurrent capsules contend for VRAM, distort each other's timings, and can
OOM each other mid-run. Numbers from a contended run are **junk, not merely
imprecise** — they are worse than no number, because they look real.

This actually happened on 2026-09-27: two agents ran capsules concurrently and
held 9.18 GiB between them. See FINDINGS.md F-009.

## How to run GPU work — always this way

```bash
bash /mnt/ssd/b70-vllm-lab/tools/with_gpu_lock.sh --name "<your-label>" <command>
```

The lock blocks until the lane is free, prints who holds it and for how long,
and **hard-fails** if VRAM is already dirty from a process that bypassed it.

Do **not** invoke a capsule script directly. Do **not** set
`VLLM_XPU_ENABLE_XPU_GRAPH` or `LD_LIBRARY_PATH` yourself — the capsule drivers
already do that, and getting it wrong reintroduces the SYCL segfault
(FINDINGS.md F-004).

## Before you claim a benchmark number

- Interleave your arms (A B B A B A). Never compare against a previous run.
- Record: exact versions, the lock receipt line, and that the desktop was live
  (~1.0 GiB baseline, nondeterministic compositor load).
- A greedy decode can flip at an **exact 0.000000-nat tie** between graph-on and
  graph-off. That is benign and already documented (F-005). Only divergence
  larger than that is a real finding.
- Keep each run under ~3 minutes. If something needs longer, say why.

## Environment (already built — do NOT reinstall)

```bash
# Always: LD_LIBRARY_PATH shim + the venv python
LD_LIBRARY_PATH=$HOME/.local/lib /mnt/ssd/b70-venv/bin/python ...
```

- venv `/mnt/ssd/b70-venv` (py3.12), torch 2.13.0+xpu, vllm 0.30.0+xpu,
  vllm_xpu_kernels 0.1.15.4
- **Never** source `oneapi setvars.sh` — it shadows the system UR loader and
  breaks torch XPU with `undefined symbol: urDeviceWaitExp`.
- triton's Intel backend carries a **local patch**
  (`tools/apply_f004_fix.sh`). Never revert it: without it every engine start
  segfaults. Upstream issue intel/intel-xpu-backend-for-triton#8200.
- The GPU is **shared with the user's live X11 desktop**. Never kill it, never
  stop plasmashell, never run anything that starves the compositor.

## Division of labour

The scarce resource is GPU-seconds, not agent-hours. So:

- **Most agents must not touch the GPU.** Write patches, tests, reproducers,
  and upstream issues. That work is unlimited and parallel.
- Only a small number of capsules run at a time, through the lock.
- If a workstream needs GPU time, get to a *specific, testable hypothesis* first.
  "Run it and see" is not a hypothesis and does not deserve the lane.

## Honesty rules (non-negotiable)

- Never report a benchmark number you did not measure.
- Never claim hardware verification you did not perform.
- A negative result is a real result. "Ran it 60 times, could not reproduce,
  here is what that constrains" is valuable — and must **not** be phrased as
  refuting a bug reported on different hardware (e.g. 2x-B70).
- Never file an upstream issue without checking for an existing one first,
  including closed issues and PRs. Two issues filed in this lab turned out to
  be already-fixed upstream.
- If a fix does not work, say so and leave the negative result in FINDINGS.md.

## Where things live

| Path | What |
|---|---|
| `/mnt/ssd/b70-vllm-lab/FINDINGS.md` | every result, including dead ends |
| `/mnt/ssd/b70-vllm-lab/ENVIRONMENT.md` | full env + trap list |
| `/mnt/ssd/b70-vllm-lab/capsules/` | GPU capsules, numbered |
| `/mnt/ssd/b70-vllm-lab/patches/` | verified patches |
| `/mnt/ssd/b70-vllm-lab/tools/` | lock, VRAM probe, upgrade scripts |
| `/mnt/ssd/b70-vllm-lab/results/` | machine-readable results |
| `riceharvest/vllm` | public fork — branches per issue |
| `riceharvest/vllm-xpu-kernels` | public kernels fork |
| `riceharvest/b70-vllm-lab` | this lab (no vLLM source here) |

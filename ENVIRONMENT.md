# B70 Lab — Environment Manifest

Captured 2026-09-27. This is the ground truth every experiment must be measured against.

## Host

| Field | Value |
|---|---|
| OS | Fedora Linux 43 (kernel 7.1.8-100.fc43.x86_64) |
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 threads |
| RAM | 62 GiB total |
| Kernel GPU driver | `xe` (Battlemage / Gen20), `i915` not in use |
| DRM node | `/dev/dri/renderD128`, card1 |

## GPU

| Field | Value |
|---|---|
| Device | Intel Arc Pro B70 (Battlemage G31) |
| PCI | `09:00.0` `8086:e223` |
| VRAM | 32 GB |
| Level Zero driver | 26.18.38308.4 |
| OpenCL driver | 26.18.38308.4 |
| UR (Unified Runtime) | 1.15.38308+4 |
| IGC | 2.34.4 |
| Compute Runtime | 26.18.38308.4 |
| oneAPI | 2025.3 at `/home/dario/oneapi` |

`sycl-ls` reports:
```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2,
    Intel(R) Arc(TM) Pro B70 Graphics 20.2.0 [1.15.38308+4]
```

## Toolchain availability — NON-OBVIOUS, read this

oneAPI 2025.3 is installed but **not on PATH by default**. Every XPU shell must start with:

```bash
source /home/dario/oneapi/setvars.sh
```

| Tool | Status | Notes |
|---|---|---|
| `sycl-ls` | works, needs setvars | canonical device probe |
| `clinfo` | works | `/usr/bin/clinfo`, no setvars needed |
| `gputop` | installed to `~/.local/bin` | **requires a TTY** — prints nothing when piped/redirected |
| `intel_gpu_top` | installed to `~/.local/bin` | **does NOT support the `xe` driver**; it errors and points at `gputop` |
| `xpu-smi` | **NOT AVAILABLE on Fedora 43** | not packaged in any enabled repo; `oneapi-level-zero` ships only the loader, not the tool |
| `sudo` | password-gated | used only for installs, never for experiment runs |

`gputop` is the authoritative GPU monitor here. Invocation:

```bash
gputop -d 1 -n 3      # 1s delay, 3 samples
```

It reports per-process rcs/vcs/vecs/bcs engine busy %, MEM/RSS, and
`Frequency(MHz) GT0-<cur>/<max> GT1-<cur>/<max>`. Use GT clocks as the
XMX-utilization proxy: if GT0 is pinned at max while decode throughput is low,
the GPU is genuinely saturated; if it is far below max, the bottleneck is host-side.

## KNOWN MEASUREMENT CONTAMINATION

**The X11 desktop shares this GPU.** Confirmed GPU-resident processes at capture time:

```
 6449      37M      37M | Xwayland
 6212      96M      96M | plasma-keyboard
 6855     215M     215M | plasmashell
 8364      69M      28M | xwaylandvideobridge
56732     227M     227M | brave
```

Consequences for benchmarking:
- ~0.6 GB VRAM is unavailable to vLLM versus a headless box.
- Compositor repaints add non-deterministic engine load and can distort
  short-window kernel timings.
- Any decode tok/s measured on the desktop session is a **lower bound** and is
  not comparable to a headless run.

Mitigation for any performance claim: record whether the run was headless, and
prefer stopping the compositor for benchmark windows. Never compare a desktop
number against a headless number.

## Python environments — state as found

| Path | State |
|---|---|
| `/mnt/ssd/b70-prep/spark-dspark/.venv` | **BROKEN** — `ModuleNotFoundError: No module named 'torch.library'` |
| `/mnt/ssd/b70-prep/quant-venv` | **BROKEN** — same torch failure |
| `/home/dario/.venvs/sglang-quant` | works but **CUDA** torch 2.10.0+cu128, `xpu.is_available() == False` |
| `/home/dario/vllm-gemma4-venv` | works but **CUDA** torch 2.11.0+cu130, `xpu.is_available() == False` |

`/mnt/ssd/b70-prep/vllm-src` is a **non-git partial dump** (`vllm/` + `csrc/`
only, no `.git`, no requirements files). Its
`vllm/platforms/xpu.py` does a hard `import vllm_xpu_kernels._C` / `._moe_C` /
`._xpu_C`, so the kernels package is a mandatory runtime dep on XPU.

**As found, there was no working vLLM-on-XPU environment on this host.** The
only prebuilt kernels wheel present was
`/mnt/ssd/b70-prep/xpu-wheel-out/vllm_xpu_kernels-0.1.15.dev0+g6d92b1b.d20260917-cp312-cp312-linux_x86_64.whl`.

## Disk

| Mount | Size | Free |
|---|---|---|
| `/` | 929G | 115G (88% used) |
| `/mnt/ssd` | 932G | 179G (81% used) |

Put large model caches and venvs on `/mnt/ssd`. Watch `/` — 88% full.

## Repos

| Repo | Role |
|---|---|
| `riceharvest/vllm` | public fork of `vllm-project/vllm`, staging area for PRs |
| `riceharvest/vllm-xpu-kernels` | public fork of `vllm-project/vllm-xpu-kernels` |
| `riceharvest/b70-vllm-lab` | coordination only: runner, benchmarks, manifests, results DB. **No vLLM source.** |

## Reproduce the probe

```bash
source /home/dario/oneapi/setvars.sh
sycl-ls
lspci | grep -i battlemage
lsmod | grep xe
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

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

### CRITICAL: do NOT source setvars.sh before running torch XPU

This is the single most expensive trap on this host, and it is the opposite of
what you would guess.

`setvars.sh` prepends oneAPI 2025.3's bundled `libsycl.so.9` / older Unified
Runtime to `LD_LIBRARY_PATH`. That **shadows the newer system UR loader**
(`/usr/lib64/libze_loader.so.1`, from `oneapi-level-zero` 1.28.6) that
`torch 2.13.0+xpu` is linked against. Result:

```
ImportError: libsycl.so.9: undefined symbol: urDeviceWaitExp, version LIBUR_LOADER_0.12
```

**Rule: run every vLLM/torch XPU process with oneAPI NOT sourced.**

```bash
env -u LD_LIBRARY_PATH python ...        # correct
source /home/dario/oneapi/setvars.sh && python ...   # BROKEN, undefined symbol
```

Use oneAPI **only** for offline compilation (`icpx`, `sycl-ls`, CMake builds),
never in the same shell as a torch XPU run. If you must have both, re-append the
system loader *after* setvars rather than before it.

### Verified working stack

| Field | Value |
|---|---|
| Python | 3.12.12 (**mandatory** — the kernels wheel is cp312-specific) |
| venv | `/mnt/ssd/b70-venv` |
| torch | 2.13.0+xpu (matches vLLM 0.30.0's `torch==2.13.0` pin) |
| triton | triton-xpu 3.7.2 (+ `triton==3.7.2+xpu` shim from wheels.vllm.ai) |
| oneMKL | 2026.0.0 |
| vLLM | 0.30.0 **XPU wheel** + vllm_xpu_kernels 0.1.14.1 |

### CRITICAL: `pip install vllm` gives you the CUDA build

**The default `vllm==0.30.0` on PyPI is the CUDA wheel.** Installing it into an
XPU venv silently succeeds and produces a broken environment. Observed damage:

- ~2 GB of `nvidia-*`, `nvidia-cutlass-dsl`, `flashinfer-python` packages pulled in
- `triton==3.8.0` (CUDA) installed **alongside** `triton-xpu 3.7.2`, shadowing it
- vLLM logs `triton not found; flop counting will not work for triton kernels`
- **`vllm.platforms.current_platform` resolves to `None`** — the CUDA wheel has no
  XPU platform plugin at all, so nothing selects the B70

`torch.xpu.is_available()` still returns `True` after this, which makes the
breakage easy to miss. **Check `current_platform`, not just torch.**

Correct install — the XPU wheel is a GitHub release asset, and **both** extra
indexes are mandatory (`wheels.vllm.ai/xpu/` serves the only wheel satisfying
vLLM's `triton==3.7.2+xpu` pin; without it the resolve fails):

```bash
uv venv --python 3.12 /mnt/ssd/b70-venv
uv pip install --python /mnt/ssd/b70-venv/bin/python \
  torch==2.13.0+xpu torchvision==0.28.0+xpu torchaudio==2.11.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu

uv pip install --python /mnt/ssd/b70-venv/bin/python \
  "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+xpu-cp38-abi3-manylinux_2_34_x86_64.whl" \
  "vllm_xpu_kernels==0.1.14.1" \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  --extra-index-url https://wheels.vllm.ai/xpu/ \
  --index-strategy unsafe-best-match
```

Pin `vllm_xpu_kernels` exactly — 0.30.0 pairs with **0.1.14.1** (PyPI also has
0.1.15.4, but the ABI is matched to the release; don't float it).
`vllm_xpu_kernels` comes from **PyPI**, not wheels.vllm.ai.

**Post-install verification — all three must hold:**

```bash
env -u LD_LIBRARY_PATH /mnt/ssd/b70-venv/bin/python -c "
import vllm, torch, vllm_xpu_kernels
from vllm.platforms import current_platform
print(vllm.__version__, torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())
print('platform:', current_platform.device_name)"
```

Expect `0.30.0 2.13.0+xpu True 1` and `platform: Intel(R) Arc(TM) Pro B70 Graphics`.
Also confirm no contamination: `uv pip list | grep -i nvidia` must be empty.

### Where the XPU kernels actually live

XPU custom SYCL kernels were **migrated out of vLLM** into a separate repo:
`vllm-project/vllm-xpu-kernels` (RFC vllm-project/vllm#33214, closed Jun 2026).
`csrc/` in vLLM main has **no `xpu/` directory** any more.

Integration is import-time op registration: `vllm/platforms/xpu.py` does
`import vllm_xpu_kernels._C / ._moe_C / ._xpu_C`, and that import *is* the
integration. There is no vLLM-side build step for XPU. Extensions: `_C`
(norm/activation/RoPE/cache/quant/topk), `_vllm_fa2_C` (flash attn), `_moe_C`,
`_xpu_C` (LoRA, grouped GEMM, GDN, MQA logits, samplers), `xpumem_allocator`.

Building from source is only justified when the `default` attention kernel
presets miss a model/head_size combination (e.g. Gemma-2 at head_size 256 needs
the `full` preset, ~60 min build vs ~2 min).

### Device properties (from torch, authoritative)

```
name                  Intel(R) Arc(TM) Pro B70 Graphics
device_id             0xE223
driver_version        1.15.38308+4
total_memory          32656 MB
gpu_eu_count          256
gpu_subslice_count    32
sub_group_sizes       [16, 32]
max_compute_units     256
max_work_group_size   1024
memory_clock_rate     2800 MHz
memory_bus_width      64-bit
last_level_cache      24576 KB
local_mem_size        128 KB
has_fp16/bf16/fp64    yes / yes / yes
```

Note `local_mem_size = 128 KB` — very small scratch. This is the kind of
hardware constraint that shapes kernel design: no SLM-heavy tiling strategy will
fit, so XPU kernels must lean on the 24 MB LLC and plain USM instead.

### Measured GEMM baseline (4096³, 30 iters, desktop session running)

| dtype | time | throughput |
|---|---|---|
| float16 | 75.9 ms | **54.3 TFLOP/s** |
| bfloat16 | 115.6 ms | **35.7 TFLOP/s** |

All outputs finite. This is the reference point for any future kernel work —
bf16 running 1.5x slower than fp16 is itself a lead worth a workstream
(oneDNN bf16 path vs fp16 path on Battlemage).

### Monitoring tools

| Tool | Status | Notes |
|---|---|---|
| `gputop` | `~/.local/bin` | **authoritative monitor. Requires a TTY** — piped it prints nothing and exits 0, which falsely reads as "GPU idle". |
| `intel_gpu_top` | `~/.local/bin` | does **not** support the `xe` driver; errors and points at `gputop` |
| `xpu-smi` | **does not exist on Fedora 43** | not in any enabled repo; `oneapi-level-zero` ships only the loader. Don't hunt for it. |
| `sycl-ls` | oneAPI 2025.3 | device probe; needs oneAPI sourced, which is fine (no torch involved) |
| `clinfo` | `/usr/bin/clinfo` | works with no oneAPI |
| VRAM free | `torch.xpu.mem_get_info(0)` | the only reliable source; see below |

The `xe` driver exposes **no** sysfs VRAM accounting
(`/sys/class/drm/card1/device/mem_info_vram_*` absent, `card1/client*` empty).
Query VRAM through torch, not sysfs.

> A ctypes probe against the `ext_intel_free_memory` Level Zero aspect was
> attempted and **segfaulted** — the vendor struct layout is version-sensitive and
> undocumented in the installed headers. Don't repeat it; use
> `torch.xpu.mem_get_info()`.

## KNOWN MEASUREMENT CONTAMINATION

**The X11 desktop shares this GPU**, but the cost is small and quantified.
Measured at env-build time with torch:

```
VRAM free: 30.90 GiB / 31.89 GiB total   ->   desktop holds ~1.0 GiB
```

That is **exactly the 1 GB reserve requested**, and it is already available
without touching the live desktop session. No compositor stop is needed for
capacity reasons.

Confirmed GPU-resident desktop processes:

```
 6449      37M      37M | Xwayland
 6212      96M      96M | plasma-keyboard
 6855     215M     215M | plasmashell
 8364      69M      28M | xwaylandvideobridge
56732     227M     227M | brave
```

What still matters for measurement quality, even with the memory available:

- Compositor repaints add **nondeterministic engine load**, which distorts
  short-window kernel timings. The GEMM numbers above (54.3 / 35.7 TFLOP/s) were
  taken with the desktop live, so treat them as a **slightly pessimistic**
  reference, not a clean-room maximum.
- For any headline performance claim, record whether the run was headless.
  **Never compare a desktop number against a headless number.**

### Clean-room option (opt-in, needs explicit approval)

The compositor can be suspended for a benchmark window to remove the noise:

```bash
systemctl --user suspend-targetplasma-wayland   # or stop plasma-plasmashell
# ... run benchmark ...
systemctl --user start plasma-plasmashell
```

This takes down the visible desktop and is therefore **not** run automatically.
It is a human decision, made per measurement window. It buys measurement
cleanliness, not VRAM.

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

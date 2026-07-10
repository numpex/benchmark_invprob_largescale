"""
torch.compile Denoiser Benchmark: Architectures x 2D/3D
---------------------------------------------------------

Benchmarks deepinv's **UNet**, **DnCNN**, and **DRUNet** denoisers, each in
both 2D and 3D, with and without :func:`torch.compile` (``fullgraph=True,
mode="max-autotune"``).

For each (architecture, dim, shape) combination we measure:

- **1st-call latency** - JIT compilation (compiled) / cold-GPU (eager)
- **Steady-state latency** - mean +/- std over ``N_RUNS`` calls
- **Speedup** - eager steady-state / compiled steady-state
- **Roofline metrics** - FLOPs, memory traffic, arithmetic intensity (eager
  mode), used to classify each case as compute- or memory-bound.

All architectures use random (untrained) weights with 3 input/output
channels, so 2D and 3D are directly comparable.

Produces exactly two figures:

1. ``compile_speedup.png`` - grouped bars of steady-state speedup per
   (architecture, dim, shape).
2. ``compile_roofline.png`` - arithmetic intensity vs. compile speedup
   scatter, with the hardware ridge point marked.

**Run:**

.. code-block:: bash

    python -m toolsbench.scripts.demo_compile_benchmark
"""

# %%
import inspect
import shutil
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo
from matplotlib.lines import Line2D
from torch._inductor.codecache import cache_dir

import deepinv
from deepinv.models import DRUNet, DnCNN, UNet

# H100 SXM peak specs (source: nvidia.com/en-us/data-center/h100/)
# PyTorch uses TF32 tensor cores by default (allow_tf32=True on Ampere/Hopper).
H100_PEAK_TFLOPS = 494.0  # TF32 tensor-core TFLOPS (dense, no sparsity)
H100_PEAK_BW_TBs = 3.35  # HBM3 bandwidth in TB/s
H100_PEAK_FLOPS = H100_PEAK_TFLOPS * 1e12
H100_PEAK_BW = H100_PEAK_BW_TBs * 1e12  # bytes/s
RIDGE_POINT = H100_PEAK_FLOPS / H100_PEAK_BW  # FLOP/byte where compute = memory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHANNELS = 3
NOISE_SIGMA = 0.1
BATCH_SIZE = 1
N_WARMUP_COMPILED = 5
N_WARMUP_EAGER = 5
N_RUNS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

ARCHS = [("UNet", UNet), ("DnCNN", DnCNN), ("DRUNet", DRUNet)]
ARCH_COLORS = {"UNet": "steelblue", "DnCNN": "seagreen", "DRUNet": "darkorange"}
DIM_MARKERS = {2: "o", 3: "s"}

# 2D shapes: (B, C, H, W)
SHAPES_2D = [
    (BATCH_SIZE, CHANNELS, 512, 512),
    (BATCH_SIZE, CHANNELS, 1024, 1024),
    (BATCH_SIZE, CHANNELS, 2048, 2048),
]

# 3D shapes: (B, C, D, H, W)
SHAPES_3D = [
    (BATCH_SIZE, CHANNELS, 128, 128, 128),
    (BATCH_SIZE, CHANNELS, 4, 512, 512),
    (BATCH_SIZE, CHANNELS, 16, 512, 512),
]

FONT_SIZE = 13
ANNOTATION_SIZE = 10


def shape_label(shape):
    if len(shape) == 4:
        return f"{shape[2]}x{shape[3]}"
    return f"{shape[2]}x{shape[3]}x{shape[4]}"


def build_model(arch_cls, dim):
    kwargs = dict(in_channels=CHANNELS, out_channels=CHANNELS, dim=dim, device=DEVICE)
    if "pretrained" in inspect.signature(arch_cls.__init__).parameters:
        kwargs["pretrained"] = None
    return arch_cls(**kwargs).eval()


print("=" * 70)
print("torch.compile Denoiser Benchmark - Architectures x 2D/3D")
print("=" * 70)
print(f"  DeepInv version : {deepinv.__version__}")
print(f"  PyTorch version : {torch.__version__}")
print(f"  Device          : {DEVICE}")
print(f"  Architectures   : {[name for name, _ in ARCHS]}")
print(f"  Channels        : {CHANNELS} (random init, no pretrained weights)")
print(f"  2D shapes       : {[shape_label(s) for s in SHAPES_2D]}")
print(f"  3D shapes       : {[shape_label(s) for s in SHAPES_3D]}")
print(f"  torch.compile   : fullgraph=True, mode='max-autotune'")
print("=" * 70)


# %%
# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_one(fn):
    """Time a single call; return (output, elapsed_seconds)."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def benchmark(fn, n_warmup):
    """1st call (timed) -> n_warmup discarded -> N_RUNS steady-state calls."""
    torch._dynamo.reset()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    out, t_first = time_one(fn)
    for _ in range(n_warmup):
        with torch.no_grad():
            fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    steady = []
    for _ in range(N_RUNS):
        out, t = time_one(fn)
        steady.append(t)
    return out, t_first, steady


# %%
# ---------------------------------------------------------------------------
# Speed benchmark: eager vs compiled, one (architecture, shape)
# ---------------------------------------------------------------------------


def run_shape(model_eager, x, label):
    """Benchmark one (model, shape); return a result dict."""
    call = lambda m: m(x, sigma=NOISE_SIGMA)

    torch._dynamo.reset()
    compile_supported = True
    try:
        model_compiled = torch.compile(model_eager, fullgraph=True, mode="max-autotune")
        print(f"    [compile] ", end="", flush=True)
        _, cf_sec, c_ss = benchmark(lambda: call(model_compiled), N_WARMUP_COMPILED)
    except Exception as exc:
        print(f"    torch.compile failed for {label} ({exc}); falling back to eager.")
        model_compiled = model_eager
        compile_supported = False
        _, cf_sec, c_ss = benchmark(lambda: call(model_compiled), N_WARMUP_COMPILED)

    cf, cr = cf_sec * 1e3, np.array(c_ss) * 1e3
    crm, crs = float(np.mean(cr)), float(np.std(cr))
    if compile_supported:
        print(f"1st={cf:.1f} ms  steady={crm:.1f}±{crs:.1f} ms")

    print(f"    [eager]   ", end="", flush=True)
    _, ef_sec, e_ss = benchmark(lambda: call(model_eager), N_WARMUP_EAGER)
    ef, er = ef_sec * 1e3, np.array(e_ss) * 1e3
    erm, ers = float(np.mean(er)), float(np.std(er))
    speedup_steady = erm / crm if crm > 0 and compile_supported else float("nan")
    print(f"1st={ef:.1f} ms  steady={erm:.1f}±{ers:.1f} ms  "
          f"speedup={speedup_steady:.2f}×" if compile_supported else
          f"1st={ef:.1f} ms  steady={erm:.1f}±{ers:.1f} ms  (compile unsupported)")

    return dict(
        eager_mean=erm, eager_std=ers,
        compiled_mean=crm, compiled_std=crs,
        speedup_steady=speedup_steady, compile_supported=compile_supported,
    )


# %%
# ---------------------------------------------------------------------------
# Roofline profiling (eager mode): FLOPs, memory traffic, arithmetic intensity
# ---------------------------------------------------------------------------


def profile_model(model, x):
    """Profile one eager forward pass: FLOPs, memory bytes, achieved TFLOPS."""
    with torch.no_grad():
        for _ in range(3):
            model(x, sigma=NOISE_SIGMA)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    bytes_per_elem = 4  # float32
    total_flops, total_bytes = [0], [0]

    def make_hook():
        def hook(module, inp, out):
            for t in inp:
                if isinstance(t, torch.Tensor):
                    total_bytes[0] += t.numel() * bytes_per_elem
            if isinstance(out, torch.Tensor):
                total_bytes[0] += out.numel() * bytes_per_elem
            if hasattr(module, "weight") and module.weight is not None:
                total_bytes[0] += module.weight.numel() * bytes_per_elem

            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d,
                                    torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                w = module.weight
                out_spatial = out.numel() // (out.shape[0] * out.shape[1]) if isinstance(out, torch.Tensor) else 0
                kernel_ops = int(np.prod(w.shape[1:]))
                batch = inp[0].shape[0] if isinstance(inp[0], torch.Tensor) else 1
                total_flops[0] += 2 * kernel_ops * w.shape[0] * out_spatial * batch
            elif isinstance(module, torch.nn.Linear):
                inp_t = inp[0] if isinstance(inp[0], torch.Tensor) else None
                if inp_t is not None:
                    batch = int(np.prod(inp_t.shape[:-1]))
                    total_flops[0] += 2 * module.in_features * module.out_features * batch

        return hook

    hooks = [m.register_forward_hook(make_hook()) for m in model.modules() if not list(m.children())]
    with torch.no_grad():
        model(x, sigma=NOISE_SIGMA)
    for h in hooks:
        h.remove()

    flops, mem_bytes = total_flops[0], total_bytes[0]

    times = []
    for _ in range(N_RUNS):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x, sigma=NOISE_SIGMA)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    wall_time = float(np.mean(times))

    arith_intensity = flops / mem_bytes if mem_bytes > 0 else 0.0
    achieved_tflops = flops / wall_time / 1e12
    regime = "COMPUTE-bound" if arith_intensity >= RIDGE_POINT else "MEMORY-bound"

    return dict(
        flops=flops, mem_bytes=mem_bytes, wall_time=wall_time,
        arith_intensity=arith_intensity, achieved_tflops=achieved_tflops, regime=regime,
    )


# %%
# ---------------------------------------------------------------------------
# Main loop: architecture x dim x shape
# ---------------------------------------------------------------------------
# results[(arch_name, dim, shape_label)] = {**speed_result, **roofline_result}

results = {}
all_cases = [(2, s) for s in SHAPES_2D] + [(3, s) for s in SHAPES_3D]

for arch_name, arch_cls in ARCHS:
    print(f"\n{'=' * 70}\n  Architecture: {arch_name}\n{'=' * 70}")
    for dim, shape in all_cases:
        lbl = shape_label(shape)
        print(f"\n  {dim}D  shape={lbl}")

        # Clear Inductor's on-disk kernel cache before each case: max-autotune
        # writes many candidate kernels per case and never deletes old ones,
        # which can exhaust small /tmp quotas after just a few cases.
        shutil.rmtree(cache_dir(), ignore_errors=True)

        torch.manual_seed(SEED)
        model = build_model(arch_cls, dim)
        x = torch.rand(*shape, device=DEVICE)

        speed = run_shape(model, x, f"{arch_name} {dim}D {lbl}")
        roofline = profile_model(model, x)
        results[(arch_name, dim, lbl)] = {**speed, **roofline}

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

# %%
# ---------------------------------------------------------------------------
# Summary table (text)
# ---------------------------------------------------------------------------

print("\n\n" + "=" * 100)
print(f"{'Arch':<8} {'Dim':>4} {'Shape':>12}  {'E steady':>10} {'C steady':>10}  "
      f"{'Speedup':>8}  {'AI(F/B)':>9} {'Regime':<14}")
print("-" * 100)
for arch_name, _ in ARCHS:
    for dim, shape in all_cases:
        lbl = shape_label(shape)
        r = results[(arch_name, dim, lbl)]
        print(f"{arch_name:<8} {dim:>4} {lbl:>12}  "
              f"{r['eager_mean']:>10.1f} {r['compiled_mean']:>10.1f}  "
              f"{r['speedup_steady']:>8.2f}  {r['arith_intensity']:>9.1f} {r['regime']:<14}")
    print("-" * 100)
print("=" * 100)

# %%
# ---------------------------------------------------------------------------
# Figure 1: Speedup - one line per architecture, 2D and 3D in one panel
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({"font.size": FONT_SIZE})

fig1, ax1 = plt.subplots(figsize=(11, 6.5))

n2d = len(SHAPES_2D)
for arch_name, _ in ARCHS:
    for dim, shapes, xs in [(2, SHAPES_2D, np.arange(n2d)), (3, SHAPES_3D, np.arange(n2d, n2d + len(SHAPES_3D)))]:
        ys, unsupported = [], []
        for shape in shapes:
            r = results[(arch_name, dim, shape_label(shape))]
            ys.append(r["speedup_steady"])
            unsupported.append(not r["compile_supported"])
        ys = np.array(ys, dtype=float)
        # 2D and 3D are plotted as separate line segments (no connecting line
        # between them - the transition isn't a meaningful continuous quantity).
        ax1.plot(xs, ys, "-", marker=DIM_MARKERS[dim], color=ARCH_COLORS[arch_name],
                 markersize=9, linewidth=2, label=arch_name if dim == 2 else None)
        bad = np.where(unsupported)[0]
        if len(bad):
            ax1.scatter(xs[bad], ys[bad], marker="x", s=120, color=ARCH_COLORS[arch_name],
                        linewidths=2.5, zorder=11)
        for xi, yi in zip(xs, ys, strict=False):
            if np.isfinite(yi):
                ax1.annotate(f"{yi:.2f}x", (xi, yi), xytext=(0, 8),
                             textcoords="offset points", ha="center",
                             fontsize=ANNOTATION_SIZE - 1, color=ARCH_COLORS[arch_name])

ax1.axvline(n2d - 0.5, color="black", ls="--", lw=1.2, alpha=0.4)
ax1.axhline(1.0, color="gray", ls=":", lw=1.3)
all_shapes = list(SHAPES_2D) + list(SHAPES_3D)
ax1.set_xticks(np.arange(len(all_shapes)))
ax1.set_xticklabels([shape_label(s) for s in all_shapes])
ax1.set_xlabel("Shape  (2D  |  3D)")
ax1.set_ylabel("Speedup (eager / compiled)")

arch_handles = [Line2D([0], [0], color=c, marker="o", markersize=9, label=name)
                for name, c in ARCH_COLORS.items()]
dim_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="None", markersize=9, label=f"{d}D")
               for d, m in DIM_MARKERS.items()]
ax1.legend(handles=arch_handles + dim_handles, title="Architecture / dim")
ax1.set_title(
    "torch.compile Steady-State Speedup by Architecture and Shape\n"
    "('x' marker = compile unsupported, fell back to eager)",
    fontweight="bold",
)
fig1.tight_layout()
fig1.savefig("compile_speedup.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: compile_speedup.png")

# %%
# ---------------------------------------------------------------------------
# Figure 2: Roofline - arithmetic intensity vs. compile speedup (one panel)
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(9.5, 6.5))

for arch_name, _ in ARCHS:
    for dim, shapes in [(2, SHAPES_2D), (3, SHAPES_3D)]:
        for shape in shapes:
            r = results[(arch_name, dim, shape_label(shape))]
            if not np.isfinite(r["speedup_steady"]):
                continue
            ax2.scatter(
                r["arith_intensity"], r["speedup_steady"],
                s=170, color=ARCH_COLORS[arch_name], marker=DIM_MARKERS[dim],
                edgecolors="black", linewidths=0.8, zorder=10,
            )

ax2.axvline(RIDGE_POINT, color="gray", ls="--", lw=1.5)
ax2.axhline(1.0, color="black", ls=":", lw=1, alpha=0.4)
ax2.set_xscale("log")
ax2.set_xlabel("Arithmetic Intensity (FLOP / byte, eager mode)")
ax2.set_ylabel("Compile speedup (eager / compiled)")
ax2.set_title(
    "Roofline: Arithmetic Intensity vs. Compile Speedup\n"
    "Left of ridge = memory-bound, right = compute-bound"
)

arch_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=name)
                for name, c in ARCH_COLORS.items()]
dim_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="None", markersize=9, label=f"{d}D")
               for d, m in DIM_MARKERS.items()]
ridge_handle = [Line2D([0], [0], color="gray", ls="--", lw=1.5, label=f"Ridge point ({RIDGE_POINT:.0f} FLOP/byte)")]
ax2.legend(handles=arch_handles + dim_handles + ridge_handle, fontsize=ANNOTATION_SIZE, loc="best")

fig2.tight_layout()
fig2.savefig("compile_roofline.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: compile_roofline.png")

print("\nDone.")

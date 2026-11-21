"""How much error INT4 g128 adds on top of the checkpoint's NVFP4/MXFP8 experts.

Weight-space only (no activations): relative Frobenius error of the INT4
reconstruction against the dequantized checkpoint weights, per variant.

    venv/bin/python prepare/requant_error.py [layer ...]
"""
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "tests")
from test_marlin_experts import CKPT, _dequant_mxfp8, _dequant_nvfp4  # noqa: E402

G = 128


def int4(w, sym, clip=1.0):
    g = w.unflatten(-1, (-1, G))
    if sym:
        s = (g.abs().amax(-1, keepdim=True) * clip / 7.5).clamp_min(1e-12)
        q = (g / s - 0.5).round().clamp(-8, 7) + 0.5  # uint4b8-style midpoints
        return (q * s).flatten(-2)
    lo, hi = g.amin(-1, keepdim=True) * clip, g.amax(-1, keepdim=True) * clip
    s = ((hi - lo) / 15).clamp_min(1e-12)
    z = (-lo / s).round().clamp(0, 15)
    q = (g / s + z).round().clamp(0, 15)
    return ((q - z) * s).flatten(-2)


def best_clip(w, sym):
    g = w.unflatten(-1, (-1, G))
    best = torch.full(g.shape[:-1], float("inf"), device=w.device)
    out = torch.empty_like(g)
    for c in torch.linspace(1.0, 0.7, 13):
        r = int4(w, sym, float(c)).unflatten(-1, (-1, G))
        err = (r - g).pow(2).sum(-1)
        better = err < best
        best = torch.where(better, err, best)
        out[better] = r[better]
    return out.flatten(-2)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


for layer in [int(x) for x in sys.argv[1:]] or [0, 12, 20]:
    p = f"model.llm.layers.{layer}.mlp.experts."
    with safe_open(CKPT, "pt", device="cuda") as f:
        t = {k[len(p):]: f.get_tensor(k) for k in f.keys() if k.startswith(p)}
    for name in ("w13", "w2"):
        if f"{name}_weight_amax" in t:
            gs = t[f"{name}_weight_amax"] / t[f"{name}_weight_outer_scale_denominator"]
            w = _dequant_nvfp4(t[f"{name}_weight"], t[f"{name}_weight_scale"], gs)
            fmt = "nvfp4"
        else:
            w = _dequant_mxfp8(t[f"{name}_weight"], t[f"{name}_weight_scale"])
            fmt = "mxfp8"
        w = w[:32]  # 32 experts are enough for the statistic
        print(f"layer {layer:2d} {name} ({fmt}): "
              f"sym {rel(int4(w, True), w):.4f}  asym {rel(int4(w, False), w):.4f}  "
              f"sym+clip {rel(best_clip(w, True), w):.4f}  "
              f"asym+clip {rel(best_clip(w, False), w):.4f}")

"""GPTQ (INT4, group 128, asymmetric, static groups) of Inkling 2 Mini's experts.

Inputs are the activations `prepare/replay_capture.py` recorded on the
reference checkpoint: per layer (all loop passes together, since they share
the weights) each routed token's MoE input `x`, expert ids and routing
weights `gamma`.

Per expert:
    w13   H = sum over its tokens of gamma^2 x x^T (a token's output reaches
          the layer scaled by gamma, after the SiTU)
    w2    H = sum of a a^T, where a = SiTU(x W13_q^T) * gamma uses the already
          quantized w13 (`moe_sm90.swiglu`, the kernel the server runs), so
          GPTQ on w2 also absorbs w13's error.

Experts that saw few tokens get an identity prior (`PRIOR_TOKENS`), which
pulls them toward plain round-to-nearest instead of trusting a noisy Hessian.
No act-order: Marlin's AWQ layout has no g_idx.

    venv/bin/python prepare/gptq_experts.py models/Inkling-2-Mini data/capture \
        models/Inkling-2-Mini-W4A16-g128-gptq
"""

import glob
import os
import sys
import time
from functools import partial

import torch

sys.path.insert(0, os.path.dirname(__file__))
import int4_awq  # noqa: E402
from quantize_experts import dense_experts, write_checkpoint  # noqa: E402

from vllm.models.inkling2.nvidia.ops.moe_sm90 import swiglu  # noqa: E402

G = int4_awq.GROUP
DAMP = 0.01
PRIOR_TOKENS = 256
EXPERT_CHUNK = 16
GATE_CAP, UP_CAP = 4.0, 25.0


def load_capture(capture_dirs: str, key: str):
    """`capture_dirs`: one capture directory, or several separated by commas."""
    files = sorted(f for d in capture_dirs.split(",")
                   for f in glob.glob(os.path.join(d, f"{key}_pass*_*.pt")))
    parts = [torch.load(p) for p in files]
    x = torch.cat([p["x"] for p in parts]).cuda()
    ids = torch.cat([p["ids"] for p in parts]).cuda().long()
    w = torch.cat([p["w"] for p in parts]).cuda().float()
    return x, ids, w


def _group_params(w1: torch.Tensor):
    """Clip-searched (z, s) of one group of columns: w1 [E, R, G]."""
    best = None
    for c in int4_awq.CLIPS:
        q, z, s = int4_awq._quantize_groups(w1, c)
        s = s.to(torch.bfloat16).float()
        q = (w1 / s + z).round().clamp(0, 15)
        err = ((q - z) * s - w1).pow(2).sum(-1, keepdim=True)
        if best is None:
            best = [err, z, s]
        else:
            better = err < best[0]
            best = [torch.where(better, a, b) for a, b in zip((err, z, s), best)]
    return best[1], best[2]


def gptq(w: torch.Tensor, h: torch.Tensor, n_tokens: torch.Tensor, prior: float):
    """w [E, R, C] fp32, h [E, C, C] -> codes [E, R, C], zeros, scales [E, R, C/G]."""
    w = w.clone()
    e, r, c = w.shape
    eye = torch.eye(c, device=w.device)
    mean_diag = h.diagonal(dim1=1, dim2=2).mean(-1)
    weak = ((PRIOR_TOKENS - n_tokens).clamp(min=0) / PRIOR_TOKENS).float()
    h = h + (DAMP * mean_diag + weak * prior)[:, None, None] * eye
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    codes = torch.empty((e, r, c), dtype=torch.uint8, device=w.device)
    zeros = torch.empty((e, r, c // G), dtype=torch.uint8, device=w.device)
    scales = torch.empty((e, r, c // G), dtype=torch.bfloat16, device=w.device)
    for g0 in range(0, c, G):
        g1 = g0 + G
        w1 = w[:, :, g0:g1].clone()
        z, s = _group_params(w1)
        zeros[:, :, g0 // G] = z.squeeze(-1).to(torch.uint8)
        scales[:, :, g0 // G] = s.squeeze(-1).to(torch.bfloat16)
        err1 = torch.empty_like(w1)
        u1 = u[:, g0:g1, g0:g1]
        for i in range(G):
            col = w1[:, :, i]
            q = (col / s[..., 0] + z[..., 0]).round().clamp(0, 15)
            codes[:, :, g0 + i] = q.to(torch.uint8)
            deq = (q - z[..., 0]) * s[..., 0]
            err = (col - deq) / u1[:, i, i][:, None]
            w1[:, :, i:] -= err[:, :, None] * u1[:, i, i:][:, None, :]
            err1[:, :, i] = err
        if g1 < c:
            w[:, :, g1:] -= torch.bmm(err1, u[:, g0:g1, g1:])
    return codes, zeros, scales


def _hessians(make_inputs, num_experts, dim):
    """One expert's inputs at a time (`make_inputs(e)` -> [n, dim] fp32)."""
    h = torch.zeros((num_experts, dim, dim), device="cuda")
    ns = torch.zeros(num_experts, dtype=torch.long, device="cuda")
    for e in range(num_experts):
        a = make_inputs(e)
        ns[e] = a.shape[0]
        if a.shape[0]:
            torch.mm(a.T, a, out=h[e])
        del a
    return h, ns


def _run_gptq(w, h, n):
    mean_diag = h.diagonal(dim1=1, dim2=2).mean(-1)
    prior = mean_diag[n >= PRIOR_TOKENS].mean().item() if (n >= PRIOR_TOKENS).any() else 1.0
    out = [
        gptq(w[i : i + EXPERT_CHUNK], h[i : i + EXPERT_CHUNK], n[i : i + EXPERT_CHUNK], prior)
        for i in range(0, w.shape[0], EXPERT_CHUNK)
    ]
    return tuple(torch.cat([o[j] for o in out]) for j in range(3))


def _w13_inputs(x, tok, gam, e):
    return x[tok[e]].float() * gam[e][:, None]


def _w2_inputs(x, tok, gam, w13q, e):
    """Expert e's w2 inputs: swiglu of its quantized w13 outputs, 8k tokens at a time."""
    tok, gam, w13q = tok[e], gam[e], w13q[e]
    if not len(tok):
        return torch.zeros((0, w13q.shape[0] // 2), device="cuda")
    out = []
    for t, g in zip(tok.split(8192), gam.split(8192)):
        gate_up = (x[t].float() @ w13q.T).to(torch.bfloat16)
        src2dst = torch.arange(len(t), device="cuda", dtype=torch.int32)
        out.append(swiglu(gate_up, g.contiguous(), src2dst, GATE_CAP, UP_CAP).float())
    return torch.cat(out)


def gptq_layer(f, layer: int, capture_dir: str, keys: list[str]):
    t0 = time.time()
    x, ids, gammas = load_capture(capture_dir, keys[layer])
    num_experts = 128
    # Rows (token, slot) per expert.
    flat = ids.flatten()
    rows = [torch.nonzero(flat == e).squeeze(1) for e in range(num_experts)]
    tok = [r // ids.shape[1] for r in rows]
    gam = [gammas.flatten()[r] for r in rows]

    w13, _ = dense_experts(f, layer, "w13")
    h13, n = _hessians(partial(_w13_inputs, x, tok, gam), num_experts, w13.shape[2])
    c13 = _run_gptq(w13, h13, n)
    del h13, w13
    w13q = int4_awq.dequantize(*c13)

    h2, _ = _hessians(partial(_w2_inputs, x, tok, gam, w13q), num_experts, w13q.shape[1] // 2)
    del w13q
    w2, _ = dense_experts(f, layer, "w2")
    c2 = _run_gptq(w2, h2, n)
    del h2, w2
    print(f"layer {layer:2d}: {flat.numel() // 8} tokens, experts with < {PRIOR_TOKENS} rows: "
          f"{int((n < PRIOR_TOKENS).sum())}, min/median rows {int(n.min())}/{int(n.median())}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"w13": c13, "w2": c2}


def main(src, capture_dir, dst):
    keys = sorted({os.path.basename(p).split("_pass")[0]
                   for d in capture_dir.split(",")
                   for p in glob.glob(os.path.join(d, "visit*_pass*_*.pt"))})
    assert len(keys) == 24, keys
    write_checkpoint(src, dst, lambda f, layer: gptq_layer(f, layer, capture_dir, keys))


if __name__ == "__main__":
    main(*sys.argv[1:])

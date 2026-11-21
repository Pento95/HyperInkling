"""INT4 group-128 asymmetric quantization of expert weights, in AWQ layout.

A weight `W [E, N, K]` (out x in, as the checkpoint stores it) is quantized
along K in groups of 128: `W ~ (q - z) * s`, `q, z` in [0, 15]. The AWQ layout
transposes to `[K, N]` and packs 8 nibbles per int32 along N in the order
`AWQ_ORDER` (AutoAWQ's), which `ops.awq_marlin_moe_repack` expects:

    qweight [E, K, N / 8] int32    scales [E, K / 128, N] bf16
    qzeros  [E, K / 128, N / 8] int32

`clip_search` picks, per group, the range shrink factor that minimizes the
group's squared reconstruction error (AWQ's clipping step, here weight-space
only; activation-aware scaling is a separate step).
"""

import torch

GROUP = 128
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
CLIPS = tuple(1.0 - 0.025 * i for i in range(13))  # 1.0 .. 0.7


def _quantize_groups(g: torch.Tensor, clip: float):
    """g [..., G]: integer codes, zero points and scales at one clip ratio."""
    lo = g.amin(-1, keepdim=True) * clip
    hi = g.amax(-1, keepdim=True) * clip
    lo, hi = lo.clamp(max=0), hi.clamp(min=0)  # keep 0 representable
    s = ((hi - lo) / 15).clamp_min(1e-10)
    z = (-lo / s).round().clamp(0, 15)
    q = (g / s + z).round().clamp(0, 15)
    return q, z, s


def quantize(w: torch.Tensor, clip_search: bool = True):
    """w [E, N, K] float -> codes [E, N, K] uint8, zeros and scales [E, N, K/G]."""
    g = w.float().unflatten(-1, (-1, GROUP))
    best_q, best_z, best_s = _quantize_groups(g, 1.0)
    if clip_search:
        best_err = ((best_q - best_z) * best_s - g).pow(2).sum(-1, keepdim=True)
        for c in CLIPS[1:]:
            q, z, s = _quantize_groups(g, c)
            err = ((q - z) * s - g).pow(2).sum(-1, keepdim=True)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q, best_q)
            best_z = torch.where(better, z, best_z)
            best_s = torch.where(better, s, best_s)
    # Round the scales to bf16 first and requantize against them, so the codes
    # match the scales the kernel will actually use.
    s = best_s.to(torch.bfloat16).float()
    q = (g / s + best_z).round().clamp(0, 15)
    return (
        q.flatten(-2).to(torch.uint8),
        best_z.squeeze(-1).to(torch.uint8),
        s.squeeze(-1).to(torch.bfloat16),
    )


def dequantize(q, z, s):
    """Inverse of `quantize`'s outputs (codes, zeros, scales) -> [E, N, K] fp32."""
    g = q.float().unflatten(-1, (-1, GROUP))
    return ((g - z.float()[..., None]) * s.float()[..., None]).flatten(-2)


def _pack_n(x: torch.Tensor) -> torch.Tensor:
    """[..., N] values in [0, 15] -> [..., N / 8] int32, AWQ nibble order."""
    x = x.to(torch.int32).unflatten(-1, (-1, 8))
    out = torch.zeros(x.shape[:-1], dtype=torch.int32, device=x.device)
    for i, col in enumerate(AWQ_ORDER):
        out |= x[..., col] << (4 * i)
    return out


def to_awq_layout(q, z, s):
    """Codes [E, N, K], zeros / scales [E, N, K/G] -> AWQ qweight, scales, qzeros."""
    qweight = _pack_n(q.transpose(1, 2))  # [E, K, N/8]
    qzeros = _pack_n(z.transpose(1, 2))  # [E, K/G, N/8]
    scales = s.transpose(1, 2).contiguous()  # [E, K/G, N]
    return qweight.contiguous(), scales, qzeros.contiguous()

"""MarlinExperts against a dequantize-then-matmul reference, on real layers.

Layer 0 holds NVFP4 experts and layer 20 MXFP8 ones; `int4` requantizes
layer 0 with `prepare/int4_awq.py`. The reference dequantizes every expert
to fp32 (for `int4`, the INT4 codes themselves, so the test checks packing
and repack rather than quantization quality), runs each (token, slot)
through its expert in fp32, and shares `moe_sm90.swiglu` with the kernel
path, so the comparison isolates the Marlin repack and GEMMs.

    venv/bin/python -m pytest -q tests/test_marlin_experts.py
"""

import os
import sys

import pytest
import torch
from safetensors import safe_open

from vllm.models.inkling2.nvidia.ops.moe_marlin import MarlinExperts
from vllm.models.inkling2.nvidia.ops.moe_sm90 import swiglu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prepare"))
import int4_awq  # noqa: E402

CKPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "Inkling-2-Mini",
    "model-00001-of-00001.safetensors",
)
HIDDEN, INTER, TOPK, EXPERTS = 2304, 1152, 8, 128
GATE_CAP, UP_CAP = 4.0, 25.0
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _load(layer: int) -> dict[str, torch.Tensor]:
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    with safe_open(CKPT, "pt", device="cuda") as f:
        return {
            k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)
        }


def _dequant_nvfp4(values, scales, global_scale):
    """[E, N, K/2] packed e2m1 (low nibble first), [E, N, K/16] e4m3 scales."""
    table = torch.cat([E2M1, -E2M1]).to(values.device)
    lo = table[(values & 0xF).long()]
    hi = table[(values >> 4).long()]
    w = torch.stack([lo, hi], dim=-1).flatten(-2)
    w = w.unflatten(-1, (-1, 16)) * scales.float()[..., None]
    return w.flatten(-2) * global_scale.view(-1, 1, 1)


def _dequant_mxfp8(values, scales):
    """[E, N, K] e4m3, [E, N, K/32] e8m0 exponents."""
    w = values.float().unflatten(-1, (-1, 32))
    return (w * torch.exp2(scales.float() - 127)[..., None]).flatten(-2)


def _weights(t, nvfp4):
    out = {}
    for name in ("w13", "w2"):
        global_scale = None
        if nvfp4:
            global_scale = (
                t[f"{name}_weight_amax"] / t[f"{name}_weight_outer_scale_denominator"]
            )
        out[name] = (t[f"{name}_weight"], t[f"{name}_weight_scale"], global_scale)
    return out


def _dense(t, nvfp4):
    weights = _weights(t, nvfp4)
    if nvfp4:
        return [_dequant_nvfp4(*weights[n]) for n in ("w13", "w2")]
    return [_dequant_mxfp8(*weights[n][:2]) for n in ("w13", "w2")]


def _int4(t):
    """Layer 0's NVFP4 experts requantized to INT4, and their exact dequant."""
    weights, dense = {}, []
    for name, w in zip(("w13", "w2"), _dense(t, True)):
        q, z, s = int4_awq.quantize(w)
        dense.append(int4_awq.dequantize(q, z, s))
        weights[name] = int4_awq.to_awq_layout(q, z, s)
    return weights, dense


def _reference(x, ids, gammas, src2dst, dense):
    w13, w2 = dense
    tokens = x.shape[0]
    flat_ids = ids.flatten().long()
    rows = x.repeat_interleave(TOPK, dim=0).float()
    gate_up = torch.empty((tokens * TOPK, 2 * INTER), device=x.device)
    for e in flat_ids.unique():
        sel = flat_ids == e
        gate_up[sel] = rows[sel] @ w13[e].T
    gate_up = gate_up.to(torch.bfloat16)
    act = swiglu(gate_up, gammas, src2dst, GATE_CAP, UP_CAP).float()
    down = torch.empty((tokens * TOPK, HIDDEN), device=x.device)
    for e in flat_ids.unique():
        sel = flat_ids == e
        down[sel] = act[sel] @ w2[e].T
    return down


@pytest.mark.parametrize("layer,fmt", [(0, "nvfp4"), (20, "mxfp8"), (0, "int4")])
@pytest.mark.parametrize("tokens", [1, 6, 48, 512])
def test_marlin_experts(layer, fmt, tokens):
    torch.manual_seed(0)
    t = _load(layer)
    if fmt == "int4":
        weights, dense = _int4(t)
    else:
        weights, dense = _weights(t, fmt == "nvfp4"), _dense(t, fmt == "nvfp4")
    experts = MarlinExperts(
        {
            k: tuple(v.clone() if v is not None else None for v in w)
            for k, w in weights.items()
        },
        ((2 * INTER, HIDDEN), (HIDDEN, INTER)),
        fmt,
    )
    x = (torch.randn(tokens, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(EXPERTS, device="cuda")[:TOPK] for _ in range(tokens)]
    ).to(torch.int32)
    gammas = torch.rand(tokens * TOPK, device="cuda") + 0.5
    src2dst = torch.arange(tokens * TOPK, device="cuda", dtype=torch.int32)

    got = experts(x, ids, gammas, src2dst, EXPERTS, GATE_CAP, UP_CAP).float()
    want = _reference(x, ids, gammas, src2dst, dense)
    rel = (got - want).norm() / want.norm()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    print(f"{fmt} layer {layer} tokens {tokens}: rel err {rel:.2e}, cos {cos:.6f}")
    assert torch.isfinite(got).all()
    assert rel < 2e-2 and cos > 0.9998

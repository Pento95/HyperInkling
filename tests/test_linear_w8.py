"""W8Linear on real attention weights (layers 4: local, 5: full).

Two checks per weight: the kernel against the exact dequantized int8 weight
(packing, repack, GEMM), and the quantization error against bf16.

    venv/bin/python -m pytest -q -s tests/test_linear_w8.py
"""

import os

import pytest
import torch
from safetensors import safe_open

from vllm.models.inkling2.nvidia.ops.linear_w8 import (
    W8Linear,
    dequantize_int8,
    quantize_int8,
)

CKPT = os.path.join(
    os.path.dirname(__file__), "..", "models", "Inkling-2-Mini",
    "model-00001-of-00001.safetensors",
)


def _weights():
    out = {}
    with safe_open(CKPT, "pt", device="cuda") as f:
        for k in f.keys():
            if not (k.startswith(("model.llm.layers.4.attn.", "model.llm.layers.5.attn."))
                    and k.endswith(".weight")):
                continue
            shape = f.get_slice(k).get_shape()
            if len(shape) == 2 and shape[1] % 128 == 0 and shape[0] >= 64:
                out[k] = f.get_tensor(k)
    return out


@pytest.mark.parametrize("tokens", [1, 6, 48, 512, 2048])
def test_w8_linear(tokens):
    torch.manual_seed(0)
    weights = _weights()
    assert weights
    for name, w in weights.items():
        lin = W8Linear(w)
        x = torch.randn(tokens, w.shape[1], device="cuda").to(torch.bfloat16)
        got = lin(x).float()
        exact = x.float() @ dequantize_int8(*quantize_int8(w)).T
        full = x.float() @ w.float().T
        kernel = ((got - exact).norm() / exact.norm()).item()
        quant = ((got - full).norm() / full.norm()).item()
        print(f"{name} {tuple(w.shape)} tokens {tokens}: "
              f"kernel {kernel:.1e}, vs bf16 {quant:.1e}")
        assert got.shape == full.shape
        assert kernel < 5e-3 and quant < 1e-2


@pytest.mark.parametrize("tokens", [1, 6, 40])
def test_w8_multi(tokens):
    """Three inputs, three column ranges (a full layer's q / k / v)."""
    torch.manual_seed(0)
    w = (torch.randn(8600, 2304, device="cuda") * 0.02).to(torch.bfloat16)
    lin = W8Linear(w)
    ranges = [(0, 6552), (6552, 7576), (7576, 8600)]
    xs = [torch.randn(tokens, 2304, device="cuda").to(torch.bfloat16) for _ in ranges]
    dense = dequantize_int8(*quantize_int8(w))
    for got, x, (lo, hi) in zip(lin.multi(xs, ranges), xs, ranges):
        want = x.float() @ dense[lo:hi].T
        assert got.shape == want.shape
        assert ((got.float() - want).norm() / want.norm()).item() < 5e-3

"""W4Linear (int4 g128 asymmetric, packed nibbles) at the DSpark drafter's shapes.

Checks: the kernel against the exact dequantized weight, the round-trip error
of the quantization, the nibble packing on a hand-built row; plus timings
against W8Linear and bf16 (printed only).

    venv/bin/python -m pytest -q -s tests/test_linear_w4.py
"""

import time

import pytest
import torch

from vllm.models.inkling2.nvidia.ops.linear_w4 import (
    W4Linear,
    dequantize_int4,
    quantize_int4,
)
from vllm.models.inkling2.nvidia.ops.linear_w8 import W8Linear

# Drafter MLP gate_up and down, attention qkv-like, and a small one.
SHAPES = [(18432, 2304), (2304, 9216), (4096, 2304), (2304, 1152)]


def _weight(n, k):
    torch.manual_seed(0)
    return (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 8, 16, 48, 512])
@pytest.mark.parametrize("shape", SHAPES)
def test_kernel_matches_dequantized(shape, tokens):
    w = _weight(*shape)
    lin = W4Linear(w)
    x = torch.randn(tokens, shape[1], device="cuda").to(torch.bfloat16)
    ref = x.float() @ dequantize_int4(lin.packed, lin.scale, lin.zero).T
    out = lin(x).float()
    err = (out - ref).abs().max() / ref.abs().max()
    assert err < 2e-2, f"{shape} x {tokens}: {err:.3e}"


@pytest.mark.parametrize("shape", SHAPES)
def test_round_trip(shape):
    w = _weight(*shape)
    deq = dequantize_int4(*quantize_int4(w))
    rel = (deq - w.float()).norm() / w.float().norm()
    assert rel < 0.12, f"{shape}: {rel:.3f}"


def test_packing():
    codes = torch.arange(128, device="cuda") % 16
    # min -3, max 12: scale 1, zero 3
    w = (codes.float() - 3.0).to(torch.bfloat16)[None, :]
    packed, scale, zero = quantize_int4(w)
    assert scale.item() == 1.0 and zero.item() == 3
    assert packed[0, 0].item() == 0 | (1 << 4)
    assert packed[0, 7].item() == 14 | (15 << 4)
    assert torch.equal(dequantize_int4(packed, scale, zero), w.float())


def _bench(fn, n=200):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n // 20):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6


@pytest.mark.parametrize("tokens", [8, 48])
@pytest.mark.parametrize("shape", SHAPES[:2])
def test_timing(shape, tokens):
    w = _weight(*shape)
    x = torch.randn(tokens, shape[1], device="cuda").to(torch.bfloat16)
    w4, w8 = W4Linear(w), W8Linear(w)
    n, k = shape
    rows = []
    for name, fn, nbytes in (
        ("w4", lambda: w4(x), n * k // 2),
        ("w8", lambda: w8(x), n * k),
        ("bf16", lambda: torch.nn.functional.linear(x, w), n * k * 2),
    ):
        us = _bench(fn)
        rows.append(f"{name} {us:6.1f} us {nbytes / us / 1e3:5.0f} GB/s")
    print(f"\n{shape} x {tokens}: " + " | ".join(rows))

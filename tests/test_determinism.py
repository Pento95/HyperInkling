"""Run-to-run determinism of the kernels on the prefill path: the same inputs twice must
give bitwise the same outputs, or the KL eval's noise floor (bench/kl_eval.py) rises.

    venv/bin/python -m pytest -q tests/test_determinism.py
"""

import pytest
import torch
from test_marlin_experts import EXPERTS, GATE_CAP, HIDDEN, INTER, TOPK, UP_CAP, _load, _weights

from vllm.models.inkling2.nvidia.ops.moe_marlin import MarlinExperts

REPEAT = 20


@pytest.mark.parametrize("tokens", [6, 512])
def test_marlin_experts_repeat(tokens):
    torch.manual_seed(0)
    weights = _weights(_load(0), True)
    experts = MarlinExperts(weights, ((2 * INTER, HIDDEN), (HIDDEN, INTER)), "nvfp4")
    x = (torch.randn(tokens, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(EXPERTS, device="cuda")[:TOPK] for _ in range(tokens)]
    ).to(torch.int32)
    gammas = torch.rand(tokens * TOPK, device="cuda") + 0.5
    src2dst = torch.arange(tokens * TOPK, device="cuda", dtype=torch.int32)
    first = experts(x, ids, gammas, src2dst, EXPERTS, GATE_CAP, UP_CAP).clone()
    for i in range(REPEAT):
        got = experts(x, ids, gammas, src2dst, EXPERTS, GATE_CAP, UP_CAP)
        diff = (got.float() - first.float()).abs().max().item()
        assert torch.equal(got, first), f"repeat {i}: max abs diff {diff:.3e}"


def _repeat(fn):
    first = fn().clone()
    for i in range(REPEAT):
        got = fn()
        diff = (got.float() - first.float()).abs().max().item()
        assert torch.equal(got, first), f"repeat {i}: max abs diff {diff:.3e}"


@pytest.mark.parametrize("tokens", [6, 512])
def test_w8_linear_repeat(tokens):
    from vllm.models.inkling2.nvidia.ops.linear_w8 import W8Linear

    torch.manual_seed(0)
    lin = W8Linear((torch.randn(3072, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16))
    x = torch.randn(tokens, HIDDEN, device="cuda").to(torch.bfloat16)
    _repeat(lambda: lin(x))


@pytest.mark.parametrize("tokens", [1, 6])
def test_bgemv_repeat(tokens):
    from vllm.models.inkling2.nvidia.ops.bgemv import bgemv

    torch.manual_seed(0)
    down = (torch.randn(3, 64, 2 * HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    pair = torch.randn(3, tokens, 2 * HIDDEN, device="cuda").to(torch.bfloat16)
    _repeat(lambda: bgemv(pair, down))


@pytest.mark.parametrize("q", [1, 300])
def test_short_conv_repeat(q):
    from test_sconv_padding import WS, H, N, W, _batch

    from vllm.models.inkling2.nvidia.ops.sconv import short_conv

    torch.manual_seed(0)
    reqs, blocks = 4, 32
    ctx = [10, 50, 90, 130]
    cache = torch.randn(reqs * blocks, H, N, WS, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(H * WS, W, device="cuda") * 0.5).to(torch.bfloat16)
    x = torch.randn(reqs * q, H * WS, device="cuda").to(torch.bfloat16)
    meta = _batch([q] * reqs, ctx, 0, blocks)
    _repeat(lambda: short_conv(x, weight, cache.clone(), *meta, 0, WS))

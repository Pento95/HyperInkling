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


def _poison():
    """Garbage in the memory torch.empty hands out next: a kernel that reads what it did
    not write (or a padding row) then differs between repeats."""
    junk = [torch.empty(n, dtype=torch.float32, device="cuda").normal_(0, 1e3)
            for n in (1 << 8, 1 << 16, 1 << 20, 1 << 24) for _ in range(4)]
    del junk


def _repeat(fn):
    first = fn().clone()
    for i in range(REPEAT):
        _poison()
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
    cache = torch.randn(1 + reqs * blocks, H, N, WS, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(H * WS, W, device="cuda") * 0.5).to(torch.bfloat16)
    x = torch.randn(reqs * q, H * WS, device="cuda").to(torch.bfloat16)
    meta = _batch([q] * reqs, ctx, 0, blocks)
    _repeat(lambda: short_conv(x, weight, cache.clone(), *meta, 0, WS))


@pytest.mark.parametrize("tokens", [1, 6, 129, 300, 512, 2048])
def test_router_repeat(tokens):
    from vllm.models.inkling2.nvidia.ops.moe_sm90 import select_experts

    torch.manual_seed(0)
    gate = (torch.randn(EXPERTS, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(tokens, HIDDEN, device="cuda").to(torch.bfloat16)
    bias = torch.zeros(EXPERTS, device="cuda")
    bounds = torch.tensor([0, EXPERTS], dtype=torch.int32, device="cuda")
    for out in range(3):  # ids, weights, src2dst
        _repeat(lambda out=out: select_experts(x, gate, bias, TOPK, 1.0, bounds)[out])


@pytest.mark.parametrize("tokens", [6, 512])
def test_attn_res_repeat(tokens):
    from vllm.models.inkling2.nvidia.ops.attn_res import attn_res

    torch.manual_seed(0)
    stack = torch.randn(3, tokens, HIDDEN, device="cuda").to(torch.bfloat16)
    last = torch.randn(tokens, HIDDEN, device="cuda").to(torch.bfloat16)
    query = torch.randn(1, HIDDEN, device="cuda") * HIDDEN**-0.5
    norm = torch.ones(1, HIDDEN, device="cuda")
    _repeat(lambda: attn_res(stack, last, None, query, norm, 1e-6, True))


@pytest.mark.parametrize("tokens", [1, 6, 16])
def test_ll_gemm_repeat(tokens):
    from vllm.models.inkling2.nvidia.ops.ll_gemm_sm120 import sm12x_bf16_ll_gemm

    torch.manual_seed(0)
    w = (torch.randn(1024, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(tokens, HIDDEN, device="cuda").to(torch.bfloat16)
    _repeat(lambda: sm12x_bf16_ll_gemm(x, w))


@pytest.mark.parametrize("tokens", [17, 512, 2048])
def test_cublas_linear_repeat(tokens):
    torch.manual_seed(0)
    w = (torch.randn(8608, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(tokens, HIDDEN, device="cuda").to(torch.bfloat16)
    _repeat(lambda: torch.nn.functional.linear(x, w))

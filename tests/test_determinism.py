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


@pytest.mark.parametrize("q_lens,past", [([1, 1, 1], 500), ([7, 5, 3], 500), ([512], 0), ([300], 700)])
@pytest.mark.parametrize("window", [0, 256])
def test_rel_attention_repeat(q_lens, past, window):
    """The prefill attention with FP8 keys and values; positions past each request's end
    and the rel-logits outside the extent hold garbage."""
    from vllm.models.inkling2.nvidia.ops.attention_sm90 import MODE_OUT_LSE, rel_attention

    torch.manual_seed(0)
    heads, kv_heads, dim, page, extent = 16, 8, 64, 64, 128
    reqs, tokens, ctx = len(q_lens), sum(q_lens), past + max(q_lens)
    blocks = -(-ctx // page)
    # Codes without NaN (0x7f, 0xff); `_poison` covers the memory around them.
    cache = torch.randint(0, 0x60, (2, 1 + reqs * blocks, page, kv_heads, dim),
                          dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    q = torch.randn(tokens, heads, dim, device="cuda").to(torch.bfloat16)
    rel = torch.randn(tokens + 128, heads, extent + 256, device="cuda").to(torch.bfloat16)
    table = 1 + torch.arange(reqs * blocks, dtype=torch.int32, device="cuda").view(reqs, blocks)
    lens = torch.tensor([past + n for n in q_lens], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, *q_lens], device="cuda").cumsum(0).to(torch.int32)

    def run(want):
        out = torch.empty(tokens, heads, dim, device="cuda")
        lse = torch.empty(heads, tokens, device="cuda")
        rel_attention(q, cache[0], cache[1], out, lse, rel, block_table=table,
                      cache_seqlens=lens, cu_seqlens_q=cu, max_query_len=max(q_lens),
                      softmax_scale=dim**-0.5, window=window, num_splits=1, mode=MODE_OUT_LSE)
        return out if want == "out" else lse

    for want in ("out", "lse"):
        assert torch.isfinite(run(want)).all()
        _repeat(lambda want=want: run(want))


@pytest.mark.parametrize("tokens,block_m", [(48, 8), (375, 32), (2048, 64)])
def test_moe_rows_ordered(tokens, block_m):
    """`order_rows` on a skewed routing (some experts take many blocks, some none): the same
    table every time, each expert's rows ascending, every row once, invalid rows skipped."""
    from vllm.models.inkling2.nvidia.ops.moe_marlin import moe_align_block_size, order_rows

    torch.manual_seed(0)
    weights = 1 / torch.arange(1, EXPERTS + 1, device="cuda") ** 1.5
    ids = torch.stack([torch.multinomial(weights, TOPK) for _ in range(tokens)]).to(torch.int32)
    ids[-1] = -1  # a padding token
    rows = tokens * TOPK
    tables = []
    for _ in range(REPEAT):
        table = moe_align_block_size(ids, block_m, EXPERTS, ignore_invalid_experts=True)[:3]
        order_rows(ids, *table, block_m, EXPERTS)
        padded = int(table[2][0])
        tables.append(table[0][:padded].clone())
        assert torch.equal(tables[-1], tables[0])
    blocks = tables[0].view(-1, block_m)
    experts = table[1][: blocks.shape[0]]
    for e in range(EXPERTS):
        mine = blocks[experts == e].flatten()
        mine = mine[mine < rows]
        assert torch.equal(mine.long(), (ids.flatten() == e).nonzero().flatten())

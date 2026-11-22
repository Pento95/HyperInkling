"""bgemv against torch.bmm at the loop controllers' shapes, plus timings.

    venv/bin/python -m pytest -q -s tests/test_bgemv.py
"""

import time

import pytest
import torch

from vllm.models.inkling2.nvidia.ops.bgemv import bgemv, controller_gemv

RANK, WIDTH = 64, 2304


def _bench(fn, n=200):
    for _ in range(5):
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


@pytest.mark.parametrize("streams", [3, 4])
@pytest.mark.parametrize("tokens", [1, 6, 16, 32, 48, 64])
def test_bgemv(streams, tokens):
    torch.manual_seed(0)
    down = (torch.randn(streams, RANK, 2 * WIDTH, device="cuda") * 0.02).to(torch.bfloat16)
    up = (torch.randn(streams, 2 * WIDTH, RANK, device="cuda") * 0.02).to(torch.bfloat16)
    pair = torch.randn(streams, tokens, 2 * WIDTH, device="cuda").to(torch.bfloat16)

    hidden = bgemv(pair, down)
    want_h = torch.bmm(pair, down.transpose(1, 2))
    out = bgemv(hidden, up, torch.float32)
    want_o = torch.bmm(want_h, up.transpose(1, 2), out_dtype=torch.float32)
    rel_h = ((hidden.float() - want_h.float()).norm() / want_h.float().norm()).item()
    rel_o = ((out - want_o).norm() / want_o.norm()).item()

    t_new = _bench(lambda: bgemv(bgemv(pair, down), up, torch.float32))
    t_old = _bench(lambda: torch.bmm(torch.bmm(pair, down.transpose(1, 2)),
                                     up.transpose(1, 2), out_dtype=torch.float32))
    print(f"S={streams} T={tokens}: rel err {rel_h:.1e}/{rel_o:.1e}, "
          f"bgemv {t_new:.1f} us vs bmm {t_old:.1f} us (down + up)")
    assert rel_h < 1e-2 and rel_o < 1e-2


@pytest.mark.parametrize("streams", [3, 4])
@pytest.mark.parametrize("tokens", [1, 6, 16, 32, 48, 64])
def test_controller_gemv(streams, tokens):
    torch.manual_seed(0)
    down = (torch.randn(streams, RANK, 2 * WIDTH, device="cuda") * 0.02).to(torch.bfloat16)
    up = (torch.randn(streams, 2 * WIDTH, RANK, device="cuda") * 0.02).to(torch.bfloat16)
    pair = torch.randn(streams, tokens, 2 * WIDTH, device="cuda").to(torch.bfloat16)
    got = controller_gemv(pair, down, up)
    hidden = torch.bmm(pair, down.transpose(1, 2))
    want = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
    rel = ((got - want).norm() / want.norm()).item()
    t_new = _bench(lambda: controller_gemv(pair, down, up))
    t_two = _bench(lambda: bgemv(bgemv(pair, down), up, torch.float32))
    print(f"S={streams} T={tokens}: rel err {rel:.1e}, split-K {t_new:.1f} us vs bgemv x2 {t_two:.1f} us")
    assert rel < 1e-2

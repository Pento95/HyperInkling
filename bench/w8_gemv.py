"""W8A16 matmul tiles at decode and verify sizes: the tiles `ops/linear_w8.py` picks against a
grid search, on the attention projections and the LM head.

    venv/bin/python bench/w8_gemv.py [rows ...]
"""

import itertools
import sys
import time

import torch

from vllm.models.inkling2.nvidia.ops.linear_w8 import GROUP, _w8_kernel, quantize_int8, w8_matmul
from vllm.triton_utils import triton

# (N, K): qkvr (q, rel, gate, sink, k, v of one layer: one launch), o, the q side and k/v
# of a full layer (separate inputs), the LM head.
SHAPES = [(8600, 2304), (2304, 3072), (6552, 2304), (1024, 2304), (201024, 2304)]
ROWS = [int(a) for a in sys.argv[1:]] or [1, 3, 6, 12, 18, 30, 48, 64, 96, 128]


def _time(fn, n=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n // 10):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def tiled(x, q, s, out, bm, bn, warps, stages):
    m, k = x.shape
    n = q.shape[0]
    _w8_kernel[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
        x, q, s, out, m, n, k, x.stride(0), q.stride(0), s.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=GROUP, num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return out


def main():
    torch.manual_seed(0)
    for (n, k) in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        q, s = quantize_int8(w)
        nbytes = q.numel() + s.numel() * 2
        for m in ROWS:
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            ref = w8_matmul(x, q, s).float()
            t_now = _time(lambda: w8_matmul(x, q, s))
            out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            best = None
            top = min(128, max(16, triton.next_power_of_2(m)))
            bms = [b for b in (16, 32, 64, 128) if b <= top]
            for bm, bn, warps, stages in itertools.product(
                bms, (16, 32, 64, 128, 256), (2, 4, 8), (2, 3, 4, 5)
            ):
                try:
                    fn = lambda: tiled(x, q, s, out, bm, bn, warps, stages)  # noqa: E731
                    fn()
                    if not torch.allclose(out.float(), ref, atol=1e-2, rtol=1e-2):
                        continue
                    t = _time(fn)
                except Exception:  # out of shared memory etc.
                    continue
                if best is None or t < best[0]:
                    best = (t, bm, bn, warps, stages)
            t, bm, bn, warps, stages = best
            print(f"N {n:6d} K {k:5d} M {m:3d}: now {nbytes / t_now / 1e9:5.0f} GB/s, "
                  f"best {nbytes / t / 1e9:5.0f} GB/s (BM {bm}, BN {bn}, warps {warps}, "
                  f"stages {stages})", flush=True)


if __name__ == "__main__":
    main()

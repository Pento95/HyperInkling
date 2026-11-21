"""W4A16 decode GEMV: the split-K path of ops/linear_w4.py against `_w4_kernel` with
W8-style tiles (one launch, no fp32 partials), on the DSpark drafter's shapes.

    venv/bin/python bench/w4_gemv.py
"""

import itertools
import time

import torch

from vllm.models.inkling2.nvidia.ops.linear_w4 import (
    GROUP,
    _w4_kernel,
    dequantize_int4,
    quantize_int4,
    w4_matmul,
)
from vllm.triton_utils import triton

# (N, K): qkv, o, gate_up, down of a drafter layer; the folded Markov projection.
SHAPES = [(5120, 2304), (2304, 3072), (18432, 2304), (2304, 9216), (201024, 256)]
ROWS = [1, 7, 21, 42]


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


def direct(x, packed, scale, zero, out, bm, bn, warps, stages):
    m, k = x.shape
    n = packed.shape[0]
    _w4_kernel[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
        x, packed, scale, zero, out, m, n, k,
        x.stride(0), packed.stride(0), scale.stride(0), zero.stride(0), out.stride(0),
        BM=bm, BN=bn, G=GROUP, num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return out


def main():
    torch.manual_seed(0)
    for (n, k), m in itertools.product(SHAPES, ROWS):
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        packed, scale, zero = quantize_int4(w)
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        ref = (x.float() @ dequantize_int4(packed, scale, zero).T)
        nbytes = packed.numel() + scale.numel() * 2 + zero.numel()
        t_old = _time(lambda: w4_matmul(x, packed, scale, zero))
        best = None
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        bm = max(16, triton.next_power_of_2(m))
        for bn, warps, stages in itertools.product((16, 32, 64, 128), (2, 4, 8), (2, 3, 4, 5)):
            try:
                fn = lambda: direct(x, packed, scale, zero, out, bm, bn, warps, stages)  # noqa: E731
                fn()
                err = ((out.float() - ref).norm() / ref.norm()).item()
                if err > 1e-2:
                    continue
                t = _time(fn)
            except Exception:  # out of shared memory etc.
                continue
            if best is None or t < best[0]:
                best = (t, bn, warps, stages)
        t, bn, warps, stages = best
        print(f"N {n:6d} K {k:5d} M {m:2d}: split-K {nbytes / t_old / 1e9:5.0f} GB/s, "
              f"direct {nbytes / t / 1e9:5.0f} GB/s (BN {bn}, warps {warps}, stages {stages})",
              flush=True)


if __name__ == "__main__":
    main()

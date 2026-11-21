"""vLLM's moe_wna16_gemm (CUDA, small batches) vs Marlin MoE on one projection.

Same INT4 g128 asymmetric codes in both layouts; checks each against the
dequantized reference and times them under CUDA graphs over 8 copies of the
layer's weights (so L2 never holds them).

    venv/bin/python bench/moe_kernels.py
"""

import os
import sys
import time

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_wna16_block_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.models.inkling2.nvidia.ops.moe_marlin import _repack_int4
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_workspace,
)
from vllm.scalar_type import scalar_types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prepare"))
import int4_awq  # noqa: E402

E, TOPK, G = 128, 8, 128


def to_wna16(q, z, s):
    """Codes [E, N, K], zeros / scales [E, N, K/G] -> moe_wna16 layout."""
    qweight = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()  # [E, N, K/2]
    qzeros = (z[:, 0::2, :] | (z[:, 1::2, :] << 4)).contiguous()  # [E, N/2, K/G]
    return qweight, s.contiguous(), qzeros


def run_wna16(x, w, ids, n, k, top_k):
    qweight, scales, qzeros = w
    rows = ids.numel()
    bm = 16
    cfg = get_moe_wna16_block_config(
        config={"BLOCK_SIZE_M": bm}, use_moe_wna16_cuda=True, num_valid_tokens=rows,
        size_k=k, size_n=n, num_experts=E, group_size=G, real_top_k=top_k,
        block_size_m=bm,
    )
    sorted_ids, expert_ids, ntp = moe_align_block_size(ids, bm, E)[:3]
    out = torch.zeros((x.shape[0], top_k, n), dtype=x.dtype, device=x.device)
    ops.moe_wna16_gemm(x, out, qweight, scales, qzeros, None, sorted_ids, expert_ids,
                       ntp, top_k, bm, cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"], 4)
    return out.view(-1, n)


def run_triton(x, w, ids, n, k, top_k):
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_wna16_triton_kernel,
    )
    from vllm.triton_utils import tl

    qweight, scales, qzeros = w
    bm = 16
    sorted_ids, expert_ids, ntp = moe_align_block_size(ids, bm, E)[:3]
    out = torch.zeros((x.shape[0], top_k, n), dtype=x.dtype, device=x.device)
    invoke_fused_moe_wna16_triton_kernel(
        x, qweight, out, scales, qzeros, None, sorted_ids, expert_ids, ntp,
        False, top_k, {"BLOCK_SIZE_M": bm, "GROUP_SIZE_M": 1, "SPLIT_K": 1}, tl.bfloat16,
        False, True, [0, G])
    return out.view(-1, n)


def run_marlin(x, w, ids, n, k, top_k, ws):
    qweight, scales, zeros = w
    rows = ids.numel()
    for bm in (8, 16, 32, 48, 64):
        if rows / E / bm < 0.9:
            break
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        ids, bm, E, ignore_invalid_experts=True)[:3]
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    tw = torch.ones(ids.shape, dtype=torch.float32, device=x.device)
    ops.moe_wna16_marlin_gemm(
        x, out, qweight, None, scales, None, zeros, ws, sorted_ids, expert_ids, ntp, tw,
        moe_block_size=bm, top_k=top_k, mul_topk_weights=False,
        b_q_type=scalar_types.uint4, size_m=x.shape[0], size_n=n, size_k=k,
        use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    return out


def bench(fns, n=400):
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    torch.cuda.synchronize()
    reps = max(1, n // len(fns))
    t = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (reps * len(fns)) * 1e6


def main():
    torch.manual_seed(0)
    ws = get_marlin_workspace(torch.device("cuda"))
    for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
        layers = []
        for _ in range(8):
            w = torch.randn(E, n, k, device="cuda") * 0.02
            q, z, s = int4_awq.quantize(w, clip_search=False)
            dense = int4_awq.dequantize(q, z, s) if not layers else None
            layers.append((to_wna16(q, z, s),
                           _repack_int4(*int4_awq.to_awq_layout(q, z, s), k), dense))
            del w, q, z, s
        for tokens in (1, 2, 6):
            # w13 gathers each token's row topk times; w2 reads row by row.
            top_k = TOPK if name == "w13" else 1
            m = tokens if name == "w13" else tokens * TOPK
            x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
            ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK]
                               for _ in range(tokens)]).to(torch.int32)
            if name == "w2":
                ids = ids.view(-1, 1)
            dense = layers[0][2]
            flat = ids.flatten().long()
            src = torch.arange(flat.numel(), device="cuda") // top_k
            want = torch.einsum("rk,rnk->rn", x[src].float(), dense[flat])
            for label, fn in (
                ("triton", lambda L=layers[0]: run_triton(x, L[0], ids, n, k, top_k)),
                ("wna16", lambda L=layers[0]: run_wna16(x, L[0], ids, n, k, top_k)),
                ("marlin", lambda L=layers[0]: run_marlin(x, L[1], ids, n, k, top_k, ws)),
            ):
                got = fn().float()
                err = ((got - want).norm() / want.norm()).item()
                if label == "triton":
                    fns = [lambda L=L: run_triton(x, L[0], ids, n, k, top_k) for L in layers]
                elif label == "wna16":
                    fns = [lambda L=L: run_wna16(x, L[0], ids, n, k, top_k) for L in layers]
                else:
                    fns = [lambda L=L: run_marlin(x, L[1], ids, n, k, top_k, ws) for L in layers]
                t = bench(fns)
                bytes_read = len(flat.unique()) * n * k / 2
                print(f"{name} tokens {tokens}: {label:6s} {t:7.1f} us "
                      f"({bytes_read / t / 1e3:5.0f} GB/s)  err {err:.1e}")


if __name__ == "__main__":
    main()

"""DFlash2's draft attention on SM8x (BlockDraftImpl, vllm/models/inkling2/nvidia/dspark.py):
FlashAttention's metadata into `block_decode_attention`, against a torch reference."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.inkling2.nvidia.dspark import BlockDraftImpl

DEVICE = "cuda"


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("sliding_window", [None, 24])
def test_block_draft_attention_matches_reference(sliding_window, cache_dtype):
    g = torch.Generator().manual_seed(7)
    block, heads, kv_heads, dim, page, pages = 8, 32, 8, 128, 16, 12
    context_lens = [0, 5, 40, 150]  # keys before each request's block
    num_reqs = len(context_lens)

    impl = BlockDraftImpl.__new__(BlockDraftImpl)
    impl.num_heads, impl.head_size, impl.num_kv_heads, impl.block = (
        heads,
        dim,
        kv_heads,
        block,
    )
    impl.window = sliding_window - 1 + block if sliding_window else 0

    # The model's [blocks, page, kv_heads, (K | V)] cache, as the impl sees it
    # (it transposes back); requests own shuffled pages.
    kv = torch.randn(num_reqs * pages, page, kv_heads, 2 * dim, generator=g)
    kv = kv.to(cache_dtype).to(DEVICE)
    kv_cache = kv.transpose(1, 2)
    perm = torch.randperm(num_reqs * pages, generator=g).int().to(DEVICE)
    block_table = perm.view(num_reqs, pages)
    seq_lens = torch.tensor([n + block for n in context_lens], dtype=torch.int32)
    md = SimpleNamespace(
        seq_lens=seq_lens.to(DEVICE),
        block_table=block_table,
        query_start_loc=(torch.arange(num_reqs + 1) * block).int().to(DEVICE),
    )
    # Two padding requests' rows past the real ones, as a CUDA graph pads.
    q = (
        torch.randn((num_reqs + 2) * block, heads, dim, generator=g)
        .bfloat16()
        .to(DEVICE)
    )
    out = torch.zeros(q.shape[0], heads * dim, dtype=torch.bfloat16, device=DEVICE)
    impl.forward(None, q, None, None, kv_cache, md, out)

    key_cache, value_cache = kv.split(dim, dim=-1)
    out = out.view(-1, heads, dim)
    for r, n in enumerate(context_lens):
        rows = torch.cat(
            [torch.arange(p * page, (p + 1) * page) for p in block_table[r].tolist()]
        )
        start = max(0, n - (sliding_window - 1)) if sliding_window else 0
        keys = (
            key_cache.flatten(0, 1)[rows[start : n + block]]
            .float()
            .repeat_interleave(4, 1)
        )
        vals = (
            value_cache.flatten(0, 1)[rows[start : n + block]]
            .float()
            .repeat_interleave(4, 1)
        )
        qs = q[r * block : (r + 1) * block].float()
        scores = torch.einsum("qhd,khd->hqk", qs, keys) * dim**-0.5
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals)
        torch.testing.assert_close(
            out[r * block : (r + 1) * block].float(), ref, atol=2e-2, rtol=2e-2
        )

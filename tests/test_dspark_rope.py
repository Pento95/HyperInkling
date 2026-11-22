"""The DSpark draft's RoPE with in-kernel angles against vLLM's cos/sin table
(`RotaryEmbedding._compute_cos_sin_cache`), up to the drafter's 1M positions.

    venv/bin/python -m pytest -q tests/test_dspark_rope.py
"""

import torch

from vllm.models.inkling2.nvidia.ops.dspark import rope_

D, THETA = 128, 1_000_000.0


def test_rope_matches_the_table_far_out():
    torch.manual_seed(0)
    inv_freq = 1.0 / THETA ** (torch.arange(0, D, 2, dtype=torch.float) / D)
    inv_freq = inv_freq.cuda()
    positions = torch.cat([
        torch.arange(0, 64), torch.randint(0, 1 << 20, (4096,)), torch.tensor([(1 << 20) - 1])
    ]).cuda()
    x = torch.randn(positions.numel(), 4, D, device="cuda").to(torch.bfloat16)

    freqs = positions.float()[:, None] * inv_freq[None, :]  # the table's rows
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
    x1, x2 = x[..., : D // 2].float(), x[..., D // 2 :].float()
    want = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(torch.bfloat16)

    got = x.clone()
    rope_(got, positions, inv_freq)
    diff = (got.view(torch.int16).int() - want.view(torch.int16).int()).abs()
    assert diff.max().item() <= 1  # at most one bf16 ulp
    assert (diff == 0).float().mean().item() > 0.999

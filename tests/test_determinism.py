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

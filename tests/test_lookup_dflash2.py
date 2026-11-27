"""The context lookup on DFlash2's sparse draft-logits cache: a point mass must
not outlive the step it was taken in (the cache clears only its last candidates)."""

from types import SimpleNamespace

import torch

from vllm.models.inkling2.nvidia.ops.lookup import ContextLookup, _point_mass_kernel
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler

DEVICE = "cuda"


def test_point_masses_are_cleared_by_the_next_step():
    reqs, steps, top_k, vocab = 2, 3, 4, 64
    sampler = CandidateSampler(reqs, steps, top_k, DEVICE)
    cache = SimpleNamespace(
        values=torch.full((reqs, steps, vocab), float("-inf"), device=DEVICE),
        row_mapping=None,
    )
    idx_mapping = torch.arange(reqs, device=DEVICE).repeat_interleave(steps).int()

    def step(candidates):
        sampler.scores.copy_(torch.randn(reqs, steps, top_k, device=DEVICE))
        sampler._cache_draft_logits(
            candidates.view(-1, top_k), reqs * steps, idx_mapping, cache
        )

    step(torch.arange(reqs * steps * top_k, device=DEVICE).view(reqs, steps, top_k))
    # The lookup takes request 0's first two drafts (tokens 60, 61).
    lookup = ContextLookup(reqs, steps, torch.device(DEVICE))
    lookup.used.copy_(torch.tensor([2, 0], dtype=torch.int32))
    drafts = torch.zeros(reqs, steps, dtype=torch.int32, device=DEVICE)
    drafts[0, :2] = torch.tensor([60, 61], dtype=torch.int32)
    _point_mass_kernel[(reqs, steps)](
        drafts, drafts.stride(0), lookup.used, idx_mapping, steps,
        cache.values, cache.values.stride(0), cache.values.stride(1), vocab,
        BLOCK_V=64,
    )  # fmt: skip
    lookup.claim_candidates(
        drafts, idx_mapping, steps, reqs, sampler.cached_candidate_ids
    )
    assert cache.values[0, 0].isfinite().sum() == 1

    fresh = 30 + torch.arange(reqs * steps * top_k, device=DEVICE) % 20
    step(fresh.view(reqs, steps, top_k))
    for r in range(reqs):
        for s in range(steps):
            finite = cache.values[r, s].isfinite().nonzero().flatten().tolist()
            assert sorted(finite) == sorted(
                set(fresh.view(reqs, steps, top_k)[r, s].tolist())
            )

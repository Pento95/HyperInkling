"""Context lookup drafting (ops/lookup.py): what it proposes, when it overrides the
drafter, and the point mass it leaves in the draft-logits cache.

    venv/bin/python -m pytest -q tests/test_lookup.py
"""

import torch

from vllm.models.inkling2.nvidia.ops.lookup import ContextLookup

K, VOCAB = 7, 5000


def _run(history, drafts, with_cache=True, **kw):
    """One request in request-state slot 1; idx_mapping has stride K, as the speculator's."""
    hist = torch.zeros(3, 4096, dtype=torch.int32, device="cuda")
    hist[1, : len(history)] = torch.tensor(history, dtype=torch.int32)
    total_len = torch.tensor([0, len(history), 0], dtype=torch.int32, device="cuda")
    idx_mapping = torch.full((K,), 1, dtype=torch.int32, device="cuda")
    draft = torch.tensor([drafts], dtype=torch.int64, device="cuda")
    cache = torch.zeros(3, K, VOCAB, dtype=torch.bfloat16, device="cuda") if with_cache else None
    look = ContextLookup(1, K, "cuda", **kw)
    look.apply(hist, total_len, idx_mapping, K, 1, draft, cache)
    return draft[0].tolist(), cache, look


def test_copies_the_continuation_of_the_longest_match():
    doc = list(range(100, 160))
    # The answer is quoting doc[10:20]; the continuation is doc[20:27].
    history = [1, 2, 3, *doc, 7, 8, *doc[10:20]]
    drafts, cache, look = _run(history, [9] * K)
    assert drafts == doc[20:27]
    assert look.used[0].item() == 1
    for step, tok in enumerate(doc[20:27]):
        row = cache[1, step].float()
        assert row[tok].item() == 0.0 and torch.isinf(row).sum().item() == VOCAB - 1
    assert (cache[0] == 0).all() and (cache[2] == 0).all()  # other requests untouched


def test_short_match_needs_the_drafter_to_agree():
    doc = list(range(200, 230))
    history = [*doc, 5, *doc[:6]]  # a 6-token match: below nstrong (8)
    drafts, cache, _ = _run(history, [9] * K)
    assert drafts == [9] * K and (cache == 0).all()
    agreeing = doc[6:8] + [9] * (K - 2)
    drafts, _, _ = _run(history, agreeing)
    assert drafts == doc[6:13]


def test_period_shorter_than_the_block_fills_only_what_follows():
    # Period 3: the most recent long match ends 3 tokens before the history does.
    drafts, cache, _ = _run([1, 2, 3] * 6, [9] * K)
    assert drafts == [1, 2, 3] + [9] * (K - 3)
    assert (cache[1, 3:] == 0).all()


def test_no_match_leaves_everything():
    drafts, cache, look = _run(list(range(400, 450)), [9] * K)
    assert drafts == [9] * K and (cache == 0).all() and look.used[0].item() == 0


def test_greedy_without_cache():
    doc = list(range(500, 540))
    drafts, _, _ = _run([*doc, *doc[:12]], [9] * K, with_cache=False)
    assert drafts == doc[12:19]

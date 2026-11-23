"""SconvStateManager (the conv-state cache's KV manager) at Inkling's sizes: 8-token
blocks, a 4-tap window, 7 drafts, prefix hits aligned to 64 tokens. No GPU.

    venv/bin/python -m pytest -q tests/test_sconv_state_manager.py
"""

import pytest
import torch

from vllm.models.inkling.nvidia.sconv_swa_attn import SconvStateManager
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import (
    DFlashSlidingWindowManager,
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup
from vllm.v1.request import Request

BLOCK, WINDOW, DRAFTS, ALIGN, CHUNK = 8, 4, 7, 64, 1024


def _layer(max_in_flight_steps=1):
    spec = AttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
    return SconvStateManager.layer_config(
        spec,
        sliding_window=WINDOW,
        num_speculative_tokens=DRAFTS,
        max_in_flight_steps=max_in_flight_steps,
        num_lookahead_tokens=DRAFTS,
        hit_alignment_tokens=ALIGN,
    )


def _manager(layer, num_blocks=4096, caching=False):
    group = KVCacheGroup({"layer": layer.spec}, layer.manager_config, BLOCK, False)
    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=caching, hash_block_size=BLOCK)
    return SconvStateManager(
        group,
        block_pool=pool,
        enable_caching=caching,
        kv_cache_group_id=0,
        scheduler_block_size=ALIGN,
        max_admission_blocks_per_request=10**9,
    )


def _resident(manager, req_id):
    return [i for i, b in enumerate(manager.req_to_blocks[req_id]) if not b.is_null]


def test_prefill_chunks_hold_only_their_tails():
    """A 1024-token chunk holds a tail of (4 + 7 + 64) tokens, not 128 blocks."""
    manager = _manager(_layer())
    tail_blocks = -(-(WINDOW + DRAFTS + ALIGN + DRAFTS) // BLOCK) + 1
    computed = 0
    for end in (CHUNK, 2 * CHUNK, 3000):
        manager.remove_skipped_blocks("r", computed)
        manager.allocate_new_blocks("r", end + DRAFTS, end)
        computed = end
        assert len(_resident(manager, "r")) <= 2 * tail_blocks
    # The next step reads the taps of the last 3 prompt tokens.
    assert set(range((3000 - WINDOW + 1) // BLOCK, 3000 // BLOCK)) <= set(
        _resident(manager, "r")
    )


@pytest.mark.parametrize("max_in_flight_steps", [1, 2])
def test_taps_backed_after_rejected_drafts(max_in_flight_steps):
    """Every draft in flight rejected: the next step's cached taps (the 3 tokens
    below the accepted ones) still have real blocks."""
    manager = _manager(_layer(max_in_flight_steps))
    for target in range(2000, 2000 + BLOCK):
        req_id = str(target)
        manager.allocate_new_blocks(req_id, target + DRAFTS, target)
        accepted = target - DRAFTS * max_in_flight_steps
        blocks = manager.req_to_blocks[req_id]
        assert blocks[0].is_null
        assert not any(b.is_null for b in blocks[(accepted - WINDOW + 1) // BLOCK :])


@pytest.mark.parametrize("prompt_len", [3000, 3008, 2049])
def test_prefix_hit_at_the_aligned_position_below_the_prompt_end(prompt_len):
    """A follow-up that extends a prompt prefilled in 1024-token chunks hits at
    the last 64-aligned position of that prompt, not at a chunk boundary."""
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK, sha256)
    manager = _manager(_layer(), caching=True)

    def request(req_id, n):
        return Request(
            request_id=req_id,
            prompt_token_ids=list(range(n)),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=hasher,
        )

    first = request("a", prompt_len)
    computed = 0
    while computed < prompt_len:
        end = min(computed + CHUNK, prompt_len)
        manager.remove_skipped_blocks("a", computed)
        manager.allocate_new_blocks("a", end, end)
        manager.cache_blocks(first, end, replay_boundaries=[])
        computed = end

    follow_up = request("b", prompt_len + 500)
    _, hit = SconvStateManager.find_longest_cache_hit(
        block_hashes=follow_up.block_hashes,
        max_length=prompt_len + 499,
        kv_cache_group_ids=[0],
        block_pool=manager.block_pool,
        block_size=BLOCK,
        manager_config=manager.manager_config,
        alignment_tokens=ALIGN,
    )
    assert hit == prompt_len // ALIGN * ALIGN


@pytest.mark.parametrize("chunk", [1024, 2000, 2048])
@pytest.mark.parametrize("shared,total", [(1000, 6052), (2344, 3076), (3000, 6052), (4090, 6052)])
def test_shared_prefix_hit_away_from_a_step_end(shared, total, chunk):
    """Requests sharing only their first tokens (a system prompt, then different pages)
    resume at the junction from the third on, with every KV group of the served model
    (full, window 1024, conv state, DSpark drafter)."""
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK, sha256)

    def spec(block):
        return AttentionSpec(block_size=block, num_kv_heads=1, head_size=1, dtype=torch.float32)

    layers = [
        FullAttentionManager.layer_config(spec(ALIGN)),
        SlidingWindowManager.layer_config(spec(ALIGN), sliding_window=1024),
        _layer(2),
        DFlashSlidingWindowManager.layer_config(
            spec(ALIGN), sliding_window=1024, num_speculative_tokens=DRAFTS,
            max_in_flight_steps=2, num_lookahead_tokens=DRAFTS),
    ]
    groups = [KVCacheGroup({f"l{i}": c.spec}, c.manager_config, c.spec.block_size, False)
              for i, c in enumerate(layers)]
    manager = KVCacheManager(
        KVCacheConfig(num_blocks=20000, kv_cache_tensors=[], kv_cache_groups=groups,
                      prefix_cache_retention_interval=0, hash_block_size=BLOCK),
        max_model_len=100000, scheduler_block_size=ALIGN, hash_block_size=BLOCK,
        enable_caching=True, max_in_flight_tokens=2 * chunk,
    )
    free = manager.block_pool.get_num_free_blocks()
    hits = []
    for k in range(4):
        ids = [*range(shared), *range(10**6 * (k + 1), 10**6 * (k + 1) + total - shared)]
        request = Request(request_id=str(k), prompt_token_ids=ids, pooling_params=None,
                          sampling_params=SamplingParams(max_tokens=1), block_hasher=hasher)
        blocks, hit, request.shared_prefix_boundary = manager.get_computed_blocks(request)
        hits.append(hit)
        pos, first = hit, {"num_new_computed_tokens": hit, "new_computed_blocks": blocks}
        while pos < total:
            step = min(chunk, total - pos)
            assert manager.allocate_slots(
                request, step, num_lookahead_tokens=DRAFTS, **first) is not None
            first, pos = {}, pos + step
            request.num_computed_tokens = pos
        manager.free(request)
        assert manager.block_pool.get_num_free_blocks() == free  # nothing leaks
    assert hits[2:] == [shared // ALIGN * ALIGN] * 2

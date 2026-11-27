"""The online verify length (vllm/v1/spec_decode/dynamic/adaptive_k.py). No GPU."""

from vllm.v1.spec_decode.dynamic.adaptive_k import AdaptiveVerifyLength


def test_a_step_without_drafts_keeps_the_acceptance():
    # Async scheduling can schedule draft slots that end up with no draft: such
    # a verdict must not pull predictable text's acceptance back to the prior.
    k = AdaptiveVerifyLength([2, 4, 7], 7)
    for _ in range(200):
        k.observe_request(7, 7)
    before = [k.p(i) for i in range(7)]
    for _ in range(50):
        k.observe_request(0, 0)
    assert [k.p(i) for i in range(7)] == before

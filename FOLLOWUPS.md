# Follow-ups

Where the work stopped on 2026-11-21, in the order to pick it up.

## Plan

Any item that trades something for something else ships as a wizard option, not a default.

| phase | items | then |
|---|---|---|
| 2, the checkpoint | 6, 8, 11, 12, 13, 14 | checkpoint on Hugging Face (2) |
| 3, context | 17, 18, 19, 30 | |
| 4, speed and polish | 7, 9, 10, 21, 22, 31 | Reddit post |

Phase 2 holds everything that changes the published checkpoint, so it is uploaded once. Wizard
alternatives to the default tensors go in the same HF repo as separate files, fetched
with `hf download --include` only when chosen.

## Release (in progress)

2. **Published 0.1.1** (GitHub release, `ghcr.io/pento95/hyperinkling:0.1.1`, the HF checkpoint;
   re-upload the HF README from docs/MODEL_CARD.md when the card changes; a new image goes
   with a new tag and the compose default).

## Still to try

Where each idea comes from: **[HQ]** HyperQwen (github.com/syv-ai/HyperQwen, its
`patches/` and docs), **[BS]** basalt (github.com/jesdga95/basalt, MIT), **[IK]** the
structure of Inkling 2 Mini itself, **[NS]** ours (measurements in this repo). Tried and
dropped ideas are at the end, with the reason, so they are not re-tried blind.

### DSpark

6. **Lookup drafting from the context** [HQ `dflash2-lookup-drafting`, BS suffix drafter]:
   `ops/lookup.py`, `HYPERINKLING_LOOKUP`, on by default with the adaptive length; lossless (greedy text equals
   `SPEC=off`). Copying a 120-line table from a 2-50k document (`bench/copy_test.py`): at
   the default k it adds 2% (4.88 -> 5.00 tokens per step, the ceiling); at k=7 7-13%
   (7.07-7.43 -> 8.00; 438 -> 494 tok/s at 50k). The drafter copies tables well on its own;
   the verify length is the bigger lever (k=7: +33% while copying, -14 to -21% on chat and
   code), hence 7. On spec_bench and the extraction bench the lookup is within noise.
7. **Per-request verify length** [BS draft policy, vLLM adaptive verification]: needs decode
   graphs with per-request query lengths. docs/varlen-decode-investigation.md has the map,
   with file:line references and open doubts;
   mapped the changes: the sconv metadata builder's varlen bound, the decode combine kernel
   zeroing rows from the device total, the CPU `max_query_len` as an upper bound, and a
   varlen test (the `SKIP_IDLE_CACHE` padding bug it found is fixed). Today `adaptive_k.py` picks one length per step for the whole batch (default, candidates 2,4,7).
8. **Drafter GPTQ** [NS]: low priority, INT4 RTN accepts like bf16 on counting (6.6 vs 6.7).
9. **Kernels compiled at boot** [HQ `spec-sampler-prewarm`]: vLLM's warmup prefills
   9-token requests only, so the first long prefill and first mixed batches compile ~10
   Triton variants (`_sconv_kernel` above 256 tokens, `_rel_attention_kernel`, ...): a
   ~5 s first TTFT, once per Triton cache (a compose volume). A long-prompt warmup step
   in `v1/worker/gpu/warmup.py` would move it to boot.

### Speed

10. **Small-kernel fusion in the loop** [IK: 48 layer visits x ~15 kernels of 2-5 us]
    (~1.5 ms of a 7.5 ms step at one request): `_attn_res_row_kernel`, `_sconv_kernel`,
    `_k_group_kernel` (router), `_attn_prologue`, `_blend_rms_norm`, routing. The reference
    numerics (fixed summation orders, no FMA contraction) live in these kernels: fuse with
    bitwise checks against the unfused path.
11. **Unembed INT4 GPTQ** [HQ fast variant: int4 GPTQ lm_head]: ~0.25 ms per step, and the
    drafter's base logits would read half as much. Judge by the KL median.
12. **Attention 4-bit** [NS] (GPTQ + int4 GEMV): ~0.7 ms per step at one request.
13. **A faster W4A16 GEMV** [NS]: `ops/linear_w4.py` reaches 400-550 GB/s at decode (INT8
    ~900). Outputs of 16k columns or more now take the single-launch kernel (Markov fold
    786 vs 660 GB/s at 1 row, 679 vs 431 at 7; gate_up 724 vs 554), the rest keep split-K,
    which the single launch loses to (`bench/w4_gemv.py`). The remaining cost is the
    nibble unpack and conversion; a faster kernel there helps 11 and 12.

### Quality and measurement

14. **Run-to-run nondeterminism** [NS]: the same request scored twice on one server differs
    from position 1 (~0.003 in logprob), and an expert flip carries it down the sequence
    (KL mean floor 0.060; the median, 0.0003, is what decisions use). Bitwise repeatable
    (`tests/test_determinism.py`): Marlin MoE, the W8 linear, `bgemv`, `short_conv`. Not yet
    tested: `_rel_attention_kernel` (prefill attention), the router (`_k_group_kernel`,
    `select_experts`), `attn_res`, cuBLAS.

### Context, best effort (no DSpark, vision off the GPU)

17. **sconv state cache** as a fixed 3-token state instead of a fake sliding window [IK]:
    frees the per-chunk-token reserve (0.85 GiB per step in flight at NBT 1024).
18. **4-bit KV for the global layers** [HQ KVarN]: halves the per-token KV. Then the wizard
    needs a 4-bit column.
19. **Least-routed experts in host RAM when the context needs room** [BS]: the way to
    approach 1M without DSpark and without the vision tower on the GPU.

### Other

21. **Rust frontend** for the native build: vendored OpenSSL needs Perl `FindBin` (the
    container builds it).
22. **Upstream**: the processing.py dummy-size fix and the oversized-image downscale are real
    upstream bugs; worth a vLLM PR.
### Added with the plan


30. **Memory profile after warmup** [HQ `memory-profile-after-warmup`]: vLLM checks the
    context limit before the warmup, hence the two-boot measurement (NOTES). Profiling
    after it makes the first boot's estimate the real one.
31. **Field-report issue template** [HQ]: for the Reddit post's invitation to share results.

### Tried and dropped (2026-11-21)

- Verify-side MoE kernel work [NS]: Marlin already reads ~900 GB/s; a second verified token
  costs less than double because tokens share experts.
- Screening the Markov head on the base logits' top-k [BS-inspired]: acceptance collapsed
  (counting 6.6 -> 2.9 tokens/step); the right token comes from the Markov bias.
- A drafter-only INT4 copy of the LM head for base logits [HQ fast variant / BS 2-bit
  screen]: -1.5% step time for 0.23 GiB; revisit with item 13.
- Adaptive verification as shipped by vLLM [BS/vLLM]: the sconv backend refuses varlen
  decode graphs (item 7 is the way).
- HyperQwen's `kv-cache-seat-budget` (its `KVCacheGroup` API differs here),
  `terminal-request-progress` (does not apply), `tokenize-v1-route` (not needed).
- The release's MXFP8 experts on layers 20-23 [BS] (`prepare/mix_experts.py`, 2026-11-21):
  KL median 0.0027 vs 0.0030 (the noise floor is 0.0003), top-1 92.5% vs 92.35%, for
  +1.94 GiB (~110k tokens of context); layer 23 alone changes nothing.
- A draft vocabulary subset [HQ `qwen3_5-mtp-draft-vocab`] (2026-11-22): base logits and the
  Markov fold on 32k or 48k head rows (the most frequent in the model's own calibration
  output, 9.5k distinct, filled by lowest id). Steps 5% shorter, but acceptance fell
  (extraction bench 42.5% -> 34-35%, 419 -> 394-400 tok/s; struct -5 to -12%); only chat
  gained (+3-7%). Document text needs the full vocabulary.

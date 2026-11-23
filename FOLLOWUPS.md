# Follow-ups

What comes next, by phase. Any item that trades something for something else ships as a
wizard option, not a default. Alternative tensors go in the same HF repo as separate files,
fetched with `hf download --include` only when chosen.

| phase | items |
|---|---|
| 2, the checkpoint (new files on HF) | 8, 11, 12, 33 |
| 3, context (best effort) | 18, 19, 13, 38, 39, 55 |
| 4, speed and upstream | 10, 7, 21, 22, 32, 51 |

Released: `ghcr.io/pento95/hyperinkling:0.1.3` (compose default) and two HF checkpoints, the
original (README = docs/MODEL_CARD.md) and the uncensored one (README =
docs/MODEL_CARD_UNCENSORED.md): re-upload a card when it changes. A new image goes with a new
tag (`scripts/build-image.sh` refuses a published one) and the compose default.

Where each idea comes from: **[HQ]** HyperQwen (github.com/syv-ai/HyperQwen, `patches/`, docs),
**[BS]** basalt (github.com/jesdga95/basalt, MIT), **[AR]** Ar4ikov's vLLM PRs
[#69165](https://github.com/vllm-project/vllm/pull/69165) (Inkling 2 on SM80/86/89) and
[#69141](https://github.com/vllm-project/vllm/pull/69141) (compressed-tensors experts),
**[AX]** github.com/agent2x0r/inkling2-sm86-kernels (2x3090 bring-up, `labs/`, `csrc/`),
**[C3]** club-3090 (github.com/noonghunna/club-3090, its `vllm-inkling` image recipe),
**[LC]** [llm-compressor](https://github.com/vllm-project/llm-compressor), **[IK]** the model's
own structure, **[NS]** ours (measurements in this repo, nsys profiles).

## Phase 2: the checkpoint

8. **Drafter GPTQ** [NS]: low priority, INT4 RTN accepts like bf16 on counting (6.6 vs 6.7).
11. **Unembed INT4 GPTQ** [HQ fast variant]: ~0.25 ms per step, and the drafter's base
    logits read half as much. Judge by the KL median. Needs 13 (the W4 kernel reaches ~500
    GB/s on 201k x 2304, INT8 ~900).
12. **Attention 4-bit** [NS] (GPTQ + int4 GEMV): halves the bytes, but the small attention
    matrices are latency-bound (35): measure the step, not GB/s.
    Also only layers 0-5 and 18-23, the loop core staying INT8: half the bytes saved but
    12 of the 48 visits. Keep it if the KL median does not get worse and the context or the step
    gains. Does not apply to 11: the head runs once per token.
33. **llm-compressor / compressed-tensors** [LC, AR #69141]: llm-compressor calibrates GPTQ,
    AWQ and SmoothQuant schemes on the HF model and writes compressed-tensors checkpoints;
    #69141 runs such experts (W4A16, W4A8, W8A8, FP8) on Humming. Three uses here:
    (a) a second calibration of the experts, attention and head (11, 12) to compare with
    `prepare/gptq_experts.py` by KL median; (b) loading compressed-tensors W4A16 experts by
    repacking to the Marlin layout, so community quantizations run unchanged;
    (c) W8A8 INT8 for prefill: sm86's INT8 tensor cores run at twice the bf16 rate and
    prefill is compute-bound (vLLM has W8A8 int8 GEMMs for sm80; the MoE side needs a kernel).
    #69141 also fixes the DSpark draft quantization config and casts fp32 checkpoint tensors
    to bf16: check both against our drafter loader at the rebase (32).

## Phase 3: context (best effort: no DSpark, vision off the GPU)

18. **4-bit KV for the global layers** [HQ KVarN]: halves the per-token KV. Then the wizard
    needs a 4-bit column.
19. **Least-routed experts in host RAM when the context needs room** [BS]: the way toward
    1M without DSpark and without the vision tower on the GPU.
13. **Drafter W4 at 6-16 requests** [NS]: past ~21 rows the W4 kernels
    are tensor-core bound (int4 halves the bytes per MAC, so the bound comes at half the
    rows of INT8): 1.3 ms per step at 6 requests. fp16 accumulation doubles the MMA rate on
    GeForce; for the drafter it moves only acceptance (est. +2-3% at 6-16 requests).
    Building bf16 from bits (`code | 0x4300` = 128 + code, the zero point corrected per
    group from the row sums) gave +15-20% at 1-7 rows on split-K, worth ~0.03 ms a step.
55. **Sources in bf16** (other fine-tunes): `prepare/quantize_experts.py` and
    `gptq_experts.py` read only the release's NVFP4/MXFP8 experts (as the uncensored source
    is), and a bf16 Inkling 2 Mini does not fit 24 GiB to generate the corpus and capture.
38. **Drafter SWA layers padded into the global groups' size** [NS]: the KV manager groups 9
    layers per block; the drafter's 2 SWA layers pad to 9 (the "350% padding" warning).
    Small real waste per block; merging that bucket with another frees it.
39. **Fewer bits for the least-routed experts** [NS, BS]: captured routing gives each
    expert's frequency; 3-bit (or 2-bit) for the tail frees GiBs of context. Marlin is 4/8
    bit only: needs a mixed-bit MoE kernel. KL median decides.

## Phase 4: speed and polish

7. **Per-request verify length** [BS draft policy, vLLM adaptive verification]: needs decode
   graphs with per-request query lengths; docs/varlen-decode-investigation.md maps the
   changes (sconv builder varlen bound, the decode combine kernel's row bound from the
   device total, the CPU `max_query_len` as an upper bound, a varlen test). Today
   `adaptive_k.py` picks one length per step for the whole batch.
10. **Small-kernel fusion in the loop** [IK, AX `labs/lab1`, `csrc/convmix_sm86.cu`]: ~3 ms
    of a 22 ms step at 6 requests (48 layer visits x ~15 kernels of 2-5 us):
    `_attn_res_row_kernel`, `_sconv_kernel`, `_k_group_kernel` (router), `_attn_prologue`,
    `_blend_rms_norm`, routing. AX fuses short conv + residual + RMSNorm in one kernel
    (`cp.async` staging) and blend + RMSNorm with the row kept in registers. The reference
    numerics (fixed summation orders, no FMA contraction, exact `tanh`: AX measured
    `tanh.approx` flipping 0.23% of bf16 outputs) must hold: check bitwise against the
    unfused path.
    Measured at 6 requests: the controller chain (`pair_rms`, `down_split`,
    `up_from_partials`, `blend_rms_norm`) is 18 us per visit, but `blend`'s RMSNorm sums the
    whole row in one `tl.sum` and `up` must stay a `tl.dot`, so they cannot merge bitwise;
    `pair_rms` into `down_split` saves ~0.1 ms, `attn_res_row` (0.53 ms) is the other
    candidate: ~1-2% in all.
21. **Rust frontend** for the native build: vendored OpenSSL needs Perl `FindBin` (the
    container builds it).
22. **Upstream PRs**: the processing.py dummy-size fix and the oversized-image downscale are
    real upstream bugs; so is the sconv padding-row fix (NOTES).
32. **BASE to vLLM main when #69165 merges** [AR]: `patches/BASE` is its head today
    (`scripts/setup-vllm.sh` stops if a force-push removed it). #69165 hit a sampler-warmup
    OOM at `max_num_seqs` 256: keep the wizard's cap.
51. **Against club-3090's image** [C3]: PR noonghunna/club-3090#2135 (open) adds one-card
    slugs on `ghcr.io/noonghunna/vllm-inkling-club3090` (private so far): `single-max` 180k
    without drafter, `single-fast` 100k with DSpark, both NBT 512, 2 requests, release
    checkpoint. Once it pulls: `bench/context_bench.py` against both on this GPU.

## Tried and dropped

- Loop-core attention (layers 6-17) in bf16, as [hxyulin's recipe](https://huggingface.co/hxyulin/Inkling-2-Mini-Outer-Attention-FP8)
  does: all load-time conversions and the FP8 KV together cost 0.0001 of KL median (0.0030 ->
  0.0031), for ~17k tokens of context.
- Verify-side MoE kernel work [NS]: Marlin already reads ~900 GB/s; a second verified token
  costs less than double because tokens share experts. A warp-GEMV MoE for 1-4 tokens [AX]
  reaches ~390 GB/s per expert: below Marlin here.
- W8 attention/head kernels [NS, item 35]: up to 32 rows they read 75-90% of bandwidth
  (qkvr 765-800 GB/s, the head 870-930); past 32 rows they are bound by the dequant and
  MMAs (qkvr at 96 rows ~62 TFLOPS). Ordering the grid so row tiles of a column run
  together (weights from L2) changed nothing. Left: o (few column tiles), <1% of a step.
- SiTU fused into the experts' w13 epilogue [AX, item 34]: `_swiglu_kernel` is 0.12 ms of a
  22 ms step at 6 requests (0.5%), and each try rebuilds Marlin's C++.
- Screening the Markov head on the base logits' top-k [BS]: acceptance collapsed (counting
  6.6 -> 2.9 tokens/step); the right token comes from the Markov bias.
- A drafter-only INT4 copy of the LM head [HQ / BS 2-bit screen]: -1.5% step time for 0.23
  GiB; revisit with 13.
- vLLM's adaptive verification as shipped: the sconv backend refuses varlen decode graphs
  (7 is the way).
- HyperQwen's `kv-cache-seat-budget` (its `KVCacheGroup` API differs here),
  `terminal-request-progress` (does not apply), `tokenize-v1-route` (not needed).
- The release's MXFP8 experts on layers 20-23 [BS]: KL median
  0.0027 vs 0.0030 (noise floor 0.0003), top-1 92.5% vs 92.35%, for +1.94 GiB (~110k tokens
  of context); layer 23 alone changes nothing.
- A draft vocabulary subset [HQ `qwen3_5-mtp-draft-vocab`]: base logits and Markov fold on
  the 32k or 48k most frequent head rows. Steps 5% shorter, acceptance lower (extraction
  bench 42.5% -> 34-35%, struct -5 to -12%); document text needs the full vocabulary.
- Context lookup at the default verify length: +2% while copying (the drafter copies tables
  well on its own); it pays with the adaptive length (on by default, `ops/lookup.py`).

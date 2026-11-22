# Follow-ups

What comes next, by phase. Any item that trades something for something else ships as a
wizard option, not a default. Alternative tensors go in the same HF repo as separate files,
fetched with `hf download --include` only when chosen.

| phase | items | then |
|---|---|---|
| 2, the checkpoint (alternatives, after the post) | 8, 11, 12, 14, 33 | new files on HF |
| 3, context (best effort) | 54 (with 53) | |
| after the post | 38 | |
| 4, speed and polish | 35, 13 (+37), 10, 34, 53, 51, 52, 50, in this order | Reddit post |

Released: `ghcr.io/pento95/hyperinkling:0.1.1` (compose default), the HF checkpoint (its README
is docs/MODEL_CARD.md: re-upload it when the card changes). A new image goes with a new tag
and the compose default.

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
13. **A faster W4A16 GEMV** [NS, AX]: `ops/linear_w4.py` reaches 400-550 GB/s at decode
    (outputs of 16k columns or more take the single-launch kernel, the rest split-K;
    `bench/w4_gemv.py`). The cost is the nibble unpack: try the bit-shift pair unpack of
    `csrc/nvfp4_moe_sm86.cu` [AX] (two weights to bf16x2 in three integer ops, `lop3`).
    Unblocks 11, 12 and the drafter (1.1-1.8 ms per step, 37).
14. **Run-to-run nondeterminism** [NS]: the same request scored twice differs from position 1
    (~0.003 in logprob); an expert flip carries it down the sequence (KL mean floor 0.060,
    median 0.0003). Bitwise repeatable (`tests/test_determinism.py`): Marlin MoE, the W8
    linear, `bgemv`, `short_conv`. Not yet tested: `_rel_attention_kernel` (prefill),
    the router (`_k_group_kernel`, `select_experts`), `attn_res`, cuBLAS.
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
54. **`MAX_LEN=auto` in the wizard**: `start.sh` defaults to it (vLLM fits the longest
    request after warmup; the drafter computes its RoPE angles, so nothing grows with the
    limit) and `ASYNC` to 1; `bench/memory_matrix.sh` boots once per row. Left: the wizard
    asks for an optional cap instead of computing `MAX_LEN`, and shows the expected context
    from the new table (with 53, on the matrix of 50). Its `NBT` question still has the old
    logic (2048 forced async off at -68k): `NBT=2048` is now the default.
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
21. **Rust frontend** for the native build: vendored OpenSSL needs Perl `FindBin` (the
    container builds it).
22. **Upstream PRs**: the processing.py dummy-size fix and the oversized-image downscale are
    real upstream bugs; so is the sconv padding-row fix (NOTES).
32. **BASE to vLLM main when #69165 merges** [AR]: `patches/BASE` is its head today
    (`scripts/setup-vllm.sh` stops if a force-push removed it). #69165 hit a sampler-warmup
    OOM at `max_num_seqs` 256: keep the wizard's cap.
34. **SiTU fused into the experts' w13 epilogue** [AX `labs/lab3b`]: today Marlin writes
    gate_up, a separate kernel applies the softcapped SiTU, w2 reads it back: one launch and
    a round trip per visit, 48 per token. AX fuses it in registers (and the top-8 w2
    reduction). Needs a Marlin epilogue change, bitwise against the unfused output.
35. **W8 small matrices are latency-bound** [NS nsys]: `_w8_kernel` takes 2.8/3.6/5.5 ms per
    step at 1/6/16 requests; qkvr (8600 x 2304) and o (2304 x 3072) stay at 300-500 GB/s
    whatever the tile. Fewer launches, not fewer bytes: o fused with the next qkvr input
    path, or a persistent kernel over a layer's projections. A single-launch kernel over a
    global layer's three q/k/v inputs helps only the 8 global visits (not worth it alone).
37. **Drafter W4 time** [NS nsys]: 1.1-1.8 ms per step; follows 13.
51. **Against club-3090's image** [C3]: once its single-GPU profile ships, run the same
    benches on both for the post (it serves the release checkpoint, bf16 KV).
52. **Release guards** [C3 `build.sh`]: `scripts/build-image.sh` refuses to retag a published
    tag and the push pushes the image it built and checked; a `--verify` that the compiled
    libraries carry sm86 code (stock libraries without it leave `moe_align_block_size`'s
    buffers unwritten, silently).
53. **Image budget in the wizard** [NS]: ask the largest image as tokens per image (default
    1024, shown also as megapixels: 1024 tokens = 1024 cells of 40 x 40 px = 1.64 MP, an A4
    page at ~128 ppi), and set `IMAGE_MAX_W`/`_H` from it. With `NBT=2048` a step then
    encodes two such pages (the encoder budget is the prefill chunk). Measure the context
    and prefill of `NBT=2048` with 1024-token images after 17, in the memory matrix, and
    move the owner's profile to it (a little under 262k is acceptable).
50. **Final measurements** (once, after phases 3-4, before the Reddit post), for README and
    the model card, with the defaults (`NBT=2048`): the speed table is one table from
    `bench/context_bench.sh` (8k and 32k context; prefill per request and aggregate, decode
    per request and aggregate without prefill; 1 and 6 requests, with and without DSpark);
    plus `bench/spec_bench.py 8080 512 user`, `bench/copy_test.py` at 50k,
    `bench/extract_bench.py`, `bench/memory_matrix.sh`, `bench/kl_eval.py`,
    `bench/quality_battery.py` against the original. The post credits HyperQwen, basalt, AR,
    AX, C3 and invites field reports (`.github/ISSUE_TEMPLATE/field-report.yml`).

## Tried and dropped

- Verify-side MoE kernel work [NS]: Marlin already reads ~900 GB/s; a second verified token
  costs less than double because tokens share experts. A warp-GEMV MoE for 1-4 tokens [AX]
  reaches ~390 GB/s per expert: below Marlin here.
- Screening the Markov head on the base logits' top-k [BS]: acceptance collapsed (counting
  6.6 -> 2.9 tokens/step); the right token comes from the Markov bias.
- A drafter-only INT4 copy of the LM head [HQ / BS 2-bit screen]: -1.5% step time for 0.23
  GiB; revisit with 13.
- vLLM's adaptive verification as shipped: the sconv backend refuses varlen decode graphs
  (7 is the way).
- HyperQwen's `kv-cache-seat-budget` (its `KVCacheGroup` API differs here),
  `terminal-request-progress` (does not apply), `tokenize-v1-route` (not needed).
- The release's MXFP8 experts on layers 20-23 [BS] (`prepare/mix_experts.py`): KL median
  0.0027 vs 0.0030 (noise floor 0.0003), top-1 92.5% vs 92.35%, for +1.94 GiB (~110k tokens
  of context); layer 23 alone changes nothing.
- A draft vocabulary subset [HQ `qwen3_5-mtp-draft-vocab`]: base logits and Markov fold on
  the 32k or 48k most frequent head rows. Steps 5% shorter, acceptance lower (extraction
  bench 42.5% -> 34-35%, struct -5 to -12%); document text needs the full vocabulary.
- Context lookup at the default verify length: +2% while copying (the drafter copies tables
  well on its own); it pays with the adaptive length (on by default, `ops/lookup.py`).

# Follow-ups

What comes next, by phase. Any item that trades something for something else ships as a
wizard option, not a default. Alternative tensors go in the same HF repo as separate files,
fetched with `hf download --include` only when chosen.

| phase | items |
|---|---|
| 2, the checkpoint (new files on HF) | 33 |
| 3, context | 58, 18, 60, 13, 55 |
| 4, speed and upstream | 58, 64, 22, 10, 7, 21, 32 |
| 5, tool calls | 61 |

Released: `ghcr.io/pento95/hyperinkling:0.1.6` (compose default) and two HF checkpoints, the
original (README = docs/MODEL_CARD.md) and the uncensored one (README =
docs/MODEL_CARD_UNCENSORED.md): re-upload a card when it changes. A new image goes with a new
tag (`scripts/build-image.sh` refuses a published one) and the compose default.

Where each idea comes from: **[HQ]** HyperQwen (github.com/syv-ai/HyperQwen, `patches/`, docs),
**[BS]** basalt (github.com/jesdga95/basalt, MIT), **[AR]** Ar4ikov's vLLM PRs
[#69165](https://github.com/vllm-project/vllm/pull/69165) (Inkling 2 on SM80/86/89) and
[#69141](https://github.com/vllm-project/vllm/pull/69141) (compressed-tensors experts),
**[AX]** github.com/agent2x0r/inkling2-sm86-kernels (2x3090 bring-up, `labs/`, `csrc/`),
**[C3]** club-3090 (github.com/noonghunna/club-3090, its `vllm-inkling` image recipe),
**[LU]** github.com/lulakien/inkling2-mini-3090-recipe,
**[CD]** crazydiamond-on-ampere (github.com/Mr-Moonsilver/crazydiamond-on-ampere, KV in
host RAM), **[LC]** [llm-compressor](https://github.com/vllm-project/llm-compressor), **[IK]** the model's
own structure, **[NS]** ours (measurements in this repo, nsys profiles).

## Phase 2: the checkpoint

33. **llm-compressor / compressed-tensors** [LC, AR #69141]: llm-compressor calibrates GPTQ,
    AWQ and SmoothQuant schemes on the HF model and writes compressed-tensors checkpoints;
    #69141 runs such experts (W4A16, W4A8, W8A8, FP8) on Humming. Three uses here:
    (a) a second calibration of the experts, attention and head to compare with
    `prepare/gptq_experts.py` by KL median; (b) loading compressed-tensors W4A16 experts by
    repacking to the Marlin layout, so community quantizations run unchanged;
    (c) W8A8 INT8 for prefill: sm86's INT8 tensor cores run at twice the bf16 rate and
    prefill is compute-bound (vLLM has W8A8 int8 GEMMs for sm80; the MoE side needs a kernel).
    #69141 also fixes the DSpark draft quantization config and casts fp32 checkpoint tensors
    to bf16: check both against our drafter loader at the rebase (32).

## Phase 3: context (best effort: no DSpark, vision off the GPU)

18. **K8V4: K in E4M3, V in 4 bits after a randomized Hadamard rotation, one scale and zero point
    per vector (g128)** [HQ KVarN, vLLM `int4_per_token_head`]. Decided: the new default, with 8-bit V as a
    wizard option. Global layers only: ~+27% of the context they hold (before padding and what
    does not scale). Simulated at 131k tokens (3 AA-LCR documents, against bf16 KV on the same
    weights, KL median / mean / top-1):

    | V | 8-bit E4M3 | RHT g128 | RHT g64 | RHT g32 | RHT g128 symmetric | g16 | g32 | g64 | g128 |
    |---|---|---|---|---|---|---|---|---|---|
    | median | 0.0005 | **0.0006** | 0.0006 | 0.0006 | 0.0007 | 0.0006 | 0.0006 | 0.0007 | 0.0007 |
    | mean | 0.0029 | **0.0036** | 0.0035 | 0.0034 | 0.0037 | 0.0033 | 0.0035 | 0.0037 | 0.0039 |
    | top-1 | 98.15% | **97.96%** | 97.96% | 98.00% | 97.85% | 98.03% | 97.97% | 97.90% | 97.82% |

    5-bit V (RHT g128) is E4M3's equal (0.0005 / 0.0031 / 98.12%) and K in 6 bits with the rotation costs nothing more
    (0.0006 / 0.0035 / 97.96%), but neither packs like 4 and 8 bits; K at 4-5 bits and V at 3 bits are out (K 0.0012 /
    0.0008, V 0.0010).
    No worse with depth (96-131k: 0.0004 for RHT g128, 0.0003 for E4M3); needles at 64k and 120k are
    40/40 for every variant, too easy to tell them apart. K is the sensitive half (K and V 4-bit
    g128: 0.0047 on the short transcripts, against 0.0031); V 4-bit on the windowed layers too costs
    0.0046. Missing: the kernels (write with the rotation; decode and prefill reading packed V,
    the rotation undone once on the attention output, which is linear in V), a planner change (vLLM has
    equal pages per group; vllm#69031 gives full and windowed layers different formats),
    then decode speed, acceptance and the KL of the real format. Starting points:
    `vllm/v1/attention/ops/int4_per_token_head.py` (nibbles, RHT, zero point in the scale's
    mantissa); `turboquant_k8v4` is a backend of its own. Upstream as an issue first, once it
    works, and only after #69165.

13. **Drafter W4 at 6-16 requests** [NS]: past ~21 rows the W4 kernels
    are tensor-core bound (int4 halves the bytes per MAC, so the bound comes at half the
    rows of INT8): 1.3 ms per step at 6 requests. fp16 accumulation doubles the MMA rate on
    GeForce; for the drafter it moves only acceptance (est. +2-3% at 6-16 requests).
    Building bf16 from bits (`code | 0x4300` = 128 + code, the zero point corrected per
    group from the row sums) gave +15-20% at 1-7 rows on split-K, worth ~0.03 ms a step.
55. **Sources in bf16** (other fine-tunes): `prepare/quantize_experts.py` and
    `gptq_experts.py` read only the release's NVFP4/MXFP8 experts (as the uncensored source
    is), and a bf16 Inkling 2 Mini does not fit 24 GiB to generate the corpus and capture.

## Phase 4: speed and polish

58. **DFlash2, what is left** [NS] (NOTES "DFlash2"): the selector's codebooks in INT8 (2 x 103 MB
    bf16, ~10k tokens of KV); the draft KV in bf16 (the drafter's values are not unit-scale, DSpark keeps
    bf16 on SM12x for that reason), which needs `SconvStateManager` to take a hit alignment larger than
    the block size (its assert fires: the alignment is set before the draft's pages are known);
    long-document acceptance (DSpark is 10-19% ahead at 8-32k); our block-attention fix for SM8x as a
    comment on #69165 once both PRs are in; published checkpoints keep `mtp.safetensors` (636 MB) for
    `SPEC=dspark`.

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
22. **Upstream PRs**, one at a time; vLLM's `AGENTS.md` wants a human who reviewed every line and ran the
    tests, the `pr-checklist` skill, its PR template, disclosure of AI use.
    - Truncated tool arguments (misplaced brace): branch `inkling-misplaced-brace` on the owner's fork,
      ready but not opened. On a real server: 16 of 16 broken calls were cut, 0 of 20 with the fix
      (in 8 of them the wrapper's brace stays: the model wrote one closer too few, so it
      cannot be told apart; the description must say it). Neighbours: #51141 (replaces the streaming
      conversion), #69297 (in our patch), #69347.
    - Tool grammar: comment on #54120 (open, stalled; does not fit Inkling 2: the model writes its own
      `<|message_model|>`, reasoning comes first, calls carry the name before the marker).
    - #69297: our timings as a comment.
    - Images over the encoder budget that wait forever: #68203 / #68204 (open) cover it; a comment there, no PR of ours.
    - Candidates to agree on first (they come from HyperQwen): continuous usage while a parser
      buffers, `--api-key` on every path, the stall sentinel.
32. **BASE to vLLM main when #69165 merges** [AR]: `patches/BASE` is its head today
    (`scripts/setup-vllm.sh` stops if a force-push removed it). #69165 hit a sampler-warmup
    OOM at `max_num_seqs` 256: keep the wizard's cap.

64. **No speculation above a batch size** [NS]: DSpark only breaks even at 32 requests and loses 23% at
    64 (prose, `scripts/quick_bench.py`); DFlash2 is still +35% over no speculation at 64. The adaptive
    verify length could take 0 as a candidate (vLLM's schedule accepts k = 0); the wizard warns instead.

## Phase 5: tool calls

61. **Undeclared tools**: without `strict` the grammar leaves the name free, and the model sometimes
    calls a tool that does not exist (2-3 of 24 on a one-tool request). Constraining only the
    payload's name would be worse: the model has already named the tool in the block's header, and
    would then call another one with the wrong arguments. It has to hold from the header, that is a grammar of
    the whole answer (reasoning, text or a declared name).
60. **CPU-tier miss at prompt length 63 modulo 64** [NS]: such a prompt is not found in host RAM the first
    time (13 other lengths are; a second recompute makes it hit). Also before the rebase. Look at
    `SconvStateManager`'s tail against the chunk that the first generated token completes.

## To consider

59. **From [lulakien/inkling2-mini-3090-recipe](https://github.com/lulakien/inkling2-mini-3090-recipe)** [LU]
    (one or two 3090s, the release's NVFP4 experts; its single-GPU decode and context are below ours). Worth trying:
    - far path in the prefill attention (`FAR_ATTN`): the relative bias reaches 1,024 keys back, so the key blocks
      before that skip bias, masks and `-inf` handling: x1.13-1.15 on the kernel with FP8 KV, x1.25 with bf16 (500k cold -13%).
      It changes outputs at fp32 rounding level, and that kernel serves decode there;
    - `moe_align` as a parallel kernel (193 -> 42 us at 8,192 routes) and a two-pass `attn_res`, both
      bit-identical: +3.4% prefill on one GPU together;
      Already here or worse here: FP8 KV, sconv tail, INT8 at load, the draft length by batch size.
56. **A server-side brake on reasoning loops** [NS]: vLLM has `repetition_detection` (per
    request) and `thinking_token_budget` (needs `--reasoning-config`). As wizard options, after
    checking them with DSpark and tool calls, if loops remain at temperature 1.0.
57. **Public cases for `bench/loop_eval.py`** [NS]: loops were measured on one private
    session, too few cases to publish rates; synthetic agent transcripts (8k-100k tokens) would
    make them repeatable.

## Tried and dropped

- Fewer bits, KL median (as served 0.0031, round to nearest with clip search): attention INT4
  0.0081 (g32 0.0064, only layers 0-5 and 18-23 0.0048, g32 there 0.0042), INT6 0.0036; LM head
  INT4 0.0042 (g64 0.0039) for ~13k tokens of context and a slower kernel; 3-bit for the 32 / 64
  / 128 least-routed experts 0.0040 / 0.0060 / 0.0138, 2-bit for 16 0.0040.
- Least-routed experts in host RAM [BS]: routing is flat (the 8 rarest of 128 take 1.7% of the
  routes, 32 take 11%, 64 29%): 8 experts free ~0.75 GiB and cost ~1 ms per decoded token.
- DSpark tuned for temperature 1.0: adaptive candidates 1,2,4,7 / 1,2,3,4,5,7 / 1,2,3, greedy
  drafts and no lookup are within noise of the defaults or worse; `SPEC=off` is 5% (chat) to
  2.1x (structured) slower.
- A bf16 drafter (a bound for drafter GPTQ): 1-9% slower at one request, +4% at 6 (noise).
- Padding of the drafter's sliding-window layers: ~15 MiB per live request (~850 tokens).
- Against club-3090's image [C3] (`single-fast`, release checkpoint, 100k limit, NBT 512, 2 requests), decode of one
  request at temperature 1.0, `bench/spec_bench.py`: chat 86 tok/s (here 144), code 144 (227),
  struct 195 (288), prose 90 (157), doc 178 (246); KV pool 113k tokens (268k).
- `NBT` above 2048: 3072 costs ~27k tokens of context and gives +1-2% prefill at 8k and 32k;
  4096 ~65k and +5-6%. With the KV overflow in host RAM (2 GiB) the context goes 370k -> 342k and a 299k request decodes
  at 35 tok/s instead of 52 (more of it in RAM) with no faster prefill. With 18 the cost per
  chunk is the same and its KL does not depend on the chunk (0.0034 at 2048 and 3072).
- KV overflow in host RAM [CD]: works, left out by choice (branch `hyperinkling/kv-host` of the
  local `vllm/`). Blocks past the GPU's in host memory mapped in the same address range, prefill
  through a GPU stage; lossless (KL scores and greedy text identical to the GPU-only ones).
  Longest request 318k with 1 GiB, 370k with 2, 421k with 3. Past the GPU's share decode pays
  ~0.7 ms per step per 1k tokens in RAM: at 1 GiB 130 tok/s at 266k and 63 at 299k, at 3 GiB 49
  at 299k and 23 at 401k (154 at 266k all on the GPU, 76 at 308k with `SPEC=off`).
- The whole KV in host RAM: dense global attention reads all of it at each step (25 GB/s over
  PCIe against 900); CD's engine can because its attention is sparse.
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

<!-- Line numbers refer to the vllm/ tree at commit 0120e0317 (patch of 2026-11-21). -->

# Inkling 2 Mini: what breaks with per-request varying query length in one decode batch

Paths are relative to `vllm/vllm/`. Everything below was read from the code. Items I could not verify are in "Open doubts".

Shorthand: `CPU_total` is `query_start_loc_cpu[-1]`. `dev_total` is `query_start_loc[num_reqs]` on the device. With trimming, `dev_total <= CPU_total <= padded`.

## 0. Backend questions (`v1/attention/backend.py`, `flash_attn.py`)

- `supports_device_cpu_query_lens_mismatch()` (backend.py:234-247):
  - It asks whether a backend can run a batch whose device `query_start_loc` disagrees with the CPU one.
  - Backends that plan from the CPU query lengths must opt out.
  - The default is `return not cls.is_ssm()` (backend.py:247), and `is_ssm()` is `False` (backend.py:398-399). So the default is **True**.
  - The attention selector uses it at backend.py:369-375.
  - `get_query_lens_mismatch_unsupported_backend` (attn_utils.py:342-360) checks it for backends that are hard-wired, such as the sconv one.
- `get_varlen_cudagraph_max_query_len()` (backend.py:646-): the largest per-request query length `L` that a FULL graph of this builder can replay.
  - Every real request must have between 1 and `L` tokens, and padding requests must have 0.
  - Lengths come from the device `query_start_loc`. The host only has the token count and an upper bound.
  - It returns `None` (backend.py:661-685 area) for builders that report ALWAYS, and for builders that cannot replay varlen batches.
  - `get_varlen_cudagraph_unsupported_backend` (attn_utils.py:363-393):
    - ALWAYS builders are skipped.
    - For NEVER, the bound is `None`.
    - Otherwise it takes the builder's bound.
    - It fails if the bound is `None` or `< max_query_len`.
    - The message is "... allows none" (adaptive_verification.py:644-649).
- `FlashAttentionBackend`:
  - `_cudagraph_support = ALWAYS` if FA3 or FA4 (or XPU), else `UNIFORM_BATCH` (flash_attn.py:715-719).
  - It does **not** override `supports_device_cpu_query_lens_mismatch` or `get_varlen_cudagraph_max_query_len` (grep finds neither name in flash_attn.py). So it answers mismatch=True and bound=None.
  - Under FA3/4 it is ALWAYS, so the bound is never consulted. Under FA2 (UNIFORM_BATCH) it would fail with "allows none".
  - `Inkling2AttentionBackend` inherits this (attention.py:108-111), so it passes whenever FA3/4 is selected.
- **Where "InklingSconvBackend allows none" comes from:**
  - `InklingSconvMetadataBuilder._cudagraph_support = UNIFORM_BATCH` (inkling/nvidia/sconv_swa_attn.py:53).
  - It does not override `get_varlen_cudagraph_max_query_len`, so the default `None` applies.
  - `InklingSconvBackend` (sconv_swa_attn.py:104-116) extends `AttentionBackend` directly, so mismatch=True and that check passes.
  - Only the varlen-cudagraph check fails. Nothing in the kernels needs UNIFORM_BATCH (see §1).

## 1. `InklingSconvMetadataBuilder.build` (inkling/nvidia/sconv_swa_attn.py:73-101)

| Line | Input | Source |
|---|---|---|
| 79 | `num_reqs` | host int |
| 80 | `num_actual_tokens = int(query_start_loc_cpu[-1])` | CPU tensor read, `CPU_total` |
| 81 | `num_padded_tokens = slot_mapping.shape[0]` | host int (shape) |
| 88 | `query_start_loc` | device tensor |
| 97-100 | `block_table`, `slot_mapping[:padded]`, persistent `seq_idx`/`query_start` slices | device |

- **No uniformity assumption.** `sconv_seq_metadata` (inkling2 uses `inkling/nvidia/ops/sconv.py:224-290`) does a per-token binary search over the device `qsl` (largest `j` with `qsl[j] <= tok`). `n_iters = (num_reqs-1).bit_length()` is host. There is no reshape and no query-length constexpr.
- **Rows past `CPU_total`.** `actual = offs < num_actual_tokens` (l.244), with `seq_idx = 0` and `query_start = 0` for the rest. Rows are never skipped.
- **Rows in `[dev_total, CPU_total)`.** These get `seq_idx` = the last request and `query_start` = that request's start. They are not marked as padding by this kernel.
  - What protects them is `slot_mapping == -1`. `_compute_slot_mappings_kernel` (v1/worker/gpu/block_table.py:375-384) fills PAD from `qsl[num_reqs]` (the device total) up to the padded size. So the tail after `dev_total` is -1.
  - This holds provided `prepare_attn` (model_runner.py:1709) is called with the trimmed device `query_start_loc`. I did not verify that call order.
- **Change needed.** None for correctness. Add
  ```python
  @classmethod
  def get_varlen_cudagraph_max_query_len(cls, vllm_config, kv_cache_spec):
      return max_decode_query_len(vllm_config)  # or 1 + num_speculative_tokens
  ```
  and/or set `_cudagraph_support = ALWAYS`. Optionally replace line 80 by a value that does not depend on the CPU total (the kernel only needs an upper bound; padded rows are already protected by slot_mapping).

## 2. `Inkling2AttentionMetadataBuilder` (inkling2/nvidia/attention.py:75-105)

Reads:
- l.90: `seq_lens_cpu_upper_bound` is asserted not None.
- l.93: `query_lens_cpu = query_start_loc_cpu.diff().numpy()`. This is a host array, it is exact only for prefill, and the field comment (l.70) says so.
- l.94: `seq_lens_cpu = seq_lens_cpu_upper_bound.numpy()`.
- l.92: `super().build()` (FlashAttentionMetadataBuilder.build, flash_attn.py:912-1070). It reads `common_attn_metadata.max_query_len`, `num_actual_tokens` and `max_seq_len` as host ints, and `query_start_loc` and `seq_lens` as device tensors.
  - `_get_scheduler_metadata` gets `max_query_len` and `cu_query_lens` (flash_attn.py:1038-1047).
  - `max_num_splits` is chosen from `num_actual_tokens <= max_cudagraph_size` (flash_attn.py:953-962).
- l.102-104: `split_decodes_and_prefills(cm, decode_threshold)` (backends/utils.py:~930-990). It reads `max_query_len`, `num_actual_tokens` and the **CPU** `query_start_loc_cpu`. It is only ever uniform-checked when `require_uniform` is set, which it is not here. If `max_query_len <= decode_threshold`, the whole batch is returned as decode (utils.py:963-968).

Uniformity: none assumed, because `require_uniform=False`. The decode/prefill split relies on `max_query_len` being a valid upper bound and on the CPU `qsl[0]` being the first request's length (utils.py:971).

What it needs:
- `max_query_len` in the CPU metadata must be an upper bound of the true device lengths, e.g. `1+k_max` and not an even share of the budget. With even distribution, a request that gets more tokens on the device than the CPU thinks would violate the kernel assertion (see §3).
- `query_lens_cpu` and `seq_lens_cpu` feed only `plan_attention` (l.533-539). That is the prefill part, only on batches with prefill. For a pure decode batch this does not matter.
- Nothing else to change in the builder. The FA scheduler metadata (`aot_schedule`) is not used by the Inkling2 decode or prefill kernels as far as I can see, because `_attention` calls its own kernels. See doubts.

## 3. Decode path in `Inkling2Attention._attention` (attention.py:437-560)

| Line | What | Host or device |
|---|---|---|
| 460 | `num_tokens = q.shape[0]` | host (padded) |
| 462-467 | `attn_metadata[layer_name]` | host dict of the metadata above |
| 470-471 | `_plan_decode(md, num_tokens)` | host ints only: `md.seq_lens.shape[0]`, `md.max_query_len`, `md.num_decode_reqs`, `md.num_decode_tokens` |
| 476-478 | `rel_logits` allocated `(num_tokens+128, H, E+256)` | host shape |
| 483-512 | `attention_prologue(... md.slot_mapping, md.query_start_loc, md.seq_lens ...)` | device |
| 513-525 | `_decode_attention` | see below |

Under FULL cudagraph this whole function is **captured** (`eager_break_during_capture` returns `fn` directly if `cudagraph_runtime_mode == FULL`; breakable_cudagraph.py:114-118). In piecewise mode it runs eagerly, so there it re-plans from the current host metadata. So under FULL, `num_tokens`, `num_decodes`, `decode_splits`, `max_query_len` and `md.num_actual_tokens` are Python values baked at capture. They must therefore be the same for every replay of the graph (captured at the padded/uniform shape, with max `L`).

### 3.1 `_plan_decode` (attention.py:562-588) and `plan_decode_attention` (ops/attention.py:375-403)

- Inputs: `num_reqs = md.seq_lens.shape[0]` (padded batch size), `max_query_len = md.max_query_len`, `num_heads`, `num_kv_heads`, `rel_extent`. All host. `work = num_reqs*num_kv_heads`, `num_sms` from the device.
- Limit: `max_query_len > 64 // (H/HKV)` returns 0, which means "use FA4". This is an exact-value use of the bound. It assumes `max_query_len` is `>=` every request's length.
- The split count depends only on the batch size and `max_query_len` (docstring l.392-393 says this deliberately), not on KV lengths. It is therefore replay-safe.
- Varlen: with a captured `max_query_len = L_max` the plan is correct for every shorter request. A smaller `max_query_len` gives a plan that is valid only if it is still `>=` the true max, otherwise `num_splits` could differ from capture.
- Mixed batches (l.579-588): `max_query_len = min(max_decode_query_len, num_decode_tokens - num_decodes + 1)` (CPU). This is still an upper bound for any single request, so it is safe with trimming.
- Change: none, if the CPU `max_query_len` is documented as an upper bound.

### 3.2 `_decode_attention` (attention.py:590-624)

- Host: `num_reqs` (arg), `mixed = num_reqs < md.seq_lens.shape[0]`, `num_tokens = md.num_decode_tokens if mixed else md.num_actual_tokens`, `max_query_len`.
- Device: `md.block_table`, `md.seq_lens`, `md.query_start_loc[:num_reqs+1]`.
- It slices `block_table[:num_reqs]`, `seq_lens[:num_reqs]` and `cu_seqlens_q[:num_reqs+1]` (device views at fixed addresses, fine for graphs).
- `num_actual_tokens=num_tokens` is passed as a host int. In FULL graphs it is baked at capture.
- Change: see §3.4, which handles `num_actual_tokens`.

### 3.3 `rel_decode_attention` / `block_decode_attention` -> `_decode_attention` (ops/attention.py:1106-1299)

- `_decode_attention` (l.1193-1299):
  - l.1221: `assert max_query_len <= _VERIFY_BLOCK_M // group`. This is a host check against the bound.
  - Output buffers: `o_part (S, T, H, D)` fp32 and `lse_part (S, T, H)` fp32, both allocated with `new_empty` over **`num_tokens = q.shape[0]`** (padded). Rows that no program writes stay uninitialized.
  - l.1272-1274: if one split, then `out[num_actual_tokens:].zero_()`. The rows in `[dev_total, num_actual_tokens)` are not touched by the kernel (see below) and not zeroed.
  - l.1281: the combine kernel runs on a `(num_tokens, H, D/block_d)` grid.
- SM100 kernel `_rel_verify_kernel` (l.856-1103). **This is already varlen-ready:**
  - Grid = `(num_splits, HKV, cache_seqlens.shape[0])` (l.1323). It does not depend on `max_query_len`.
  - l.911-914: `q_start = CuSeqlensQ[req]`, `q_len = CuSeqlensQ[req+1] - q_start`, `seqlen_k = SeqLens[req]`, `past = seqlen_k - q_len` all **from the device**.
  - `BLOCK_M = 64` is a constexpr independent of `q_len`: rows `= tokens x G`, and `tok = rows // G`.
  - l.941: `row_valid = (rows < BLOCK_M//G*G) & (tok < q_len)`. Rows past the request's real length are masked on Q load (l.1002), on the Rel/bias loads (l.762, 772), on the Sink/Gate loads (l.836, 841) and on every store (l.832, 845, 848, 851).
  - So with `q_len` smaller than `64//G`, the extra rows are never read (apart from masked loads) and never written.
  - `pos0 = past`, `n_bias`, `rel_base = cdiv(seqlen_k,128)*128 - REL_PAD` (l.929) and `num_iters` are all derived from device `seqlen_k` and `q_len`.
  - A padding request (`q_len = 0`, `seqlen_k = 0`): `num_iters = 0`, and every row is invalid. No stores happen in the single-split path. In the multi-split path the combine kernel zeros requests without keys (comment at l.486).
- SM90/SM12x path `_rel_attention_kernel` (ops/attention_sm90.py:74-252) via `rel_attention` (l.260-364):
  - Host: `block_m = _block_m(max_query_len, group)` (l.311) and grid dim 0 `= cdiv(max_query_len, tokens_per_tile)` (l.314-318). Both use `max_query_len` as a bound. Programs with `first_tok >= q_len` return at once (l.123-124).
  - `q_len` and `seqlen_k` come from the device (l.120-125).
  - `row_valid = (rows < TOKENS*G) & (tok < q_len)` (l.130) masks Q (l.153), the bias (l.194-195) and every store (l.226, 231, 247, 252).
  - Uniform assumption: none, as long as `max_query_len` is the largest per-request length. `tile_end`/`rel_base` (l.157-158) use the per-request device `q_len`.
- `_rel_decode_combine_kernel` (ops/attention.py:442-492):
  - l.471: `if row >= num_actual_tokens: store zeros` where `num_actual_tokens` is a host int baked into the launch. Row `r` of the combine reads `part = (split*num_tokens + row)*H + head`, so it reads fp32 partials at rows in `[dev_total, num_actual_tokens)` that **no program wrote** (uninitialized).
  - Those rows are not masked by device lengths. The result there is garbage (possibly NaN).
  - They are slot-mapping padding rows, so no KV or conv write is done for them, but the row still flows through `wo_ud`, the all-reduce, the MoE router, etc.

### 3.4 Concrete changes (decode path)

1. In the combine kernel and in the single-split path, replace the baked `num_actual_tokens` by the device total `CuSeqlensQ[num_reqs]` (a device load, one extra pointer arg). Rows `>= dev_total` get zeros. This removes the uninitialized-read issue, and it is the only thing in the decode kernels I found that is tied to a host-known token count.
2. Keep `max_query_len` as a documented *upper bound* in `plan_decode_attention`, `_decode_attention` and the SM90 grid. Capture and replay must use the same value.

## 4. `attention_prologue` and its kernels (inkling2/nvidia/ops/sconv.py:495-1023, ops/norm.py:85-185)

Host inputs: `num_tokens = kv_slot_mapping.shape[0]` (l.888, the padded count), the shapes, `block_t`, `qr_tokens`. No `.item()`, no `.tolist()`, no loop over requests.
Device inputs: `positions`, `seq_idx`, `slot_mapping`, `query_start`, `block_table`, `cu_seqlens_q`, `seq_lens`, `kv_slot_mapping`.

- `_kv_conv_tile` / `_conv` (l.189-301, 710-820):
  - Grid is `(cdiv(T, BT), num_heads, 2+...)` (l.920-924). It is over padded tokens and independent of the query length.
  - Per token: `slot` (l.227), `pos` (l.229), `req` (l.230) and `qstart` (l.231) come from the device. `valid = slot >= 0`.
  - The cache write is masked by `valid` (l.266) and the KV write by `kv_slot >= 0` (l.807-820). So padding rows in the batch are not written.
  - Taps: same-forward taps are read from `x` when `row >= qstart` (intra), and cache taps when `row < qstart` (l.137-165). Both depend only on the per-token `qstart` and the device `req`. Nothing assumes a fixed per-request length.
- `qr_clip_rel_tokens` (ops/norm.py:85-185):
  - `seq = SeqIdx[t]`, `q_start = qsl[seq]`, `q_len = qsl[seq+1]-q_start`, `pos = SeqLens[seq] - q_len + q_idx` (l.150-154). All device.
  - `tile_end = min((q_idx//128+1)*128, q_len)` (l.155).
  - l.156-160 clamps `diag` to `[E+1, E_PAD-1]` for "CUDA-graph padding rows (seq_idx 0, q_idx past the request)". Rows in `[dev_total, CPU_total)` (seq_idx = last request, `q_idx >= q_len`) fall in the same class and are clamped too. They write garbage but in-bounds `rel_out`.
  - The shear convention matches the decode kernels: SM100 uses `rel_base = cdiv(seqlen_k,128)*128 - REL_PAD` and SM90 uses `(past+tile_end+127)//128*128 - E_PAD`, both with `tile_end = q_len` for `q_len <= 128`.
- Change needed: none for correctness. Document that the padding rows of `rel_out` and `q_out` hold clamped garbage.

## 5. `Inkling2ShortConv` and `ops/sconv.py` (inkling2/nvidia/sconv.py, ops/sconv.py)

- `conv_metadata` (sconv.py:33-37) reads the forward-context metadata dict. No CPU-length input.
- `Inkling2ShortConv.forward` / `conv_mix` / `all_reduce_mix`: pass `m.block_table`, `m.seq_idx`, `m.slot_mapping`, `m.query_start` (device). No `.item()`, `.tolist()` or request loop.
- `short_conv` (ops/sconv.py:391-466) and `_sconv_kernel` (l.304-388):
  - Host: `num_tokens` (padded) decides `block_t, block_c` and `prefill = num_tokens > 256`. The grid is per token.
  - No uniform assumption, no query-length constexpr.
  - Writes: `tl.store(dst, cur, mask=tc_mask & valid)` (l.266), so rows with `slot_mapping == -1` are not written. Outputs for such rows are computed and written to `out` (l.388, masked only by `t_mask`). They are harmless values.
  - Read side for padding rows: `valid` False gives `in_win` False, so `intra` and `cached` are False and the taps are zero.
- Pre-existing risk: `SKIP_IDLE_CACHE` / `tile_reads_cache` (see end of §1).
- Change needed: none for varlen. Fix or confirm the 256-token tile skip.

## 6. Other model parts (grep only, no varlen-relevant input found)

`model.py`, `loop.py`, `moe.py`, `ops/loop.py`, `ops/conv_mix.py`, `ops/allreduce.py`, `ops/seq_parallel.py`, `ops/attn_res.py`: no `query_len`, `query_start_loc_cpu`, `.item()` or `.tolist()` on the decode path. The `.tolist()` hits in model.py (l.619-708) are the vision encoder. Per-token ops use the padded token count.
`dspark.py` (draft model) is out of scope: it reshapes to `[num_reqs, block]` (l.229-251, 394-401, 457-460), a uniform draft block, and it is not part of target verification.

## Changes needed, smallest first

1. `InklingSconvMetadataBuilder`: override `get_varlen_cudagraph_max_query_len` to return the maximum decode query length (or set `_cudagraph_support = ALWAYS`). This removes the "allows none" error. Nothing else in the sconv builder or kernels is uniform-dependent.
2. Combine kernel / single-split fuse path (ops/attention.py:471, 1273): zero rows `>= dev_total` (load `CuSeqlensQ[num_reqs]` on the device) instead of the baked `num_actual_tokens`, so uninitialized partials are never consumed.
3. Document and assert that the CPU `max_query_len` passed to the Inkling2 builder is an upper bound (`1 + k_max`), the same value as at graph capture. It drives `plan_decode_attention`, `_VERIFY_BLOCK_M // group` (ops/attention.py:395, 1221) and the SM90 grid.
4. Confirm or fix the `SKIP_IDLE_CACHE` tile skip for padded batches over 256 tokens (ops/sconv.py:341-346, 423, 461 and builder kernel l.246-251). Padding `query_start = 0` breaks the "non-decreasing starts" assumption.
5. Add a varlen test (e.g. A=3 and B=8 tokens, cudagraph captured at uniform `L_max`) comparing against per-request eager runs, for both SM90 and SM100 paths.

## Open doubts

- Does the runner's device `seq_lens` get updated to `computed + device q_len` for trimmed requests? Every kernel computes `past = seq_lens - q_len` from the device. If `seq_lens` still counts the untrimmed drafts, positions and the bias shear would be wrong. I did not check how adaptive verification writes `seq_lens` and `positions`.
- Are `positions` and `input_ids` compacted to match the device `query_start_loc` (packed request layout)? `compute_slot_mappings` and the kernels assume it.
- Is `prepare_attn` (model_runner.py:1709) called with the trimmed `query_start_loc`, i.e. is the `slot_mapping` tail from `dev_total` really -1 before the builders run?
- Does the `fast_build` / scheduler-metadata work in `FlashAttentionMetadataBuilder.build` matter for Inkling2, given `_attention` never calls FA for decode? It may be wasted CPU work. It can also embed `max_query_len` and `num_actual_tokens` in graph-captured metadata (flash_attn.py:1038-1048).
- Is `query_lens[0]` from the CPU `query_start_loc_cpu` (utils.py:971) meaningful if the CPU splits the draft budget unevenly (e.g. 3,3,2)? It is only compared with `decode_threshold`, so probably fine. Worth confirming.
- Is `num_actual_tokens` a Python constant during FULL capture and replay (padded size), and does anything outside the decode kernels (e.g. `output[num_decode_tokens:]` in the mixed path) rely on it? I only found the two uses in §3.3.
- Is the `tile_reads_cache` skip really wrong for padded decode graphs above 256 tokens, or does the runner guarantee `query_start` non-decreasing for padding? This looks like an existing bug, independent of varlen.

## Collateral suggestions (outside the question)

- `Inkling2AttentionMetadata.query_lens_cpu` / `seq_lens_cpu` are built (attention.py:93-94) on every `build()`, also for pure-decode batches that never use them. Build them lazily, or only when `num_prefill_reqs > 0`.
- `InklingSconvMetadataBuilder.build` reads a CPU tensor element (`int(query_start_loc_cpu[-1])`) on every step. `CommonAttentionMetadata.num_actual_tokens` is a cheaper host int, if it is the same value.
- `sconv_seq_metadata` docstring says it "replaces the arange + searchsorted ... chain". That is development history, not behaviour, and can be dropped.

# HyperInkling

**[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines
Lab, Apache-2.0) on one RTX 3090 / 3090 Ti, with vLLM.** Text and images (document pages),
tool calls, strict JSON output, 262k tokens of context per request, several requests at once,
and the model's own DSpark drafter for speculative decoding.

Upstream vLLM runs Inkling 2 Mini only on Hopper and Blackwell (PR
[#69028](https://github.com/vllm-project/vllm/pull/69028)). This repository is the port to
Ampere (sm86) plus a requantized checkpoint that fits in 24 GB: a pinned vLLM with one patch,
a model-preparation pipeline, and a launcher with measured defaults.

> **Scope, honestly.** This is a personal project, written almost entirely with an AI coding
> assistant ("vibe coded"), published because it may be useful to other 3090 owners. It
> targets **one RTX 3090 or 3090 Ti and nothing else**: no other GPU generation, no
> multi-GPU, no promise of support. Issues with a reproduction are welcome; feature requests
> for other hardware will be closed.

## Quick start (Docker)

```bash
git clone https://github.com/Pento95/HyperInkling hyperinkling && cd hyperinkling
python3 scripts/wizard.py         # a few questions -> .env (context, vision, DSpark, ...)
docker compose up -d              # first start pulls the image (~26 GB) and the checkpoint (~16 GB)
docker compose logs -f hyperinkling
```

The wizard's answers are in `.env`, which compose reads at every start: next time, just
`docker compose up -d`. To keep several setups, save copies and pick one before starting:
`mkdir -p profiles && cp .env profiles/long-context.env`, later
`cp profiles/long-context.env .env && docker compose up -d`.

To build the image yourself instead (~1 hour, compiles vLLM for sm86):
`scripts/setup-vllm.sh && scripts/build-image.sh`, then `IMAGE=hyperinkling:latest` in `.env`.

The OpenAI-compatible API is at `http://127.0.0.1:8080/v1`, model `inkling-2-mini` (alias `local`, see `SERVED_ALIASES`)
(`GET /health` when it is up; `GET /` is a 404 by design). The port is published on
127.0.0.1 only: to serve other machines set `BIND=0.0.0.0` and a `VLLM_API_KEY` in `.env`
(the wizard does both). With a key, every path except `/health`, `/ping`, `/load` and
`/version` needs it, `/metrics` and `/tokenize` included.

Requirements: an RTX 3090 or 3090 Ti with nothing else on it (a display on the same GPU costs
about 1 GiB: tell the wizard), an NVIDIA driver recent enough for CUDA 13 (580 or newer),
Docker with the NVIDIA Container Toolkit, ~20 GB of disk for the checkpoint. 32 GB of RAM
works; 64 GB lets the prefix-cache CPU tier hold long shared prompts.

### Without Docker

```bash
scripts/setup-vllm.sh            # vllm/ = PR #69028 at patches/BASE + our patch
scripts/build-vllm.sh            # venv/, sm86 only (~30-60 min)
hf download Pentium95/Inkling-2-Mini-W4A16-g128-gptq --local-dir models/Inkling-2-Mini-W4A16-g128-gptq
python3 scripts/wizard.py         # the same questions, writes .env
./start.sh                       # foreground; reads .env, every knob is also an env var
```

Tested on Fedora 44 (gcc 15 next to the default gcc 16, CUDA 13.3). Other distributions
should work where CUDA 13 and a gcc it accepts are installed; that is best effort.

## Speed (RTX 3090 Ti, 2026-11-22)

Measured on an RTX 3090 Ti undervolted to 350 W, with
[gpu-unlock](https://github.com/zagitta/gpu-unlock) active; KV cache FP8 E4M3; DSpark on
(the default). "Aggregate" is the sum over all requests running at once.

Decode with short prompts (~30 tokens in, 512 out, greedy; `bench/profile_spec.sh`):

| requests at once | 1 | 6 | 16 |
|---|---|---|---|
| decode, per request | 124 tok/s | 72 tok/s | 49 tok/s |
| decode, aggregate | 124 tok/s | **434 tok/s** | **785 tok/s** |
| time per step | 12.6 ms | 24.0 ms | 33.8 ms |

Each step reads the experts the batch routes to: ~17 of 128 per layer visit for one request,
~60 for 6, ~90 for 16 (measured on real routing), so more requests cost little more per step.

Prompt processing (prefill), 14.5k-token prompts, context limit 262k:

| prompts at once | 1 | 6 |
|---|---|---|
| prefill, aggregate | 5,500 tok/s | 5,600 tok/s |
| prefill, aggregate, `NBT=2048` (context limit 65k: 2048 does not fit 262k) | 6,050 tok/s | 6,050 tok/s |

One prompt already fills the GPU (prefill is compute-bound), so 6 at once share the same
~5,600 tok/s. gpu-unlock raises the bf16 tensor-core rate (102 TFLOPS on a cuBLAS GEMM instead
of ~40): that changes prefill, not decode, which is bandwidth-bound. A stock card prefills
slower; decode is the same.

Decode of one request, by kind of text (`bench/spec_bench.py`, temperature 0.2 for structured
output and 0.5 for the rest, reasoning on):

| | without DSpark | with DSpark (default) |
|---|---|---|
| chat (Italian, English) | 136 tok/s | ~150 |
| code | 135 | ~220 |
| structured output (JSON from text) | 135 | **~295** |
| prose | 135 | ~155 |
| questions over a document, quoting it | 133 | ~275 |
| copying a 120-line table from a 50k-token document (greedy) | not measured | **~490** |

Long prompts are bound by that prefill (AA-LCR documents, 256 to 64k tokens, mean 12k; 512
tokens out, so 24 prompt tokens per generated one; `bench/evalscope_perf.sh`, 10 minutes each).
With 6 at once the GPU prefills 6 x 12k tokens per round: the output cannot exceed
~5,600 / 24 = ~230 tok/s aggregate however fast the decode, and each request's decode steps
share the GPU with the others' prefill chunks.

| | 1 request, no DSpark | 1 request, DSpark | 6 requests, no DSpark | 6 requests, DSpark |
|---|---|---|---|---|
| decode, per request | 128 tok/s | **217 tok/s** | 26 tok/s | 31 tok/s |
| output, aggregate, prefill time included | 77 tok/s | 102 tok/s | 125 tok/s | 141 tok/s |
| time to first token (6: queueing included) | 2.66 s | 2.64 s | 4.64 s | 5.29 s |

Document extraction, 6 requests at once (`bench/extract_bench.py`, 10 minutes: a shared ~3.5k
token Italian system prompt, 1-3 DocVQA pages, half strict JSON at temperature 0.2, half
Italian summaries at 0.5; prompts 6.4k tokens, answers 1.6k): 377 tok/s decode aggregate,
time to first token 0.84 s median and 1.45 s p90, 2.4 tokens per verify step.

DSpark is lossless: the target verifies every draft. It helps most where the text is
predictable (structured output, code, quoting) and least on free prose, because a verify step
on a mixture-of-experts model reads many more experts than a plain step (NOTES.md).

## Context and the knobs that trade it

All measured with the 8-bit KV cache (`docs/memory_matrix.tsv`, `bench/memory_matrix.sh`).
The wizard reads that table; these are the defaults and what each knob costs or frees.

| knob | default | what it does | context / VRAM | speed |
|---|---|---|---|---|
| `SPEC` | `dspark` | speculative decoding with the bundled drafter | ~80k tokens less for the longest request (213k vs 291k at the same scheduling, vision on the GPU) | 1.3-2.3x decode on one request |
| `MAX_LEN` | 262144 | longest request | with DSpark, every 100k allowed costs ~3k tokens of KV (the drafter's RoPE table) | none |
| `ASYNC` | `auto` (off with DSpark above 192k) | async scheduling (two steps in flight) | off: ~55k tokens more for the longest request | off: -3 to -4% decode |
| `IMAGES` | 50 | images per request; 0 = no vision tower | 0: ~34k tokens more | no images |
| `HYPERINKLING_VISION_HOST` | 0 | vision tower weights in host RAM, copied to the GPU layer by layer per image | 1: ~34k tokens more (as much as no vision) | +27 ms per A4 page (0.39 s vs 0.37 s to first token for one page) |
| `IMAGE_MAX_W` / `_H` | 1240 x 1760 | largest image (A4 at 150 ppi, 1,364 tokens); larger ones are scaled down to it | A4 at 300 ppi (2480 x 3508, with `ENCODER_CACHE=16384`): about the same context, but each page costs ~4x the tokens | 300 ppi: ~4x the prefill per page |
| `MAX_SEQS` | 6 | requests at once | longest request with DSpark: 279k at 1, 268k at 6, 257k at 16, 228k at 64 | 6 -> 16 requests: ~430 -> 740 tok/s decode in total |
| `KV_OFFLOAD_GB` | `auto` (60% of RAM - 8 GiB, at most 36) | prefix-cache tier in host RAM (`/dev/shm`) | host RAM only | evicted prefixes come back in 0.1 s instead of seconds |
| `NBT` | 1024 | prefill chunk | 2048: ~68k tokens less (268k -> 200k), async off; 512: more | 2048: +7-8% prefill (6,050 vs 5,610 tok/s), same decode; 512: -22% prefill |
| `HYPERINKLING_SPEC_ADAPTIVE` | `2,4,7` | verify length chosen per step from these; empty = the fixed `SPEC_K` schedule | ~5k tokens (1,2,3,4,5,7: ~14k) | with the lookup: +41-43% copying, +20% at 6 requests |
| `HYPERINKLING_LOOKUP` | 1 | drafts copied from the request's own context, fused with DSpark's (lossless) | 6 MB | +7-13% while copying |
| `SPEC_K` | `[[1,2,4],[3,6,3]]` | drafts verified per step, by batch size, when the adaptive length is off | none | |
| `SPEC_DRAFT` | `probabilistic` | drafts sampled from the draft distribution | none | +15-20% on structured output at temperature > 0 |
| `HOST` / `VLLM_API_KEY` | 127.0.0.1 / none | who can connect | none | none |

Longest single request, 6 requests at once, KV at 8 bits:

| vision tower | DSpark, async on | DSpark, async off | no DSpark, async on | no DSpark, async off |
|---|---|---|---|---|
| on the GPU | 213k | **268k** (the default) | 291k | 345k |
| in host RAM | 249k | 302k | 331k | 386k |
| none | 249k | 302k | 336k | 386k |

With DSpark the adaptive verify length (the default) takes ~5k tokens of these: 263k in the
default column. A long context without DSpark and without the vision tower on the GPU is best effort: it
grows as far as the 24 GiB allow (386k), not to the model's 1M.

## What runs where

| part | format | kernel |
|---|---|---|
| routed experts | INT4 g128 asymmetric (AWQ layout), GPTQ-calibrated | Marlin MoE |
| attention projections | INT8 g128, quantized at load | Triton W8A16 GEMV/GEMM |
| LM head | INT8 g128, fp32 logits (shared with the drafter) | same |
| token embeddings | INT8 g128, in pinned host RAM (UVA) | lookup |
| KV cache | unit-scale FP8 E4M3, emulated (sm86 has no FP8) | Triton attention, fp16 MMAs |
| loop controllers, sconv taps | bf16 | Triton batched GEMV |
| DSpark drafter | layers INT4 g128; Markov fold INT4; KV FP8 | Triton W4A16 split-K |
| vision tower | bf16, on the GPU (or streamed from host RAM) | FlashAttention-2 + cuBLAS |
| prefix cache | GPU + CPU tier | vLLM OffloadingConnector |

The patch's commit message (`patches/vllm-hyperinkling-sm86.patch`) lists every change.
NOTES.md has the reasons, the measurements and the gotchas; FOLLOWUPS.md what comes next.

## Quality

Teacher-forced KL against the original NVFP4/MXFP8 checkpoint on 22 held-out transcripts
(images, tool calls, code, Italian, chat; `bench/kl_eval.py`, one request at a time). The
median is the number to compare: a few positions where the run-to-run noise flips an expert
dominate the mean.

| | KL median | KL mean | top-1 agreement |
|---|---|---|---|
| the original against itself (noise) | 0.0003 | 0.060 | 95.8% |
| experts INT4, round to nearest | 0.0039 | 0.133 | 91.2% |
| experts INT4 GPTQ | 0.0028 | 0.112 | 92.7% |
| **as served**: + INT8 attention, head, embeddings, FP8 KV | **0.0030** | 0.117 | 92.4% |

Document pages are close to the noise floor (mean 0.009). Speculative decoding does not
change the output distribution.

Task accuracy, the same server settings for both (`bench/quality_battery.py`, greedy, 200
items each; about +-1.3 points of noise):

| | original checkpoint | HyperInkling as served |
|---|---|---|
| GSM8K | 0.965 | 0.970 |
| tool calls, exact (names only) | 0.595 (0.685) | 0.610 (0.700) |
| DocVQA as strict JSON, ANLS (valid JSON) | 0.926 (99.5%) | 0.930 (99.0%) |

## Model preparation

The published checkpoint is the output of this pipeline; you only need it to redo the
quantization.

```bash
V=venv/bin/python
# 1. calibration corpus from public datasets (data/calib-src, see NOTES.md), answers
#    generated by the original checkpoint (needs a server on :18030 with tool/reasoning parsers)
$V prepare/build_calib.py data/calib/calib.jsonl data/calib/eval.jsonl
# 2. capture MoE inputs + routing (eager server with HYPERINKLING_CAPTURE_DIR set)
$V prepare/replay_capture.py data/calib/calib.jsonl data/capture
# 3. GPTQ the experts into the INT4 checkpoint (~5 min, GPU)
$V prepare/gptq_experts.py models/Inkling-2-Mini data/capture models/Inkling-2-Mini-W4A16-g128-gptq
```

## Repository

| path | what |
|---|---|
| `start.sh` | the launcher; every knob is an env var |
| `scripts/wizard.py` | asks the questions, writes `.env` |
| `Dockerfile`, `docker-compose.yml`, `scripts/build-image.sh` | the container |
| `patches/` | the vLLM patch and its base commit (`BASE`) |
| `scripts/` | `setup-vllm.sh`, `build-vllm.sh`, `serve-try.sh`, `quick_bench.py` |
| `prepare/` | INT4 packing, RTN and GPTQ converters, calibration corpus and capture |
| `bench/` | speed (per kind of text, long prompts, server-side metrics), memory matrix, KL eval, profiles |
| `tests/` | kernel and processor tests (`venv/bin/python -m pytest -q tests/`) |
| `docs/` | measured tables |

## Credits

- [HyperQwen](https://github.com/syv-ai/HyperQwen): the idea, the name, and several of the
  approaches here (a pinned vLLM plus a patch, embeddings in host RAM, the launcher and
  wizard style, the one-step-in-flight trade for context).
- [basalt](https://github.com/jesdga95/basalt) (MIT): its DSpark draft policy (verify window
  by expected tokens per millisecond) and its handling of oversized images informed ours.
- vLLM PR [#69028](https://github.com/vllm-project/vllm/pull/69028) for Inkling 2 support,
  and Thinking Machines Lab for the model and its drafter.

## License

Apache-2.0, as the model and vLLM.

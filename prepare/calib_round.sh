#!/bin/bash
# One more calibration round: generate a further corpus (new seed), capture it,
# run GPTQ on every capture so far, and score the result on the eval set.
# Usage: prepare/calib_round.sh <seed> <tag>   e.g. 20261122 r2
set -uo pipefail
cd "$(dirname "$0")/.."
seed=$1; tag=$2
corpus=data/calib/calib-$tag.jsonl
capture=data/capture-$tag
out=models/Inkling-2-Mini-W4A16-g128-gptq-$tag
common=(--max-num-batched-tokens 512 --reasoning-parser inkling
	--enable-auto-tool-choice --tool-call-parser inkling)

stop_server() {
	for p in $(pgrep -f '[v]env/bin/vllm serve'); do kill "$p"; done
	while pgrep -f '[v]env/bin/vllm serve' > /dev/null; do sleep 1; done
}
start_server() {  # <log name> [env...] -- [args...]
	local log=".local/$1.log"; shift
	rm -f "$log"
	env "$@" > /dev/null 2>&1 &
	until grep -q -E 'Application startup complete|Traceback' "$log" 2>/dev/null; do sleep 2; done
	! grep -q Traceback "$log"
}

echo "== generate $corpus"
start_server "$tag-gen" MM_LIMIT='{"image":1}' MAX_LEN=16384 EAGER= MAX_SEQS=6 TRY_TIMEOUT=7200 \
	scripts/serve-try.sh "$tag-gen" "${common[@]}" \
	--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6]}' \
	|| { echo "gen server failed"; exit 1; }
HYPERINKLING_CALIB_SEED=$seed venv/bin/python prepare/build_calib.py "$corpus" data/calib/eval.jsonl
stop_server

echo "== capture $capture"
rm -rf "$capture"; mkdir -p "$capture"
start_server "$tag-capture" HYPERINKLING_CAPTURE_DIR="$PWD/$capture" MM_LIMIT='{"image":1}' \
	MAX_LEN=16384 MAX_SEQS=2 TRY_TIMEOUT=7200 scripts/serve-try.sh "$tag-capture" "${common[@]}" \
	|| { echo "capture server failed"; exit 1; }
venv/bin/python prepare/replay_capture.py "$corpus" "$capture"
stop_server

echo "== gptq $out"
dirs=$(ls -d data/capture data/capture-* | paste -sd,)
rm -rf "$out"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True taskset -c 0-30 \
	venv/bin/python prepare/gptq_experts.py models/Inkling-2-Mini "$dirs" "$out" \
	| grep -E '^layer [ 0-9]+:|done'

echo "== score"
bench/kl_sweep.sh data/calib/eval.jsonl bench/kl/eval22 "gptq-$tag=$out"
venv/bin/python bench/kl_eval.py compare bench/kl/eval22/ref.json \
	"bench/kl/eval22/gptq-$tag.json" data/calib/eval.jsonl

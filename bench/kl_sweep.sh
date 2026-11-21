#!/bin/bash
# Score a corpus on several checkpoints, one server at a time, same settings.
# Usage: bench/kl_sweep.sh <corpus> <out dir> <name>=<model dir> [...]
# A name may repeat (e.g. ref=..., ref2=... on the same dir) for a noise floor.
# The HYPERINKLING_* load-time conversions come from the caller's env (off by default);
# SERVE_ARGS adds vllm serve flags (e.g. --kv-cache-dtype fp8).
set -uo pipefail
cd "$(dirname "$0")/.."
corpus=$1; out=$2; shift 2
mkdir -p "$out"
for spec in "$@"; do
	name=${spec%%=*}; model=${spec#*=}
	log=".local/sweep-$name.log"
	rm -f "$log"  # a previous run's ready line must not count
	MODEL=$model MM_LIMIT='{"image":1}' MAX_LEN=16384 EAGER= MAX_SEQS=1 TRY_TIMEOUT=3600 \
		scripts/serve-try.sh "sweep-$name" --max-num-batched-tokens 512 \
		--reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
		--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
		${SERVE_ARGS:-} \
		> /dev/null 2>&1 &
	for _ in $(seq 600); do  # 20 min
		grep -q -E 'Application startup complete|Traceback' "$log" 2>/dev/null && break
		sleep 2
	done
	if ! grep -q 'Application startup complete' "$log"; then
		echo "$name: server failed, see $log"
		for p in $(pgrep -f '[v]env/bin/vllm serve'); do kill "$p"; done
		wait
		continue
	fi
	venv/bin/python bench/kl_eval.py score "$corpus" "$out/$name.json"
	for p in $(pgrep -f '[v]env/bin/vllm serve'); do kill "$p"; done
	while pgrep -f '[v]env/bin/vllm serve' > /dev/null; do sleep 1; done
	wait
done

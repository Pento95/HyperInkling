#!/bin/bash
# DSpark verify-length sweep. The block drafter always drafts 7; k is how many of them the target
# verifies (vLLM's dynamic speculative decoding with one batch-size range): for each k, boot start.sh with DSpark k, then quick_bench at
# concurrency 1 and 6 (512 tokens, ignore_eos) and the acceptance over that window.
#
#   bench/dspark_k_sweep.sh "2 3 4 5" [port]      # log in .local/dspark-sweep/
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
KS=${1:-"2 3 4 5 7"}; PORT=${2:-18030}; OUT=.local/dspark-sweep; mkdir -p "$OUT"
stop() { local p; p=$(pgrep -f '[v]env/bin/vllm serve'); [ -n "$p" ] && kill $p
	for _ in $(seq 60); do pgrep -f '[v]env/bin/vllm serve' >/dev/null || return 0; sleep 2; done; }
metric() { curl -s "localhost:$PORT/metrics" | awk -v k="$1" 'index($0,k"{")==1 {s+=$NF} END {print s+0}'; }
for k in $KS; do
	stop
	KV_OFFLOAD_GB=0 WARMUP=0 ENV_FILE= PORT=$PORT MAX_LEN=${MAX_LEN:-131072} SPEC_K="[[1,6,$k]]" \
		./start.sh > "$OUT/server-k$k.log" 2>&1 &
	sp=$!
	ok=0; for _ in $(seq 120); do curl -sf --max-time 3 "localhost:$PORT/health" >/dev/null && { ok=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
	[ $ok = 1 ] || { echo "k=$k: server did not come up: $(grep -m1 -E '^\S*Error' "$OUT/server-k$k.log")" | tee -a "$OUT/results.txt"; continue; }
	venv/bin/python scripts/quick_bench.py 1 64 "$PORT" > /dev/null   # warm-up
	for c in 1 6; do
		d0=$(metric vllm:spec_decode_num_drafts_total); a0=$(metric vllm:spec_decode_num_accepted_tokens_total)
		r=$(venv/bin/python scripts/quick_bench.py $c 512 "$PORT")
		d1=$(metric vllm:spec_decode_num_drafts_total); a1=$(metric vllm:spec_decode_num_accepted_tokens_total)
		echo "k=$k $r | $(python3 -c "print(f'{1+($a1-$a0)/max($d1-$d0,1):.2f} tokens/step, {($d1-$d0)} steps')")" | tee -a "$OUT/results.txt"
	done
	grep -m1 'GPU KV cache size' "$OUT/server-k$k.log" | sed 's/.*GPU KV/  KV/' | tee -a "$OUT/results.txt"
done
stop

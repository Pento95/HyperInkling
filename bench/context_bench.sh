#!/bin/bash
# bench/context_bench.py with and without DSpark, one server each with start.sh's settings
# (the .env's, then the defaults), no CPU tier. Then the README table.
#
#   bench/context_bench.sh <name>        # labels <name>-off, <name>-dspark; GPU must be free
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; PORT=${PORT:-18030}
for spec in off dspark; do
	log=.local/context-bench-$name-$spec.log
	SPEC=$spec PORT=$PORT KV_OFFLOAD_GB=0 ./start.sh > "$log" 2>&1 &
	for _ in $(seq 200); do
		curl -sf --max-time 2 "localhost:$PORT/health" > /dev/null && break
		sleep 3
	done
	if curl -sf --max-time 2 "localhost:$PORT/health" > /dev/null; then
		venv/bin/python bench/context_bench.py run "$name-$spec" --port "$PORT"
	else
		echo "server failed: $log"
	fi
	for p in $(pgrep -f '[v]env/bin/vllm serve'); do kill "$p"; done
	while pgrep -f '[v]env/bin/vllm serve' > /dev/null; do sleep 1; done
	wait
done
venv/bin/python bench/context_bench.py table "$name-off" "$name-dspark"

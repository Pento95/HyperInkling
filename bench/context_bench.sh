#!/bin/bash
# bench/context_bench.py with and without DSpark, one server each with start.sh's defaults
# (not the .env), no CPU tier. Then the README table.
#
#   bench/context_bench.sh <name>        # labels <name>-off, <name>-dspark; GPU must be free
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/server-lib.sh
name=$1; PORT=${PORT:-18030}
for spec in off dspark; do
	log=.local/context-bench-$name-$spec.log
	start_server "$PORT" "$log" SPEC=$spec PORT=$PORT KV_OFFLOAD_GB=0 WARMUP=0 ENV_FILE= ./start.sh
	if wait_health "$PORT"; then
		venv/bin/python bench/context_bench.py run "$name-$spec" --port "$PORT"
	else
		echo "server failed: $log"
	fi
	stop_server
done
venv/bin/python bench/context_bench.py table "$name-off" "$name-dspark"

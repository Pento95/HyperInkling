#!/bin/bash
# Start one server (env from the caller), run a command against it, stop it.
# Usage: bench/serve_and_run.sh <name> "<command>" [serve args...]
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; cmd=$2; shift 2
log=".local/$name.log"; rm -f "$log"
EAGER= TRY_TIMEOUT="${TRY_TIMEOUT:-3600}" scripts/serve-try.sh "$name" "$@" > /dev/null 2>&1 &
until grep -q -E 'Application startup complete|Traceback' "$log" 2>/dev/null; do sleep 2; done
if grep -q Traceback "$log"; then echo "$name: server failed, see $log"; else
	grep -o 'GPU KV cache size: [0-9,]* tokens' "$log"
	bash -c "$cmd"
fi
for p in $(pgrep -f '[v]env/bin/vllm serve'); do kill "$p"; done
while pgrep -f '[v]env/bin/vllm serve' > /dev/null; do sleep 1; done
wait

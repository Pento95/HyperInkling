#!/bin/bash
# Kernel time per decode step under nsys: one server, a warm-up, a profiled
# run of <concurrency> streams x <tokens>, then the top kernels per step.
# Usage: bench/profile_decode.sh <name> <concurrency> <tokens> [serve args...]
# The server's env (MODEL, HYPERINKLING_*) is taken from the caller.
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; conc=$2; tokens=$3; shift 3
rep=".local/prof/$name"
log=".local/prof-$name.log"
mkdir -p .local/prof; rm -f "$log"
EAGER= MAX_SEQS=6 TRY_TIMEOUT=1800 \
	WRAP="nsys profile -o $rep --force-overwrite true --trace cuda,nvtx --cuda-graph-trace node --capture-range cudaProfilerApi --capture-range-end stop" \
	scripts/serve-try.sh "prof-$name" --max-num-batched-tokens 512 \
	--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6]}' \
	--profiler-config '{"profiler":"cuda"}' "$@" > /dev/null 2>&1 &
until grep -q -E 'Application startup complete|Traceback' "$log" 2>/dev/null; do sleep 2; done
grep -q Traceback "$log" && { echo "server failed: $log"; exit 1; }
venv/bin/python scripts/quick_bench.py "$conc" 64 > /dev/null
curl -s -X POST http://127.0.0.1:18030/start_profile
venv/bin/python scripts/quick_bench.py "$conc" "$tokens" | grep conc
curl -s -X POST http://127.0.0.1:18030/stop_profile
sleep 8
for p in $(pgrep -f '[v]env/bin/vllm serve'); do kill "$p"; done
while pgrep -f '[v]env/bin/vllm serve' > /dev/null; do sleep 1; done
wait
nsys stats -q --report cuda_gpu_kern_sum --format csv "$rep.nsys-rep" 2> /dev/null | python3 -c "
import csv, sys
rows = list(csv.DictReader(sys.stdin)); steps = $tokens
tot = sum(float(r['Total Time (ns)']) for r in rows)
print(f'per step: {tot / steps / 1e6:.2f} ms')
for r in rows[:14]:
    print(f\"{float(r['Total Time (ns)']) / steps / 1e3:8.1f} us/step {int(r['Instances']) // steps:4d}x {float(r['Avg (ns)']) / 1e3:7.1f}us  {r['Name'][:80]}\")
"

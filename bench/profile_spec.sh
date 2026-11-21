#!/bin/bash
# Kernel time per decode step under nsys, with start.sh's own settings (so CUDA graphs and
# speculation are exactly the served ones). One server on PORT (18030), a warm-up, a profiled
# run of <concurrency> streams x <tokens>, then the top kernels per engine step.
#
#   bench/profile_spec.sh <name> <concurrency> <tokens> [EXTRA_ARGS for start.sh]
#   (speculation follows start.sh: SPEC=off, or SPEC_K="[[1,6,k]]")
#
# Steps are counted from /metrics (vllm:iteration_tokens_total_count), so a speculative step
# (draft + verify) is one step.
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; conc=$2; tokens=$3; extra=${4:-}
PORT=${PORT:-18030}; log=.local/prof-$name.log
mkdir -p .local/prof
PROFILE=$name PORT=$PORT KV_OFFLOAD_GB=0 MAX_LEN=${MAX_LEN:-131072} EXTRA_ARGS="$extra" ./start.sh > "$log" 2>&1 &
for _ in $(seq 150); do curl -sf --max-time 2 "localhost:$PORT/health" > /dev/null && break; sleep 3; done
curl -sf --max-time 2 "localhost:$PORT/health" > /dev/null || { echo "server failed: $log"; exit 1; }
steps() { curl -s "localhost:$PORT/metrics" | awk 'index($0,"vllm:iteration_tokens_total_count{")==1 {s+=$NF} END {print s+0}'; }
venv/bin/python scripts/quick_bench.py "$conc" 64 "$PORT" > /dev/null
curl -s -X POST "localhost:$PORT/start_profile"
s0=$(steps)
venv/bin/python scripts/quick_bench.py "$conc" "$tokens" "$PORT" | grep conc
s1=$(steps)
curl -s -X POST --max-time 600 "localhost:$PORT/stop_profile"
for p in $(pgrep -f '[v]env/bin/vllm serve'); do kill "$p"; done
while pgrep -f '[v]env/bin/vllm serve' > /dev/null; do sleep 1; done
nsys stats -q --report cuda_gpu_kern_sum --format csv ".local/prof/$name.nsys-rep" 2> /dev/null | python3 -c "
import csv, sys
rows = list(csv.DictReader(sys.stdin)); steps = max($s1 - $s0, 1)
tot = sum(float(r['Total Time (ns)']) for r in rows)
print(f'{steps} steps, kernels {tot / steps / 1e6:.2f} ms/step')
for r in rows[:${TOP:-25}]:
    print(f\"{float(r['Total Time (ns)']) / steps / 1e3:8.1f} us/step {int(r['Instances']) / steps:6.1f}x {float(r['Avg (ns)']) / 1e3:7.1f}us  {r['Name'][:90]}\")
"

#!/bin/bash
# Short-context eager boot of the original checkpoint, for porting work.
# Usage: scripts/serve-try.sh <log name> [extra vllm serve args...]
set -uo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
# nvcc 13.3 rejects gcc 16 (Fedora 44 default): FlashInfer's JIT builds use gcc-15.
export CC=gcc-15 CXX=g++-15 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-15"
log=".local/$1.log"; shift
timeout "${TRY_TIMEOUT:-900}" taskset -c 0-30 ${WRAP:-} vllm serve "${MODEL:-models/Inkling-2-Mini}" \
	--served-model-name inkling-2-mini --host 127.0.0.1 --port 18030 \
	--max-model-len "${MAX_LEN:-8192}" --max-num-seqs "${MAX_SEQS:-1}" ${EAGER---enforce-eager} \
	--gpu-memory-utilization "${GPU_UTIL:-0.95}" ${PREFIX_CACHING---no-enable-prefix-caching} \
	--limit-mm-per-prompt "${MM_LIMIT:-{\"image\":0\}}" "$@" > "$log" 2>&1
echo "exit=$?"
grep -n -m1 -B2 -A0 'Traceback' "$log"
grep -n -E '^\S*(\w+Error|\w+Exception)\b|Error:|error:' "$log" | grep -v -i warn | tail -5 | cut -c1-400

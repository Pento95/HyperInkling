#!/bin/bash
# 10-minute evalscope perf run on prompts of varied length (bench/make_varlen_prompts.py),
# with bench/metrics_window.py measuring the same window server-side. Lengths are set by the
# generator (no tokenizer here: evalscope would count characters, hence the huge max length).
#
#   bench/evalscope_perf.sh <name> [port]     # results in .local/perf/<name>/
#
# Env: PARALLEL (6), DURATION (600 s), OUT_TOK (512, fixed: ignore_eos), MODEL (inkling-2-mini),
#      PERF_WARMUP (2 requests, not measured), PROMPTS (bench/perf/varlen.jsonl).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME=$1; PORT=${2:-8080}
PARALLEL=${PARALLEL:-6}; DURATION=${DURATION:-600}; OUT_TOK=${OUT_TOK:-512}
MODEL=${MODEL:-inkling-2-mini}; PERF_WARMUP=${PERF_WARMUP:-2}
PROMPTS=${PROMPTS:-$REPO/bench/perf/varlen.jsonl}
OUT=$REPO/.local/perf/$NAME; mkdir -p "$OUT"
URL=http://127.0.0.1:$PORT

[ -f "$PROMPTS" ] || "$REPO/venv/bin/python" -I "$REPO/bench/make_varlen_prompts.py" "$PROMPTS"
curl -sf --max-time 5 "$URL/health" > /dev/null || { echo "no server on $URL" >&2; exit 1; }

perf() {  # perf <name> <number> [extra evalscope args...]
	local n=$1 num=$2; shift 2
	evalscope perf --url "$URL/v1/chat/completions" --api openai --api-key x --model "$MODEL" \
		--dataset openqa --dataset-path "$PROMPTS" --max-prompt-length 100000000 \
		--max-tokens "$OUT_TOK" --min-tokens "$OUT_TOK" --temperature 0 --seed 0 \
		--extra-args '{"ignore_eos":true}' --stream \
		--outputs-dir "$OUT" --name "$n" --no-timestamp --number "$num" "$@"
}

echo "[perf] warm-up: $PERF_WARMUP requests (not measured)"
perf warmup "$PERF_WARMUP" --parallel 1 --dataset-offset 500 > "$OUT/warmup.out" 2>&1

echo "[perf] $DURATION s at parallel $PARALLEL, $OUT_TOK output tokens; metrics window alongside"
"$REPO/venv/bin/python" "$REPO/bench/metrics_window.py" "$((DURATION - 30))" "$PORT" \
	> "$OUT/metrics_window.out" 2>&1 &
mw=$!
perf main 10000 --parallel "$PARALLEL" --duration "$DURATION" 2>&1 | tee "$OUT/main.out"
wait "$mw" || true
echo "== server side (metrics_window) =="; cat "$OUT/metrics_window.out"

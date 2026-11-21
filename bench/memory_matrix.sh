#!/bin/bash
# The longest context each knob combination can serve, measured: boot start.sh and read
# vLLM's verdict (the estimated maximum model length when it does not fit, the KV pool when
# it does), in two boots (see below). Feeds the wizard's table (scripts/wizard.py).
#
#   bench/memory_matrix.sh [out.tsv]        # ~3 min per combination, GPU must be free
#
# Columns: vision spec async max_seqs  max_len_fits  pool_tokens  weights_gib  measured_at
# (max_len_fits is vLLM's estimate at MAX_LEN = measured_at; the drafter's RoPE table
# follows MAX_LEN, see scripts/wizard.py).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OUT=${1:-.local/memory_matrix.tsv}; PORT=${PORT:-18030}
mkdir -p .local/memmatrix
COMBOS=${COMBOS:-"gpu:dspark:1:6 gpu:dspark:0:6 gpu:off:1:6 host:dspark:1:6 host:dspark:0:6 host:off:1:6 none:dspark:1:6 none:dspark:0:6 none:off:1:6 gpu:dspark:0:1 gpu:dspark:0:16 gpu:dspark:0:32 gpu:dspark:0:64 gpu:off:1:64"}
[ -f "$OUT" ] || printf 'vision\tspec\tasync\tmax_seqs\tmax_len_fits\tpool_tokens\tweights_gib\tmeasured_at\n' > "$OUT"
for c in $COMBOS; do
	IFS=: read -r vision spec async seqs <<< "$c"
	log=.local/memmatrix/$vision-$spec-a$async-s$seqs.log
	env_args=(PORT=$PORT KV_OFFLOAD_GB=0 SPEC=$spec ASYNC=$async MAX_SEQS=$seqs)
	case $vision in
	host) env_args+=(HYPERINKLING_VISION_HOST=1) ;;
	none) env_args+=(IMAGES=0) ;;
	esac
	boot() {  # boot <max_len>: sets fits, pool, weights from one start
		env "${env_args[@]}" MAX_LEN="$1" ./start.sh > "$log" 2>&1 &
		local sp=$!
		for _ in $(seq 200); do
			grep -qE 'GPU KV cache size|Engine core initialization failed' "$log" && break
			kill -0 $sp 2>/dev/null || break
			sleep 3
		done
		fits=$(grep -m1 -oE 'estimated maximum model length is [0-9]+' "$log" | grep -oE '[0-9]+$')
		pool=$(grep -m1 -oE 'GPU KV cache size: [0-9,]+' "$log" | grep -oE '[0-9,]+$' | tr -d ,)
		[ -n "$pool" ] && fits=$1
		weights=$(grep -m1 -oE 'Model loading took [0-9.]+ GiB' "$log" | grep -oE '[0-9.]+')
		for p in $(pgrep -f '[v]llm serve'); do kill "$p"; done
		for _ in $(seq 60); do pgrep -f '[v]llm serve' > /dev/null || break; sleep 2; done
		wait $sp 2>/dev/null
	}
	# vLLM checks the context twice: before the warmup (optimistic: without the transient
	# activations and CUDA graphs) and after it. A first boot at 1M stops at the first
	# check; a second at that estimate reaches the real one.
	boot 1048576
	at=$(( ${fits:-0} / 1024 * 1024 ))
	if [ -z "$pool" ] && [ "$at" -gt 0 ]; then boot "$at"; fi
	printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$vision" "$spec" "$async" "$seqs" "${fits:-?}" "${pool:--}" "${weights:-?}" "$at" | tee -a "$OUT"
done

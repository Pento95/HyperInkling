#!/bin/bash
# Build the HyperInkling container image: vLLM's own Dockerfile on our patched checkout
# (./vllm, from scripts/setup-vllm.sh) for sm86 only, then the launcher layer on top.
#
#   scripts/build-image.sh [tag]            # default tag hyperinkling:latest
#
# Env: MAX_JOBS (compile jobs, default: nproc - 1), NVCC_THREADS (2), DOCKER (docker).
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-hyperinkling:latest}
DOCKER=${DOCKER:-docker}
JOBS=${MAX_JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
[ -d vllm/.git ] || { echo "no ./vllm checkout: run scripts/setup-vllm.sh first" >&2; exit 1; }
git -C vllm log -1 --format='%s' | grep -q '^\[hyperinkling\]' \
	|| { echo "./vllm HEAD is not the HyperInkling commit" >&2; exit 1; }

"$DOCKER" build --network=host -f vllm/docker/Dockerfile --target vllm-openai \
	--build-arg torch_cuda_arch_list='8.6' \
	--build-arg max_jobs="$JOBS" --build-arg nvcc_threads="${NVCC_THREADS:-2}" \
	-t hyperinkling-vllm:sm86 vllm
"$DOCKER" build --network=host --build-arg VLLM_IMAGE=hyperinkling-vllm:sm86 -t "$TAG" .
echo "built $TAG"

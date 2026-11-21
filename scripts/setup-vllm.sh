#!/bin/bash
# Recreate ./vllm: vllm-project/vllm at the PR #69028 commit in patches/BASE,
# plus HyperInkling's sm86 port (patches/vllm-hyperinkling-sm86.patch) as one
# commit on branch hyperinkling/sm86. Then build with scripts/build-vllm.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
base=$(cat patches/BASE)
if [ ! -d vllm/.git ]; then
	git clone --filter=blob:none https://github.com/vllm-project/vllm.git vllm
fi
cd vllm
# By SHA first (GitHub serves any commit it still has, whatever happens to the PR
# branch later); the PR ref only if that fails.
git fetch --filter=blob:none origin "$base" || git fetch origin pull/69028/head
git checkout -B hyperinkling/sm86 "$base"
git am --3way --keep-non-patch ../patches/vllm-hyperinkling-sm86.patch
git log --oneline -2

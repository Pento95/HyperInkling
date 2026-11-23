"""Warm-up requests at the shapes vLLM's boot warmup leaves out, so their Triton kernels
compile (or load from the JIT cache) before the first real request: prefills of 65-256
tokens and over 256 not a multiple of 16, a full prefill chunk, decode batches of 2-8
requests (the LM head runs outside the CUDA graphs at every verify row count), and a
prefill next to running decodes. start.sh runs it in the background once /health answers
(WARMUP=1).

    python3 scripts/warmup.py [port] [model] [host]   # VLLM_API_KEY from the environment
"""

import json
import os
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8080"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "inkling-2-mini"
HOST = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
if HOST in ("0.0.0.0", "::", ""):  # listening everywhere
    HOST = "127.0.0.1"
URL = f"http://[{HOST}]:{PORT}" if ":" in HOST else f"http://{HOST}:{PORT}"
failures = []
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("VLLM_API_KEY"):
    HEADERS["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"
WORDS = ("the model reads every page of the report and writes a careful summary of the "
         "numbers in each table ").split()


def _wait_health(deadline_s=1800):
    end = time.time() + deadline_s
    server = os.getppid()  # start.sh becomes the server
    while time.time() < end and os.getppid() == server:
        try:
            urllib.request.urlopen(f"{URL}/health", timeout=2)
            return True
        except OSError:
            time.sleep(3)
    return False


def _chat(words, max_tokens):
    text = " ".join(WORDS[i % len(WORDS)] for i in range(words))
    body = {"model": MODEL, "temperature": 0.5, "max_tokens": max_tokens,
            "min_tokens": max_tokens, "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(),
                                 HEADERS)
    try:
        urllib.request.urlopen(req, timeout=600).read()
    except OSError as e:
        failures.append(e)
        print(f"warm-up request failed: {e}", file=sys.stderr)


def _at_once(*calls):
    threads = [threading.Thread(target=_chat, args=c) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    if not _wait_health():
        sys.exit("warm-up: the server never answered /health")
    t0 = time.time()
    # Prefills (~1.3 tokens per word with the chat template): 65-256 tokens, over 256
    # not a multiple of 16, one full 2048-token chunk and its remainder.
    for words in (60, 115, 170, 530, 1150, 2300):
        _chat(words, 2)
    # Decode batches of 2..8 requests (verify lengths vary with the adaptive k).
    for n in range(2, 9):
        _at_once(*[(20, 48)] * n)
    # A prefill while three requests decode (a mixed step).
    threads = [threading.Thread(target=_chat, args=(20, 160)) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    _chat(530, 2)
    for t in threads:
        t.join()
    if failures:
        sys.exit(f"warm-up: {len(failures)} requests failed")
    print(f"warm-up done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()

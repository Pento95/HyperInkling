"""Prefix reuse after eviction: send P1..P4 (different ~17k-token prompts), then
P1 again, and report each TTFT. With a small GPU KV pool, P4 evicts P1, so the
last TTFT is a full recompute unless a CPU tier brings P1's prefix back.

    venv/bin/python bench/offload_test.py [prompt tokens] [count]
"""

import json
import sys
import time
import urllib.request

import pyarrow.parquet as pq

URL = "http://127.0.0.1:18030/v1/chat/completions"
target = int(sys.argv[1]) if len(sys.argv) > 1 else 17000
count = int(sys.argv[2]) if len(sys.argv) > 2 else 4

texts = [t for t in pq.read_table(
    "data/calib-src/wiki-it/20231101.it/train-00002-of-00010.parquet", columns=["text"]
).column("text").to_pylist()[8000:] if len(t) > 3000]
prompts, i = [], 0
for _ in range(count):
    buf = []
    while sum(len(b) for b in buf) < target * 3.2:
        buf.append(texts[i])
        i += 1
    prompts.append("\n\n".join(buf))


def ttft(p):
    body = json.dumps({"model": "inkling-2-mini",
                       "messages": [{"role": "user", "content": p + "\n\nRiassumi."}],
                       "max_tokens": 8, "temperature": 0, "stream": True,
                       "reasoning_effort": "low"}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    for line in urllib.request.urlopen(req, timeout=3600):
        if line.startswith(b"data: {") and json.loads(line[6:])["choices"]:
            return time.time() - t0


for n, p in enumerate(prompts, 1):
    print(f"P{n}: TTFT {ttft(p):.2f}s")
print(f"P1 again: TTFT {ttft(prompts[0]):.2f}s")
print(f"P{count} again (still on GPU): TTFT {ttft(prompts[-1]):.2f}s")

"""Decode speed after a long prompt: N concurrent requests, each a different
~<tokens>-token Italian Wikipedia prompt (prefix caching cannot share them).

    venv/bin/python bench/long_decode.py <concurrency> <prompt tokens> [gen tokens]
"""

import json
import sys
import threading
import time
import urllib.request

import pyarrow.parquet as pq

URL = "http://127.0.0.1:18030/v1/chat/completions"
conc = int(sys.argv[1])
target = int(sys.argv[2])
gen = int(sys.argv[3]) if len(sys.argv) > 3 else 256

table = pq.read_table("data/calib-src/wiki-it/20231101.it/train-00002-of-00010.parquet",
                      columns=["text"])
texts = [t for t in table.column("text").to_pylist()[5000:] if len(t) > 3000]
prompts, i = [], 0
for _ in range(conc):
    buf = []
    while sum(len(b) for b in buf) < target * 3.2:  # ~3.2 chars per token
        buf.append(texts[i])
        i += 1
    prompts.append("\n\n".join(buf) + "\n\nRiassumi in dettaglio i testi precedenti.")

res = []


def run(p):
    body = json.dumps({"model": "inkling-2-mini", "messages": [{"role": "user", "content": p}],
                       "max_tokens": gen, "temperature": 0, "stream": True, "ignore_eos": True,
                       "reasoning_effort": "low", "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0, first, usage = time.time(), None, None
    for line in urllib.request.urlopen(req, timeout=3600):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        elif d["choices"] and first is None:
            first = time.time()
    res.append((first - t0, usage["prompt_tokens"], usage["completion_tokens"], time.time() - first))


ts = [threading.Thread(target=run, args=(p,)) for p in prompts]
[t.start() for t in ts]
[t.join() for t in ts]
per = [r[2] / r[3] for r in res]
print(f"conc {conc}, prompt ~{sum(r[1] for r in res) // conc} tok: decode {min(per):.1f}-{max(per):.1f} "
      f"tok/s per stream, {sum(per):.0f} aggregate; TTFT max {max(r[0] for r in res):.1f}s")

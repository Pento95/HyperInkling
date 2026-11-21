"""Decode tok/s against the local server: N concurrent streams, fixed length.

    venv/bin/python scripts/quick_bench.py [concurrency] [max_tokens] [port]
"""
import json
import sys
import threading
import time
import urllib.request

PORT = sys.argv[3] if len(sys.argv) > 3 else "8080"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
conc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
PROMPTS = [
    "Spiega in dettaglio come funziona un motore a quattro tempi.",
    "Descrivi la storia dell'Impero romano d'Occidente.",
    "Write a detailed guide to binary search trees in Python.",
    "Elenca e spiega i principi della termodinamica.",
    "Racconta una lunga storia su un faro in Sardegna.",
    "Explain how TCP congestion control works, step by step.",
]
res = []

def run(i):
    body = json.dumps({"model": "inkling-2-mini",
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "ignore_eos": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0, first, toks = time.time(), None, 0
    for line in urllib.request.urlopen(req):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            toks = d["usage"]["completion_tokens"]
        elif d["choices"] and first is None:
            first = time.time()
    res.append((first - t0, toks, time.time() - first))

t0 = time.time()
ts = [threading.Thread(target=run, args=(i,)) for i in range(conc)]
for t in ts:
    t.start()
for t in ts:
    t.join()
wall = time.time() - t0
total = sum(r[1] for r in res)
per = [r[1] / r[2] for r in res]
print(f"conc {conc}: {total} tok in {wall:.1f}s -> {total / wall:.1f} tok/s aggregate, "
      f"{min(per):.1f}-{max(per):.1f} tok/s per stream, TTFT max {max(r[0] for r in res):.2f}s")

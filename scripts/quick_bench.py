"""Decode tok/s against the local server: N concurrent streams, fixed length.

Free prose, where DSpark gains least: not comparable with the README table
(bench/context_bench.sh). "All decoding" is the rate from the last first token to the first
finished stream, as that table counts it; "wall" includes prefill and the tail.

    venv/bin/python scripts/quick_bench.py [concurrency] [max_tokens] [port]
"""
import bisect
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
        "ignore_eos": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0, seen = time.time(), []  # (time, tokens so far)
    for line in urllib.request.urlopen(req):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage", {}).get("completion_tokens") and d["choices"]:
            seen.append((time.time(), d["usage"]["completion_tokens"]))
    res.append((t0, seen))

t0 = time.time()
ts = [threading.Thread(target=run, args=(i,)) for i in range(conc)]
for t in ts:
    t.start()
for t in ts:
    t.join()
wall = time.time() - t0
total = sum(seen[-1][1] for _, seen in res)
per = [(seen[-1][1] - seen[0][1]) / (seen[-1][0] - seen[0][0]) for _, seen in res]
lo, hi = max(seen[0][0] for _, seen in res), min(seen[-1][0] for _, seen in res)


def tokens_at(seen, t):
    return seen[bisect.bisect_right(seen, (t, float("inf"))) - 1][1]


both = sum(tokens_at(s, hi) - tokens_at(s, lo) for _, s in res) / (hi - lo) if hi > lo else 0
print(f"conc {conc}: {total} tok in {wall:.1f}s -> {total / wall:.1f} tok/s wall, "
      f"{both:.1f} tok/s all decoding, {min(per):.1f}-{max(per):.1f} per stream, "
      f"TTFT max {max(seen[0][0] - t for t, seen in res):.2f}s")

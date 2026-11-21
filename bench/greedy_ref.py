"""Greedy answers to a fixed prompt set, saved or compared: a quick check that a change
(e.g. speculative decoding) leaves the output unchanged.

    venv/bin/python bench/greedy_ref.py save    <out.json> [port]
    venv/bin/python bench/greedy_ref.py compare <ref.json> [port]

Compare prints, per prompt, how many leading tokens match. Exact equality is not guaranteed:
a verify step runs several tokens through different kernel shapes, so greedy paths can part
after a near-tie, but they should agree for a long prefix, not diverge in the first few tokens.
"""

import json
import sys
import urllib.request

PROMPTS = [
    "Spiega in dettaglio come funziona un motore a quattro tempi.",
    "Write a Python function that merges two sorted lists, with tests.",
    "Riassumi la storia dell'Impero romano d'Occidente in dieci punti.",
    "List the planets of the solar system with one fact about each.",
    "Scrivi una email formale per chiedere un preventivo di manutenzione.",
    'Return a JSON object with keys "name", "age", "city" for a fictional person.',
]

mode, path = sys.argv[1], sys.argv[2]
port = sys.argv[3] if len(sys.argv) > 3 else "8080"


def ask(prompt):
    body = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 384,
        "logprobs": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return [t["token"] for t in r["choices"][0]["logprobs"]["content"]]


got = [ask(p) for p in PROMPTS]
if mode == "save":
    json.dump(got, open(path, "w"), ensure_ascii=False)
    print(f"saved {len(got)} answers, {sum(map(len, got))} tokens")
else:
    ref = json.load(open(path))
    for p, a, b in zip(PROMPTS, ref, got):
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"{n:4d}/{len(a):4d} tokens equal  {p[:50]}")

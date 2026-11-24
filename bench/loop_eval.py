"""How often generation falls into a repetition loop, per sampling setting.

    venv/bin/python bench/loop_eval.py cases.jsonl out.json [--port 8080] [--seeds 4] [--jobs 2]
        [--max-tokens 8192] [--set name='{"temperature":1.0}' ...]

A case is a JSON line: id, messages and any other chat field (tools, reasoning_effort).
Tool calls are never executed. A run loops when its last 4000 characters (reasoning and
text) are mostly repeated sentences; `length` counts runs that reach max_tokens.
"""

import argparse
import collections
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SETS = {
    "model": {"temperature": 1.0},
    "t0.8-trunc": {"temperature": 0.8, "top_p": 0.9, "top_k": 20, "min_p": 0.05},
    "t0.5": {"temperature": 0.5},
}


def repeated(text, tail=4000):
    """Share of the tail's sentences that occur at least 3 times in it."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text[-tail:])]
    parts = [s for s in parts if len(s) > 20]
    if len(text) < tail or not parts:
        return 0.0
    counts = collections.Counter(parts)
    return sum(n for n in counts.values() if n >= 3) / len(parts)


def run(port, max_tokens, case, name, params, seed):
    body = {k: v for k, v in case.items() if k != "id"}
    body |= {"model": "local", "max_tokens": max_tokens, "seed": seed, **params}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"},
    )  # fmt: skip
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.load(r)
    choice, msg = res["choices"][0], res["choices"][0]["message"]
    text = (msg.get("reasoning") or msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
    share = repeated(text)
    return {"case": case["id"], "set": name, "seed": seed, "finish": choice["finish_reason"],
            "prompt": res["usage"]["prompt_tokens"], "out": res["usage"]["completion_tokens"],
            "repeated": round(share, 3), "loop": share >= 0.5, "tail": text[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--set", action="append", help="name=JSON sampling fields")
    a = ap.parse_args()
    sets = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)} if a.set else SETS
    cases = [json.loads(line) for line in Path(a.cases).read_text().splitlines()]
    todo = [(c, n, p, s) for s in range(a.seeds) for c in cases for n, p in sets.items()]
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda t: run(a.port, a.max_tokens, *t), todo))
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for n in sets:
        r = [x for x in rows if x["set"] == n]
        print(f"{n:12s} loop {sum(x['loop'] for x in r)}/{len(r)}  length "
              f"{sum(x['finish'] == 'length' for x in r)}  mean out {sum(x['out'] for x in r) / len(r):.0f}")
    for x in rows:
        print(x["case"], x["set"], x["seed"], x["finish"], x["prompt"], x["out"], x["repeated"])


if __name__ == "__main__":
    main()

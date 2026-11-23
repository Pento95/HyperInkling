"""Where two runs of the same request first differ: the MoE inputs, routing ids and weights
of every layer visit, from the capture hook (moe.py), one prefill at a time.

Needs an eager server with prefix caching off and HYPERINKLING_CAPTURE_DIR=<dir> (an empty
directory), e.g. scripts/serve-try.sh with MAX_SEQS=1:

    venv/bin/python bench/diverge.py send <dir> [repeats] [port]
    venv/bin/python bench/diverge.py compare <dir> [repeats]
"""

import glob
import json
import os
import re
import sys
import urllib.request

import torch

WORDS = "the model reads every page of the report and writes a summary of each table ".split()


def _chat(port, text):
    body = {"model": "inkling-2-mini", "temperature": 0, "max_tokens": 1,
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["usage"]["prompt_tokens"]


def send(capture_dir, repeats=4, port=18030):
    flag = os.path.join(capture_dir, "ACTIVE")
    open(flag, "w").close()
    try:
        for _ in range(int(repeats)):
            print(_chat(port, " ".join(WORDS * 24)), "prompt tokens")
    finally:
        os.remove(flag)
    _chat(port, "ok")  # the first forward without ACTIVE writes the buffers


def compare(capture_dir, repeats=4):
    """Visits in file order are layer by layer; the first difference in execution order is
    the smallest (layer, pass) inside the loop's order, so every differing visit is listed."""
    repeats = int(repeats)
    names = sorted({re.sub(r"_\d{4}\.pt$", "", f)
                    for f in glob.glob(os.path.join(capture_dir, "*_pass*_*.pt"))})
    if not names:
        sys.exit(f"no captures in {capture_dir}: did `send` finish?")
    for name in names:
        parts = [torch.load(f) for f in sorted(glob.glob(f"{name}_[0-9]*.pt"))]
        line = []
        for key in ("x", "ids", "w"):
            t = torch.cat([p[key] for p in parts])
            n = t.shape[0] // repeats
            if t.shape[0] % repeats:
                sys.exit(f"{name}: {t.shape[0]} rows do not split in {repeats}")
            runs = t.view(repeats, n, *t.shape[1:])
            # Against run 2: run 1 may still be compiling or warming caches.
            differ = [(runs[i] != runs[1]).flatten(1).any(1) for i in range(repeats)]
            rows = [int(d.sum()) for d in differ]
            first = [int(d.nonzero()[0]) if d.any() else None for d in differ]
            line.append(f"{key} rows {rows} first {first}")
        print(os.path.basename(name), "|", " | ".join(line))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"send": send, "compare": compare}[cmd](*args)

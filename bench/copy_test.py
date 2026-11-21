"""Draft acceptance where the answer reproduces the prompt: copying a 120-line table from
the first third of a long document (needs the drafter to see the far context), and
counting (needs only recent tokens). Prints tokens per verify step and how much of the copy is verbatim.

    venv/bin/python bench/copy_test.py [port] [context tokens ~ chars/4]
"""

import difflib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

PORT = sys.argv[1] if len(sys.argv) > 1 else "8080"
CTX_CHARS = int(sys.argv[2]) * 4 if len(sys.argv) > 2 else 80000
URL = f"http://127.0.0.1:{PORT}"
src = Path.home() / ".cache/evalscope/aa_lcr/lcr/Industry_Reports"
text = "\n\n".join(f.read_text(errors="ignore") for f in sorted(src.rglob("*.txt")))
body = text[:CTX_CHARS]
cut = body.rfind("\n\n", 0, len(body) // 3) + 2  # the table sits in the first third
# 120 invoice lines: structured text the model copies to the end (prose it stops early,
# tables of contents it fills with dots).
rng = random.Random(0)
passage = "\n".join(
    f"{i:03d} | {rng.choice('ABCDEFGHKLMNPRSTVZ')}{rng.choice('ABCDEFGHKLMNPRSTVZ')}-"
    f"{rng.randint(1000, 9999)} | qty {rng.randint(1, 48)} | EUR {rng.randint(1, 999)},"
    f"{rng.randint(0, 99):02d}" for i in range(1, 121))
# The table is marked, so the model copies instead of searching for it.
doc = f"{body[:cut]}<<<COPY>>>\n{passage}\n<<<END>>>\n\n{body[cut:]}"
anchor = passage.split("\n")[0] + "\n"


def metrics():
    t = urllib.request.urlopen(f"{URL}/metrics", timeout=10).read().decode()
    return {k: sum(float(x.split()[-1]) for x in t.splitlines()
                   if x.startswith(f"vllm:spec_decode_num_{k}_total{{"))
            for k in ("drafts", "accepted_tokens")}


tests = {
    "copy-far": f"{doc}\n\n---\nReproduce exactly the table between <<<COPY>>> and "
                "<<<END>>> in the document above, all 120 lines. Output only the table.",
    "count": "Count from 1 to 300, numbers separated by commas, nothing else.",
}
def stream(prompt, max_tokens, prefill=None):
    """(reasoning text, content text, output tokens, content seconds). `prefill` starts
    the answer, skipping the reasoning (the model otherwise searches the document in it)."""
    messages = [{"role": "user", "content": prompt}]
    body = {"model": "inkling-2-mini", "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True}}
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
        body.update(continue_final_message=True, add_generation_prompt=False)
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    reasoning, content, times, tokens = [], [], [], 0
    for line in urllib.request.urlopen(req, timeout=1800):
        if not line.startswith(b"data: {"):
            continue
        chunk = json.loads(line[6:])
        if chunk.get("usage"):
            tokens = chunk["usage"]["completion_tokens"]
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            content.append(delta["content"])
            times.append(time.time())
        elif delta.get("reasoning_content") or delta.get("reasoning"):
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning"))
    secs = times[-1] - times[0] if len(times) > 1 else 0.0
    # A chunk carries several tokens when drafts are accepted: count tokens, not chunks.
    return "".join(reasoning), "".join(content), tokens, secs


for name, prompt in tests.items():
    m0 = metrics()
    prefill = anchor if name == "copy-far" else None
    reasoning, out, tokens, secs = stream(prompt, 1500 if prefill else 4000, prefill)
    out = (prefill or "") + out
    m1 = metrics()
    d, a = m1["drafts"] - m0["drafts"], m1["accepted_tokens"] - m0["accepted_tokens"]
    line = (f"{name:8s} whole answer {1 + a / max(d, 1):.2f} tok/step; content "
            f"{tokens} tokens in {secs:.1f} s ({tokens / max(secs, 1e-6):.0f} tok/s), "
            f"reasoning {len(reasoning)} chars")
    if name == "copy-far":
        sm = difflib.SequenceMatcher(None, out, passage, autojunk=False)
        line += f", copy verbatim {sum(b.size for b in sm.get_matching_blocks()) / max(len(out), 1):.0%}"
    print(line)

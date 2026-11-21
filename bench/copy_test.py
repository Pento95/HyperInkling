"""Draft acceptance where the answer reproduces the prompt: copying a passage from a
long document (needs the drafter to see the far context), and counting (needs only
recent tokens). Prints tokens per verify step and how much of the copy is verbatim.

    venv/bin/python bench/copy_test.py [port] [context tokens ~ chars/4]
"""

import difflib
import json
import sys
import time
import urllib.request
from pathlib import Path

PORT = sys.argv[1] if len(sys.argv) > 1 else "8080"
CTX_CHARS = int(sys.argv[2]) * 4 if len(sys.argv) > 2 else 80000
URL = f"http://127.0.0.1:{PORT}"
src = Path.home() / ".cache/evalscope/aa_lcr/lcr/Industry_Reports"
text = "\n\n".join(f.read_text(errors="ignore") for f in sorted(src.rglob("*.txt")))
doc = text[:CTX_CHARS]
cut = doc.rfind("\n\n", 0, len(doc) // 3)  # a passage from the first third
passage = doc[cut:cut + 2500].strip()
anchor = passage[:80]


def metrics():
    t = urllib.request.urlopen(f"{URL}/metrics", timeout=10).read().decode()
    return {k: sum(float(x.split()[-1]) for x in t.splitlines()
                   if x.startswith(f"vllm:spec_decode_num_{k}_total{{"))
            for k in ("drafts", "accepted_tokens")}


tests = {
    "copy-far": f"{doc}\n\n---\nIn the document above, find the paragraph that starts with "
                f'"{anchor}" and reproduce it and the text that follows it exactly, '
                "word for word, for about 400 words. Output only the copied text.",
    "count": "Count from 1 to 300, numbers separated by commas, nothing else.",
}
def stream(prompt, max_tokens, prefill=None):
    """(reasoning text, content text, content tokens, content seconds). `prefill` starts
    the answer, skipping the reasoning (the model otherwise searches the document in it)."""
    messages = [{"role": "user", "content": prompt}]
    body = {"model": "inkling-2-mini", "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "reasoning_effort": "none", "stream": True}
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
        body.update(continue_final_message=True, add_generation_prompt=False)
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    reasoning, content, times = [], [], []
    for line in urllib.request.urlopen(req, timeout=1800):
        if not line.startswith(b"data: {"):
            continue
        delta = json.loads(line[6:])["choices"][0]["delta"]
        if delta.get("content"):
            content.append(delta["content"])
            times.append(time.time())
        elif delta.get("reasoning_content") or delta.get("reasoning"):
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning"))
    secs = times[-1] - times[0] if len(times) > 1 else 0.0
    return "".join(reasoning), "".join(content), len(times), secs


for name, prompt in tests.items():
    m0 = metrics()
    prefill = anchor if name == "copy-far" else None
    reasoning, out, chunks, secs = stream(prompt, 1500 if prefill else 4000, prefill)
    out = (prefill or "") + out
    m1 = metrics()
    d, a = m1["drafts"] - m0["drafts"], m1["accepted_tokens"] - m0["accepted_tokens"]
    line = (f"{name:8s} whole answer {1 + a / max(d, 1):.2f} tok/step; content "
            f"{chunks} chunks in {secs:.1f} s ({chunks / max(secs, 1e-6):.0f} chunks/s), "
            f"reasoning {len(reasoning)} chars")
    if name == "copy-far":
        sm = difflib.SequenceMatcher(None, out, doc[cut:cut + len(out) + 500], autojunk=False)
        line += f", copy verbatim {sum(b.size for b in sm.get_matching_blocks()) / max(len(out), 1):.0%}"
    print(line)

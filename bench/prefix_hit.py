"""Prefix-cache hits of requests that share a system prompt and differ in the page, and
whether a hit answers like a cold prefill.

Sequential requests per system-prompt length: the same system prompt, then a different DocVQA
page each. The first two find nothing; from the third on the shared prefix must be a hit. Then
unseen pages are answered warm and cold (a fresh `cache_salt`): the token
distributions (mean total variation, up to the first different token) must differ no more than
two cold runs do.

    venv/bin/python bench/prefix_hit.py [port]        # needs an otherwise idle server
"""

import base64
import json
import math
import os
import re
import sys
import time
import urllib.request
import uuid

import pyarrow.parquet as pq
from extract_bench import PAGES, system_prompt

PORT = sys.argv[1] if len(sys.argv) > 1 else "8080"
URL = f"http://127.0.0.1:{PORT}"
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("VLLM_API_KEY"):
    HEADERS["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"
RUN = uuid.uuid4().hex  # a system prompt no earlier run has cached


def hits():
    req = urllib.request.Request(f"{URL}/metrics", headers=HEADERS)
    text = urllib.request.urlopen(req, timeout=10).read().decode()
    return sum(float(v) for v in re.findall(r"^vllm:prefix_cache_hits_total\{[^}]*\} (\S+)",
                                            text, re.M))


def ask(image, lines, salt=None, max_tokens=1):
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    body = {"model": "inkling-2-mini", "temperature": 0, "max_tokens": max_tokens, "logprobs": True,
            "top_logprobs": 20, "reasoning_effort": "low", "messages": [
                {"role": "system", "content": f"{RUN}-{lines}\n" + "\n".join(system_prompt().split("\n")[:lines])},
                {"role": "user", "content": [
                    {"type": "text", "text": "Ti mostro 1 pagina."},
                    {"type": "image_url", "image_url": {"url": url}}]}]}
    if salt:
        body["cache_salt"] = salt
    before, t0 = hits(), time.time()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(), HEADERS)
    r = json.load(urllib.request.urlopen(req, timeout=300))
    ms = (time.time() - t0) * 1e3
    time.sleep(0.3)
    tokens = [(p["token"], {t["token"]: t["logprob"] for t in p["top_logprobs"]})
              for p in r["choices"][0]["logprobs"]["content"]]
    return r["usage"]["prompt_tokens"], int(hits() - before), ms, tokens


def distance(a, b):
    """(mean total variation, tokens compared)"""
    tv = []
    for (ta, pa), (tb, pb) in zip(a, b):
        tv.append(sum(abs(math.exp(pa.get(t, -99)) - math.exp(pb.get(t, -99)))
                      for t in {*pa, *pb}) / 2)
        if ta != tb:
            break
    return f"{sum(tv) / max(len(tv), 1):.4f} ({len(tv)} tokens)"


def main():
    rows = pq.ParquetFile(PAGES).read_row_group(0, columns=["image"]).to_pylist()
    images, seen = [], set()
    for row in rows:
        data = row["image"]["bytes"]
        if data not in seen:
            seen.add(data)
            images.append(data)
    for lines in (40, 68, 95, 123):
        print(f"system prompt, {lines} lines:")
        for image in images[:5]:
            prompt, hit, ms, _ = ask(image, lines)
            print(f"  prompt {prompt}, hit {hit}, {ms:.0f} ms")
        for image in images[5:9]:
            _, hit, _, warm = ask(image, lines, max_tokens=48)
            cold, cold2 = (ask(image, lines, uuid.uuid4().hex, 48)[3] for _ in range(2))
            print(f"  hit {hit}: warm-cold {distance(warm, cold)}, "
                  f"cold-cold {distance(cold, cold2)}")


if __name__ == "__main__":
    main()

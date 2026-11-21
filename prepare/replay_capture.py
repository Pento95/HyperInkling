"""Replay the calibration transcripts through a capture server (prefill only).

The server runs with HYPERINKLING_CAPTURE_DIR set and in eager mode; this
script creates `<dir>/ACTIVE`, sends every transcript of the corpus with the
final assistant turn teacher-forced (max_tokens 1), and removes ACTIVE.

    venv/bin/python prepare/replay_capture.py data/calib/calib.jsonl data/capture
"""

import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:18030/v1/chat/completions"


def send(rec):
    msgs = rec["messages"]
    body = {"model": "inkling-2-mini", "messages": msgs, "max_tokens": 1,
            "temperature": 0}
    if rec.get("tools"):
        body["tools"] = rec["tools"]
    if rec.get("effort") is not None:
        body["reasoning_effort"] = rec["effort"]
    if msgs[-1]["role"] == "assistant":
        body.update(continue_final_message=True, add_generation_prompt=False)
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        return r["usage"]["prompt_tokens"]
    except urllib.error.HTTPError as e:
        print(f"skipped {rec['source']}: HTTP {e.code} {e.read()[:160]!r}")
        return 0


def main(corpus, capture_dir):
    recs = [json.loads(line) for line in open(corpus)]
    flag = os.path.join(capture_dir, "ACTIVE")
    open(flag, "w").close()
    try:
        with ThreadPoolExecutor(2) as ex:
            tokens = list(ex.map(send, recs))
    finally:
        os.remove(flag)
    print(f"{len(recs)} transcripts, {sum(tokens)} prompt tokens replayed")


if __name__ == "__main__":
    main(*sys.argv[1:])

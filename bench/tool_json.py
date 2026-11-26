"""Tool calls the model breaks without HYPERINKLING_TOOL_GRAMMAR: an invoice whose array of
objects is followed by another field. Also tool_choice, and a long streamed argument next to
/health. Only validity is scored.

    venv/bin/python bench/tool_json.py [port] [repeats]
"""

import json
import sys
import threading
import time
import urllib.request
from collections import Counter

PORT = sys.argv[1] if len(sys.argv) > 1 else "18030"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
URL = f"http://127.0.0.1:{PORT}"


def tool(name, description, **props):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": list(props)}}}


STRING, NUMBER = {"type": "string"}, {"type": "number"}
INVOICE = tool("create_invoice", "Create a draft invoice.", customer=STRING, items={
    "type": "array", "items": {
        "type": "object", "required": ["description", "qty", "unit_price"],
        "properties": {"description": STRING, "qty": {"type": "integer"}, "unit_price": NUMBER}},
}, vat_pct=NUMBER)
LOOKUP = tool("get_weather", "Weather in a city.", city=STRING)
WRITE = tool("write_file", "Write a text file.", path=STRING, content=STRING)
ASK = ("Make a draft invoice for Jan de Vries: one Batavus Finez E-go at 700 euro and two bike "
       "locks at 35 euro each. Used bikes go under the margin scheme, so use 0% VAT.")


def request(prompt, tools, **kw):
    body = {"model": "inkling-2-mini", "messages": [{"role": "user", "content": prompt}],
            "tools": tools, "max_tokens": 4000, "temperature": 0.6, **kw}
    return urllib.request.urlopen(urllib.request.Request(
        f"{URL}/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=600)


def calls(prompt, tools, **kw):
    """The calls of a finished answer; None when it hit max_tokens (a reasoning loop)."""
    choice = json.load(request(prompt, tools, **kw))["choices"][0]
    if choice["finish_reason"] == "length":
        return None
    return [c["function"] for c in choice["message"].get("tool_calls") or []]


def verdict(got, ok):
    if got is None:
        return "cut at max_tokens"
    if not got:
        return "no call"
    return "valid" if ok(got) else f"NOT VALID: {got}"


def complete(call, spec):
    """Whether the arguments are JSON with every required field."""
    try:
        return set(spec["function"]["parameters"]["required"]) <= set(json.loads(call["arguments"]))
    except ValueError:
        return False


def invoice(strict):
    spec = json.loads(json.dumps(INVOICE))
    if strict:
        spec["function"]["strict"] = True
    return verdict(calls(ASK, [spec]), lambda got: all(complete(c, spec) for c in got))


def forced(choice, names):
    got = calls("Hello, how are you?", [INVOICE, LOOKUP], tool_choice=choice)
    specs = {t["function"]["name"]: t for t in (INVOICE, LOOKUP)}
    return verdict(got, lambda got: all(
        c["name"] in names and complete(c, specs[c["name"]]) for c in got))


def long_stream():
    """(verdict, characters, slowest /health in s) while one long argument streams."""
    slowest, done = [0.0], threading.Event()

    def probe():
        while not done.wait(0.5):
            t = time.time()
            urllib.request.urlopen(f"{URL}/health", timeout=120).read()
            slowest[0] = max(slowest[0], time.time() - t)

    thread = threading.Thread(target=probe)
    thread.start()
    args, end = "", None
    try:
        for line in request("Write a story of about 800 words to story.txt with write_file, all "
                            "in one call.", [WRITE], stream=True, max_tokens=16000,
                            tool_choice="required"):
            if line.startswith(b"data: {"):
                for choice in json.loads(line[6:])["choices"]:
                    end = choice["finish_reason"] or end
                    for c in choice["delta"].get("tool_calls") or []:
                        args += (c.get("function") or {}).get("arguments") or ""
    finally:
        done.set()
        thread.join()
    got = None if end == "length" else [{"arguments": args}] if args else []
    ok = verdict(got, lambda got: complete(got[0], WRITE))
    return ok[:200], len(args), slowest[0]


named = {"type": "function", "function": {"name": "get_weather"}}
for label, test in (("invoice", lambda: invoice(False)), ("invoice, strict", lambda: invoice(True)),
                    ("required", lambda: forced("required", ("create_invoice", "get_weather"))),
                    ("named tool", lambda: forced(named, ("get_weather",)))):
    counts = Counter(test() for _ in range(N))
    print(f"{label:16s} {counts.pop('valid', 0)}/{N} valid", dict(counts) or "", flush=True)
ok, chars, health = long_stream()
print(f"{'long stream':16s} {ok}, {chars} characters, slowest /health {health * 1e3:.0f} ms")

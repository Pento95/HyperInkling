"""Decode speed and draft acceptance per kind of text (speculative decoding A/B).

Each category runs its prompts one at a time (concurrency 1), then all prompts run together
(concurrency 6). Decode speed is per stream, from the streamed tokens after the first one, so
prefill does not count. Acceptance comes from the server's /metrics over the same requests.

    venv/bin/python bench/spec_bench.py [port] [max_tokens] [temperature|user] [--only c1|c6]

Categories: chat (Italian and English), code, struct (JSON out of a text), prose, doc (a
question over a ~3k-token AA-LCR excerpt, answer quoting it).
"""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PORT = args[0] if len(args) > 0 else "8080"
MAX_TOKENS = int(args[1]) if len(args) > 1 else 768
# A number, or "user": 0.2 for struct, 0.5 for everything else.
TEMP = args[2] if len(args) > 2 else "0"
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
URL = f"http://127.0.0.1:{PORT}"

doc_dir = Path.home() / ".cache/evalscope/aa_lcr/lcr/Legal/legal_eu_ai"
doc = ""
for f in sorted(doc_dir.glob("*.txt")):
    doc += f.read_text(errors="ignore")
doc = doc[:12000]

PROMPTS = {
    "chat": [
        "Ciao! Mi spieghi in modo semplice cos'è l'inflazione e perché le banche centrali alzano i tassi?",
        "What are some good habits for staying productive when working from home?",
    ],
    "code": [
        "Write a Python class implementing an LRU cache with get/put in O(1), with docstrings and a few unit tests.",
        "Scrivi una funzione TypeScript che fa il parsing di una data in formato DD/MM/YYYY e la valida, con test.",
    ],
    "struct": [
        "Estrai i dati in JSON con chiavi nome, cognome, data_nascita, citta, email, telefono da questo testo: "
        "'Mario Rossi, nato il 12/03/1985 a Torino, scrivere a mario.rossi@example.com o chiamare il 333 1234567. "
        "Laura Bianchi, nata il 01/09/1990 a Bari, laura.b@example.org, 340 7654321.' Rispondi solo con il JSON.",
        "Convert this CSV to a JSON array of objects, nothing else:\nid,name,price,stock\n1,Widget,9.99,120\n"
        "2,Gadget,24.50,15\n3,Doohickey,3.75,900\n4,Thingamajig,49.00,7\n5,Whatsit,12.30,64",
    ],
    "prose": [
        "Scrivi un racconto breve su un vecchio guardiano del faro in Sardegna che riceve una lettera inattesa.",
        "Write a short essay on how the printing press changed European society.",
    ],
    "doc": [
        f"{doc}\n\n---\nIn base al testo qui sopra, quali obblighi impone l'AI Act ai fornitori di sistemi ad alto "
        "rischio? Cita le frasi del testo che lo dicono.",
        f"{doc}\n\n---\nList every date and deadline mentioned in the text above, quoting the sentence for each.",
    ],
}


def metrics():
    text = urllib.request.urlopen(f"{URL}/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        for k in ("drafts", "draft_tokens", "accepted_tokens"):
            if line.startswith(f"vllm:spec_decode_num_{k}_total{{"):
                out[k] = out.get(k, 0.0) + float(line.split()[-1])
    return out


def temp_for(cat):
    if TEMP == "user":
        return 0.2 if cat == "struct" else 0.5
    return float(TEMP)


def run(prompt, res, temp=0.0):
    body = json.dumps({
        "model": "inkling-2-mini", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": temp, "seed": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0, first, last, toks = time.time(), None, None, 0
    for line in urllib.request.urlopen(req, timeout=1200):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            toks = d["usage"]["completion_tokens"]
        elif d["choices"]:
            last = time.time()
            first = first or last
    res.append((first - t0, toks, (toks - 1) / max(last - first, 1e-6)))


def batch(prompts, temps):
    m0, res = metrics(), []
    ts = [threading.Thread(target=run, args=(p, res, t)) for p, t in zip(prompts, temps)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall, m1 = time.time() - t0, metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("drafts", "draft_tokens", "accepted_tokens")}
    acc = f"{1 + d['accepted_tokens'] / d['drafts']:.2f} tok/step" if d["drafts"] else "no spec"
    return res, wall, acc


run(PROMPTS["chat"][1], [])  # warm-up
print(f"port {PORT}, max_tokens {MAX_TOKENS}, temperature {TEMP}")
if ONLY != "c6":
    for cat, prompts in PROMPTS.items():
        rows, acc = [], []
        for p in prompts:
            res, _, a = batch([p], [temp_for(cat)])
            rows += res
            acc.append(a)
        tps = [r[2] for r in rows]
        print(f"c1 {cat:7s} {sum(tps) / len(tps):6.1f} tok/s per stream  ({', '.join(acc)}; "
              f"{sum(r[1] for r in rows)} tokens)")
if ONLY != "c1":
    pairs = [(p, temp_for(c)) for c, ps in PROMPTS.items() for p in ps][:6]
    res, wall, acc = batch([p for p, _ in pairs], [t for _, t in pairs])
    tps = [r[2] for r in res]
    print(f"c6 mixed   {sum(tps) / len(tps):6.1f} tok/s per stream, "
          f"{sum(r[1] for r in res) / wall:6.1f} aggregate ({acc})")

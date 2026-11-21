"""A document-extraction workload on public data.

Each request: one shared Italian system prompt (~3.5k tokens, served from the prefix cache
after the first), 1-3 document pages (DocVQA validation, 1.76 per request on average) and an
Italian instruction; half ask for strict JSON (temperature 0.2), half for an Italian summary
(temperature 0.5), with reasoning_effort low. Outputs end naturally (reasoning included): ~6.4k prompt tokens,
~1.8k output tokens, most of the time in decode.

    venv/bin/python bench/quality_battery.py download     # the DocVQA pages, once
    venv/bin/python bench/extract_bench.py [--port 8080] [--jobs 6] [--seconds 600]

Closed loop: `--jobs` requests in flight for `--seconds`, then the ones running finish.
Prints client-side latency and bench/metrics_window.py's server-side view of the window.
"""

import argparse
import base64
import json
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = HERE / "quality-data/docvqa-single-page-questions/data/validation-00000-of-00004.parquet"

FIELDS = [
    "ragione sociale del mittente", "partita IVA del mittente", "codice fiscale", "indirizzo",
    "data del documento", "numero del documento", "importo imponibile", "aliquota IVA",
    "importo totale", "valuta", "data di scadenza", "modalità di pagamento", "IBAN",
    "nome del referente", "telefono", "email", "oggetto", "riferimento d'ordine",
    "quantità", "prezzo unitario", "sconto", "firma presente", "timbro presente", "lingua",
]
RULES = [
    "formato AAAA-MM-GG", "solo cifre, senza spazi", "come scritto nel documento",
    "numero con il punto come separatore decimale", "testo libero, al massimo 20 parole",
    "true o false", "codice ISO a tre lettere",
]


def system_prompt():
    rng = random.Random(0)
    lines = [
        "Sei un assistente che estrae dati da documenti aziendali (fatture, lettere, moduli, "
        "contratti, report) per un gestionale italiano. Leggi ogni pagina con attenzione, non "
        "inventare valori, e quando un dato manca usa null. Rispondi sempre in italiano.",
        "",
        "Catalogo dei campi (nome, significato, formato):",
    ]
    for i in range(120):
        f = FIELDS[i % len(FIELDS)]
        lines.append(f"- campo_{i:03d}: {f} (variante {i // len(FIELDS) + 1}); "
                     f"{rng.choice(RULES)}; se assente o illeggibile usa null.")
    return "\n".join(lines)


SCHEMA = {
    "type": "object",
    "properties": {
        "tipo_documento": {"type": "string"},
        "mittente": {"type": ["string", "null"]},
        "destinatario": {"type": ["string", "null"]},
        "data": {"type": ["string", "null"]},
        "importi": {"type": "array", "items": {"type": "object", "properties": {
            "descrizione": {"type": "string"}, "valore": {"type": "number"}},
            "required": ["descrizione", "valore"]}},
        "riassunto": {"type": "string"},
    },
    "required": ["tipo_documento", "mittente", "destinatario", "data", "importi", "riassunto"],
}


def load_pages(n=300):
    import pyarrow.parquet as pq

    out, seen = [], set()
    for batch in pq.ParquetFile(PAGES).iter_batches(batch_size=64, columns=["image", "other_metadata"]):
        for r in batch.to_pylist():
            doc = r["other_metadata"]["image"]
            if doc in seen:  # several questions per page: one copy each
                continue
            seen.add(doc)
            out.append("data:image/png;base64," + base64.b64encode(r["image"]["bytes"]).decode())
            if len(out) == n:
                return out
    return out


def make_request(rng, pages, system, effort=None):
    k = rng.choices([1, 2, 3], weights=[40, 44, 16])[0]
    images = [{"type": "image_url", "image_url": {"url": u}} for u in rng.sample(pages, k)]
    body = {"model": "inkling-2-mini", "max_tokens": 8192, "stream": True,
            "stream_options": {"include_usage": True}}
    if effort:
        body["reasoning_effort"] = effort
    if rng.random() < 0.5:
        text = ("Estrai i dati principali di queste pagine nel formato JSON richiesto. "
                "Il riassunto in italiano, 3-5 frasi.")
        body["temperature"] = 0.2
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "documento", "schema": SCHEMA}}
    else:
        text = ("Descrivi in italiano il contenuto di queste pagine: che documento è, chi lo "
                "invia, a chi, i dati e gli importi rilevanti, e cosa richiede al destinatario.")
        body["temperature"] = 0.5
    body["messages"] = [{"role": "system", "content": system},
                        {"role": "user", "content": [*images, {"type": "text", "text": text}]}]
    return body


def one(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first, usage, finish = None, None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices"):
                first = first or time.time()
                finish = d["choices"][0].get("finish_reason") or finish
    end = time.time()
    out = usage["completion_tokens"]
    return {"ttft": first - t0, "e2e": end - t0, "prompt": usage["prompt_tokens"], "out": out,
            "decode": (out - 1) / (end - first) if end > first else 0.0,
            "cut": finish == "length"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default="8080")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default="low",
                    help="reasoning_effort (none, low, medium, high; omitted with an empty string)")
    args = ap.parse_args()

    pages, system = load_pages(), system_prompt()
    rng, lock, res = random.Random(args.seed), threading.Lock(), []
    one(args.port, make_request(random.Random(-1), pages, system, args.effort))  # warm the prefix
    window = subprocess.Popen([sys.executable, str(HERE / "metrics_window.py"),
                               str(args.seconds), args.port], stdout=subprocess.PIPE, text=True)
    stop = time.time() + args.seconds

    def worker():
        while time.time() < stop:
            with lock:
                body = make_request(rng, pages, system, args.effort)
            r = one(args.port, body)
            with lock:
                res.append(r)

    ts = [threading.Thread(target=worker) for _ in range(args.jobs)]
    [t.start() for t in ts]
    [t.join() for t in ts]

    def q(key, p):
        v = sorted(r[key] for r in res)
        return v[min(len(v) - 1, int(p * len(v)))]

    print(f"{len(res)} requests, {args.jobs} in flight: prompt {statistics.mean(r['prompt'] for r in res):.0f} "
          f"tokens, output {statistics.mean(r['out'] for r in res):.0f} tokens (mean), "
          f"{sum(r['cut'] for r in res)} cut at max_tokens")
    print(f"TTFT p50 {q('ttft', .5):.2f} s, p90 {q('ttft', .9):.2f} s; end to end p50 "
          f"{q('e2e', .5):.1f} s, p90 {q('e2e', .9):.1f} s; decode per request "
          f"{statistics.mean(r['decode'] for r in res):.0f} tok/s")
    print("--- server, measured window ---")
    print(window.communicate()[0], end="")


if __name__ == "__main__":
    main()

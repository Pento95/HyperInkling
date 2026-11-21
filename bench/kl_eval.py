"""Teacher-forced KL of a quantized variant against the reference checkpoint.

    gen     the server (reference) answers PROMPTS greedily; saves the texts
    score   the server rereads those texts with prompt_logprobs; saves top-k
    compare two score files: mean / p99 / max KL(ref || variant) over the
            reference's top-k, and top-1 agreement

    venv/bin/python bench/kl_eval.py gen bench/kl/corpus.json
    venv/bin/python bench/kl_eval.py score bench/kl/corpus.json bench/kl/ref.json
    venv/bin/python bench/kl_eval.py compare bench/kl/ref.json bench/kl/int4.json

KL over the reference's top-k only: the mass outside it is lumped into one
bucket on both sides, so the number is a slight underestimate of the full KL.

One request at a time (KL_JOBS=1, with a MAX_SEQS=1 server as bench/kl_sweep.sh boots it).
Decide on the median: the same request scored twice still differs slightly (from the first
positions on), and where that flips an expert the difference carries down the sequence, so a
few positions dominate the mean. The original against itself: mean 0.060, median 0.0003.
"""

import json
import math
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = f"http://127.0.0.1:{os.environ.get('PORT', '18030')}/v1"
JOBS = int(os.environ.get("KL_JOBS", "1"))
MODEL = "inkling-2-mini"
TOPK = 20
# Inkling2's image pad token id (processor_config.json `image_token`,
# <|unused_200054|>). prompt_logprobs are keyed by token id and always hold
# the prompt's own token, so a position whose dict has it is an image pad.
_IMAGE_PAD_ID = "200054"

PROMPTS = [
    # Italian documents and extraction: the served workload.
    "Riassumi in italiano, in modo dettagliato, il contenuto tipico di un contratto di locazione ad uso abitativo 4+4.",
    "Estrai da questo testo i dati in JSON (ragione sociale, partita IVA, data, importo totale): 'Fattura n. 1287 del 14/03/2026 emessa da Officine Meccaniche Brambilla S.r.l., P.IVA 03481920163, per un totale di EUR 12.480,50 IVA inclusa.' Poi spiega ogni campo.",
    "Scrivi una lettera formale a un comune italiano per chiedere l'accesso agli atti di una pratica edilizia, citando la legge 241/1990.",
    "Spiega le differenze tra S.r.l., S.p.A. e società in nome collettivo nel diritto italiano.",
    "Descrivi come si legge un bilancio d'esercizio italiano: stato patrimoniale, conto economico, nota integrativa.",
    "Racconta la storia della Repubblica di Venezia dalle origini alla caduta nel 1797.",
    "Spiega il funzionamento della fatturazione elettronica in Italia tramite il Sistema di Interscambio.",
    "Scrivi una guida pratica per preparare la dichiarazione dei redditi con il modello 730.",
    "Analizza i temi principali dei Promessi Sposi di Alessandro Manzoni.",
    "Descrivi una ricetta tradizionale del risotto alla milanese con tutti i passaggi e i consigli.",
    # English.
    "Explain how a modern SSD's flash translation layer works: mapping tables, garbage collection, wear leveling and TRIM.",
    "Write a detailed explanation of how transformers use attention, with the math.",
    "Describe the causes and consequences of the 2008 financial crisis.",
    "Explain TCP congestion control (slow start, AIMD, CUBIC, BBR) step by step.",
    # Code.
    "Write a complete Python module implementing an LRU cache with TTL expiry, thread safety and type hints, with pytest tests.",
    "Implement a thread-safe bounded blocking queue in Rust and explain the design.",
    "Write a SQL schema and queries for a library management system, with explanations.",
    "Write a TypeScript function that parses ISO 8601 durations, with tests.",
]


def _post(path, body):
    req = urllib.request.Request(
        f"{URL}/{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def gen(out):
    def one(p):
        r = _post("chat/completions", {
            "model": MODEL, "messages": [{"role": "user", "content": p}],
            "max_tokens": 512, "temperature": 0,
        })
        return {"prompt": p, "answer": r["choices"][0]["message"]["content"]}

    with ThreadPoolExecutor(6) as ex:
        corpus = list(ex.map(one, PROMPTS))
    json.dump(corpus, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(corpus)} texts -> {out}")


def _load_corpus(path):
    """`gen`'s JSON list of {prompt, answer}, or `prepare/build_calib.py`'s
    JSONL transcripts (images, tools, reasoning) as request bodies."""
    if path.endswith(".jsonl"):
        out = []
        for line in open(path):
            rec = json.loads(line)
            body = {"messages": rec["messages"]}
            if rec.get("tools"):
                body["tools"] = rec["tools"]
            if rec.get("effort") is not None:
                body["reasoning_effort"] = rec["effort"]
            out.append(body)
        return out
    return [{"messages": [{"role": "user", "content": it["prompt"]},
                          {"role": "assistant", "content": it["answer"]}]}
            for it in json.load(open(path))]


def score(corpus_path, out):
    corpus = _load_corpus(corpus_path)

    def one(body):
        # The chat template renders the prompt; the answer is teacher-forced.
        r = _post("chat/completions", {
            "model": MODEL, **body,
            "max_tokens": 1, "temperature": 0, "prompt_logprobs": TOPK,
            "continue_final_message": True, "add_generation_prompt": False,
        })
        rows = []
        for pos in r["prompt_logprobs"]:
            if pos is None:
                rows.append(None)
                continue
            rows.append({k: v["logprob"] for k, v in pos.items()})
        return rows

    with ThreadPoolExecutor(JOBS) as ex:
        scores = list(ex.map(one, corpus))
    json.dump(scores, open(out, "w"))
    print(f"{sum(len(s) for s in scores)} positions -> {out}")


def compare(ref_path, var_path, corpus_path=None):
    """With the `.jsonl` corpus, also a breakdown per record source."""
    ref, var = json.load(open(ref_path)), json.load(open(var_path))
    if corpus_path:
        sources = [json.loads(line)["source"] for line in open(corpus_path)]
        for src in sorted(set(sources)):
            keep = [i for i, s in enumerate(sources) if s == src]
            print(f"  {src:14s}", end="")
            _compare([ref[i] for i in keep], [var[i] for i in keep])
        print(f"  {'ALL':14s}", end="")
    _compare(ref, var)


def _compare(ref, var):
    kls, agree, n = [], 0, 0
    for rs, vs in zip(ref, var):
        assert len(rs) == len(vs)
        for r, v in zip(rs, vs):
            if r is None or v is None:
                continue
            if _IMAGE_PAD_ID in r:
                continue  # inside an image: no text to predict
            top = sorted(r, key=r.get, reverse=True)[:TOPK]
            p = [math.exp(r[t]) for t in top]
            floor = min(v.values())  # unseen in the variant's top-k: at most this
            q = [math.exp(v.get(t, floor)) for t in top]
            rest_p, rest_q = max(1e-9, 1 - sum(p)), max(1e-9, 1 - sum(q))
            kl = sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)
            kl += rest_p * math.log(rest_p / rest_q)
            kls.append(max(kl, 0.0))
            agree += top[0] == max(v, key=v.get)
            n += 1
    kls.sort()
    mean = sum(kls) / n
    print(f"{n} positions: KL mean {mean:.4f}  median {kls[n // 2]:.4f}  "
          f"p99 {kls[int(n * 0.99)]:.4f}  max {kls[-1]:.3f}  top-1 agree {agree / n:.2%}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"gen": gen, "score": score, "compare": compare}[cmd](*args)

"""Prompts of varied length (log-uniform, 256 to 64k tokens) for `evalscope perf --dataset openqa`.

Text comes from the AA-LCR documents in evalscope's cache (real English prose: reports, legal,
academia), with an Italian instruction at the end. Each prompt starts at a different document and
offset, so the prefix cache does not serve one prompt from another.

    venv/bin/python -I bench/make_varlen_prompts.py [out.jsonl] [count] [max tokens] [seed]
"""

import json
import math
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
SRC = Path.home() / ".cache/evalscope/aa_lcr/lcr"
INSTR = (
    "\n\n---\nLeggi il testo qui sopra e rispondi in italiano: riassumi i punti principali, "
    "poi elenca tre domande a cui il testo non risponde."
)

out = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "bench/perf/varlen.jsonl")
count = int(sys.argv[2]) if len(sys.argv) > 2 else 600
max_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 64000
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0)

tok = Tokenizer.from_file(str(REPO / "models/Inkling-2-Mini/tokenizer.json"))
sets = sorted(p for p in SRC.glob("*/*") if p.is_dir())
corpus = []  # one token stream per document set, documents concatenated
for s in sets:
    text = "\n\n".join(f.read_text(errors="ignore") for f in sorted(s.glob("*.txt")))
    corpus.append(tok.encode(text, add_special_tokens=False).ids)
corpus = [c for c in corpus if len(c) > 1000]
print(f"{len(corpus)} document sets, {sum(map(len, corpus)) / 1e6:.1f}M tokens", file=sys.stderr)

instr_len = len(tok.encode(INSTR, add_special_tokens=False).ids)
lengths = []
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for _ in range(count):
        n = int(math.exp(rng.uniform(math.log(256), math.log(max_tok)))) - instr_len
        ids = rng.choice(corpus)
        n = min(n, len(ids))
        start = rng.randrange(0, len(ids) - n + 1)
        text = tok.decode(ids[start : start + n]) + INSTR
        lengths.append(n + instr_len)
        f.write(json.dumps({"question": text}, ensure_ascii=False) + "\n")

lengths.sort()
q = lambda p: lengths[int(p * (len(lengths) - 1))]  # noqa: E731
print(
    f"wrote {out}: {count} prompts, tokens p10 {q(0.1)} p50 {q(0.5)} p90 {q(0.9)} max {lengths[-1]}, "
    f"mean {sum(lengths) / len(lengths):.0f}",
    file=sys.stderr,
)

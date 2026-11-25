#!/usr/bin/env python3
"""HyperInkling launch wizard: a few questions, then the .env (and the command) that serves
Inkling 2 Mini on one RTX 3090 / 3090 Ti with those choices.

    python3 scripts/wizard.py            # interactive; writes .env next to docker-compose.yml
    python3 scripts/wizard.py --print    # only print, write nothing

The server fits the longest request to the GPU at boot (MAX_LEN=auto); the context shown
for each choice comes from docs/memory_matrix.tsv, measured on an RTX 3090 Ti with nothing
else on the GPU (bench/memory_matrix.sh), KV cache at 8 bits per value.
Standard library only, so it runs wherever Python 3.9+ does (Linux, macOS, Windows).
"""

import argparse
import csv
import math
import os
import secrets
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TABLE = REPO / "docs" / "memory_matrix.tsv"
MAX_CONTEXT = 1_048_576  # the model's own limit
# Per-token KV of a long request (the 8 full-attention layer visits, 8-bit; the DSpark
# drafter adds its own full-attention layer), to turn GiB into tokens.
KIB_PER_TOKEN = {"dspark": 18, "off": 16}
# Decode tok/s of one request, 8k context, RTX 3090 Ti (README).
DECODE_1 = {"dspark": 254, "off": 131}
# Context the adaptive verify length (its CUDA graphs) and the context lookup take.
ADAPTIVE_COST = 5_120
CELL = 40  # px per image token side (40 px patches, 2x2 merge)
A4_WIDTH_IN = 8.27
# (HF repo, local directory under models/) of each checkpoint.
CHECKPOINTS = {
    "original": ("Pentium95/Inkling-2-Mini-W4A16-g128-gptq", "Inkling-2-Mini-W4A16-g128-gptq"),
    "uncensored": ("Pentium95/Inkling-2-Mini-Uncensored-W4A16-g128-gptq",
                   "Inkling-2-Mini-Uncensored-W4A16-g128-gptq"),
}


def load_table():
    """{(vision, spec, async, max_seqs, nbt): longest request}."""
    rows = {}
    with TABLE.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["max_len_fits"] in ("", "?"):
                continue
            key = (r["vision"], r["spec"], r["async"] == "1", int(r["max_seqs"]),
                   int(r["nbt"]))
            rows[key] = int(r["max_len_fits"])
    return rows


def expected(rows, vision, spec, async_, seqs, nbt):
    """Longest request for a combination: measured, or the 6-request async-on NBT-2048 row
    of this vision tower and spec shifted by what async, NBT and the request count shift
    on the GPU (a shift not measured for `spec` is taken from DSpark's)."""
    if (vision, spec, async_, seqs, nbt) in rows:
        return rows[(vision, spec, async_, seqs, nbt)]

    def row(v, s, a=True, q=6, n=2048):
        return rows.get((v, s, a, q, n))

    def shift(change):  # change(spec) -> (from, to) row values, or None
        for s in (spec, "dspark"):
            a, b = change(s)
            if a is not None and b is not None:
                return b - a
        return 0

    base = row(vision, spec)
    if base is None:  # this tower measured with the other spec only
        other = "off" if spec == "dspark" else "dspark"
        if None in (row("gpu", spec), row(vision, other), row("gpu", other)):
            return None
        base = row("gpu", spec) + row(vision, other) - row("gpu", other)
    if not async_:
        base += shift(lambda s: (row("gpu", s), row("gpu", s, False)))
    if nbt != 2048:
        base += shift(lambda s: (row("gpu", s, async_), row("gpu", s, async_, n=nbt))) or \
            shift(lambda s: (row("gpu", s), row("gpu", s, n=nbt)))
    if seqs != 6:
        # Piecewise-linear in the request count over the measured gpu series.
        series = sorted((q, v) for (vi, s, a, q, n), v in rows.items()
                        if (vi, s, a, n) == ("gpu", "dspark", True, 2048))
        lo = max((p for p in series if p[0] <= seqs), default=series[0])
        hi = min((p for p in series if p[0] >= seqs), default=series[-1])
        ref = lo[1] if hi[0] == lo[0] else (
            lo[1] + (hi[1] - lo[1]) * (seqs - lo[0]) / (hi[0] - lo[0]))
        base += int(ref - (row("gpu", "dspark") or ref))
    return max(0, min(MAX_CONTEXT, base))


def image_size(tokens):
    """The largest A4-shaped image of at most `tokens` tokens: (width, height) px."""
    w = max(1, math.floor(math.sqrt(tokens / math.sqrt(2))))
    return w * CELL, (tokens // w) * CELL


def ask(prompt, choices=None, default=None, cast=str, check=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            raw = str(default)
        try:
            value = cast(raw)
        except ValueError:
            print("  not a valid answer")
            continue
        if choices and str(value).lower() not in choices:
            print(f"  choose one of: {', '.join(choices)}")
            continue
        if check and not check(value):
            continue
        return str(value).lower() if choices else value


def ram_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def fmt_tokens(n):
    return f"{n / 1000:.0f}k" if n < 1_000_000 else f"{n / 1_048_576:.2f}M"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--print", action="store_true", help="print only, do not write .env")
    args = ap.parse_args()
    rows = load_table()
    print(__doc__.split("\n\n")[0], "\n")

    def ctx(*a):
        n = expected(rows, *a)
        return fmt_tokens(n) if n else "?"

    print("0) Which checkpoint?")
    print("   a) Inkling 2 Mini as released by Thinking Machines")
    print("   b) Inkling 2 Mini Uncensored (OpenUncensored's abliteration: refusals removed, only")
    print("      the attention output projections edited), quantized the same way")
    checkpoint = "uncensored" if ask("   choice", "ab", "a") == "b" else "original"

    seqs = ask("\n1) How many requests should run at the same time? (1-64)", default=8,
               cast=int, check=lambda v: 1 <= v <= 64 or print("  1 to 64"))

    print("\n2) Where should the vision tower (images) run?")
    towers = (("a", "gpu", "on the GPU: fastest image encoding"),
              ("b", "host", "weights in host RAM, streamed per image: ~27 ms more per A4 page"),
              ("c", "none", "no vision: text only"))
    for key, vision, note in towers:
        print(f"   {key}) {note}  (longest request: "
              f"{ctx(vision, 'dspark', True, seqs, 2048)})")
    vision = {k: v for k, v, _ in towers}[ask("   choice", "abc", "a")]

    print("\n3) Speculative decoding with the bundled DSpark drafter?")
    print(f"   y) yes: one request decodes ~{DECODE_1['dspark']} tok/s instead of "
          f"~{DECODE_1['off']} (more on code and structured output, less on free prose); "
          f"longest request {ctx(vision, 'dspark', True, seqs, 2048)}")
    print(f"   n) no: longest request {ctx(vision, 'off', True, seqs, 2048)}")
    spec = "dspark" if ask("   choice", "yn", "y") == "y" else "off"

    adaptive = "n"
    if spec == "dspark":
        print("\n3a) Adaptive verify length and drafts from the context: +43% while the "
              "answer copies the prompt, +4-10% on free text and code, for "
              f"~{fmt_tokens(ADAPTIVE_COST)} of context.")
        adaptive = ask("   enable", "yn", "y")

    width = height = None
    if vision != "none":
        print("\n4) Largest image, in tokens (larger images are scaled down to it). 1024 "
              "tokens is an A4 page at ~126 ppi; the prefill chunk of 2048 encodes two per "
              "step.")
        for t in (1024, 1364, 2048, 5488):
            w, h = image_size(t)
            print(f"   {t:5d} tokens = {w * h / 1e6:.1f} MP, A4 at ~{w / A4_WIDTH_IN:.0f} ppi"
                  f" ({w} x {h})")
        tokens = ask("   tokens", default=1024, cast=int,
                     check=lambda v: 256 <= v <= 16384 or print("  256 to 16384"))
        width, height = image_size(tokens)

    nbt_gain = (expected(rows, vision, spec, True, seqs, 1024) or 0) - (
        expected(rows, vision, spec, True, seqs, 2048) or 0)
    print(f"\n5) Prefill chunk (NBT): 2048, or 1024 for ~{fmt_tokens(max(0, nbt_gain))} more "
          "context at ~8% slower prefill (same decode).")
    nbt = int(ask("   chunk", ("2048", "1024"), "2048"))

    on = expected(rows, vision, spec, True, seqs, nbt)
    off = expected(rows, vision, spec, False, seqs, nbt)
    gain = f"~{fmt_tokens(off - on)} more context" if on and off else "more context"
    print(f"\n6) Async scheduling (two steps in flight): +3-4% decode; off: {gain}.")
    async_ = ask("   enable", "yn", "y") == "y"

    headroom = ask("\n7) GiB of GPU memory to leave free (a display on this GPU: 1)",
                   default=0.0, cast=float, check=lambda v: 0 <= v <= 4)

    longest = expected(rows, vision, spec, async_, seqs, nbt)
    if longest:
        if adaptive == "n" and spec == "dspark":
            longest += ADAPTIVE_COST
        longest -= int(headroom * 2**20 / KIB_PER_TOKEN[spec])
    print(f"\n8) Longest request: auto fits it to the GPU at boot (expected "
          f"~{fmt_tokens(longest) if longest else '?'}); a number caps it (prompts above it "
          "are refused; a cap that does not fit stops the boot).")

    def ok(v):
        if v == "auto":
            return True
        if not 1024 <= v <= MAX_CONTEXT:
            return print(f"  1024 to {MAX_CONTEXT}, or auto") or False
        if longest and v > longest:
            return print(f"  {fmt_tokens(v)} is above the expected ~{fmt_tokens(longest)}"
                         ) or False
        return True

    max_len = ask("   tokens or auto", default="auto",
                  cast=lambda v: v.lower() if v.lower() == "auto" else int(v), check=ok)

    ram = ram_gib()
    auto = "unknown" if ram is None else f"{int(max(0, min(36, ram * 0.6 - 8)))} GiB"
    print(f"\n9) Prefix-cache tier in host RAM (/dev/shm): prompts evicted from the GPU come "
          f"back in 0.1 s instead of being recomputed. RAM: "
          f"{'unknown' if ram is None else f'{ram:.0f} GiB'}.")
    tier = ask(f"   GiB, 0 = off, auto = 60% of RAM minus 8, at most 36 (here {auto})",
               default="auto", cast=lambda v: v if v.lower() == "auto" else int(v),
               check=lambda v: v == "auto" or v >= 0 or print("  0 or more, or auto"))

    expose = ask("\n10) Reachable from other machines? (needs an API key)", "yn", "n")
    dash = ask("\n11) Status page on http://127.0.0.1:8082: live decode and prefill rates, "
               "prefix hits, draft\n     acceptance, KV usage, the card's temperature and watts. "
               "Read-only, runs beside\n     the engine, localhost only unless you set DASH_BIND.",
               "yn", "y")
    env = {
        "MAX_SEQS": seqs,
        "MAX_LEN": max_len,
        "SPEC": spec,
        "ASYNC": int(async_),
        "NBT": nbt,
        "KV_OFFLOAD_GB": tier,
        "GPU_UTIL": round(0.95 - headroom / 24, 3),
        "DASHBOARD": 1 if dash == "y" else 0,
    }
    repo, local = CHECKPOINTS[checkpoint]
    if checkpoint != "original":  # compose downloads HF_REPO into models/CHECKPOINT
        env["HF_REPO"], env["CHECKPOINT"], env["MODEL"] = repo, local, f"models/{local}"
    if width:
        env["IMAGE_MAX_W"], env["IMAGE_MAX_H"] = width, height
        if (width // CELL) * (height // CELL) > 2048:
            env["ENCODER_CACHE"] = 4 * (width // CELL) * (height // CELL)
    if vision == "none":
        env["IMAGES"] = 0
    if vision == "host":
        env["HYPERINKLING_VISION_HOST"] = 1
    if spec == "dspark" and adaptive == "n":
        env["HYPERINKLING_SPEC_ADAPTIVE"] = ""
        env["HYPERINKLING_LOOKUP"] = 0
    if tier != "auto" and tier > 36:  # compose's /dev/shm default is 48
        env["SHM_GB"] = tier + 12
    if hasattr(os, "getuid"):  # compose's prepare writes ./models as this user
        env["UID"], env["GID"] = os.getuid(), os.getgid()
    if expose == "y":
        env["BIND"] = "0.0.0.0"  # Docker: the published port
        env["HOST"] = "0.0.0.0"  # native start.sh
        env["VLLM_API_KEY"] = secrets.token_hex(24)

    text = "".join(f"{k}={v}\n" for k, v in env.items())
    print("\n--- .env " + "-" * 50 + f"\n{text}" + "-" * 59)
    if seqs > 1 and longest:
        print(f"{seqs} long requests at once share the KV pool, about "
              f"{fmt_tokens(longest / seqs)} each; shorter ones fit more.")
    if not args.print:
        dest = REPO / ".env"
        if dest.exists() and ask(f"\n{dest} exists. Overwrite?", "yn", "n") != "y":
            print("Not written.")
        else:
            dest.write_text(text)
            # Docker would create a missing bind-mount source as root.
            (REPO / "models").mkdir(exist_ok=True)
            print(f"Written {dest}")
    print("\nStart it:  docker compose up -d     (logs: docker compose logs -f; the first start"
          " downloads the checkpoint)")
    print(f"Native:    hf download {repo} --local-dir models/{local}   (once)")
    exports = " ".join(f"{k}={v}" for k, v in env.items()
                       if k not in ("UID", "GID", "SHM_GB", "BIND", "HF_REPO", "CHECKPOINT"))
    print(f"Native:    {exports} ./start.sh")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)

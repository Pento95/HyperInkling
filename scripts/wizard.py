#!/usr/bin/env python3
"""HyperInkling launch wizard: a few questions, then the .env (and the command) that serves
Inkling 2 Mini on one RTX 3090 / 3090 Ti with those choices.

    python3 scripts/wizard.py            # interactive; writes .env next to docker-compose.yml
    python3 scripts/wizard.py --print    # only print, write nothing

Every number it shows comes from docs/memory_matrix.tsv, measured on an RTX 3090 Ti with
nothing else on the GPU (bench/memory_matrix.sh), with the KV cache at 8 bits per value.
A GPU that also drives a display has less free memory: answer the last question.
Standard library only, so it runs wherever Python 3.9+ does (Linux, macOS, Windows).
"""

import argparse
import csv
import os
import secrets
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TABLE = REPO / "docs" / "memory_matrix.tsv"
MAX_CONTEXT = 1_048_576  # the model's own limit
# Per-token KV of a long request (the 8 full-attention layer visits, 8-bit; the DSpark
# drafter adds its own full-attention layer), to turn freed GiB into tokens.
KIB_PER_TOKEN = {"dspark": 18, "off": 16}
# The DSpark drafter's RoPE table is sized by the context limit (MAX_LEN): fp32, 128
# values per position. Table rows are brought to a common MAX_LEN = 1M basis (each row's
# measured_at), and a lower limit gets the difference back as KV.
ROPE_BYTES_PER_TOKEN = 512
MEASURED_AT = 1_048_576
# Speeds, RTX 3090 Ti, 2026-11-21 (README): decode tok/s of one request, AA-LCR prompts.
DECODE_1 = {"dspark": 214, "off": 128}
# NBT=2048 (prefill chunk): +7-8% prefill, same decode; context lost against NBT=1024 with
# async scheduling off (measured 2026-11-21, 6 requests, vision on the GPU). With async on
# it costs far more (213k -> 94k), so 2048 runs with async off.
NBT2048_COST = {"dspark": 67_712, "off": 73_728}


def load_table():
    rows = []
    with TABLE.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["max_len_fits"] in ("", "?"):
                continue
            fits, spec = int(r["max_len_fits"]), r["spec"]
            at = int(r.get("measured_at") or MEASURED_AT)
            if spec == "dspark":
                # As if measured at MAX_LEN = 1M: the RoPE table that a 1M limit adds.
                fits -= (MEASURED_AT - at) * ROPE_BYTES_PER_TOKEN // (
                    KIB_PER_TOKEN[spec] * 1024
                )
            rows.append(
                {
                    "vision": r["vision"],
                    "spec": spec,
                    "async": r["async"] == "1",
                    "max_seqs": int(r["max_seqs"]),
                    "fits": fits,
                }
            )
    return rows


def lookup(rows, vision, spec, async_, seqs):
    """Longest single request for this combination, in tokens. Measured at 6 requests;
    other request counts shift it by what the measured gpu/dspark series shifts."""
    exact = [r for r in rows if (r["vision"], r["spec"], r["async"], r["max_seqs"])
             == (vision, spec, async_, seqs)]
    if exact:
        return exact[0]["fits"]
    base = [r for r in rows if (r["vision"], r["spec"], r["async"], r["max_seqs"])
            == (vision, spec, async_, 6)]
    if not base:
        return None
    series = sorted((r["max_seqs"], r["fits"]) for r in rows
                    if (r["vision"], r["spec"], r["async"]) == ("gpu", "dspark", False))
    ref6 = dict(series).get(6)
    if not series or ref6 is None:
        return base[0]["fits"]
    # Piecewise-linear in the request count over the measured series.
    lo = max((s for s in series if s[0] <= seqs), default=series[0])
    hi = min((s for s in series if s[0] >= seqs), default=series[-1])
    if hi[0] == lo[0]:
        ref = lo[1]
    else:
        ref = lo[1] + (hi[1] - lo[1]) * (seqs - lo[0]) / (hi[0] - lo[0])
    return int(base[0]["fits"] + (ref - ref6))


def fits_at(measured, spec, limit):
    """Longest request with MAX_LEN = `limit`, from one measured at MAX_LEN = 1M."""
    if measured is None:
        return None
    if spec != "dspark":
        return measured
    freed = (MEASURED_AT - limit) * ROPE_BYTES_PER_TOKEN
    return measured + freed // (KIB_PER_TOKEN[spec] * 1024)


def max_limit(measured, spec):
    """Largest MAX_LEN = L with L <= fits_at(measured, spec, L), in closed form."""
    if measured is None:
        return None
    if spec != "dspark":
        return min(MAX_CONTEXT, measured)
    r = ROPE_BYTES_PER_TOKEN / (KIB_PER_TOKEN[spec] * 1024)
    return min(MAX_CONTEXT, int((measured + MEASURED_AT * r) / (1 + r)))


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

    seqs = ask("1) How many requests should run at the same time? (1-64)", default=6,
               cast=int, check=lambda v: 1 <= v <= 64 or print("  1 to 64"))

    print("\n2) Where should the vision tower (images) run?")
    for key, vision, note in (
        ("a", "gpu", "on the GPU: fastest image encoding"),
        ("b", "host", "weights in host RAM, streamed per image: ~27 ms more per A4 page"),
        ("c", "none", "no vision: text only"),
    ):
        ctx = max_limit(lookup(rows, vision, "dspark", False, seqs), "dspark")
        print(f"   {key}) {note}" + (f"  (longest request with DSpark: {fmt_tokens(ctx)})"
                                       if ctx else ""))
    vision = {"a": "gpu", "b": "host", "c": "none"}[ask("   choice", "abc", "a")]

    print("\n3) Speculative decoding with the bundled DSpark drafter?")
    on = max_limit(lookup(rows, vision, "dspark", False, seqs), "dspark")
    off = max(
        max_limit(lookup(rows, vision, "off", a, seqs), "off") or 0 for a in (True, False)
    )
    print(f"   y) yes: one request decodes ~{DECODE_1['dspark']} tok/s instead of "
          f"~{DECODE_1['off']} (1.3-2.3x by kind of text), longest request "
          f"{fmt_tokens(on) if on else '?'}")
    print(f"   n) no: longest request {fmt_tokens(off) if off else '?'}")
    spec = "dspark" if ask("   choice", "yn", "y") == "y" else "off"

    print("\n3b) Prefill chunk (NBT): 1024, or 2048 for +7-8% prefill (same decode) at "
          f"~{fmt_tokens(NBT2048_COST[spec])} less context, async scheduling off.")
    nbt = ask("   chunk", ("1024", "2048"), "1024")

    # Context: async scheduling first (faster); one step in flight frees the second
    # in-flight chunk's sliding-window and conv-state reserves (-3 to -4% decode).
    fast_m = lookup(rows, vision, spec, True, seqs)
    slow_m = lookup(rows, vision, spec, False, seqs) or fast_m
    if nbt == "2048":
        fast_m = None
        slow_m = slow_m and slow_m - NBT2048_COST[spec]
    fast = max_limit(fast_m, spec) or 0
    best = max(fast, max_limit(slow_m, spec) or 0)
    print(f"\n4) Context limit per request (MAX_LEN), in tokens. Up to {fmt_tokens(fast or 0)}"
          f" at full speed; up to {fmt_tokens(best)} with async scheduling off (-3 to -4%"
          " decode).")
    if spec == "dspark":
        per_100k = 100_000 * ROPE_BYTES_PER_TOKEN // (KIB_PER_TOKEN[spec] * 1024)
        print(f"   The drafter's RoPE table grows with the limit: every 100k tokens "
              f"allowed costs ~{fmt_tokens(per_100k)} of KV pool, so a limit above what "
              "you need is not free.")
    default = min(262_144, best) // 1024 * 1024

    def ok(v):
        if not 1024 <= v <= MAX_CONTEXT:
            return print(f"  1024 to {MAX_CONTEXT}") or False
        if v > (fits_at(slow_m, spec, v) or 0):
            return print(f"  {fmt_tokens(v)} does not fit; at most "
                         f"{fmt_tokens(fits_at(slow_m, spec, v) or 0)}") or False
        return True

    ctx = ask("   context", default=default, cast=int, check=ok)
    async_ = 1 if ctx <= (fits_at(fast_m, spec, ctx) or 0) else 0
    best = fits_at(slow_m if not async_ else fast_m, spec, ctx)
    pool_note = ""
    if seqs > 1:
        pool_note = (f" With {seqs} long requests at once they share the KV pool, about "
                     f"{fmt_tokens((best or ctx) / seqs)} each; shorter ones fit more.")

    headroom = ask("\n5) GiB of GPU memory to leave free (a display on this GPU: 1)",
                   default=0.0, cast=float, check=lambda v: 0 <= v <= 4)
    if headroom:
        ctx_cut = int(headroom * 2**20 / KIB_PER_TOKEN[spec])
        ctx = max(1024, (ctx - ctx_cut) // 1024 * 1024)
        print(f"   longest request lowered to {fmt_tokens(ctx)}")

    ram = ram_gib()
    auto = "unknown" if ram is None else f"{int(max(0, min(36, ram * 0.6 - 8)))} GiB"
    print(f"\n6) Prefix-cache tier in host RAM (/dev/shm): prompts evicted from the GPU come "
          f"back in 0.1 s instead of being recomputed. RAM: "
          f"{'unknown' if ram is None else f'{ram:.0f} GiB'}.")
    tier = ask(f"   GiB, 0 = off, auto = 60% of RAM minus 8, at most 36 (here {auto})",
               default="auto", cast=lambda v: v if v.lower() == "auto" else int(v),
               check=lambda v: v == "auto" or v >= 0 or print("  0 or more, or auto"))

    expose = ask("\n7) Reachable from other machines? (needs an API key)", "yn", "n")
    env = {
        "MAX_SEQS": seqs,
        "MAX_LEN": ctx,
        "SPEC": spec,
        "ASYNC": async_,
        "KV_OFFLOAD_GB": tier,
        "NBT": int(nbt),
        "GPU_UTIL": round(0.95 - headroom / 24, 3),
    }
    if vision == "none":
        env["IMAGES"] = 0
    if vision == "host":
        env["HYPERINKLING_VISION_HOST"] = 1
    if hasattr(os, "getuid"):  # compose's prepare writes ./models as this user
        env["UID"], env["GID"] = os.getuid(), os.getgid()
    if expose == "y":
        env["BIND"] = "0.0.0.0"  # Docker: the published port
        env["HOST"] = "0.0.0.0"  # native start.sh
        env["VLLM_API_KEY"] = secrets.token_hex(24)

    text = "".join(f"{k}={v}\n" for k, v in env.items())
    print("\n--- .env " + "-" * 50 + f"\n{text}" + "-" * 59)
    print(f"Longest request {fmt_tokens(ctx)}, {seqs} at a time.{pool_note}")
    if not async_:
        print("Async scheduling off for this context: -3 to -4% decode.")
    if not args.print:
        dest = REPO / ".env"
        if dest.exists() and ask(f"\n{dest} exists. Overwrite?", "yn", "n") != "y":
            print("Not written.")
        else:
            dest.write_text(text)
            # Docker would create a missing bind-mount source as root.
            (REPO / "models").mkdir(exist_ok=True)
            print(f"Written {dest}")
    print("\nStart it:  docker compose up -d     (logs: docker compose logs -f)")
    exports = " ".join(f"{k}={v}" for k, v in env.items())
    print(f"Native:    {exports} ./start.sh")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)

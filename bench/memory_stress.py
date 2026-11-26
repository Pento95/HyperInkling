#!/usr/bin/env python3
"""GPU memory under load: one request at the longest length, then waves of concurrent long
requests, and peak `memory.used` (nvidia-smi). Run `bench/quality_battery.py run docvqa
--jobs 8` alongside for images. A failed request is printed; an engine OOM ends the server.

    bench/memory_stress.py [port] [burst]     # burst: one more wave this wide (MAX_SEQS)
"""

import json
import random
import subprocess
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18030
BURST = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BASE = f"http://127.0.0.1:{PORT}"
rng = random.Random(5)
peak = [0]
done = threading.Event()


def poll_memory():
    while not done.is_set():
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        peak[0] = max(peak[0], int(out.split()[0]))
        time.sleep(0.3)


def generate(model, prompt_len, max_tokens):
    body = {"model": model, "prompt": [rng.randrange(1000, 150000) for _ in range(prompt_len)],
            "max_tokens": max_tokens, "temperature": 1.0, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1500))["usage"]["completion_tokens"]


def wave(model, count, prompt_len, max_tokens):
    errors, ok = [], []

    def one():
        try:
            ok.append(generate(model, prompt_len, max_tokens))
        except Exception as e:  # noqa: BLE001 - reported, the wave goes on
            errors.append(repr(e)[:200])

    threads = [threading.Thread(target=one) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        print(f"   {len(errors)} failed: {errors[0]}", flush=True)
    print(f"  {count} x {prompt_len}: {len(ok)} done, peak {peak[0]} MiB", flush=True)


def main():
    info = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]
    model, longest = info["id"], info["max_model_len"]
    threading.Thread(target=poll_memory, daemon=True).start()
    t = time.time()
    generate(model, longest - 600, 400)
    print(f"max_len {longest}: longest request {time.time() - t:.0f} s, peak {peak[0]} MiB",
          flush=True)
    waves = [(8, longest // 9, 300), (8, 30000, 600), (16, 6000, 400)]
    if BURST:
        waves.append((BURST, 3000, 300))
    for count, prompt_len, max_tokens in waves:
        wave(model, count, prompt_len, max_tokens)
    done.set()


if __name__ == "__main__":
    main()

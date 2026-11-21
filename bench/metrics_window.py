"""Server-side speeds over a time window, from two /metrics snapshots.

Independent of the client: decode tok/s, prompt tokens computed vs served
from the prefix cache, per-request prefill time, queueing, ITL, step time.

    venv/bin/python bench/metrics_window.py [seconds] [port]
"""

import re
import sys
import time
import urllib.request

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30
port = sys.argv[2] if len(sys.argv) > 2 else "18030"


def snap():
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    d = {}
    for line in text.splitlines():
        m = re.match(r'^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if m and 'le="' not in (m.group(2) or ""):
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(3))
    return time.time(), d


ta, a = snap()
time.sleep(secs)
tb, b = snap()
dt = tb - ta
D = lambda k: b.get(k, 0.0) - a.get(k, 0.0)  # noqa: E731

steps = D("vllm:iteration_tokens_total_count")
pt, hit = D("vllm:prompt_tokens_total"), D("vllm:prefix_cache_hits_total")
print(f"window {dt:.0f} s, {steps:.0f} steps" + (f" ({dt / steps * 1e3:.1f} ms/step)" if steps else ""))
print(f"decode      {D('vllm:generation_tokens_total') / dt:6.0f} tok/s aggregate")
print(f"prompt      {pt / dt:6.0f} tok/s arriving, {hit / dt:.0f} from prefix cache, "
      f"{(pt - hit) / dt:.0f} computed")
if (n := D("vllm:inter_token_latency_seconds_count")):
    itl = D("vllm:inter_token_latency_seconds_sum") / n
    print(f"ITL         {itl * 1e3:6.1f} ms  ({1 / itl:.0f} tok/s per request)")
if (n := D("vllm:request_queue_time_seconds_count")):
    pre = D("vllm:request_prefill_time_seconds_sum") / n
    comp = D("vllm:request_prefill_kv_computed_tokens_sum") / n
    print(f"per request queue {D('vllm:request_queue_time_seconds_sum') / n:.2f} s, "
          f"prefill {pre:.2f} s for {comp:.0f} computed of "
          f"{D('vllm:request_prompt_tokens_sum') / n:.0f} prompt tokens "
          f"({comp / pre:.0f} tok/s while sharing the GPU), {n:.0f} requests")
if (n := D("vllm:time_to_first_token_seconds_count")):
    print(f"TTFT        {D('vllm:time_to_first_token_seconds_sum') / n:6.2f} s mean")
if steps:
    print(f"tokens/step {D('vllm:iteration_tokens_total_sum') / steps:6.1f}")
if (drafts := D("vllm:spec_decode_num_drafts_total")):
    prop, acc = D("vllm:spec_decode_num_draft_tokens_total"), D("vllm:spec_decode_num_accepted_tokens_total")
    print(f"spec decode {acc / prop * 100:6.1f}% of draft tokens accepted, "
          f"{1 + acc / drafts:.2f} tokens per verify step ({prop / drafts:.1f} drafted)")
print(f"now: running {b.get('vllm:num_requests_running', 0):.0f}, "
      f"waiting {b.get('vllm:num_requests_waiting', 0):.0f}, "
      f"KV used {b.get('vllm:kv_cache_usage_perc', 0) * 100:.1f}%")

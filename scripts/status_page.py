#!/usr/bin/env python3
"""Status page for the Inkling 2 Mini server.

Stdlib only, read-only, no host access: everything comes from the server itself
(/health, /version, /load, /v1/models, /metrics) plus nvidia-smi for the card this
process can see. It runs unchanged inside the container (next to the engine, where
start.sh already put every knob) and outside it (point --url at the server).

    scripts/status_page.py [--bind 127.0.0.1] [--port 8082] [--url http://127.0.0.1:8000]

Defaults come from the launcher's own environment, so `docker compose up` needs no flags:
  PORT           the engine's port          -> --url http://127.0.0.1:$PORT
  VLLM_API_KEY   sent as Authorization      (/metrics and /v1/models need it when set)
  DASHBOARD_PORT the page's own port        -> --port
Serves:
  GET /             dashboard (server-rendered first paint, then polls every 5 s)
  GET /api/status   the same data as JSON, incl. a rolling 15-minute history
  GET /health       200 if the upstream server answers /health, else 503

Live numbers: state, and prefill / decode / engine-throughput rates computed from the
/metrics counters over a 2 s sliding window. vLLM exposes no per-request progress (no slot
states, no prompt_read), so the page is aggregate-since-boot and says so. The sampler runs
at 1 s cadence in the background; the page polls the latest snapshot every 5 s.
Uptime is `process_start_time_seconds`, which the engine exports itself; a counter reset
(the server restarted) flushes the windows and re-reads it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).parent
WEB_DIR = HERE / "web"          # optional assets only (an OFL font, say); the page needs nothing

CACHE: dict = {"ts": 0.0, "data": None}
CACHE_TTL = 5.0
LATEST = {"data": None, "ts": 0.0}
HISTORY = deque(maxlen=900)          # 15 min at 1 s cadence
SAMPLER_STATE = {"last_gen": None, "boot_time": None, "boot_checked_at": 0.0}
SAMPLE_GAP = 0.85                     # ~1 s sampling
LIVE_WINDOW_S = 2.0                   # sliding window for the live tok_s figures
PROMPT_WINDOW_S = 1.0                 # fresh prefill bursts are 1 s events — 2 s would smear them
ACCEPT_WINDOW_S = 30.0




def _api_key():
    return os.environ.get("VLLM_API_KEY", "")


def get(url, timeout=4):
    """GET with the server's own key, if the launcher set one: /metrics and /v1/models
    need it when VLLM_API_KEY is on (everything except /health, /ping, /load, /version)."""
    try:
        req = Request(url)
        key = _api_key()
        if key:
            req.add_header("Authorization", "Bearer " + key)
        with urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except Exception:
        return None, None


def parse_prom(text):
    """Prometheus text -> {metric: [(labels, value), ...]}."""
    out = {}
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        head, value = parts
        try:
            val = float(value)
        except ValueError:
            continue
        name = head.split("{", 1)[0]
        labels = {}
        m = re.search(r"\{(.*)\}", head)
        if m:
            for pair in m.group(1).split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')
        out.setdefault(name, []).append((labels, val))
    return out


def get_one(parsed, name, **labels):
    for lbls, val in parsed.get(name, []):
        if all(lbls.get(k) == v for k, v in labels.items()):
            return val
    return None


def get_all(parsed, name, **labels):
    return sum(val for lbls, val in parsed.get(name, [])
               if all(lbls.get(k) == v for k, v in labels.items()))


def gpu_info():
    """The card this process was given. Inside a container NVML renumbers it (the host's
    GPU 1 is index 0 here), and the container only sees its own - so no index is asked
    for and no index is assumed. None if nvidia-smi is absent (no GPU passthrough)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        return {"index": int(parts[0]), "util": float(parts[1]), "mem_used_mib": float(parts[2]),
                "mem_total_mib": float(parts[3]), "temp_c": float(parts[4]),
                "power_w": float(parts[5]), "power_limit_w": float(parts[6])}
    except Exception:
        return None


def shape_from_argv(pid=1, cmdline_path=None):
    """The shape the engine was actually started with, read from its own command line.

    Not the environment: start.sh derives values (LONG_PREFILL from NBT, GRAPH_SIZES from
    MAX_SEQS, MAX_LEN=auto -> -1), and what a status page should show is what the server
    is doing, not what .env said. Inside the container PID 1 is `vllm serve ...`;
    anywhere else it may not be, so this returns None and the page drops that row.
    """
    try:
        path = Path(cmdline_path) if cmdline_path else Path(f"/proc/{pid}/cmdline")
        raw = path.read_bytes().split(b"\0")
        argv = [a.decode(errors="replace") for a in raw if a]
        if not any(a.startswith("vllm") or a.endswith("/vllm") for a in argv[:3]):
            return None
        flags = {}
        i = 0
        while i < len(argv):
            a = argv[i]
            if a.startswith("--") and "=" in a:
                # `--flag=value` is the same flag to argparse, and a hand-written command line
                # (a native install, or someone copying vLLM's own docs) uses it. Reading only
                # the space form would drop the row from the panel without saying so.
                key, value = a[2:].split("=", 1)
                flags[key] = value
                i += 1
            elif a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                flags[a[2:]] = argv[i + 1]
                i += 2
            else:
                i += 1
        shape = {"max_seqs": flags.get("max-num-seqs"), "nbt": flags.get("max-num-batched-tokens"),
                 "gpu_util": flags.get("gpu-memory-utilization"), "kv_dtype": flags.get("kv-cache-dtype"),
                 "prefix_caching": any(x == "--enable-prefix-caching"
                                       or x.startswith("--enable-prefix-caching=") for x in argv),
                 "kv_offload_gib": flags.get("kv-offloading-size"),
                 "long_prefill": flags.get("long-prefill-token-threshold")}
        for key, opt in (("spec", "speculative-config"), ("images", "limit-mm-per-prompt")):
            try:
                shape[key] = json.loads(flags[opt])
            except Exception:
                pass
        return {k: v for k, v in shape.items() if v is not None}
    except Exception:
        return None


def fetch_all(url):
    """One full snapshot of the server. Returns a dict, or None if the server is unreachable."""
    data = {"url": url, "ts": time.time(), "up": False, "version": None, "model": None,
            "max_model_len": None, "load": None, "gpu": gpu_info(), "shape": shape_from_argv()}

    status, _ = get(url + "/health")
    data["up"] = status == 200

    _, vtext = get(url + "/version")
    if vtext:
        try:
            data["version"] = json.loads(vtext).get("version")
        except Exception:
            pass

    _, load = get(url + "/load")
    if load:
        try:
            data["load"] = json.loads(load).get("server_load")
        except Exception:
            pass

    _, mtext = get(url + "/v1/models")
    if mtext:
        try:
            j = json.loads(mtext)
            data["model"] = j["data"][0].get("id")
            data["max_model_len"] = j["data"][0].get("max_model_len")
        except Exception:
            pass

    status, text = get(url + "/metrics")
    if status != 200 or not text:
        return None
    data["metrics_http"] = 200
    p = parse_prom(text)
    data["start_seconds"] = get_one(p, "process_start_time_seconds")
    data["requests_running"] = get_one(p, "vllm:num_requests_running") or 0
    data["requests_waiting"] = get_one(p, "vllm:num_requests_waiting") or 0
    data["preemptions"] = get_one(p, "vllm:num_preemptions_total") or 0
    data["admission_rejections"] = get_all(p, "vllm:admission_rejections_total")

    success = {}
    for reason in ("stop", "length", "abort", "error", "repetition"):
        success[reason] = get_one(p, "vllm:request_success_total", finished_reason=reason) or 0
    data["requests_success"] = success
    data["requests_total"] = sum(success.values())

    data["prompt_tokens_total"] = get_one(p, "vllm:prompt_tokens_total") or 0
    data["prompt_tokens_cached"] = get_one(p, "vllm:prompt_tokens_cached_total") or 0
    data["prompt_local_compute"] = get_one(p, "vllm:prompt_tokens_by_source_total", source="local_compute") or 0
    data["prompt_compute_tokens"] = data["prompt_local_compute"]
    data["prompt_cache_hit"] = get_one(p, "vllm:prompt_tokens_by_source_total", source="local_cache_hit") or 0
    data["generation_tokens"] = get_one(p, "vllm:generation_tokens_total") or 0
    data["iteration_tokens_total"] = get_one(p, "vllm:iteration_tokens_total_sum") or 0

    data["spec_drafts"] = get_one(p, "vllm:spec_decode_num_drafts_total") or 0
    data["spec_verified"] = get_one(p, "vllm:spec_decode_num_verified_tokens_total") or 0
    data["spec_accepted"] = get_one(p, "vllm:spec_decode_num_accepted_tokens_total") or 0

    data["prefix_queries"] = get_one(p, "vllm:prefix_cache_queries_total") or 0
    data["prefix_hits"] = get_one(p, "vllm:prefix_cache_hits_total") or 0
    data["external_prefix_hits"] = get_one(p, "vllm:external_prefix_cache_hits_total") or 0
    data["mm_cache_queries"] = get_one(p, "vllm:mm_cache_queries_total") or 0
    data["mm_cache_hits"] = get_one(p, "vllm:mm_cache_hits_total") or 0

    data["offload_stored"] = get_one(p, "vllm:kv_offload_total_bytes_total", transfer_type="GPU_to_CPU") or 0
    data["offload_reloaded"] = get_one(p, "vllm:kv_offload_total_bytes_total", transfer_type="CPU_to_GPU") or 0
    for line in text.splitlines():
        if line.startswith("vllm:kv_offload_config_info"):
            m = re.search(r'cpu_kv_cache_size_tokens="(\d+)"', line)
            if m:
                data["offload_cpu_tokens"] = int(m.group(1))
            break
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            m = re.search(r'kv_cache_size_tokens="(\d+)"', line)
            if m:
                data["kv_cache_tokens"] = int(m.group(1))
            m = re.search(r'num_gpu_blocks="(\d+)"', line)
            if m:
                data["gpu_blocks"] = int(m.group(1))
            break
    data["kv_usage_perc"] = get_one(p, "vllm:kv_cache_usage_perc") or 0.0
    data["tool_calls"] = get_all(p, "vllm:tool_call_parser_invocations_total", outcome="tool_call")
    data["tool_call_invocations"] = get_all(p, "vllm:tool_call_parser_invocations_total")
    data["time_to_first_token_avg"] = None
    ttf_sum = get_one(p, "vllm:time_to_first_token_seconds_sum")
    ttf_count = get_one(p, "vllm:time_to_first_token_seconds_count")
    if ttf_sum and ttf_count:
        data["time_to_first_token_avg"] = round(ttf_sum / ttf_count, 2)
    return data


def live_rate(hist, now, cur_val, sample_key, window):
    """Sliding-window rate over the last `window` seconds."""
    if len(hist) < 2 or cur_val is None:
        # A counter this build does not export arrives as None; the call sites coalesce it to 0,
        # but subtracting it here would raise and the sampler thread swallows exceptions, which
        # would look like a frozen page rather than a missing figure.
        return None
    target = now - window
    for i in range(len(hist) - 1, -1, -1):
        s = hist[i]
        if s["t"] <= target:
            dt = now - s["t"]
            if dt >= 0.25 and s.get(sample_key) is not None:
                return max(0.0, (cur_val - s[sample_key]) / dt)
            break
    return None


def sample_once(base_url):
    """One 1 s snapshot. Appends to HISTORY and stores in LATEST. Returns the snapshot."""
    data = fetch_all(base_url)
    if data is None:
        # Server unreachable - mark down and keep the last known numbers, but always report
        # the card and the shape: those come from this process, and the temperature of a GPU
        # whose engine just died is the thing a person looks at first. A cold start reads
        # them too, so /api/status keeps the same keys whatever the server is doing.
        stale = LATEST["data"]
        down = dict(stale or {"url": base_url})
        down["up"] = False
        down["ts"] = time.time()
        down["metrics_http"] = None
        down["gpu"] = gpu_info()
        down["shape"] = shape_from_argv()
        down["history"] = list(HISTORY)
        LATEST["data"] = down
        return down

    now = time.time()
    # counter reset → new boot (server restart): flush the window, re-check boot time
    if SAMPLER_STATE["last_gen"] is not None and data["generation_tokens"] < SAMPLER_STATE["last_gen"]:
        SAMPLER_STATE["boot_time"] = now
        SAMPLER_STATE["boot_checked_at"] = 0.0
        HISTORY.clear()
    SAMPLER_STATE["last_gen"] = data["generation_tokens"]

    # last-known rates persist across samples: a 2 s window that hits 0 (idle) is not
    # "no request" — remember the last request that actually generated
    prev = LATEST["data"] or {}
    data["decode_tps_last"] = prev.get("decode_tps_last")
    data["decode_tps_last_at"] = prev.get("decode_tps_last_at")
    data["prompt_tps_last"] = prev.get("prompt_tps_last")
    data["prompt_tps_last_at"] = prev.get("prompt_tps_last_at")
    data["iter_tps_last"] = prev.get("iter_tps_last")
    data["iter_tps_last_at"] = prev.get("iter_tps_last_at")

    # uptime from the engine's own counter; re-read at most every 60 s (cheap, but no reason
    # to do it per second). None only if /metrics has no process_start_time_seconds.
    if SAMPLER_STATE["boot_time"] is None or now - SAMPLER_STATE["boot_checked_at"] > 60:
        bt = data.get("start_seconds")
        if bt is not None:
            SAMPLER_STATE["boot_time"] = bt
        SAMPLER_STATE["boot_checked_at"] = now

    data["uptime_s"] = round(now - SAMPLER_STATE["boot_time"]) if SAMPLER_STATE["boot_time"] else None

    # live sliding-window rates (2 s window)
    LATEST["data"] = data
    data["decode_tps"] = live_rate(HISTORY, now, data["generation_tokens"], "gen", LIVE_WINDOW_S)
    data["prompt_tps"] = live_rate(HISTORY, now, data["prompt_compute_tokens"], "prompt_compute", PROMPT_WINDOW_S)
    data["iter_tps"] = live_rate(HISTORY, now, data["iteration_tokens_total"], "iter", LIVE_WINDOW_S)
    if data["decode_tps"] is not None and data["decode_tps"] > 0:
        data["decode_tps_last"] = data["decode_tps"]
        data["decode_tps_last_at"] = now
    if data["prompt_tps"] is not None and data["prompt_tps"] > 0:
        data["prompt_tps_last"] = data["prompt_tps"]
        data["prompt_tps_last_at"] = now
    if data["iter_tps"] is not None and data["iter_tps"] > 0:
        data["iter_tps_last"] = data["iter_tps"]
        data["iter_tps_last_at"] = now

    # per-sample deltas for the sparklines
    prev = HISTORY[-1] if HISTORY else None
    sample = {
        "t": now,
        "gen": data["generation_tokens"], "prompt": data["prompt_tokens_total"],
        "prompt_compute": data["prompt_compute_tokens"],
        "iter": data["iteration_tokens_total"], "req": data["requests_total"],
        "acc": data["spec_accepted"], "ver": data["spec_verified"],
        "decode_tps": data["decode_tps"], "prompt_tps": data["prompt_tps"],
        "iter_tps": data["iter_tps"],
        "req_delta": None, "acc_d": None, "ver_d": None,
        "gpu_util": data["gpu"]["util"] if data["gpu"] else None,
        "power_w": data["gpu"]["power_w"] if data["gpu"] else None,
        "kv_usage": data["kv_usage_perc"],
        "running": data["requests_running"], "waiting": data["requests_waiting"],
    }
    if prev is not None and now - prev["t"] >= 0.5:
        sample["req_delta"] = max(0, data["requests_total"] - prev["req"])
        sample["acc_d"] = max(0, data["spec_accepted"] - prev["acc"])
        sample["ver_d"] = max(0, data["spec_verified"] - prev["ver"])
    HISTORY.append(sample)

    # rolling 30 s acceptance (from the per-sample deltas)
    window = [s for s in HISTORY if now - s["t"] <= ACCEPT_WINDOW_S]
    acc_d = sum(s.get("acc_d") or 0 for s in window)
    ver_d = sum(s.get("ver_d") or 0 for s in window)
    data["spec_acceptance_recent"] = round(acc_d / ver_d * 100, 1) if ver_d > 0 else None
    sample["acc30"] = data["spec_acceptance_recent"]
    gen_avg = [s["decode_tps"] for s in window if s.get("decode_tps") is not None]
    if gen_avg:
        data["decode_tps_avg_30s"] = round(sum(gen_avg) / len(gen_avg), 1)
        sample["decode_avg_30s"] = data["decode_tps_avg_30s"]
    else:
        data["decode_tps_avg_30s"] = None
        sample["decode_avg_30s"] = None

    data["history"] = list(HISTORY)
    LATEST["data"] = data
    return data


def sample_loop(base_url, stop_event):
    while not stop_event.is_set():
        t0 = time.time()
        try:
            sample_once(base_url)
        except Exception:
            pass
        elapsed = time.time() - t0
        time.sleep(max(0.1, 1.0 - elapsed))


def collect(base_url):
    """Read the latest sampler snapshot (or sample once if it doesn't exist yet)."""
    data = LATEST["data"]
    if data is None:
        data = sample_once(base_url)
    if data is None:
        return {"url": base_url, "ts": time.time(), "up": False, "history": []}
    # derived values for the page (cheap, no I/O)
    if data.get("metrics_http") == 200:
        if data["spec_drafts"]:
            data["spec_acceptance"] = round(data["spec_accepted"] / data["spec_verified"] * 100, 1) if data["spec_verified"] else 0
            data["spec_tokens_per_draft"] = round(data["spec_accepted"] / data["spec_drafts"], 2)
        if data["prefix_queries"]:
            data["prefix_hit_rate"] = round(data["prefix_hits"] / data["prefix_queries"] * 100, 1)
        if data["prompt_tokens_total"]:
            data["prompt_cached_rate"] = round(data["prompt_tokens_cached"] / data["prompt_tokens_total"] * 100, 1)
    data["history"] = list(HISTORY)
    return data




def fmt_bytes(n):
    if n is None:
        return "—"
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TiB"


def build_html(data):
    """Serve the dashboard with the first snapshot embedded (first paint is real)."""
    safe = json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return HTML.replace("<script>\nconst $ = id",
                        "<script>window.INITIAL=" + safe + "</script>\n<script>\nconst $ = id", 1).encode()


HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inkling 2 Mini — Status</title>
<style>
/* Design tokens for this page only: no external stylesheet, no webfont, one file to ship.
   Closest freely-licensed match for this look is Outfit (SIL OFL 1.1) - to use it, put the
   woff2 subsets and OFL.txt under web/ and uncomment:
     @font-face{font-family:"Outfit";src:url("/web/outfit-latin-wght.woff2") format("woff2-variations");font-weight:100 900;font-display:swap}
   then prepend "Outfit", to --hi-font. */
:root{
  color-scheme:dark;
  --hi-font:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --hi-font-mono:ui-monospace,"SF Mono","Cascadia Mono",Consolas,"Liberation Mono",Menlo,monospace;
  --hi-ease:cubic-bezier(.2,.7,.2,1); --hi-dur:160ms;
  --hi-bg:#0d1014; --hi-surface:#151a1f; --hi-surface-2:#1c2228;
  --hi-line:#28303a; --hi-line-soft:#212831;
  --hi-ink:#e9edf1; --hi-ink-muted:#8d97a3;
  --hi-accent:#35d0a5; --hi-accent-text:#3ddbb0; --hi-focus:#6f9bff;
  --hi-info:#4aa3ff; --hi-warn:#f0b429; --hi-warn-text:#f5c453;
  --hi-danger:#f2555f; --hi-danger-text:#ff7b83; --hi-danger-tint:rgba(242,85,95,.14);
  --hi-r-pill:9999px;
  --hi-sh-card:0 1px 2px rgba(0,0,0,.45),0 8px 24px rgba(0,0,0,.28);
}
[data-theme="light"]{
  color-scheme:light;
  --hi-bg:#f6f7f9; --hi-surface:#ffffff; --hi-surface-2:#f2f4f6;
  --hi-line:#dde3ea; --hi-line-soft:#e9edf1;
  --hi-ink:#1c2126; --hi-ink-muted:#626c76;
  --hi-accent:#0f9f78; --hi-accent-text:#0b7d5f; --hi-focus:#3557d6;
  --hi-info:#0b7fd4; --hi-warn:#b8770a; --hi-warn-text:#8a5a06;
  --hi-danger:#d62839; --hi-danger-text:#a81f2d; --hi-danger-tint:#fdecee;
  --hi-sh-card:0 1px 2px rgba(16,24,32,.06),0 8px 20px rgba(16,24,32,.05);
}
*{box-sizing:border-box}body{margin:0;background:var(--hi-bg);color:var(--hi-ink);font:15px/1.5 var(--hi-font)}
button{font:inherit;cursor:pointer;color:inherit;background:var(--hi-surface-2);border:1px solid var(--hi-line);border-radius:9px;padding:8px 13px;transition:border-color var(--hi-dur) var(--hi-ease)}button:hover{border-color:var(--hi-accent)}button:disabled{opacity:.4;cursor:default}button:focus-visible{outline:2px solid var(--hi-focus);outline-offset:3px}
[hidden]{display:none!important}
header{height:70px;padding:0 32px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--hi-line);background:var(--hi-surface)}
.brand{font-size:23px;font-weight:700;letter-spacing:-.5px}.brand span{font-size:13px;font-weight:400;margin-left:14px;color:var(--hi-ink-muted)}
.controls{display:flex;gap:8px;align-items:center}
.pill{display:inline-flex;align-items:center;gap:9px;padding:7px 14px;border:1px solid var(--hi-line);border-radius:var(--hi-r-pill);background:var(--hi-surface-2);font-size:13px;font-weight:500;white-space:nowrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:100%;background:var(--hi-ink-muted)}
.pill.ok .dot{background:var(--hi-accent);box-shadow:0 0 10px var(--hi-accent)}
.pill.busy .dot{background:var(--hi-warn);box-shadow:0 0 10px var(--hi-warn)}
.pill.down .dot{background:var(--hi-danger);box-shadow:0 0 10px var(--hi-danger)}
main{max-width:1560px;margin:auto;padding:28px 32px}
.top{display:flex;align-items:center;justify-content:space-between;gap:24px}h1{margin:0;font-size:29px;letter-spacing:-.7px}h2{font-size:17px;margin:0}.sub,.muted{color:var(--hi-ink-muted)}.sub{margin:4px 0 0}
.notice{margin-top:16px;padding:10px 14px;background:var(--hi-danger-tint);color:var(--hi-danger-text);border-radius:9px}
.cards{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:14px;margin:24px 0}
.card{padding:18px 20px;border:1px solid var(--hi-line);border-radius:13px;background:var(--hi-surface);min-width:0;box-shadow:var(--hi-sh-card);transition:border-color var(--hi-dur) var(--hi-ease)}
.card:hover{border-color:var(--hi-accent)}
.label{font-size:12px;color:var(--hi-ink-muted);text-transform:uppercase;letter-spacing:.8px}
.value{font-size:28px;font-weight:500;line-height:1.4;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.value.mono{font-family:var(--hi-font-mono);font-size:24px}
.small{font-size:12px;color:var(--hi-ink-muted);margin-top:5px;overflow-wrap:anywhere;line-height:1.5}
.model{font:12px var(--hi-font-mono);overflow-wrap:anywhere;margin-top:4px;color:var(--hi-ink-muted)}
.loaded .dot{background:var(--hi-accent)}.busy .dot{background:var(--hi-warn)}.down .dot{background:var(--hi-danger)}
.live-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:24px 0 14px}
.live-head h2{font-size:17px;margin:0;letter-spacing:-.3px}.live-head .tag{font-size:11px;color:var(--hi-ink-muted);font-family:var(--hi-font-mono)}
.live{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:0 0 20px}
.spark{padding:16px 18px 12px;border:1px solid var(--hi-line);border-radius:13px;background:var(--hi-surface);min-width:0;box-shadow:var(--hi-sh-card);position:relative;transition:border-color var(--hi-dur) var(--hi-ease)}
.spark:hover{border-color:var(--hi-accent)}
.spark .head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:6px}
.spark .value{font-size:22px;margin-top:0;font-family:var(--hi-font-mono);font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.spark .range{font-size:11px;color:var(--hi-ink-muted);font-family:var(--hi-font-mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:46%}
.spark svg{display:block;width:100%;height:64px;overflow:visible}
.spark .axis{display:flex;justify-content:space-between;font-size:10px;color:var(--hi-ink-muted);font-family:var(--hi-font-mono);margin-top:2px}
.spark .tip{position:absolute;top:6px;right:10px;background:var(--hi-surface-2);border:1px solid var(--hi-line);border-radius:7px;padding:4px 8px;font:11px var(--hi-font-mono);color:var(--hi-ink);pointer-events:none;white-space:nowrap;opacity:0;transition:opacity var(--hi-dur) var(--hi-ease);z-index:2}
.panel{border:1px solid var(--hi-line);border-radius:14px;background:var(--hi-surface);overflow:hidden;margin:16px 0;box-shadow:var(--hi-sh-card)}
.panel-bar{padding:15px 20px;border-bottom:1px solid var(--hi-line);display:flex;align-items:center;justify-content:space-between;gap:12px}
.panel-bar .tag{font-size:11px;color:var(--hi-ink-muted);font-family:var(--hi-font-mono)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px 20px;text-align:left;border-bottom:1px solid var(--hi-line-soft)}
th{font-size:11px;color:var(--hi-ink-muted);text-transform:uppercase;letter-spacing:.8px;font-weight:500}
td{font-family:var(--hi-font-mono);font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:340px}
td:first-child{font-family:var(--hi-font);color:var(--hi-ink-muted);font-size:13px}
tr:last-child td{border-bottom:0}
tr:hover td{background:var(--hi-surface-2)}
td .bar{display:inline-block;vertical-align:middle;height:6px;width:110px;border-radius:4px;background:var(--hi-line-soft);overflow:hidden;margin-left:10px}
td .bar i{display:block;height:100%;background:var(--hi-accent);border-radius:4px;transition:width .4s var(--hi-ease)}
.ok{color:var(--hi-accent-text)}.warn{color:var(--hi-warn-text)}.danger{color:var(--hi-danger-text)}
footer{margin-top:14px;font-size:12px;color:var(--hi-ink-muted);display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
#updated{font-family:var(--hi-font-mono)}
@media(max-width:1100px){.cards{grid-template-columns:1fr 1fr}.live{grid-template-columns:repeat(2,1fr)}}
@media(max-width:640px){main{padding:20px 14px}header{padding:0 16px}.cards{grid-template-columns:1fr}.value{font-size:23px}.live{grid-template-columns:1fr}.panel-bar{flex-direction:column;align-items:flex-start;gap:4px}footer{flex-direction:column;gap:4px}}
</style>
</head>
<body>
<header>
  <div class="brand">Inkling 2 Mini <span>HyperInkling status</span></div>
  <div class="controls">
    <span id="pill" class="pill"><span class="dot"></span><span id="pill-text">Connecting…</span></span>
    <button id="theme" aria-label="Switch color theme">Light / dark</button>
  </div>
</header>
<main>
  <div class="top"><div><h1>Inkling 2 Mini</h1><p class="sub" id="subtitle">Reading…</p></div></div>
  <div id="error" class="notice" role="status" hidden></div>
  <div class="cards">
    <div class="card"><div class="label">Model</div><div id="model-state" class="value"><span class="dot"></span><span id="model-state-text">Connecting…</span></div><div id="model" class="model">—</div></div>
    <div class="card"><div class="label">Requests</div><div id="requests" class="value">—</div><div class="small" id="requests-detail">—</div></div>
    <div class="card"><div class="label">Uptime</div><div id="uptime" class="value">—</div><div class="small" id="uptime-detail">—</div></div>
    <div class="card"><div class="label">Live decode</div><div id="speed" class="value mono">—</div><div class="small" id="speed-detail">tokens/second · last 2 s</div></div>
  </div>
  <div class="live-head"><h2>Live</h2><span class="tag">last 15 min · 1 s samples · 2 s window</span></div>
  <div class="live">
    <div class="spark"><div class="head"><div class="label">GPU utilization</div><span class="range" id="gpu-range">—</span></div><div class="value" id="gpu-value">—</div><svg id="spark-gpu" viewBox="0 0 300 64" preserveAspectRatio="none" role="img" aria-label="GPU utilization over the last 15 minutes"></svg><div class="axis"><span>−15 min</span><span>now</span></div><div class="tip" id="tip-gpu"></div></div>
    <div class="spark"><div class="head"><div class="label">Decode</div><span class="range" id="gen-range">—</span></div><div class="value" id="gen-value">—</div><svg id="spark-gen" viewBox="0 0 300 64" preserveAspectRatio="none" role="img" aria-label="Live decode tokens per second over the last 15 minutes"></svg><div class="axis"><span>−15 min</span><span>now</span></div><div class="tip" id="tip-gen"></div></div>
    <div class="spark"><div class="head"><div class="label">Prompt</div><span class="range" id="prompt-range">—</span></div><div class="value" id="prompt-value">—</div><svg id="spark-prompt" viewBox="0 0 300 64" preserveAspectRatio="none" role="img" aria-label="Live prompt tokens per second over the last 15 minutes"></svg><div class="axis"><span>−15 min</span><span>now</span></div><div class="tip" id="tip-prompt"></div></div>
    <div class="spark"><div class="head"><div class="label">Spec acceptance</div><span class="range" id="acc-range">—</span></div><div class="value" id="acc-value">—</div><svg id="spark-acc" viewBox="0 0 300 64" preserveAspectRatio="none" role="img" aria-label="Spec decode acceptance over the last 15 minutes"></svg><div class="axis"><span>−15 min</span><span>now</span></div><div class="tip" id="tip-acc"></div></div>
  </div>
  <div class="panel" id="shape-panel" hidden><div class="panel-bar"><h2>Started as</h2><span class="tag">from the server's own command line</span></div>
    <table>
      <thead><tr><th>Knob</th><th>Value</th><th>Where it came from</th></tr></thead>
      <tbody id="shape"></tbody>
    </table>
  </div>
  <div class="panel"><div class="panel-bar"><h2>Counters since boot</h2><span class="tag" id="boot">—</span></div>
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>Note</th></tr></thead>
      <tbody id="counters"></tbody>
    </table>
  </div>
  <div class="panel"><div class="panel-bar"><h2>Requests by finish reason</h2><span class="tag">since boot</span></div>
    <table>
      <thead><tr><th>finish_reason</th><th>Requests</th><th>%</th></tr></thead>
      <tbody id="reasons"></tbody>
    </table>
  </div>
  <footer><span>Read-only · aggregate counters since boot — vLLM exposes no per-request detail · polled every 5 s</span><span id="updated">Connecting…</span></footer>
</main>
<script>
const $ = id => document.getElementById(id);
function render(d) { renderBody(d); }
let INITIAL = window.INITIAL || null;
const fmt = n => n == null ? "—" : n.toLocaleString("en-US", {maximumFractionDigits: 1});
const fmtB = n => { if (n == null) return "—"; let u=["B","KiB","MiB","GiB"]; let i=0; while (n >= 1024 && i < u.length-1) { n/=1024; i++; } return n.toFixed(i ? 1 : 0) + " " + u[i]; };
const fmtU = s => { if (s == null) return "—"; if (s < 60) return s + " s"; if (s < 3600) return Math.floor(s/60) + " min " + (s%60) + " s"; return Math.floor(s/3600) + " h " + Math.floor((s%3600)/60) + " min"; };
const pct = n => n == null ? "—" : n.toFixed(1) + "%";
const ago = t => { const d = Math.max(0, Math.round(Date.now()/1000 - t)); if (d < 5) return "now"; if (d < 60) return d + " s ago"; if (d < 3600) return Math.floor(d/60) + " min ago"; return Math.floor(d/3600) + " h ago"; };
function theme(){ document.documentElement.dataset.theme = localStorage.getItem("inkling.theme") || "dark"; }
$("theme").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = cur;
  localStorage.setItem("inkling.theme", cur);
});
theme();

function sparkSeries(history, key) {
  const pts = (history || []).filter(p => p && p[key] != null);
  // downsample: at most ~180 points for the SVG
  const step = Math.max(1, Math.ceil(pts.length / 180));
  return pts.filter((_, i) => i % step === 0).map(p => ({t: p.t, v: p[key]}));
}
function sparkline(el, series, color, format) {
  let pts = (series || []).filter(p => p && p.v != null);
  if (pts.length < 2) { el.innerHTML = ""; return null; }
  const now = Date.now()/1000;
  const t1 = now;
  const dataSpan = pts[pts.length - 1].t - pts[0].t;
  const span = Math.max(30, Math.min(900, dataSpan + 5));   // actual span, floor 30 s
  const t0 = t1 - span;
  pts = pts.filter(p => p.t >= t0);
  if (pts.length < 2) { el.innerHTML = ""; return null; }
  let min = Math.min(...pts.map(p => p.v));
  let max = Math.max(...pts.map(p => p.v));
  if (max - min < 1e-6) { max = min + 1; }
  const W = 300, H = 64, pad = 3;
  const x = t => pad + (t - t0) / span * (W - 2 * pad);
  const y = v => pad + (1 - (v - min) / (max - min)) * (H - 2 * pad);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  const area = `${line} L${x(pts[pts.length-1].t).toFixed(1)},${H} L${x(pts[0].t).toFixed(1)},${H} Z`;
  const id = el.id + "-g";
  el.innerHTML = `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".32"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>`
    + `<path d="${area}" fill="url(#${id})"/>`
    + `<path d="${line}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>`
    + `<circle cx="${x(pts[pts.length-1].t).toFixed(1)}" cy="${y(pts[pts.length-1].v).toFixed(1)}" r="3" fill="${color}"/>`;
  const axis = el.parentElement.querySelector(".axis span");
  if (axis) axis.textContent = "−" + (span >= 60 ? Math.round(span/60) + " min" : Math.round(span) + " s");
  const tip = el.parentElement.querySelector(".tip");
  el.onmousemove = e => {
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width * W;
    const t = t0 + (px - pad) / (W - 2 * pad) * span;
    let best = pts[0], bestD = Infinity;
    for (const p of pts) { const d = Math.abs(p.t - t); if (d < bestD) { bestD = d; best = p; } }
    tip.textContent = `${ago(best.t)} · ${format(best.v)}`;
    tip.style.opacity = "1";
  };
  el.onmouseleave = () => { tip.style.opacity = "0"; };
  return {min, max, last: pts[pts.length - 1].v};
}

function renderBody(d) {
  try {
    const running = d.requests_running || 0, waiting = d.requests_waiting || 0;
    const down = !d.up;
    const generating = running > 0;
    const pill = $("pill");
    pill.className = "pill " + (down ? "down" : (generating ? "busy" : "ok"));
    $("pill-text").textContent = down ? "Down" : (generating ? `In use · ${running} running` : "Idle");
    $("error").hidden = !down;
    if (down) $("error").textContent = "Unreachable at " + d.url + " — the page keeps polling.";
    $("subtitle").innerHTML = down
      ? "Reading " + d.url + "…"
      : "Serving <b>" + (d.model || "?") + "</b> at <span id='url'>" + d.url + "</span> · " + (d.version || "?");
    const ms = $("model-state");
    ms.className = "value " + (down ? "down" : (generating ? "busy" : "loaded"));
    $("model-state-text").textContent = down ? "Down" : (generating ? "In use" : "Loaded");
    $("model").textContent = (d.model || "—") + " · " + (d.version || "?");
    $("requests").textContent = fmt(d.requests_total ?? 0);
    $("requests-detail").textContent = running + " running · " + waiting + " waiting · " + fmt(d.requests_success?.stop ?? 0) + " stop · " + fmt(d.requests_success?.length ?? 0) + " length";
    $("uptime").textContent = d.uptime_s == null ? "—" : fmtU(d.uptime_s);
    $("uptime-detail").textContent = d.uptime_s == null ? "Unknown until the server reports a counter reset" : "Since " + new Date(Date.now() - d.uptime_s * 1000).toLocaleTimeString() + " boot";
    const live = d.decode_tps, last = d.decode_tps_last;
    if (live != null && live > 0) {
      $("speed").textContent = fmt(live) + " tok/s";
      $("speed-detail").textContent = "tokens/second · last 2 s · live";
    } else if (last != null) {
      $("speed").textContent = fmt(last) + " tok/s";
      $("speed-detail").textContent = "tokens/second · last known " + ago(d.decode_tps_last_at);
    } else {
      $("speed").textContent = "—";
      $("speed-detail").textContent = "tokens/second · no request yet";
    }
    $("boot").textContent = d.uptime_s == null ? "—" : "up " + fmtU(d.uptime_s);

    const history = d.history || [];
    const gpuSeries = sparkSeries(history, "gpu_util");
    sparkline($("spark-gpu"), gpuSeries, "var(--hi-warn)", v => Math.round(v) + "%");
    $("gpu-value").textContent = d.gpu ? Math.round(d.gpu.util) + "%" : "—";
    $("gpu-range").textContent = d.gpu ? fmtB(d.gpu.mem_used_mib*1048576) + " / " + fmtB(d.gpu.mem_total_mib*1048576) + " · " + Math.round(d.gpu.temp_c) + "°C · " + Math.round(d.gpu.power_w) + " W (limit " + Math.round(d.gpu.power_limit_w) + ")" : "nvidia-smi unavailable";

    const genSeries = sparkSeries(history, "decode_tps");
    const genInfo = sparkline($("spark-gen"), genSeries, "var(--hi-accent)", v => v.toFixed(1) + " tok/s");
    $("gen-value").textContent = (d.decode_tps != null && d.decode_tps > 0) ? fmt(d.decode_tps) + " tok/s" : (d.decode_tps_last != null ? fmt(d.decode_tps_last) + " tok/s" : "—");
    $("gen-range").textContent = (d.decode_tps != null && d.decode_tps > 0)
      ? "avg " + fmt(d.decode_tps_avg_30s ?? 0) + " tok/s · 30 s"
      : (d.decode_tps_last_at ? "last known " + ago(d.decode_tps_last_at) : "waiting for samples");

    const promptSeries = sparkSeries(history, "prompt_tps");
    const promptInfo = sparkline($("spark-prompt"), promptSeries, "var(--hi-info)", v => v.toFixed(1) + " tok/s");
    $("prompt-value").textContent = (d.prompt_tps != null && d.prompt_tps > 0) ? fmt(d.prompt_tps) + " tok/s" : (d.prompt_tps_last != null ? fmt(d.prompt_tps_last) + " tok/s" : "—");
    $("prompt-range").textContent = (d.prompt_tps != null && d.prompt_tps > 0)
      ? "iter " + fmt(d.iter_tps ?? 0) + " tok/s"
      : (d.prompt_tps_last_at ? "last known " + ago(d.prompt_tps_last_at) : "waiting for samples");

    const accSeries = sparkSeries(history, "acc30");
    const accInfo = sparkline($("spark-acc"), accSeries, "var(--hi-accent-text)", v => v.toFixed(1) + "%");
    $("acc-value").textContent = d.spec_acceptance_recent != null ? pct(d.spec_acceptance_recent) : "—";
    $("acc-range").textContent = d.spec_acceptance != null ? "since boot " + pct(d.spec_acceptance) : "waiting for samples";

    const rows = [
      ["Requests", fmt(d.requests_total), "success, all reasons"],
      ["Stop / length", fmt(d.requests_success?.stop) + " / " + fmt(d.requests_success?.length), "clean finish / hit max tokens"],
      ["Running / waiting", (d.requests_running ?? 0) + " / " + (d.requests_waiting ?? 0), "in the batch / queued"],
      ["Preemptions", fmt(d.preemptions ?? 0), "evicted and recomputed"],
      ["Admission rejections", fmt(d.admission_rejections ?? 0), "refused at admission"],
      ["Generation tokens", fmt(d.generation_tokens ?? 0), "completion tokens, all requests"],
      ["Prompt tokens", fmt(d.prompt_tokens_total ?? 0), "all prompts since boot"],
      ["Prompt cached", fmt(d.prompt_tokens_cached ?? 0) + " (" + pct(d.prompt_cached_rate) + ")", "hits in GPU prefix cache"],
      ["Prefix cache hits", fmt(d.prefix_hits ?? 0) + " / " + fmt(d.prefix_queries ?? 0), "hit / queried tokens"],
      ["External prefix hits", fmt(d.external_prefix_hits ?? 0), "CPU tier / connector reloads"],
      ["Spec drafts", fmt(d.spec_drafts ?? 0), "steps drafted"],
      ["Spec verified", fmt(d.spec_verified ?? 0), "tokens verified"],
      ["Spec accepted", fmt(d.spec_accepted ?? 0), pct(d.spec_acceptance) + " acceptance (since boot)"],
      ["Tool calls", fmt(d.tool_calls ?? 0) + " / " + fmt(d.tool_call_invocations ?? 0), "parsed calls / invocations"],
      ["KV offload stored", fmtB(d.offload_stored), "GPU → CPU"],
      ["KV offload reloaded", fmtB(d.offload_reloaded), "CPU → GPU"],
      ["KV usage", d.kv_usage_perc != null ? Math.round(d.kv_usage_perc*100) + "%" : "—", "of the GPU pool"],
      ["TTFT avg", d.time_to_first_token_avg != null ? d.time_to_first_token_avg + " s" : "—", "mean first-token time, all requests"],
    ];
    $("counters").innerHTML = rows.map(([k,v,n]) => "<tr><td>" + k + "</td><td>" + v + "</td><td style='color:var(--hi-ink-muted)'>" + n + "</td></tr>").join("");

    // What the launcher actually started it with (the derived values, not what .env said).
    // Absent when the engine is not this container's PID 1: the panel hides itself.
    const sp = d.shape || {}, spec = sp.spec || {};
    const sched = spec.num_speculative_tokens_per_batch_size, img = (sp.images || {}).image || {};
    $("shape-panel").hidden = !d.shape;
    const shapeRows = [
      ["Model", d.model || "—", "served name" + (d.max_model_len != null ? " · " + fmt(d.max_model_len) + " tokens per request" : "")],
      ["Concurrent", sp.max_seqs || "—", "--max-num-seqs"],
      ["Prefill chunk", sp.nbt || "—", "--max-num-batched-tokens" + (sp.long_prefill ? ", long-prefill cap " + sp.long_prefill : "")],
      ["KV cache", [sp.kv_dtype, sp.gpu_util ? Math.round(sp.gpu_util*100) + "% of the card" : null].filter(Boolean).join(" · ") || "—",
       (sp.prefix_caching ? "prefix caching on" : "prefix caching off") + (sp.kv_offload_gib ? " · CPU tier " + sp.kv_offload_gib + " GiB" : " · no CPU tier")],
      ["Speculation", spec.method ? spec.method + " drafts " + spec.num_speculative_tokens + (sched ? ", verify " + JSON.stringify(sched) : "") : "off", "--speculative-config"],
      ["Images", img.count ? img.count + " per prompt, max " + img.width + "×" + img.height : "off", "--limit-mm-per-prompt"],
    ];
    $("shape").innerHTML = shapeRows.map(([k,v,n]) => "<tr><td>" + k + "</td><td>" + v + "</td><td style='color:var(--hi-ink-muted)'>" + n + "</td></tr>").join("");

    const rs = d.requests_success || {};
    const total = Object.values(rs).reduce((a,b)=>a+b,0);
    $("reasons").innerHTML = Object.entries(rs).filter(([k,v]) => v > 0 || k === "stop").map(([k,v]) => {
      const w = total ? Math.max(2, v / total * 100) : 0;
      return "<tr><td>" + k + "</td><td>" + fmt(v) + "<span class='bar'><i style='width:" + w.toFixed(1) + "%'></i></span></td><td>" + (total ? (v/total*100).toFixed(1) : "0.0") + "%</td></tr>";
    }).join("");

    $("updated").textContent = "Live · updated " + new Date().toLocaleTimeString();
  } catch (e) {
    $("pill-text").textContent = "Unreachable";
    $("pill").className = "pill down";
    $("updated").textContent = "Disconnected · retrying";
  }
}
async function refresh() {
  try {
    const r = await fetch("/api/status", {cache:"no-store"});
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    renderBody(d);
  } catch (e) {
    if (INITIAL) renderBody(INITIAL);
    else { $("pill-text").textContent = "Unreachable"; $("pill").className = "pill down"; $("updated").textContent = "Disconnected · retrying"; }
  }
}
if (INITIAL) renderBody(INITIAL);
setInterval(refresh, 5000);
refresh();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "InklingStatus/2.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, body, ctype, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path in ("", "/index.html"):
            now = time.time()
            if now - CACHE["ts"] < CACHE_TTL and CACHE["data"] is not None:
                data = CACHE["data"]
            else:
                data = collect(self.server.base_url)
                CACHE.update(ts=now, data=data)
            self._send(200, build_html(data), "text/html; charset=utf-8")
            return

        if path == "/api/status":
            now = time.time()
            if now - CACHE["ts"] < CACHE_TTL and CACHE["data"] is not None:
                payload = CACHE["data"]
            else:
                payload = collect(self.server.base_url)
                CACHE.update(ts=now, data=payload)
            self._send(200, json.dumps(payload).encode(), "application/json")
            return

        if path == "/health":
            status, _ = get(self.server.base_url + "/health")
            self._send(200 if status == 200 else 503, b"ok" if status == 200 else b"down", "text/plain")
            return

        if path.startswith("/web/"):
            # Optional assets for the stylesheet (see the token block). Nothing is served
            # from outside web/, and only these two extensions are ever read.
            f = (WEB_DIR / path[len("/web/"):]).resolve()
            if f.is_file() and f.is_relative_to(WEB_DIR.resolve()) and f.suffix in (".woff2", ".css"):
                self._send(200, f.read_bytes(),
                           "font/woff2" if f.suffix == ".woff2" else "text/css; charset=utf-8")
            else:
                self._send(404, b'{"error":"not found"}', "application/json")
            return

        self._send(404, b'{"error":"not found"}', "application/json")


def main():
    ap = argparse.ArgumentParser(description="Status page for the Inkling 2 Mini server")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; use --bind 0.0.0.0 to expose on the LAN/tailnet)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8082")),
                    help="port (default $DASHBOARD_PORT, else 8082)")
    ap.add_argument("--url",
                    default="http://127.0.0.1:" + os.environ.get("PORT", "8000"),
                    help="Inkling base URL (default http://127.0.0.1:$PORT, which is how the "
                         "container reaches the engine; pass it explicitly from outside)")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.base_url = args.url.rstrip("/")

    stop_event = threading.Event()
    sampler = threading.Thread(target=sample_loop,
                               args=(server.base_url, stop_event),
                               daemon=True, name="status-sampler")
    sampler.start()

    print(f"[status-page] http://{args.bind}:{args.port} -> {server.base_url}, sampler 1 s")
    try:
        server.serve_forever()
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Status page of a running server. Standard library only, read-only.

    python3 scripts/status_page.py [--bind 127.0.0.1] [--port 8082] [--url http://127.0.0.1:8080]

start.sh runs it beside the engine (DASHBOARD=1) with --parent: the page then ends with the
engine and reads the launch settings from its command line. A thread samples /metrics and
nvidia-smi every SAMPLE_S; requests only ever read the last sample.

    GET /            the page (status_page.html), which polls /api/status
    GET /api/status  JSON: the last sample and the history since ?since=<time>
    GET /health      200 while the engine answers, else 503

With VLLM_API_KEY set, all but /health need it: as a Bearer token, or as the password the
browser asks for (any user name).
"""
import argparse
import base64
import binascii
import hmac
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PAGE = Path(__file__).with_name("status_page.html")
SAMPLE_S = 2
WINDOW_S = 10  # rates: vLLM counts a prompt when its prefill ends, so short windows jump
HISTORY_S = 900
FINISH = ("stop", "length", "abort", "error", "repetition")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def fetch(url, timeout=4):
    """Body of a 200 answer, else None. --api-key guards /metrics and /v1/models."""
    req = urllib.request.Request(url)
    if key := os.environ.get("VLLM_API_KEY"):
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode()
    except (OSError, urllib.error.URLError):
        return None


def parse_metrics(text):
    """{name: [(labels, value)]}, histogram buckets left out."""
    out = {}
    for line in text.splitlines():
        head, _, value = line.rpartition(" ")
        if not head or head[0] == "#" or "_bucket{" in head:
            continue
        name, _, labels = head.partition("{")
        try:
            out.setdefault(name, []).append((dict(LABEL.findall(labels)), float(value)))
        except ValueError:
            continue
    return out


def total(metrics, name, **labels):
    """Sum of the series of vllm:`name` that carry `labels`."""
    return sum(v for have, v in metrics.get(f"vllm:{name}", [])
               if all(have.get(k) == want for k, want in labels.items()))


def mean(metrics, name):
    n = total(metrics, f"{name}_count")
    return total(metrics, f"{name}_sum") / n if n else None


def counters(m):
    return {
        "generated": total(m, "generation_tokens_total"),
        "prompt": total(m, "prompt_tokens_total"),
        "computed": total(m, "prompt_tokens_by_source_total", source="local_compute"),
        "gpu_hits": total(m, "prompt_tokens_cached_by_source_total", source="device"),
        "cpu_hits": total(m, "prompt_tokens_cached_by_source_total", source="host"),
        "drafts": total(m, "spec_decode_num_drafts_total"),
        "verified": total(m, "spec_decode_num_verified_tokens_total"),
        "accepted": total(m, "spec_decode_num_accepted_tokens_total"),
        "preemptions": total(m, "num_preemptions_total"),
        "rejected": total(m, "admission_rejections_total"),
        "offload_stored": total(m, "kv_offload_store_bytes_total"),
        "offload_loaded": total(m, "kv_offload_load_bytes_total"),
        "finished": {r: total(m, "request_success_total", finished_reason=r) for r in FINISH},
        "ttft_s": mean(m, "time_to_first_token_seconds"),
        "queue_s": mean(m, "request_queue_time_seconds"),
        "itl_s": mean(m, "inter_token_latency_seconds"),
    }


def gpu_info():
    """The first GPU this process is given, or None without nvidia-smi."""
    cmd = ["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=utilization.gpu,"
           "memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm"]
    if visible := os.environ.get("CUDA_VISIBLE_DEVICES"):
        cmd += ["-i", visible.split(",")[0]]
    try:
        row = subprocess.check_output(cmd, text=True, timeout=5).splitlines()[0]
        keys = ("util", "used_mib", "total_mib", "temp_c", "power_w", "limit_w", "sm_mhz")
        return dict(zip(keys, map(float, row.split(","))))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def launch_settings(pid):
    """What the engine was started with, from its command line and environment (start.sh
    derives some values), or None when `pid` is not `vllm serve`."""
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
    except (OSError, UnicodeDecodeError):
        return None
    if "serve" not in argv[:4]:
        return None
    try:
        environ = dict(kv.partition("=")[::2] for kv in
                       Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0"))
    except OSError:
        environ = {}
    flags = {}
    for i, arg in enumerate(argv):
        if arg.startswith("--"):
            key, eq, value = arg[2:].partition("=")
            follows = argv[i + 1] if i + 1 < len(argv) else "--"
            flags[key] = value if eq else "" if follows.startswith("--") else follows

    def parsed(flag):
        try:
            return json.loads(flags[flag])
        except (KeyError, ValueError):
            return {}

    return {
        "max_seqs": flags.get("max-num-seqs"),
        "nbt": flags.get("max-num-batched-tokens"),
        "long_prefill": flags.get("long-prefill-token-threshold"),
        "kv_dtype": flags.get("kv-cache-dtype"),
        "gpu_util": flags.get("gpu-memory-utilization"),
        "keep_free_gib": environ.get("HYPERINKLING_KV_KEEP_FREE_GIB") or None,
        "offload_gib": flags.get("kv-offloading-size"),
        "spec": parsed("speculative-config"),
        "image": parsed("limit-mm-per-prompt").get("image", {}),
    }


class Sampler:
    def __init__(self, url, parent=None):
        self.url, self.parent = url, parent
        self.lock = threading.Lock()
        self.state = {"up": False, "url": url}
        self.window = deque()  # (time, counters) over WINDOW_S
        self.history = deque(maxlen=HISTORY_S // SAMPLE_S)
        self.boot = None
        self.server = {}

    def _identify(self):
        """The server's name and settings, once per boot."""
        models = json.loads(fetch(f"{self.url}/v1/models") or "{}").get("data") or [{}]
        version = json.loads(fetch(f"{self.url}/version") or "{}").get("version")
        self.server = {"model": models[0].get("id"), "root": models[0].get("root"),
                       "max_model_len": models[0].get("max_model_len"), "version": version,
                       "settings": launch_settings(self.parent) if self.parent else None}

    def sample(self):
        now, gpu = time.time(), gpu_info()
        text = fetch(f"{self.url}/metrics")
        if text is None:
            with self.lock:
                self.window.clear()
                self.state = {**self.state, "up": False, "time": now, "gpu": gpu}
            return
        m = parse_metrics(text)
        boot = (m.get("process_start_time_seconds") or [({}, None)])[0][1]
        if boot != self.boot or not self.server.get("model"):
            self._identify()
        c = counters(m)
        with self.lock:
            if boot != self.boot:
                self.boot = boot
                self.window.clear()
                self.history.clear()
            while len(self.window) > 1 and self.window[0][0] < now - WINDOW_S:
                self.window.popleft()
            then, old = self.window[0] if self.window else (now, c)
            self.window.append((now, c))
            rate = lambda key: (c[key] - old[key]) / (now - then) if now > then else 0.0  # noqa: E731
            verified = c["verified"] - old["verified"]
            live = {
                "decode": rate("generated"), "prefill": rate("computed"),
                "acceptance": (c["accepted"] - old["accepted"]) / verified if verified else None,
                "running": total(m, "num_requests_running"),
                "waiting": total(m, "num_requests_waiting"),
                "kv": total(m, "kv_cache_usage_perc"),
            }
            self.history.append({"t": now, "gpu": gpu and gpu["util"], **{
                k: live[k] for k in ("decode", "prefill", "acceptance", "kv")}})
            self.state = {"up": True, "url": self.url, "time": now, "boot": boot, "gpu": gpu,
                          "live": live, "counters": c, **self.server}

    def status(self, since=0.0):
        with self.lock:
            return {**self.state, "window_s": WINDOW_S, "history_s": HISTORY_S,
                    "history": [h for h in self.history if h["t"] > since]}

    def run(self):
        while self.parent is None or os.getppid() == self.parent:
            t0 = time.monotonic()
            try:
                self.sample()
            except Exception as e:  # the page must outlive a bad sample
                print(f"[status] sample failed: {e!r}", flush=True)
            time.sleep(max(0.1, SAMPLE_S - (time.monotonic() - t0)))
        os._exit(0)  # the engine is gone


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype, headers=None):
        self.send_response(code)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        key = os.environ.get("VLLM_API_KEY")
        if not key:
            return True
        scheme, _, given = self.headers.get("Authorization", "").partition(" ")
        if scheme.lower() == "basic":
            try:
                given = base64.b64decode(given).decode().partition(":")[2]
            except (binascii.Error, UnicodeDecodeError):
                return False
        return hmac.compare_digest(given.encode(), key.encode())

    def do_GET(self):
        url, sampler = urlsplit(self.path), self.server.sampler
        if url.path != "/health" and not self._authorized():
            self._send(401, b"unauthorized\n", "text/plain",
                       {"WWW-Authenticate": 'Basic realm="HyperInkling"'})
            return
        if url.path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/status":
            try:
                since = float(parse_qs(url.query).get("since", ["0"])[0])
            except ValueError:
                since = 0.0
            self._send(200, json.dumps(sampler.status(since)).encode(), "application/json")
        elif url.path == "/health":
            up = fetch(f"{sampler.url}/health") is not None
            self._send(200 if up else 503, b"ok\n" if up else b"down\n", "text/plain")
        else:
            self._send(404, b"not found\n", "text/plain")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="the engine")
    ap.add_argument("--parent", action="store_true", help="the engine is the parent process")
    args = ap.parse_args()
    ThreadingHTTPServer.address_family = socket.AF_INET6 if ":" in args.bind else socket.AF_INET
    try:
        server = ThreadingHTTPServer((args.bind, args.port), Handler)
    except OSError as e:
        raise SystemExit(f"[status] not started on {args.bind}:{args.port}: {e.strerror}")
    server.daemon_threads = True
    server.sampler = Sampler(args.url.rstrip("/"), os.getppid() if args.parent else None)
    threading.Thread(target=server.sampler.run, daemon=True).start()
    print(f"[status] http://{args.bind}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

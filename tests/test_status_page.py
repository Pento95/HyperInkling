"""scripts/status_page.py against a fake engine. No GPU.

    venv/bin/python -m pytest -q tests/test_status_page.py
"""

import base64
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "status_page.py"
_spec = importlib.util.spec_from_file_location("status_page", MODULE)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


class FakeEngine:
    def __init__(self):
        self.generated, self.boot, self.key, self.up = 100, 1000.0, None, True
        engine = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                authorized = self.headers.get("Authorization") == f"Bearer {engine.key}"
                guarded = self.path in ("/metrics", "/v1/models")
                code = 503 if not engine.up else 401 if engine.key and guarded and not authorized else 200
                body = {
                    "/metrics": "# HELP x\n"
                    f'vllm:generation_tokens_total{{model_name="a,b}}"}} {engine.generated}\n'
                    "vllm:generation_tokens_created 5e9\n"
                    'vllm:admission_rejections_total{reason="a"} 1\n'
                    'vllm:admission_rejections_total{reason="b"} 2\n'
                    'vllm:ttft_bucket{le="1"} 3\n'
                    f"process_start_time_seconds {engine.boot}\n",
                    "/v1/models": json.dumps({"data": [{"id": "m", "max_model_len": 10}]}),
                    "/version": '{"version":"v"}',
                }.get(self.path, "")
                self.send_response(code)
                self.end_headers()
                self.wfile.write(body.encode())

        self.server, self.url = serve(H)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(sp, "gpu_info", lambda: None)
    e = FakeEngine()
    yield e
    e.server.shutdown()


def test_series_are_summed_and_names_are_exact(engine):
    s = sp.Sampler(engine.url)
    s.sample()
    c = s.status()["counters"]
    assert c["generated"] == 100  # not _created
    assert c["rejected"] == 3


def test_rate_and_restart(engine):
    s = sp.Sampler(engine.url)
    s.sample()
    time.sleep(0.2)
    engine.generated = 300
    s.sample()
    assert 0 < s.status()["live"]["decode"] <= 1000
    engine.generated, engine.boot = 5, 2000.0
    s.sample()
    st = s.status()
    assert st["live"]["decode"] == 0 and len(st["history"]) == 1


def test_engine_down_keeps_the_counters_and_returns(engine):
    s = sp.Sampler(engine.url)
    s.sample()
    engine.up = False
    s.sample()
    assert not s.status()["up"] and s.status()["counters"]["generated"] == 100
    engine.up = True
    s.sample()
    assert s.status()["up"]


def test_api_key_is_sent(engine, monkeypatch):
    engine.key = "k"
    s = sp.Sampler(engine.url)
    s.sample()
    assert not s.status()["up"]
    monkeypatch.setenv("VLLM_API_KEY", "k")
    s.sample()
    assert s.status()["model"] == "m"


def test_history_since(engine):
    s = sp.Sampler(engine.url)
    s.sample()
    first = s.status()["history"][0]["t"]
    s.sample()
    assert len(s.status()["history"]) == 2 and len(s.status(first)["history"]) == 1


def test_http(engine):
    server, url = serve(sp.Handler)
    server.sampler = sp.Sampler(engine.url)
    server.sampler.sample()
    assert b"api/status" in urllib.request.urlopen(url + "/").read()
    assert json.load(urllib.request.urlopen(url + "/api/status?since=x"))["model"] == "m"
    engine.up = False
    for path, code in (("/health", 503), ("/../start.sh", 404)):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(url + path)
        assert e.value.code == code
    server.shutdown()


def test_the_page_needs_the_key(engine, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "k:1")
    server, url = serve(sp.Handler)
    server.sampler = sp.Sampler(engine.url)

    def code(path, auth=None):
        req = urllib.request.Request(url + path, headers={"Authorization": auth} if auth else {})
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as e:
            return e.code

    basic = "Basic " + base64.b64encode(b"any:k:1").decode()
    assert [code("/"), code("/api/status"), code("/health")] == [401, 401, 200]
    assert [code("/", a) for a in ("Bearer k:1", basic)] == [200, 200]
    assert [code("/", a) for a in ("Bearer no", "Basic %%%", "Basic " + base64.b64encode(b"k:1").decode())] == [401] * 3
    server.shutdown()


def test_launch_settings_of_a_real_process():
    args = ["serve", "m", "--max-num-seqs", "8", "--kv-cache-dtype=fp8", "--enable-prefix-caching",
            "--speculative-config", '{"method":"dspark"}', "--limit-mm-per-prompt", "broken"]
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", *args],
                         env={**os.environ, "HYPERINKLING_KV_KEEP_FREE_GIB": "2"})
    try:
        time.sleep(0.3)
        s = sp.launch_settings(p.pid)
    finally:
        p.kill()
    assert (s["max_seqs"], s["kv_dtype"], s["spec"]["method"], s["image"]) == ("8", "fp8", "dspark", {})
    assert s["keep_free_gib"] == "2"
    assert sp.launch_settings(1 << 30) is None


def test_ends_with_its_parent():
    """`kill` of the engine alone must not leave the page holding its port."""
    out = subprocess.run(
        ["bash", "-c", f"{sys.executable} {MODULE} --parent --port 0 >/dev/null 2>&1 & echo $!; sleep 0.5"],
        capture_output=True, text=True, timeout=10)
    pid = int(out.stdout)
    for _ in range(60):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().rpartition(")")[2].split()[0] == "Z":
            return
        time.sleep(0.1)
    subprocess.run(["kill", str(pid)])
    pytest.fail("still running")

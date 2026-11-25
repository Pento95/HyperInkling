"""Status page: stdlib-only, read-only, no GPU and no running server.

    venv/bin/python -m pytest -q tests/test_status_page.py

The checks are the ones that keep it shippable: no third-party imports, one file (no
stylesheet or font to ship), every CSS custom property the markup uses actually defined,
and the launcher-facing parts (the shape read from the engine's own argv, the API key,
the optional-assets route) behaving when there is no engine to ask.
"""

import ast
import http.client
import importlib.util
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "status_page.py"
_spec = importlib.util.spec_from_file_location("status_page", MODULE)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

STDLIB = {"__future__", "argparse", "collections", "http", "json", "os", "pathlib", "re",
          "subprocess", "threading", "time", "urllib"}


def test_stdlib_only():
    """The container has vLLM's dependencies; the page must not want any of them."""
    tree = ast.parse(MODULE.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods - STDLIB == set()


def test_css_tokens_all_defined():
    """`var(--x)` with no definition renders as an unset color, not an error - silent."""
    used = set(sp.re.findall(r"var\((--[a-z0-9-]+)\)", sp.HTML))
    defined = set(sp.re.findall(r"(--[a-z0-9-]+)\s*:", sp.HTML))
    assert used - defined == set()
    assert len(used) > 10, "the stylesheet should be built from its tokens"
    assert not sp.re.search(r"--st-", sp.HTML), "the token prefix belongs to this page"


def test_the_page_is_one_file():
    """No stylesheet, no webfont: nothing to COPY and nothing to license."""
    assert "<link" not in sp.HTML
    assert "tokens.css" not in sp.HTML
    assert ":root{" in sp.HTML and '[data-theme="light"]' in sp.HTML


def test_prom_parser():
    p = sp.parse_prom(
        "# HELP vllm:x a counter\n# TYPE vllm:x counter\n"
        'vllm:x{source="local_compute",model="m"} 42\n'
        'vllm:x{source="local_cache_hit"} 7\n'
        "vllm:y 1.5\n")
    assert sp.get_one(p, "vllm:x", source="local_compute") == 42
    assert sp.get_one(p, "vllm:x", source="local_cache_hit") == 7
    assert sp.get_all(p, "vllm:x") == 49
    assert sp.get_one(p, "vllm:y") == 1.5
    assert sp.get_one(p, "vllm:missing") is None


def test_live_rate():
    hist = [{"t": 0.0, "gen": 100.0}, {"t": 1.0, "gen": 200.0}]
    assert sp.live_rate(hist, 1.0, 300.0, "gen", 1.0) == pytest.approx(200.0)
    assert sp.live_rate(hist[:1], 1.0, 300.0, "gen", 1.0) is None           # too few samples
    assert sp.live_rate(hist, 1.0, 300.0, "gen", 5.0) is None               # window older than history
    fresh = [{"t": 0.9, "gen": 100.0}, {"t": 1.0, "gen": 200.0}]
    assert sp.live_rate(fresh, 1.0, 250.0, "gen", 0.1) is None              # base sample under 0.25 s old
    assert sp.live_rate(hist, 1.0, 50.0, "gen", 1.0) == 0.0                 # counter reset: never negative
    assert sp.live_rate(hist, 1.0, None, "gen", 1.0) is None                # a counter this build lacks


def test_shape_from_argv_accepts_the_equals_form(tmp_path):
    """`vllm serve --max-num-seqs=7` is the same flag to argparse, and command lines copied from
    vLLM's own docs write it that way; reading only the space form loses the row silently."""
    f = tmp_path / "cmdline"
    f.write_bytes(b"\0".join([b"vllm", b"serve", b"m",
                              b"--max-num-seqs=7", b"--enable-prefix-caching=true",
                              b"--speculative-config={\"method\":\"dspark\"}"]) + b"\0")
    shape = sp.shape_from_argv(cmdline_path=f)
    assert shape["max_seqs"] == "7" and shape["prefix_caching"] is True
    assert shape["spec"]["method"] == "dspark"


def test_shape_from_argv_reads_the_derived_shape(tmp_path):
    """start.sh derives LONG_PREFILL from NBT and maps MAX_LEN=auto to -1: show those."""
    f = tmp_path / "cmdline"
    f.write_bytes(b"\0".join([
        b"/usr/bin/python3", b"/usr/local/bin/vllm", b"serve", b"/app/models/x",
        b"--max-num-seqs", b"6", b"--max-num-batched-tokens", b"2048",
        b"--gpu-memory-utilization", b"0.95", b"--kv-cache-dtype", b"fp8",
        b"--enable-prefix-caching", b"--kv-offloading-size", b"16",
        b"--long-prefill-token-threshold", b"1024",
        b"--speculative-config", b'{"method":"dspark","num_speculative_tokens":7}',
        b"--limit-mm-per-prompt", b'{"image":{"count":50,"width":1240,"height":1760}}']) + b"\0")
    shape = sp.shape_from_argv(cmdline_path=f)
    assert shape["max_seqs"] == "6" and shape["nbt"] == "2048"
    assert shape["long_prefill"] == "1024" and shape["prefix_caching"] is True
    assert shape["spec"]["method"] == "dspark"
    assert shape["images"]["image"]["width"] == 1240


def test_shape_none_when_pid1_is_not_the_engine(tmp_path):
    f = tmp_path / "cmdline"
    f.write_bytes(b"/bin/bash\0--login\0")
    assert sp.shape_from_argv(cmdline_path=f) is None
    assert sp.shape_from_argv(cmdline_path=tmp_path / "absent") is None


class _Echo(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        _Echo.seen.append(self.headers.get("Authorization"))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


@pytest.fixture
def echo_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def test_api_key_is_forwarded(echo_server, monkeypatch):
    """/metrics and /v1/models are guarded when VLLM_API_KEY is set; the page is inside
    the container, so it has the key in its own environment and must use it."""
    _Echo.seen.clear()
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert sp.get(echo_server)[0] == 200 and _Echo.seen[-1] is None
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    assert sp.get(echo_server)[0] == 200 and _Echo.seen[-1] == "Bearer secret"


@pytest.fixture
def page_server():
    """The dashboard itself, pointed at a port nothing listens on (CACHE cleared, see above)."""
    sp.CACHE.update(ts=0.0, data=None)
    sp.LATEST["data"] = None
    sp.HISTORY.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), sp.Handler)
    srv.base_url = "http://127.0.0.1:1"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def test_health_reports_the_server_not_itself(page_server):
    """503 means "the engine is not answering", which is what a container healthcheck wants."""
    try:
        urllib.request.urlopen(page_server + "/health", timeout=5)
        raise AssertionError("expected 503")
    except urllib.error.HTTPError as e:
        assert e.code == 503


def test_page_serves_without_a_server(page_server):
    body = urllib.request.urlopen(page_server + "/", timeout=5).read().decode()
    assert "Inkling 2 Mini" in body and "window.INITIAL" in body


def test_optional_assets_route_stays_inside_web(page_server):
    """web/ is opt-in for a licensed font; the route must not turn into a file read."""
    for path in ("/web/x.woff2", "/web/../status_page.py", "/web/../../etc/passwd", "/web/x.py"):
        conn = http.client.HTTPConnection("127.0.0.1", urllib.parse.urlparse(page_server).port)
        conn.putrequest("GET", path, skip_host=True)          # http.client: no path normalisation
        conn.putheader("Host", "127.0.0.1")
        conn.endheaders()
        assert conn.getresponse().status == 404, path
        conn.close()


def test_json_shape_of_the_api(page_server):
    """Unreachable engine: the payload is deliberately small, and says so."""
    d = json.loads(urllib.request.urlopen(page_server + "/api/status", timeout=8).read())
    assert d["up"] is False and d["metrics_http"] is None and "history" in d
    # A cold start against a dead engine still reports the card and the shape: those come
    # from this process, and they are what you want when the engine will not come up.
    assert "gpu" in d and "shape" in d


# The five endpoints fetch_all reads, with a canned /metrics in the shape vLLM really writes.
# Kept short on purpose: one line per metric the page sums, labels included.
METRICS = """# HELP vllm:num_requests_running Running requests
vllm:num_requests_running{engine="0",model_name="inkling-2-mini"} 2.0
vllm:num_requests_waiting{engine="0",model_name="inkling-2-mini"} 1.0
vllm:num_preemptions_total{model_name="inkling-2-mini"} 3.0
vllm:admission_rejections_total{model_name="inkling-2-mini",reason="max_num_queued_reqs"} 1.0
vllm:request_success_total{finished_reason="stop",model_name="inkling-2-mini"} 400.0
vllm:request_success_total{finished_reason="length",model_name="inkling-2-mini"} 12.0
vllm:prompt_tokens_total{model_name="inkling-2-mini"} 5000.0
vllm:prompt_tokens_cached_total{model_name="inkling-2-mini"} 4000.0
vllm:prompt_tokens_by_source_total{model_name="inkling-2-mini",source="local_compute"} 1000.0
vllm:prompt_tokens_by_source_total{model_name="inkling-2-mini",source="local_cache_hit"} 4000.0
vllm:generation_tokens_total{model_name="inkling-2-mini"} 900.0
vllm:iteration_tokens_total_sum{model_name="inkling-2-mini"} 1400.0
vllm:spec_decode_num_drafts_total{model_name="inkling-2-mini"} 100.0
vllm:spec_decode_num_verified_tokens_total{model_name="inkling-2-mini"} 700.0
vllm:spec_decode_num_accepted_tokens_total{model_name="inkling-2-mini"} 420.0
vllm:prefix_cache_queries_total{model_name="inkling-2-mini"} 5000.0
vllm:prefix_cache_hits_total{model_name="inkling-2-mini"} 4000.0
vllm:external_prefix_cache_hits_total{model_name="inkling-2-mini"} 250.0
vllm:mm_cache_queries_total{model_name="inkling-2-mini"} 40.0
vllm:mm_cache_hits_total{model_name="inkling-2-mini"} 33.0
vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"} 1.2e+10
vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 6.5e+09
vllm:kv_offload_config_info{cpu_kv_cache_size_tokens="910317",engine="0"} 1.0
vllm:cache_config_info{block_size="8",cache_dtype="fp8",kv_cache_size_tokens="273719",num_gpu_blocks="34215"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="inkling-2-mini"} 0.25
vllm:tool_call_parser_invocations_total{mode="streaming",outcome="tool_call"} 7.0
vllm:tool_call_parser_invocations_total{mode="streaming",outcome="no_tool_call"} 5.0
vllm:time_to_first_token_seconds_sum{model_name="inkling-2-mini"} 120.0
vllm:time_to_first_token_seconds_count{model_name="inkling-2-mini"} 40.0
process_start_time_seconds 1.0e+09
"""


class _FakeEngine(BaseHTTPRequestHandler):
    """Enough of vLLM's surface to exercise fetch_all end to end (no GPU, no container)."""

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype="application/json"):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            self._send("ok", "text/plain")
        elif p == "/version":
            self._send(json.dumps({"version": "0.34.0rc1.dev1139+g961e65267"}))
        elif p == "/load":
            self._send(json.dumps({"server_load": 0.0}))
        elif p == "/v1/models":
            self._send(json.dumps({"data": [{"id": "inkling-2-mini", "max_model_len": 273664}]}))
        elif p == "/metrics":
            self._send(METRICS, "text/plain")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture
def engine_page():
    """The dashboard pointed at the fake engine, with its module state reset.

    CACHE, LATEST, HISTORY and SAMPLER_STATE are process-global by design (one sampler per
    process); /api/status may answer from CACHE for up to CACHE_TTL seconds, so a test that
    does not clear it can read the previous test's server. Every test here that talks to a
    page clears them."""
    sp.CACHE.update(ts=0.0, data=None)
    sp.LATEST["data"] = None
    sp.LATEST["ts"] = 0.0
    sp.HISTORY.clear()
    sp.SAMPLER_STATE.update(last_gen=None, boot_time=None, boot_checked_at=0.0)

    eng = ThreadingHTTPServer(("127.0.0.1", 0), _FakeEngine)
    threading.Thread(target=eng.serve_forever, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), sp.Handler)
    srv.base_url = "http://127.0.0.1:%d" % eng.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % srv.server_address[1]
    finally:
        srv.shutdown()
        eng.shutdown()


def test_reads_a_real_engine_shape(engine_page):
    """Counters, the two info lines and the derived rates the page prints."""
    d = json.loads(urllib.request.urlopen(engine_page + "/api/status", timeout=8).read())
    assert d["up"] is True and d["metrics_http"] == 200
    assert d["model"] == "inkling-2-mini" and d["max_model_len"] == 273664
    assert d["version"].startswith("0.34.0")
    assert d["requests_running"] == 2 and d["requests_waiting"] == 1
    assert d["requests_success"]["stop"] == 400 and d["requests_total"] == 412
    assert d["preemptions"] == 3 and d["admission_rejections"] == 1
    assert d["generation_tokens"] == 900 and d["prompt_tokens_total"] == 5000
    assert d["prompt_local_compute"] == 1000 and d["prompt_cache_hit"] == 4000
    assert d["spec_drafts"] == 100 and d["spec_accepted"] == 420
    assert d["spec_acceptance"] == pytest.approx(60.0)           # 420 / 700
    assert d["spec_tokens_per_draft"] == pytest.approx(4.2)      # accepted / drafts: the yield
    assert d["prefix_hit_rate"] == pytest.approx(80.0)           # 4000 / 5000
    assert d["external_prefix_hits"] == 250 and d["mm_cache_hits"] == 33
    assert d["kv_cache_tokens"] == 273719 and d["gpu_blocks"] == 34215
    assert d["offload_cpu_tokens"] == 910317
    assert d["offload_stored"] == pytest.approx(1.2e10)
    assert d["tool_calls"] == 7 and d["tool_call_invocations"] == 12
    assert d["time_to_first_token_avg"] == pytest.approx(3.0)    # 120 / 40
    assert d["kv_usage_perc"] == pytest.approx(0.25)
    assert d["start_seconds"] == pytest.approx(1.0e9)             # uptime, no docker needed
    assert d["uptime_s"] is not None and d["uptime_s"] > 1e8

"""--api-key guards every path but the liveness allowlist (no GPU).

    venv/bin/python -m pytest -q tests/test_auth.py
"""

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware


def _client(root_path=""):
    ok = lambda request: PlainTextResponse("ok")  # noqa: E731
    methods = ["GET", "POST", "OPTIONS"]
    paths = ["/health", "/metrics", "/tokenize", "/v1/models", "/docs"]
    app = Starlette(routes=[Route(p, ok, methods=methods) for p in paths])
    app.add_middleware(AuthenticationMiddleware, tokens=["secret"])
    return TestClient(app, root_path=root_path)


@pytest.mark.parametrize("root", ["", "/"])
@pytest.mark.parametrize("path", ["/metrics", "/tokenize", "/v1/models", "/docs"])
def test_guarded(root, path):
    c = _client(root)
    assert c.get(path).status_code == 401
    assert c.options(path).status_code == 401  # a bare OPTIONS is not a preflight
    assert c.get(path, headers={"Authorization": "Bearer secret"}).status_code == 200


@pytest.mark.parametrize("root", ["", "/"])
@pytest.mark.parametrize("path", ["/health", "/health/"])
def test_liveness_open(root, path):
    assert _client(root).get(path, follow_redirects=False).status_code in (200, 307)


def test_cors_preflight_open():
    h = {"Origin": "http://x", "Access-Control-Request-Method": "POST"}
    assert _client().options("/v1/models", headers=h).status_code == 200

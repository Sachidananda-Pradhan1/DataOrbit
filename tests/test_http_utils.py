"""Tests for src/utils/http.py — the proxy-aware httpx helper.

The helper exists because a plain httpx client with trust_env=True
crashes on this sandbox's NO_PROXY (bare/bracketed IPv6 entries);
see the module docstring."""
import httpx

from src.utils import http as http_util


def test_env_proxy_prefers_https(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://other:8080")
    assert http_util.env_proxy() == "http://proxy:8080"


def test_env_proxy_none_when_unset(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    assert http_util.env_proxy() is None


def test_async_client_disables_trust_env(monkeypatch, tmp_path):
    captured = {}

    class _CaptureClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("fake bundle")
    monkeypatch.setattr(httpx, "AsyncClient", _CaptureClient)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))
    http_util.async_client(timeout=60)
    assert captured["trust_env"] is False
    assert captured["proxy"] == "http://proxy:8080"
    assert captured["verify"] == str(ca_file)
    assert captured["timeout"] == 60


def test_env_ca_bundle_falls_back_to_certifi(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ca.pem")
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    assert http_util.env_ca_bundle() is True


def test_get_uses_proxy_aware_client(monkeypatch):
    seen = {}

    class _CaptureClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            seen["url"] = url
            return "resp"

    monkeypatch.setattr(httpx, "Client", _CaptureClient)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    assert http_util.get("https://example.com/x", timeout=5) == "resp"
    assert seen["trust_env"] is False
    assert seen["proxy"] == "http://proxy:8080"
    assert seen["url"] == "https://example.com/x"

"""httpx helpers: proxy-aware clients that survive hostile env vars.

httpx with trust_env=True parses NO_PROXY entries as URL patterns and
crashes on bare/bracketed IPv6 entries (e.g. `::1`, `[::1]`) — this
sandbox's NO_PROXY contains exactly those, so a plain
httpx.AsyncClient() raises InvalidURL at construction time. The fix:
trust_env=False, with the proxy taken explicitly from
HTTPS_PROXY/HTTP_PROXY. On a machine with no proxy env vars this
behaves like a plain client. NO_PROXY bypasses are not honored — all
of this pipeline's targets are public internet hosts, so nothing is
lost in practice.

trust_env=False also makes httpx ignore SSL_CERT_FILE, which breaks
TLS in environments whose egress proxy MITMs with a custom CA (this
sandbox: /run/hatch/egress-tls/ca-bundle.pem). So the CA bundle is
read explicitly too — never verify=False.
"""
from __future__ import annotations

import os

import httpx


def env_proxy() -> str | None:
    """Proxy server URL from the environment, or None for direct."""
    return (
        os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    )


def env_ca_bundle() -> str | bool:
    """CA bundle path from the environment, or True (certifi default).

    Honors SSL_CERT_FILE / REQUESTS_CA_BUNDLE when they point at an
    existing file; otherwise falls back to httpx's default verification.
    """
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        path = os.getenv(var)
        if path and os.path.exists(path):
            return path
    return True


def async_client(**kwargs) -> httpx.AsyncClient:
    """httpx.AsyncClient with an explicit proxy, immune to bad NO_PROXY."""
    kwargs.setdefault("proxy", env_proxy())
    kwargs.setdefault("verify", env_ca_bundle())
    return httpx.AsyncClient(trust_env=False, **kwargs)


def get(url: str, **kwargs):
    """One-shot GET via a short-lived proxy-aware client."""
    with httpx.Client(proxy=env_proxy(), verify=env_ca_bundle(),
                      trust_env=False) as client:
        return client.get(url, **kwargs)

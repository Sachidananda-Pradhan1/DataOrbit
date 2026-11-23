"""Async fetcher: aiohttp first, Playwright only for JS-heavy /
Cloudflare-challenged pages (Phase V). Polite by default — rotating
user-agents, bounded concurrency, timeouts. No bypass tricks first:
APIs and RSS/sitemaps are preferred by callers.

Retries: fetch() retries network/timeout errors and 429/5xx
responses (3 attempts, exponential backoff + jitter) and raises
after the last attempt — callers must not mistake a failed fetch
for an empty page. fetch_many()/fetch_many_stream() report those
failures through the optional `failures` side channel instead of
crashing the batch.
"""
import asyncio
import os
import random
from collections.abc import AsyncIterator

import aiohttp
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import settings

USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
]

# Fetch attempts per URL before fetch() gives up and raises.
FETCH_MAX_ATTEMPTS = 3


def _is_retryable(exc: BaseException) -> bool:
    """Retry network/timeout errors and 429/5xx responses. Other 4xx
    (bad URL, gone, forbidden) fail fast — retrying will not help."""
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status == 429 or exc.status >= 500
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


@retry(
    stop=stop_after_attempt(FETCH_MAX_ATTEMPTS),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
async def fetch(url: str, session: aiohttp.ClientSession) -> str:
    async with session.get(
        url,
        headers={"User-Agent": random.choice(USER_AGENTS)},
        timeout=aiohttp.ClientTimeout(total=settings.request_timeout_secs),
    ) as resp:
        resp.raise_for_status()
        return await resp.text()


async def _fetch_one(url: str, session: aiohttp.ClientSession,
                     sem: asyncio.Semaphore,
                     failures: set | None) -> tuple[str, str]:
    """One URL: (url, text), or (url, "") after retries are
    exhausted — the URL is added to `failures` (when given) so the
    caller can tell a failed fetch from an empty page."""
    async with sem:
        try:
            return url, await fetch(url, session)
        except Exception as exc:  # logged by caller; one failure must not kill a batch
            if failures is not None:
                failures.add(url)
            print(f"fetch failed {url}: {exc}")
            return url, ""


async def fetch_many_stream(urls: list[str], batch_size: int = 100,
                            failures: set | None = None
                            ) -> AsyncIterator[tuple[str, str]]:
    """Bounded-concurrency bulk fetch as an async generator: fetches
    one batch of `batch_size` URLs at a time and yields (url, text)
    per URL, so a caller scanning 500k URLs holds at most one batch
    of pages in memory. Failed URLs yield "" and are added to
    `failures` (when given). Breaking out of the loop early stops
    further batches and closes the session."""
    sem = asyncio.Semaphore(settings.max_concurrency)
    urls = list(urls)
    async with aiohttp.ClientSession(trust_env=True) as session:
        for start in range(0, len(urls), batch_size):
            batch = urls[start:start + batch_size]
            results = await asyncio.gather(
                *[_fetch_one(u, session, sem, failures) for u in batch])
            for url, text in results:
                yield url, text


async def fetch_many(urls: list[str],
                     failures: set | None = None) -> dict[str, str]:
    """Bounded-concurrency bulk fetch. Same dict contract as before
    (every URL maps to its text, "" on failure); `failures` is an
    optional side channel filled with URLs that failed after
    retries. Scaling = raising MAX_CONCURRENCY / adding workers,
    not rewriting this. Prefer fetch_many_stream when the URL list
    is large enough that holding every page at once matters."""
    out: dict[str, str] = {}
    async for url, text in fetch_many_stream(urls, failures=failures):
        out[url] = text
    return out


async def fetch_with_playwright(url: str) -> str:
    """JS-rendered / challenge pages only. Import is lazy so the
    core pipeline does not require Playwright installed.
    Honors HTTPS_PROXY/HTTP_PROXY from the environment — Chromium
    does not read them on its own, so proxied environments (like
    this sandbox) need them passed explicitly."""
    from urllib.parse import urlparse

    from playwright.async_api import async_playwright

    proxy_url = (
        os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    )
    launch_kwargs: dict = {}
    if proxy_url:
        # Chromium does not accept credentials embedded in the server
        # URL — split them into username/password (this sandbox's
        # egress proxy requires auth).
        parsed = urlparse(proxy_url)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        proxy: dict[str, str] = {
            "server": f"{parsed.scheme or 'http'}://{host}{port}",
        }
        if parsed.username:
            proxy["username"] = parsed.username
        if parsed.password:
            proxy["password"] = parsed.password
        launch_kwargs["proxy"] = proxy
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, **launch_kwargs)
        page = await browser.new_page(user_agent=random.choice(USER_AGENTS))
        await page.goto(url, wait_until="networkidle", timeout=60_000)
        html = await page.content()
        await browser.close()
        return html

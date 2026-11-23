"""Products (Phase I): BetaList directory discovery + deterministic parse.

SOURCE — VERIFIED LIVE 2026-11-21 (plain HTTP, no API key, no login):
- Discovery: https://betalist.com/startups?page=N (server-rendered HTML).
  Page 1 returned 52 unique /startups/<slug> links, pages 2/3/20
  returned 50 each, with 0 overlap between pages 1 and 2 — ~50 new
  product URLs per page, so ~20 pages reach 1,000. Polite 1s delay
  between paginated requests.
- Startup/product page, e.g. https://betalist.com/startups/ziv-live
  HTTP 200, ~109 KB, with a clean <h1>ZIV Live</h1> and
  og:title "ZIV Live: <tagline>". The h1 is the product name; on
  BetaList the listing names the PRODUCT, not a separate company, so
  startupName is "" unless the page names a company/vendor field
  separately (see _company_from_page) or the LLM extraction names
  one — it is NEVER copied from productName.

CHECKED AND NOT USED (2026-11-21):
- Product Hunt AI topic pages: HTTP 403 (bot protection); the official
  API needs OAuth, so it is not depended on.
- AlternativeTo AI category: HTTP 403. Futurepedia homepage: JS-rendered,
  no tool links in static HTML. BetaList has no sitemap (404).

DESIGN: discovery + deterministic parsing work keyless NOW. The LLM
fallback is a small local helper mirroring src.llm.extractor
.extract_product (same prompt -> JSON -> Pydantic pattern, kept
local so this module's LLM failure mode stays self-contained); any
LLM failure yields [] rather than a crash, never an invented name
or URL.

AI RELEVANCE (verified live 2026-11-22): BetaList discovery
exposes NO category/tag data — listing rows show only the name +
tagline. Product pages DO list topics in a "Topics" section
(/browse/<group>/<slug> links, e.g.
/browse/ai/artificial-intelligence; the ai/ group has 29
subtopics). So relevance is decided per product page: an ai/
topic, or the shared keyword filter
(src.crawlers.startups.matches_ai_keywords — one source of truth
for the keyword list) on the title/tagline. parse() itself stays
unfiltered; scripts/run_bulk.py applies is_ai_relevant_page()
before the 1,000 cap.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import aiohttp
from bs4 import BeautifulSoup
from pydantic import ValidationError

from src.crawlers.base import BaseCrawler
from src.crawlers.fetcher import fetch
from src.crawlers.startups import matches_ai_keywords
from src.llm.extractor import parse_llm_json
from src.llm.orchestrator import extract_with_fallback
from src.llm.prompts import PRODUCT_PROMPT
from src.models import Product, Source

SOURCE_NAME = "BetaList"
LISTING_URL = "https://betalist.com/startups"
STARTUP_URL = "https://betalist.com/startups/{slug}"

# Polite delay between paginated listing requests (seconds).
PAGE_DELAY_SECS = 1.0
# Safety cap so a runaway listing can never loop forever.
MAX_PAGES = 100
# Cap on text sent to the LLM fallback (413 guard for long pages).
LLM_MAX_CHARS = 8000

# Only bare /startups/<slug> links — excludes /visit, /testflight, etc.
_STARTUP_LINK_RE = re.compile(r'href="/startups/([a-z0-9-]+)"')


def listing_page_url(page: int) -> str:
    """Pure helper: listing URL for page N (page 1 has no ?page=)."""
    return LISTING_URL if page <= 1 else f"{LISTING_URL}?page={page}"


def extract_startup_urls(html_text: str) -> list[str]:
    """Pure helper: product URLs from a listing page's HTML.

    Deduped and sorted."""
    return sorted({STARTUP_URL.format(slug=s)
                   for s in _STARTUP_LINK_RE.findall(html_text)})


async def _fetch_text(url: str) -> str:
    """Single polite fetch (module-level so tests can monkeypatch it)."""
    async with aiohttp.ClientSession(trust_env=True) as session:
        return await fetch(url, session)


def _name_from_page(text: str) -> str | None:
    """Product name: <h1> first, then og:title / <title> before ': '."""
    soup = BeautifulSoup(text, "html.parser")
    if soup.h1 and soup.h1.get_text(strip=True):
        return soup.h1.get_text(strip=True)
    og = soup.find("meta", property="og:title")
    title = (og.get("content") if og else None) or (
        soup.title.string if soup.title else None)
    if not title:
        return None
    title = title.replace("| BetaList", "").strip()
    return title.split(": ", 1)[0].strip() or None


# --- AI-relevance filter (deterministic; no LLM) ---
#
# BetaList discovery exposes no category/tag data (listing rows
# show name + tagline only), but each product page lists topics, so
# relevance is decided per product page from its own metadata: an
# ai/ topic, or the shared keyword filter on title/tagline. A page
# with no extractable fields FAILS the filter — only products
# verifiably described as AI are included, never guesses.

# BetaList topic group for AI products: product pages list their
# topics as /browse/<group>/<slug> links (e.g.
# /browse/ai/artificial-intelligence). The ai/ group has 29
# subtopics (verified live 2026-11-22): artificial-intelligence,
# machine-learning, generative-ai, computer-vision, ...
AI_TOPIC_GROUP = "ai"


def _tagline_from_page(text: str) -> str | None:
    """Tagline: the part of og:title / <title> after 'Name: ', else
    the meta description (which BetaList sets to the tagline)."""
    soup = BeautifulSoup(text, "html.parser")
    og = soup.find("meta", property="og:title")
    title = (og.get("content") if og else None) or (
        soup.title.string if soup.title else None)
    if title:
        title = title.replace("| BetaList", "").strip()
        if ": " in title:
            tagline = title.split(": ", 1)[1].strip()
            if tagline:
                return tagline
    for meta in soup.find_all("meta"):
        if meta.get("name") == "description":
            content = (meta.get("content") or "").strip()
            if content:
                return content
    return None


def _topics_from_page(text: str) -> list[tuple[str, str]]:
    """(topic slug path, label) pairs from the product page's
    "Topics" section, e.g. ("ai/artificial-intelligence",
    "Artificial Intelligence"). [] when the page has no Topics
    section — topic data is a bonus signal, never required."""
    soup = BeautifulSoup(text, "html.parser")
    for dt in soup.find_all("dt"):
        if dt.get_text(strip=True) != "Topics":
            continue
        dd = dt.find_next_sibling("dd")
        if dd is None:
            return []
        topics = []
        for a in dd.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("/browse/"):
                continue
            topics.append((href.removeprefix("/browse/"),
                           a.get_text(" ", strip=True)))
        return topics
    return []


# --- company + pricing extraction (deterministic; no LLM) ---
#
# BetaList exposes no dedicated company or pricing field on most
# listings, so both are best-effort and evidence-gated: a company
# name only from a labelled field, a pricing model only from
# explicit pricing phrases. Absence means "" / None — never a guess,
# and startupName is NEVER copied from productName.

# Definition-list labels under which a listing may name the company
# / maker separately from the product.
_COMPANY_LABELS = ("company", "maker", "made by", "creator", "vendor")


def _company_from_page(text: str) -> str | None:
    """Company/vendor name, if the page names one separately from the
    product (a "Company" / "Made by" / "Maker" definition-list
    entry). None when it does not — the caller leaves startupName
    "" rather than copying the product name."""
    soup = BeautifulSoup(text, "html.parser")
    for dt in soup.find_all("dt"):
        if dt.get_text(strip=True).lower().rstrip(":") not in _COMPANY_LABELS:
            continue
        dd = dt.find_next_sibling("dd")
        if dd is not None:
            name = dd.get_text(" ", strip=True)
            if name:
                return name
    return None


# Pricing phrases. "free" alone is too ambiguous ("free up your
# time"), so it must appear as a plan/tier phrase; a dollar amount
# only counts with a per-month qualifier ("raised $5M" is not
# pricing).
_PRICING_FREE_RE = re.compile(
    r"free (plan|tier|to use|forever)|completely free|100% free|"
    r"free and open source", re.IGNORECASE)
_PRICING_PAID_RE = re.compile(
    r"starting at \$|paid plan|per month|/month|"
    r"\$\d[\d,.]*\s*/\s*mo\b|monthly subscription", re.IGNORECASE)
_PRICING_ENTERPRISE_RE = re.compile(
    r"enterprise (plan|pricing|tier)|custom pricing|"
    r"contact us for pricing", re.IGNORECASE)
_FREEMIUM_RE = re.compile(r"\bfreemium\b", re.IGNORECASE)


def _pricing_from_page(text: str) -> str | None:
    """pricingModel from explicit pricing phrases in the listing
    text (tagline + page body). FREEMIUM when the page says so, or
    when it states both a free tier and a paid tier. None when the
    page does not state pricing — never guessed."""
    soup = BeautifulSoup(text, "html.parser")
    haystack = f"{_tagline_from_page(text) or ''} {soup.get_text(' ', strip=True)}"
    free = _PRICING_FREE_RE.search(haystack) is not None
    paid = _PRICING_PAID_RE.search(haystack) is not None
    if _FREEMIUM_RE.search(haystack) or (free and paid):
        return "FREEMIUM"
    if _PRICING_ENTERPRISE_RE.search(haystack):
        return "ENTERPRISE"
    if paid:
        return "PAID"
    if free:
        return "FREE"
    return None


def is_ai_relevant(name: str | None = None, tagline: str | None = None,
                   topics: list[tuple[str, str]] | None = None) -> bool:
    """Deterministic AI-relevance verdict for a BetaList product.
    True when any topic is in BetaList's ai/ group (or its label
    matches an AI keyword), or a keyword matches the title/tagline.
    Pure — no network, no LLM."""
    for slug, label in topics or []:
        if slug.split("/", 1)[0].lower() == AI_TOPIC_GROUP:
            return True
        if matches_ai_keywords(label):
            return True
    return any(matches_ai_keywords(field) for field in (name, tagline))


def _extract_product_llm(text: str, url: str,
                         collected_at: datetime) -> Product | None:
    """Local LLM fallback, mirroring src.llm.extractor.extract_product.

    Same conventions as extract_startup: prompt -> JSON -> Pydantic
    validation; anything invalid or any failure returns None. The
    prompt asks for startupName (the company, if named separately
    from the product) and pricingModel (FREE / FREEMIUM / PAID /
    ENTERPRISE); a bad pricingModel value is coerced to None by the
    model validator rather than dropping the record."""
    try:
        raw = extract_with_fallback(
            PRODUCT_PROMPT.format(chunk=text[:LLM_MAX_CHARS]))
        data = parse_llm_json(raw)
        if data is None:
            return None
        return Product(source={"name": SOURCE_NAME, "url": url},
                       collectedAt=collected_at, **data)
    except (ValidationError, Exception):
        return None


class ProductCrawler(BaseCrawler):
    kind = "products"

    async def discover(self, limit: int = 1000) -> list[str]:
        """Seed product URLs by paging the BetaList directory.

        Stops at `limit`, at an empty page, or at MAX_PAGES."""
        seen: set[str] = set()
        urls: list[str] = []
        page = 1
        while len(urls) < limit and page <= MAX_PAGES:
            try:
                html_text = await _fetch_text(listing_page_url(page))
            except Exception as exc:
                print(f"ProductCrawler.discover failed on page {page}: {exc}")
                break
            fresh = [u for u in extract_startup_urls(html_text) if u not in seen]
            if not fresh:
                break
            seen.update(fresh)
            urls.extend(fresh)
            page += 1
            if len(urls) < limit:
                await asyncio.sleep(PAGE_DELAY_SECS)
        return urls[:limit]

    def is_ai_relevant_page(self, raw: str) -> bool:
        """Runner hook: True when this product page is AI-relevant.

        Extracts the page's own metadata (title, tagline, topics)
        and applies is_ai_relevant(). A page with no extractable
        fields fails — unverifiable products are excluded.
        Deterministic, no LLM. Applied by scripts/run_bulk.py
        BEFORE the 1,000 cap, so the cap counts AI products only;
        parse() itself stays a pure extractor and is unaffected."""
        if not raw or not raw.strip():
            return False
        return is_ai_relevant(
            name=_name_from_page(raw),
            tagline=_tagline_from_page(raw),
            topics=_topics_from_page(raw),
        )

    async def parse(self, url: str, raw: str) -> list[Product]:
        """Deterministic parse first; LLM fallback only when no name
        can be extracted. Returns [] rather than inventing data."""
        collected_at = datetime.now(timezone.utc)
        name = _name_from_page(raw) if raw and raw.strip() else None
        if name:
            # BetaList names the product, not a separate company:
            # startupName comes only from a labelled company field
            # on the page, else "" — NEVER copied from productName.
            # pricingModel comes only from explicit pricing phrases
            # on the page, else None. The LLM path (here and in
            # run_bulk) can fill both when the page text names them.
            return [Product(
                source=Source(name=SOURCE_NAME, url=url),
                startupName=_company_from_page(raw) or "",
                productName=name,
                pricingModel=_pricing_from_page(raw),
                collectedAt=collected_at,
            )]
        # Unstructured page: LLM fallback; any failure yields [].
        if raw and raw.strip():
            model = _extract_product_llm(raw, url, collected_at)
            if model is not None:
                return [model]
        return []

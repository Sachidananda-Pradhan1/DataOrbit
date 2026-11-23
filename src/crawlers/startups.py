"""Startups (Phase I): Y Combinator directory discovery + deterministic parse.

SOURCE — VERIFIED LIVE 2026-11-21 (plain HTTP, no API key, no login):
- Discovery: https://www.ycombinator.com/companies/sitemap
  HTTP 200, ~635 KB, one page listing 6,410 unique company profile
  URLs (/companies/<slug>). No pagination needed; discover() slices
  to the requested limit. (The 5 non-company links on that page —
  sitemap, black-founders, founders-you-may-know,
  hispanic-latino-founders, women-founders — are excluded.)
- Company page, e.g. https://www.ycombinator.com/companies/airbnb
  HTTP 200, ~127 KB. Contains (a) an embedded HTML-escaped JSON blob
  with "slug"/"name"/"one_liner"/"website", and (b) a meta description
  of the form "... Airbnb has 6132 employees based in San Francisco."
  Both are parsed deterministically — name + employeeCount need no LLM.

CHECKED AND NOT USED (2026-11-21):
- ycombinator.com/companies listing page: JS app (Algolia); no company
  links in the static HTML and no search key in the served HTML.
- Product Hunt AI topic pages: HTTP 403 (bot protection).
- F6S /companies: HTTP 405. AlternativeTo AI category: HTTP 403.

LIMITATION: the YC sitemap covers all industries — there is no
AI-only filter in the static page, so the 1,000-company cap must
NOT be applied to raw discovery order.

AI RELEVANCE (this module): is_ai_relevant_page() decides per
company page, deterministically (no LLM). The embedded JSON blob
lists curated tags (e.g. ["SaaS", "B2B", "Sales", "AI"]) plus
one_liner and long_description; those fields are matched against
AI_KEYWORDS (case-insensitive, whole-word). There is no separate
industry field in the blob — tags are the industry signal
(verified live 2026-11-22). parse() itself stays unfiltered (a
pure extractor); scripts/run_bulk.py applies the filter before
the cap, so only AI-relevant companies count toward the 1,000.

DESIGN: discovery + deterministic parsing work keyless NOW. The LLM
extractor (src.llm.extractor.extract_startup) is only a fallback for
pages where no name can be parsed deterministically, and any LLM
failure (no keys, bad output) yields [] rather than a crash — never
an invented name or URL.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

from src.crawlers.base import BaseCrawler
from src.crawlers.fetcher import fetch
from src.llm.extractor import extract_startup
from src.models import Source, Startup

SOURCE_NAME = "Y Combinator"
SITEMAP_URL = "https://www.ycombinator.com/companies/sitemap"
COMPANY_URL = "https://www.ycombinator.com/companies/{slug}"

# Directory/filter pages linked from the sitemap — not companies.
EXCLUDE_SLUGS = {
    "sitemap",
    "black-founders",
    "founders-you-may-know",
    "hispanic-latino-founders",
    "women-founders",
}

# Matches /companies/<slug> URLs in either form the sitemap uses:
# XML sitemap entries (<loc>https://.../companies/airbnb</loc>) and
# HTML links (href="/companies/airbnb"). The lookahead stops the slug
# at a quote, slash, tag or whitespace.
_COMPANY_LINK_RE = re.compile(
    r'(?:https://www\.ycombinator\.com)?/companies/([a-z0-9-]+)(?=["/<\s])'
)
# Embedded JSON blob (HTML-escaped in the page): "slug":"x","name":"Y".
_EMBEDDED_RE = re.compile(r'"slug"\s*:\s*"([^"]+)"\s*,\s*"name"\s*:\s*"([^"]+)"')
_EMPLOYEES_RE = re.compile(r"has\s+([\d,]+)\s+employees")


# --- AI-relevance filter (deterministic; no LLM) ---
#
# The sitemap covers ALL industries, so relevance is decided per
# company page from the page's own metadata: curated `tags` from the
# embedded JSON blob, plus keyword matching on the name, one_liner,
# long_description and meta description. A page whose fields cannot
# be extracted at all FAILS the filter — only companies verifiably
# described as AI are included, never guesses.

# AI keywords, matched case-insensitively as whole words/phrases
# (word boundaries on both sides, optional plural "s"), so "said",
# "brain" and "training" never match on their "ai"/"ml" substrings.
# Deliberately NOT included: "automation", "analytics",
# "predictive", "robotics", "autonomous" — all common in non-AI
# companies, so including them would flood the tab with false
# positives. "copilot" IS included: in a 2026 startup one-liner it
# near-always means an AI assistant (GitHub Copilot usage).
AI_KEYWORDS = (
    # Standalone abbreviations.
    "ai", "ml", "nlp", "llm", "gpt", "genai", "chatgpt",
    # Phrases.
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "large language model",
    "generative ai",
    "neural network",
    "reinforcement learning",
    "speech recognition",
    "text-to-speech",
    "speech-to-text",
    "image generation",
    "ai-powered",
    "ai-driven",
    "ai agent",
    "chatbot",
    "copilot",
    "co-pilot",
    "fine-tuning",
    "fine-tuned",
    "prompt engineering",
    "foundation model",
    "diffusion model",
    "transformer model",
)

# YC tag values that on their own mean the company is AI. Tags are
# also keyword-matched (see is_ai_relevant), which catches "AI
# Agents" or "Applied AI"; this set covers short tags the regex
# would otherwise miss.
AI_TAGS = frozenset({
    "ai", "artificial intelligence", "machine learning", "ml",
    "deep learning", "nlp", "natural language processing",
    "computer vision", "llm", "large language models",
    "generative ai", "genai", "conversational ai", "chatbots",
    "ai agents", "speech recognition",
})


def _keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern:
    parts = [re.escape(kw) + r"s?\b" for kw in keywords]
    return re.compile(r"\b(?:" + "|".join(parts) + ")", re.IGNORECASE)


_AI_KEYWORD_RE = _keyword_pattern(AI_KEYWORDS)


def matches_ai_keywords(text: str | None) -> bool:
    """True when `text` contains any AI keyword (case-insensitive,
    whole-word/phrase). Empty/None text never matches. Shared by
    products.py — the keyword list has one source of truth."""
    return bool(text) and _AI_KEYWORD_RE.search(text) is not None


def is_ai_relevant(name: str | None = None, one_liner: str | None = None,
                   description: str | None = None,
                   tags: list[str] | None = None) -> bool:
    """Deterministic AI-relevance verdict from a company's own
    metadata. True when any tag names AI (exact tag match or keyword
    match on the tag), or a keyword matches the name, one-liner or
    description. Pure — no network, no LLM."""
    for tag in tags or []:
        if tag.strip().lower() in AI_TAGS or matches_ai_keywords(tag):
            return True
    return any(matches_ai_keywords(field)
               for field in (name, one_liner, description))


# Fields of the embedded JSON blob (HTML-escaped in the page, so
# these run on html.unescape(raw)). Targeted regexes, not a full
# JSON parse — the blob contains nested objects, so a naive
# brace-match mis-parses it.
_ONE_LINER_RE = re.compile(r'"one_liner"\s*:\s*"((?:[^"\\]|\\.)*)"')
_LONG_DESC_RE = re.compile(r'"long_description"\s*:\s*"((?:[^"\\]|\\.)*)"')
_WEBSITE_RE = re.compile(r'"website"\s*:\s*"((?:[^"\\]|\\.)*)"')
_TAGS_RE = re.compile(r'"tags"\s*:\s*\[(.*?)\]', re.DOTALL)
_TAG_ITEM_RE = re.compile(r'"([^"]+)"')


def _json_string(value: str) -> str:
    """Undo JSON string escapes in a regex-captured value."""
    try:
        parsed = json.loads(f'"{value}"')
    except (json.JSONDecodeError, ValueError):
        return value
    return parsed if isinstance(parsed, str) else value


def _capture(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return _json_string(match.group(1)).strip() or None


def _tags_from_text(unescaped: str) -> list[str]:
    match = _TAGS_RE.search(unescaped)
    if not match:
        return []
    return [tag.strip() for tag in _TAG_ITEM_RE.findall(match.group(1))
            if tag.strip()]


def _one_liner_from_title(text: str) -> str | None:
    """One-liner from og:title / <title>: 'Name: one-liner | YC'.
    Fallback for pages whose embedded JSON has no one_liner."""
    soup = BeautifulSoup(text, "html.parser")
    og = soup.find("meta", property="og:title")
    title = (og.get("content") if og else None) or (
        soup.title.string if soup.title else None)
    if not title:
        return None
    title = title.replace("| Y Combinator", "").strip()
    if ": " not in title:
        return None
    return title.split(": ", 1)[1].strip() or None


def _meta_description(text: str) -> str | None:
    """Meta description up to ' Founded in ...' — the first sentence
    is the one-liner; the rest names founders/employees, and a
    founder surnamed 'Ai' must not count as AI relevance."""
    soup = BeautifulSoup(text, "html.parser")
    contents = []
    for meta in soup.find_all("meta"):
        if meta.get("name") == "description":
            contents.append(meta.get("content") or "")
    for meta in soup.find_all("meta"):
        if meta.get("property") == "og:description":
            contents.append(meta.get("content") or "")
    for content in contents:
        one_liner = content.split(" Founded in ", 1)[0].strip()
        if one_liner:
            return one_liner
    return None


def extract_company_urls(html_text: str) -> list[str]:
    """Pure helper: company profile URLs from the sitemap page HTML.

    Deduped, sorted, special directory pages excluded."""
    urls = {
        COMPANY_URL.format(slug=slug)
        for slug in _COMPANY_LINK_RE.findall(html_text)
        if slug not in EXCLUDE_SLUGS
    }
    return sorted(urls)


async def _fetch_text(url: str) -> str:
    """Single polite fetch (module-level so tests can monkeypatch it)."""
    async with aiohttp.ClientSession(trust_env=True) as session:
        return await fetch(url, session)


def _slug_from_url(url: str) -> str | None:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "companies":
        return parts[1]
    return None


def _name_from_embedded_json(text: str, slug: str | None) -> str | None:
    """Company name from the page's embedded JSON blob.

    Only the entry whose slug matches the page URL is trusted — never
    attribute another company's name from the same page."""
    matches = _EMBEDDED_RE.findall(html.unescape(text))
    for entry_slug, name in matches:
        if slug and entry_slug == slug:
            return name.strip() or None
    return None


def _name_from_title(text: str) -> str | None:
    """Name from og:title / <title>: 'Airbnb: one-liner | Y Combinator'."""
    soup = BeautifulSoup(text, "html.parser")
    og = soup.find("meta", property="og:title")
    title = (og.get("content") if og else None) or (
        soup.title.string if soup.title else None)
    if not title:
        return None
    title = title.replace("| Y Combinator", "").strip()
    return title.split(": ", 1)[0].strip() or None


def _website_from_text(text: str) -> str | None:
    """Company website from the embedded JSON blob's "website" field
    (same targeted-regex style as _ONE_LINER_RE). Only an http(s)
    URL with a host counts — anything else is None, never a guess."""
    value = _capture(_WEBSITE_RE, html.unescape(text))
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return value


def _employee_count(text: str) -> int | None:
    """Headcount from the meta description ('... has 6132 employees ...')."""
    soup = BeautifulSoup(text, "html.parser")
    descs = []
    for meta in soup.find_all("meta"):
        if meta.get("name") == "description" or meta.get("property") == "og:description":
            descs.append(meta.get("content") or "")
    for desc in descs:
        m = _EMPLOYEES_RE.search(desc)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


class StartupCrawler(BaseCrawler):
    kind = "startups"

    async def discover(self, limit: int = 1000) -> list[str]:
        """Seed company URLs from the YC sitemap page (one fetch)."""
        try:
            page = await _fetch_text(SITEMAP_URL)
        except Exception as exc:
            print(f"StartupCrawler.discover failed: {exc}")
            return []
        return extract_company_urls(page)[:limit]

    def is_ai_relevant_page(self, raw: str) -> bool:
        """Runner hook: True when this company page is AI-relevant.

        Extracts the page's own metadata (tags, one_liner,
        long_description, title, meta description) and applies
        is_ai_relevant(). A page with no extractable fields fails —
        unverifiable companies are excluded. Deterministic, no LLM.
        Applied by scripts/run_bulk.py BEFORE the 1,000 cap, so the
        cap counts AI companies only; parse() itself stays a pure
        extractor and is unaffected."""
        if not raw or not raw.strip():
            return False
        unescaped = html.unescape(raw)
        one_liner = (_capture(_ONE_LINER_RE, unescaped)
                     or _one_liner_from_title(raw))
        descriptions = [_capture(_LONG_DESC_RE, unescaped),
                        _meta_description(raw)]
        description = " ".join(d for d in descriptions if d) or None
        return is_ai_relevant(
            name=_name_from_title(raw),
            one_liner=one_liner,
            description=description,
            tags=_tags_from_text(unescaped),
        )

    async def parse(self, url: str, raw: str) -> list[Startup]:
        """Deterministic parse first; LLM fallback only when no name
        can be extracted. Returns [] rather than inventing data."""
        collected_at = datetime.now(timezone.utc)
        name = None
        employees = None
        website = None
        if raw and raw.strip():
            name = (_name_from_embedded_json(raw, _slug_from_url(url))
                    or _name_from_title(raw))
            employees = _employee_count(raw)
            website = _website_from_text(raw)
        if name:
            return [Startup(
                source=Source(name=SOURCE_NAME, url=url),
                entityName=name,
                employeeCount=employees,
                website=website,
                collectedAt=collected_at,
            )]
        # Unstructured page: LLM fallback. Any failure (no keys, bad
        # output) yields [] — never a fabricated record.
        if raw and raw.strip():
            try:
                model = extract_startup(raw, SOURCE_NAME, url, collected_at)
            except Exception:
                model = None
            if model is not None:
                return [model]
        return []

"""News (Phase II): RSS-first for the 5 approved sources, full-text
extraction with trafilatura (not just summaries/meta), precise
published timestamp, 24h freshness gate.

Approved sources (user, 21 Nov 2026) — all five are RSS:
TechCrunch AI, MIT Technology Review AI, VentureBeat AI,
The Verge AI, The Decoder. Wired AI was replaced by The Decoder
(user decision 21 Nov); ArXiv and Hacker News AI are not RSS news
sources for this tab (ArXiv feeds the papers tab).

Live verification 21 Nov 2026 ~17:25 UTC:
- TechCrunch AI: HTTP 200, 20 items, newest 21 Nov 16:00 UTC — live.
- MIT Technology Review AI: HTTP 200, 10 items, newest
  20 Nov 18:15 UTC — live.
- The Verge AI: HTTP 200, 10 items, newest 21 Nov 15:00 UTC — live.
  Feed is Atom at /rss/ai-artificial-intelligence/index.xml
  (the other path 404s).
- The Decoder: HTTP 200, 10 items, newest 21 Nov 16:23 UTC — live;
  feed URL https://the-decoder.com/feed/ verified.
- VentureBeat AI: HTTP 429 on the first fetch AND on a retry 20s
  later (rate-limited from this egress IP at check time); at the
  previous check its newest visible item was 27 Aug 2026 (stale).
  One dead feed never kills the run: fetch_all_news logs a warning
  and continues with the other four.

Target: ALL fresh items in the last 24h per source (no cap). The
"100-item target" in the old docstring was wrong and is removed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import feedparser
import trafilatura
from bs4 import BeautifulSoup
from pydantic import ValidationError

from src.crawlers.date_utils import is_fresh, parse_date
from src.crawlers.fetcher import fetch_many
from src.crawlers.first_seen import FirstSeenStore
from src.models import NewsItem, Source
from src.utils import http as http_util

logger = logging.getLogger(__name__)

FEEDS: dict[str, str] = {
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "MIT Technology Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "The Decoder": "https://the-decoder.com/feed/",
}

BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def extract_published_from_html(html: str, *, now: datetime | None = None) -> datetime | None:
    """Published date from an article page's own metadata, in order:
    article:published_time, JSON-LD datePublished, <time datetime>.
    Returns None when the page carries no parseable date."""
    if not html or not html.strip():
        return None
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []
    for meta in soup.find_all("meta"):
        if (meta.get("property")
                in ("article:published_time", "article:modified_time")
                or meta.get("name")
                in ("date", "publishdate", "DC.date.issued")):
            candidates.append(meta.get("content") or "")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("datePublished"):
                candidates.append(str(item["datePublished"]))
    for time_tag in soup.find_all("time"):
        if time_tag.get("datetime"):
            candidates.append(time_tag["datetime"])
    for candidate in candidates:
        published = parse_date(candidate, now=now)
        if published is not None:
            return published
    return None


def parse_feed_xml(source_name: str, xml_text: str,
                   html_by_url: dict[str, str],
                   *, now: datetime | None = None,
                   first_seen: FirstSeenStore | None = None,
                   first_seen_urls: set[str] | None = None) -> list[NewsItem]:
    """Parse feed XML text (RSS or Atom) into fresh NewsItems.

    Applies the 24h freshness gate. An item with no feed date is
    NOT dropped outright: first try the article page's meta tags /
    JSON-LD / <time> (extract_published_from_html), and only when
    the page carries no date either, use the URL's first-seen
    timestamp from `first_seen` (the "new since last run"
    heuristic for undated sources; a fresh in-memory store is used
    when none is passed, so a first observation is `now`). The
    same 24h gate applies to a meta date and to a first-seen date
    exactly as to a feed date — a first-seen timestamp older than
    24h still drops the item. URLs dated by first-seen are added
    to `first_seen_urls` when the caller passes a set, so phase 2
    can refine them against the fetched article HTML.
    Full text is extracted with trafilatura from html_by_url (the
    fetched article HTML keyed by article URL); missing HTML yields
    an empty full_text rather than a dropped item — phase 1 keeps
    empty-text items by design, and fetch_all_news phase 2 drops
    them after the article fetch attempt."""
    now = now or datetime.now(timezone.utc)
    store = first_seen if first_seen is not None else FirstSeenStore(path=None)
    items: list[NewsItem] = []
    for entry in feedparser.parse(xml_text or "").entries:
        url = getattr(entry, "link", "") or ""
        if not url:
            continue
        published = parse_date(
            getattr(entry, "published", "") or getattr(entry, "updated", "")
            or "", now=now)
        if published is None:
            published = extract_published_from_html(
                html_by_url.get(url, ""), now=now)
        if published is None:
            published = store.observe(url, now)
            if first_seen_urls is not None:
                first_seen_urls.add(url)
        if not is_fresh(published, now=now):
            continue  # Phase II freshness guarantee
        html = html_by_url.get(url, "")
        # Skip trafilatura on empty HTML — it logs noisy lxml errors
        # for empty documents (phase 1 passes no HTML at all).
        full_text = trafilatura.extract(html) or "" if html.strip() else ""
        try:
            items.append(NewsItem(
                title=getattr(entry, "title", "") or "", source=Source(name=source_name, url=url),
                published_date=published, full_text=full_text, url=url,
            ))
        except ValidationError as exc:
            logger.debug("dropping news item with invalid URL %r: %s", url, exc)
    return items


def parse_feed(source_name: str, feed_url: str,
               html_by_url: dict[str, str]) -> list[NewsItem]:
    """Thin wrapper: fetch the feed URL (httpx, browser User-Agent),
    then parse_feed_xml. On fetch failure returns [] — the caller
    logs; one dead feed must not kill the run."""
    try:
        resp = http_util.get(feed_url, headers={"User-Agent": BROWSER_UA},
                             timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("feed fetch failed %s (%s): %s", source_name, feed_url, exc)
        return []
    return parse_feed_xml(source_name, resp.text, html_by_url)


async def fetch_all_news(
        first_seen: FirstSeenStore | None = None) -> list[NewsItem]:
    """Fetch all five feeds, then each fresh article's HTML, and
    return fresh NewsItems with full text. One dead feed logs a
    warning and the others still run. No cap: all fresh items in
    the last 24h. Undated items are dated from the first-seen store
    (default: the persistent data/processed/first_seen.json); phase
    2 refines first-seen dates against the fetched article HTML's
    meta date, and drops items whose full text is still empty after
    the article fetch attempt."""
    now = datetime.now(timezone.utc)
    store = first_seen if first_seen is not None else FirstSeenStore.default()
    items: list[NewsItem] = []
    try:
        feed_bodies = await fetch_many(list(FEEDS.values()))
    except Exception as exc:
        logger.warning("bulk news-feed fetch failed: %s", exc)
        feed_bodies = {}

    # Phase 1: find the fresh items (no HTML yet) to learn which
    # article URLs need fetching.
    fresh_by_feed: dict[str, list[NewsItem]] = {}
    first_seen_urls: set[str] = set()
    article_urls: list[str] = []
    for name, url in FEEDS.items():
        xml = feed_bodies.get(url, "")
        if not xml:
            logger.warning("news feed %s (%s) returned no content", name, url)
            continue
        try:
            fresh = parse_feed_xml(name, xml, {}, now=now, first_seen=store,
                                   first_seen_urls=first_seen_urls)
        except Exception as exc:
            logger.warning("news feed %s parse failed: %s", name, exc)
            continue
        fresh_by_feed[name] = fresh
        article_urls.extend(str(it.url) for it in fresh)
        logger.info("%s: %d fresh items", name, len(fresh))

    # Phase 2: fetch article HTML, refine first-seen dates against
    # the article's own meta date, and fill in the full text.
    # Items whose full text is still empty after the fetch attempt
    # are dropped (an empty row is worse than a missing one) — the
    # drop is logged, never silent.
    html_by_url = await fetch_many(article_urls) if article_urls else {}
    dropped_empty = dropped_stale_meta = 0
    for fresh in fresh_by_feed.values():
        for item in fresh:
            url = str(item.url)
            html = html_by_url.get(url, "")
            if url in first_seen_urls and html.strip():
                # The RSS entry carried no date, so phase 1 dated
                # this item by first-seen; now the article HTML is
                # in hand, prefer its real published date.
                meta_date = extract_published_from_html(html, now=now)
                if meta_date is not None:
                    if is_fresh(meta_date, now=now):
                        item = item.model_copy(
                            update={"published_date": meta_date})
                    else:
                        dropped_stale_meta += 1
                        logger.info(
                            "dropping news item with stale meta date %s: %s",
                            meta_date, url)
                        continue
            text = trafilatura.extract(html) or "" if html.strip() else ""
            if not text.strip():
                dropped_empty += 1
                logger.info("dropping news item with empty full_text: %s", url)
                continue
            items.append(item.model_copy(update={"full_text": text}))
    if dropped_empty or dropped_stale_meta:
        logger.info("news phase 2 dropped %d empty-text and %d "
                    "stale-meta items", dropped_empty, dropped_stale_meta)
    return items

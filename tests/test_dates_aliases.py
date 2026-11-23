"""Tests for the dates + aliases + first-seen + Hirist workstream.

All mocked/offline — no live network, no LLM calls. Payloads are
hand-built and dated relative to a fixed `now` where the parser
takes one, so the 24h freshness gate is deterministic.
"""
import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.crawlers.date_utils import is_fresh
from src.crawlers.first_seen import FirstSeenStore
from src.crawlers.jobs import (
    _hirist_field,
    parse_hirist,
    parse_remoteok,
)
from src.crawlers.news import (
    FEEDS,
    extract_published_from_html,
    fetch_all_news,
    parse_feed_xml,
)
from src.resolver.matcher import resolve_with_confidence

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 21, 17, 30, tzinfo=timezone.utc)
FRESH = NOW - timedelta(hours=1)
STALE = NOW - timedelta(hours=25)


def rfc822(dt: datetime) -> str:
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


# --- 1. Timezone regression ----------------------------------------

def test_parse_naive_timestamp_is_utc_under_ist_timezone():
    """Regression: pre-fix, "2026-11-22 10:30:00" parsed as 05:00 UTC
    under TZ=Asia/Kolkata (local-time assumption) but 10:30 UTC under
    UTC. Run the parse in a subprocess with TZ=Asia/Kolkata so the
    test fails if the TIMEZONE/TO_TIMEZONE settings are ever lost."""
    code = (
        "from src.crawlers.date_utils import parse_date;"
        "print(parse_date('2026-11-22 10:30:00').isoformat())"
    )
    env = dict(os.environ, TZ="Asia/Kolkata")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2026-11-22T10:30:00+00:00"


# --- 2a. Meta-date extraction --------------------------------------

def test_meta_date_article_published_time():
    html = ('<html><head><meta property="article:published_time" '
            'content="2026-11-21T16:30:00Z"></head><body></body></html>')
    assert extract_published_from_html(html, now=NOW) == FRESH


def test_meta_date_json_ld_date_published():
    html = ('<html><head><script type="application/ld+json">'
            '{"@context": "https://schema.org", "@type": "NewsArticle", '
            '"datePublished": "2026-11-21T16:30:00Z"}'
            '</script></head><body></body></html>')
    assert extract_published_from_html(html, now=NOW) == FRESH


def test_meta_date_time_tag():
    html = ('<html><body><time datetime="2026-11-21T16:30:00Z">'
            'Nov 21</time></body></html>')
    assert extract_published_from_html(html, now=NOW) == FRESH


def test_meta_date_prefers_meta_over_time_tag():
    html = ('<html><head><meta property="article:published_time" '
            'content="2026-11-21T16:30:00Z"></head><body>'
            '<time datetime="2026-11-20T16:30:00Z">Nov 20</time>'
            '</body></html>')
    assert extract_published_from_html(html, now=NOW) == FRESH


def test_meta_date_none_when_absent():
    assert extract_published_from_html(
        "<html><body><p>no dates here</p></body></html>", now=NOW) is None
    assert extract_published_from_html("", now=NOW) is None


# --- 2b. First-seen store ------------------------------------------

def test_first_seen_store_round_trip_and_persistence(tmp_path):
    path = tmp_path / "first_seen.json"
    store = FirstSeenStore(path)
    assert store.observe("https://example.com/a", NOW) == NOW
    # Second observation returns the FIRST timestamp, not the new one.
    assert store.observe("https://example.com/a",
                         NOW + timedelta(hours=2)) == NOW
    assert store.first_seen("https://example.com/a") == NOW
    assert store.first_seen("https://example.com/unseen") is None
    assert len(store) == 1
    # Reloaded from disk, the timestamp survives.
    reloaded = FirstSeenStore(path)
    assert reloaded.first_seen("https://example.com/a") == NOW


def test_first_seen_store_corrupt_file_starts_empty(tmp_path, caplog):
    path = tmp_path / "first_seen.json"
    path.write_text("{not json")
    store = FirstSeenStore(path)
    assert len(store) == 0
    assert store.observe("https://example.com/a", NOW) == NOW


NEWS_XML_UNDATED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>TechCrunch AI</title>
<item><title>Undated AI story</title>
<link>https://techcrunch.com/2026/11/21/undated-ai-story/</link></item>
</channel></rss>"""
UNDATED_URL = "https://techcrunch.com/2026/11/21/undated-ai-story/"


def test_first_seen_fresh_keeps_undated_news_with_first_seen_date():
    store = FirstSeenStore(path=None)
    store.observe(UNDATED_URL, FRESH)
    items = parse_feed_xml("TechCrunch AI", NEWS_XML_UNDATED, {},
                           now=NOW, first_seen=store)
    assert len(items) == 1
    assert items[0].published_date == FRESH


def test_first_seen_gate_still_enforced_stale_first_seen_drops():
    """The 24h gate applies to first-seen dates exactly as to RSS
    dates: a URL first seen 25h ago is stale, so its undated item
    is dropped — the heuristic never invents freshness."""
    store = FirstSeenStore(path=None)
    store.observe(UNDATED_URL, STALE)
    assert is_fresh(STALE, now=NOW) is False
    items = parse_feed_xml("TechCrunch AI", NEWS_XML_UNDATED, {},
                           now=NOW, first_seen=store)
    assert items == []


def test_first_seen_records_first_observation_in_parse_feed_xml():
    store = FirstSeenStore(path=None)
    parse_feed_xml("TechCrunch AI", NEWS_XML_UNDATED, {},
                   now=NOW, first_seen=store)
    assert store.first_seen(UNDATED_URL) == NOW


def test_first_seen_jobs_dateless_job_kept_with_store_dropped_without():
    data = [{"legal": "notice"},
            {"position": "ML Engineer", "company": "Acme AI",
             "tags": ["ml"], "location": "Remote",
             "url": "https://remoteok.com/jobs/1"}]
    # No date anywhere and no store: excluded, never guessed.
    assert parse_remoteok(data, now=NOW) == []
    # With a store: dated by first observation, gate still applies.
    store = FirstSeenStore(path=None)
    jobs = parse_remoteok(data, now=NOW, first_seen=store)
    assert len(jobs) == 1
    assert jobs[0].date == NOW
    assert jobs[0].company == "Acme AI"


def test_first_seen_jobs_stale_first_seen_drops():
    data = [{"legal": "notice"},
            {"position": "ML Engineer", "company": "Acme AI",
             "tags": ["ml"], "location": "Remote",
             "url": "https://remoteok.com/jobs/1"}]
    store = FirstSeenStore(path=None)
    store.observe("https://remoteok.com/jobs/1", STALE)
    assert parse_remoteok(data, now=NOW, first_seen=store) == []


# --- 3. Resolver aliases -------------------------------------------

@pytest.mark.parametrize("raw,canonical,method", [
    ("Anthropic PBC", "Anthropic", "exact"),   # "pbc" suffix stripped
    ("Mistral", "Mistral AI", "alias"),
    ("DeepMind", "Google DeepMind", "alias"),
    ("Runway ML", "Runway", "alias"),
    ("Stability", "Stability AI", "alias"),    # was fuzzy, now exact alias
])
def test_alias_cases(raw, canonical, method):
    name, confidence, got_method = resolve_with_confidence(raw)
    assert (name, confidence, got_method) == (canonical, 1.0, method)


def test_alias_does_not_break_domain_first_or_passthrough():
    # Domain evidence still outranks everything.
    assert resolve_with_confidence(
        "whatever", "https://www.anthropic.com/news") == (
            "Anthropic", 1.0, "domain")
    # Unknown names still pass through unchanged.
    assert resolve_with_confidence("Some Brand New Startup XYZ") == (
        "Some Brand New Startup XYZ", 0.0, "unmatched")
    # Exact seed matches are untouched.
    assert resolve_with_confidence("Mistral AI") == ("Mistral AI", 1.0, "exact")


# --- 4. Hirist selectors -------------------------------------------

def _card(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return soup.find("div")


def test_hirist_field_structured_itemprop():
    card = _card('<div><span itemprop="hiringOrganization">'
                 'Structured Inc</span>'
                 '<span itemprop="jobLocation">Bengaluru</span></div>')
    assert _hirist_field(card, "company") == "Structured Inc"
    assert _hirist_field(card, "location") == "Bengaluru"


def test_hirist_field_structured_data_attribute():
    card = _card('<div><span data-company="DataAttr Corp"></span></div>')
    assert _hirist_field(card, "company") == "DataAttr Corp"


def test_hirist_field_exact_class_token():
    card = _card('<div><div class="company">ExactToken Ltd</div>'
                 '<div class="job-location">Pune</div></div>')
    assert _hirist_field(card, "company") == "ExactToken Ltd"
    assert _hirist_field(card, "location") == "Pune"


def test_hirist_field_substring_class_fallback():
    # "x-company-info" matches no exact token, so the legacy
    # substring strategy still finds it.
    card = _card('<div><div class="x-company-info">Substr Co</div></div>')
    assert _hirist_field(card, "company") == "Substr Co"


def test_hirist_field_label_text_fallback():
    card = _card('<div><p>Company: LabelCorp</p>'
                 '<p>Location: Remote</p></div>')
    assert _hirist_field(card, "company") == "LabelCorp"
    assert _hirist_field(card, "location") == "Remote"


def test_hirist_field_missing_returns_empty():
    card = _card('<div><p>nothing useful here</p></div>')
    assert _hirist_field(card, "company") == ""


HIRIST_LABEL_HTML = """<html><body>
<div class="job-card">
  <a href="/j/ml-engineer-2001">ML Engineer</a>
  <p>Company: LabelCorp</p>
  <p>Location: Remote</p>
  <span>Posted 3 hours ago</span>
</div>
</body></html>"""


def test_parse_hirist_label_fallback_end_to_end():
    jobs = parse_hirist(HIRIST_LABEL_HTML, now=NOW)
    assert len(jobs) == 1
    assert jobs[0].company == "LabelCorp"
    assert jobs[0].region == "India"  # generic location + India board


HIRIST_UNPARSEABLE_HTML = """<html><body>
<div class="job-card">
  <a href="/j/ml-engineer-2003">ML Engineer</a>
  <span>Posted 1 hour ago</span>
</div>
<div class="job-card">
  <a href="/j/ai-engineer-2004">AI Engineer</a>
  <div class="company-name">Good Co</div>
  <span>Posted 2 hours ago</span>
</div>
</body></html>"""


def test_parse_hirist_unparseable_card_logged_not_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="src.crawlers.jobs"):
        jobs = parse_hirist(HIRIST_UNPARSEABLE_HTML, now=NOW)
    # The company-less card is dropped (never invented); the good
    # card is kept; the drop is logged with a count + sample.
    assert [j.title for j in jobs] == ["AI Engineer"]
    assert "1 card(s) unparseable" in caplog.text
    assert "ML Engineer" in caplog.text  # sample card text


def test_parse_hirist_first_seen_dates_card_without_posted_date():
    html = """<html><body><div class="job-card">
      <a href="/j/ml-engineer-2005">ML Engineer</a>
      <div class="company-name">Dateless Co</div>
    </div></body></html>"""
    assert parse_hirist(html, now=NOW) == []  # no date, no store
    store = FirstSeenStore(path=None)
    jobs = parse_hirist(html, now=NOW, first_seen=store)
    assert len(jobs) == 1
    assert jobs[0].company == "Dateless Co"
    assert jobs[0].date == NOW


# --- 5. Empty full_text dropped in news phase 2 --------------------

def _news_feed_xml(now: datetime) -> str:
    fresh = rfc822(now - timedelta(hours=1))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>TechCrunch AI</title>
<item><title>Fresh AI story</title>
<link>https://techcrunch.com/2026/11/21/fresh-ai-story/</link>
<pubDate>{fresh}</pubDate></item>
<item><title>Undated no-HTML story</title>
<link>https://techcrunch.com/2026/11/21/undated-no-html/</link></item>
<item><title>Undated stale-meta story</title>
<link>https://techcrunch.com/2026/11/21/undated-stale-meta/</link></item>
<item><title>Undated fresh-meta story</title>
<link>https://techcrunch.com/2026/11/21/undated-fresh-meta/</link></item>
</channel></rss>"""


ARTICLE_HTML = ("<html><body><article><p>Full article body text about "
                "the fresh AI story, with enough words to extract.</p>"
                "<p>Second paragraph of the article body.</p>"
                "</article></body></html>")


def test_fetch_all_news_phase2_drops_empty_text_and_stale_meta(
        monkeypatch, caplog):
    """Phase 2 drops items whose full text is empty after the fetch
    attempt, and refines first-seen dates against the article's meta
    date (stale meta drops the item, fresh meta re-dates it)."""
    from src.crawlers import news

    real_now = datetime.now(timezone.utc)
    meta_fresh = (real_now - timedelta(hours=2)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    meta_stale = (real_now - timedelta(hours=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    xml = _news_feed_xml(real_now)

    async def fake_fetch_many(urls):
        out = {}
        for url in urls:
            if url in FEEDS.values():
                out[url] = xml if url == FEEDS["TechCrunch AI"] else ""
            elif url.endswith("/fresh-ai-story/"):
                out[url] = ARTICLE_HTML
            elif url.endswith("/undated-no-html/"):
                out[url] = ""  # article fetch failed
            elif url.endswith("/undated-stale-meta/"):
                out[url] = (
                    '<html><head><meta property="article:published_time" '
                    f'content="{meta_stale}"></head><body><p>body text here '
                    'with enough words to extract as full text.</p>'
                    '<p>Second paragraph of the article body.</p>'
                    '</body></html>')
            elif url.endswith("/undated-fresh-meta/"):
                out[url] = (
                    '<html><head><meta property="article:published_time" '
                    f'content="{meta_fresh}"></head><body><p>body text here '
                    'with enough words to extract as full text.</p>'
                    '<p>Second paragraph of the article body.</p>'
                    '</body></html>')
        return out

    monkeypatch.setattr(news, "fetch_many", fake_fetch_many)
    with caplog.at_level(logging.INFO, logger="src.crawlers.news"):
        items = asyncio.run(
            fetch_all_news(first_seen=FirstSeenStore(path=None)))
    by_title = {i.title: i for i in items}
    # Dated item with fetched HTML: kept with full text.
    assert "Fresh AI story" in by_title
    assert "Full article body text" in by_title["Fresh AI story"].full_text
    # Undated, no HTML: empty full_text after the fetch attempt -> dropped.
    assert "Undated no-HTML story" not in by_title
    assert "dropping news item with empty full_text" in caplog.text
    # Undated, stale meta date in the fetched HTML: dropped.
    assert "Undated stale-meta story" not in by_title
    assert "dropping news item with stale meta date" in caplog.text
    # Undated, fresh meta date: kept, re-dated to the meta date.
    assert "Undated fresh-meta story" in by_title
    assert by_title["Undated fresh-meta story"].published_date == (
        real_now - timedelta(hours=2)).replace(microsecond=0)


def test_parse_feed_xml_phase1_still_keeps_empty_text():
    """Phase 1 (no HTML by design) keeps items with empty full_text;
    only phase 2 drops them."""
    items = parse_feed_xml("TechCrunch AI", NEWS_XML_UNDATED, {}, now=NOW)
    assert len(items) == 1
    assert items[0].full_text == ""


# --- 6a. re_extract.py smoke test ----------------------------------

def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STARTUP_HTML = """<html><head>
<title>TestAI: AI-powered analytics | Y Combinator</title>
<meta name="description" content="TestAI builds AI-powered analytics.">
</head><body><script type="application/json">
{"one_liner": "AI-powered analytics",
 "long_description": "Machine learning platform for teams",
 "tags": ["Artificial Intelligence"]}
</script></body></html>"""


def test_re_extract_replays_saved_raw_without_refetch(tmp_path):
    re_extract = _load_script("re_extract")
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    (raw_dir / "startups").mkdir(parents=True)
    (raw_dir / "startups" / "abc123.json").write_text(json.dumps({
        "url": "https://www.ycombinator.com/companies/test-ai",
        "fetched_at": "2026-11-21T12:00:00+00:00",
        "body": STARTUP_HTML,
    }))
    # An existing (stale/wrong) record for the same URL: re-extraction
    # replaces it rather than duplicating it.
    out_dir.mkdir(parents=True)
    (out_dir / "Startups.json").write_text(json.dumps([{
        "source": {"name": "Y Combinator",
                   "url": "https://www.ycombinator.com/companies/test-ai"},
        "entityName": "OldName",
        "collectedAt": "2026-11-21T12:00:00+00:00",
    }]))
    re_extract.main(["--kind", "startups", "--raw-dir", str(raw_dir),
                     "--out-dir", str(out_dir)])
    records = json.loads((out_dir / "Startups.json").read_text())
    assert len(records) == 1
    assert records[0]["entityName"] == "TestAI"
    mapping = json.loads((out_dir / "Entity Mapping Log.json").read_text())
    assert len(mapping) == 1
    assert mapping[0]["raw_name"] == "TestAI"

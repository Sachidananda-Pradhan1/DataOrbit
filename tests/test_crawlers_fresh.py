"""Tests for the Phase II fresh crawlers (jobs + news).

All mocked — NO live network. Payloads are hand-built and dated
relative to a fixed `now` passed into the parsers, so the 24h
freshness gate is deterministic.
"""
from datetime import datetime, timedelta, timezone

import feedparser

from src.crawlers.jobs import (
    classify_region,
    is_ai_job,
    parse_hirist,
    parse_jobicy,
    parse_remoteok,
    parse_working_nomads,
    parse_wwr,
)
from src.crawlers.news import parse_feed_xml

NOW = datetime(2026, 11, 21, 17, 30, tzinfo=timezone.utc)
FRESH = NOW - timedelta(hours=1)
STALE = NOW - timedelta(hours=25)


def iso(dt: datetime) -> str:
    return dt.isoformat()


def rfc822(dt: datetime) -> str:
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


# --- classify_region ---

def test_classify_region_indian_city():
    assert classify_region("Bengaluru, India") == "India"
    assert classify_region("Mumbai") == "India"
    assert classify_region("Hyderabad, Telangana") == "India"
    assert classify_region("Delhi NCR") == "India"


def test_classify_region_remote_india():
    assert classify_region("Remote - India") == "India"
    assert classify_region("India (Remote)") == "India"


def test_classify_region_global():
    assert classify_region("Remote (US)") == "Global"
    assert classify_region("Berlin, Germany") == "Global"
    assert classify_region("Remote") == "Global"


def test_classify_region_empty():
    assert classify_region("") == "Global"
    assert classify_region(None) == "Global"


def test_classify_region_indiana_is_not_india():
    # Word-boundary match: "Indiana" must not count as "India".
    assert classify_region("Indiana, US") == "Global"


# --- is_ai_job ---

def test_is_ai_job_matches():
    assert is_ai_job("Senior ML Engineer")
    assert is_ai_job("Data Scientist")
    assert is_ai_job("AI Researcher")
    assert is_ai_job("Software Engineer", "machine learning", "")


def test_is_ai_job_no_false_positives_on_ai_substring():
    # "ai" as a plain substring would match "training"/"domain".
    assert not is_ai_job("Training Manager")
    assert not is_ai_job("Domain Expert")
    assert not is_ai_job("Campaign Manager")


def test_is_ai_job_empty():
    assert not is_ai_job("")
    assert not is_ai_job(None)


# --- parse_remoteok ---

def _remoteok_item(**overrides):
    item = {
        "slug": "remote-ml-engineer-acme-1",
        "id": "1",
        "epoch": int(FRESH.timestamp()),
        "date": iso(FRESH),
        "company": "Acme AI",
        "position": "ML Engineer",
        "tags": ["machine learning", "python"],
        "description": "<p>Build LLM features.</p>",
        "location": "Remote",
        "url": "https://remoteok.com/remote-jobs/remote-ml-engineer-acme-1",
        "apply_url": "https://acme.example/careers/1",
    }
    item.update(overrides)
    return item


def test_parse_remoteok_skips_element_zero():
    # Even a fresh AI-looking job at index 0 (the legal notice slot)
    # must be skipped — the skip is positional, not content-based.
    legal_notice = _remoteok_item(company="LegalNotice Inc",
                                  position="AI Engineer")
    data = [legal_notice, _remoteok_item()]
    jobs = parse_remoteok(data, now=NOW)
    assert len(jobs) == 1
    assert jobs[0].company == "Acme AI"


def test_parse_remoteok_fresh_ai_only():
    data = [
        {"legal": "RemoteOK legal notice"},
        _remoteok_item(),  # fresh + AI -> kept
        _remoteok_item(id="2", date=iso(STALE),  # stale + AI -> dropped
                       epoch=int(STALE.timestamp()),
                       slug="remote-ml-engineer-old-2"),
        _remoteok_item(id="3", position="Accountant",  # fresh + non-AI
                       tags=["finance"], description="<p>Bookkeeping.</p>",
                       slug="remote-accountant-3"),
        _remoteok_item(id="4", date="", epoch=None),  # no date -> dropped
    ]
    jobs = parse_remoteok(data, now=NOW)
    assert [j.company for j in jobs] == ["Acme AI"]
    assert jobs[0].title == "ML Engineer"
    assert jobs[0].region == "Global"
    assert str(jobs[0].source_url).startswith("https://remoteok.com/")
    assert jobs[0].is_remote is True
    assert jobs[0].role_family == "Data & AI"


def test_parse_remoteok_india_region_and_epoch_fallback():
    job = _remoteok_item(date="", epoch=int(FRESH.timestamp()),
                         location="Mumbai, India")
    jobs = parse_remoteok([{}, job], now=NOW)
    assert len(jobs) == 1
    assert jobs[0].region == "India"
    assert jobs[0].date == FRESH


def test_parse_remoteok_description_mention_does_not_admit():
    # A non-AI role at an AI company: the description mentions AI,
    # but the title/tags do not — the ROLE is not AI, so it is out.
    job = _remoteok_item(
        position="Sales Manager", tags=["sales"],
        description="<p>Join our AI-powered sales platform team.</p>")
    assert parse_remoteok([{}, job], now=NOW) == []


# --- parse_wwr ---

def _wwr_feed(entries_xml: str):
    return feedparser.parse(
        f'<?xml version="1.0"?><rss version="2.0"><channel>'
        f"<title>WWR</title>{entries_xml}</channel></rss>").entries


def test_parse_wwr_splits_company_and_role():
    entries = _wwr_feed(
        f"<item><title>Acme AI: Senior ML Engineer</title>"
        f"<link>https://weworkremotely.com/remote-jobs/acme-ai-ml</link>"
        f"<pubDate>{rfc822(FRESH)}</pubDate>"
        f"<description>LLM platform role.</description></item>"
        f"<item><title>Old Co: ML Engineer</title>"
        f"<link>https://weworkremotely.com/remote-jobs/old-ml</link>"
        f"<pubDate>{rfc822(STALE)}</pubDate>"
        f"<description>ML role.</description></item>"
        f"<item><title>Biz Co: Sales Manager</title>"
        f"<link>https://weworkremotely.com/remote-jobs/biz-sales</link>"
        f"<pubDate>{rfc822(FRESH)}</pubDate>"
        f"<description>Sales role.</description></item>"
        f"<item><title>No Colon Title ML Engineer</title>"
        f"<link>https://weworkremotely.com/remote-jobs/no-colon</link>"
        f"<pubDate>{rfc822(FRESH)}</pubDate>"
        f"<description>ML role.</description></item>"
    )
    jobs = parse_wwr(entries, now=NOW)
    assert len(jobs) == 1  # stale, non-AI and colon-less entries dropped
    assert jobs[0].company == "Acme AI"
    assert jobs[0].title == "Senior ML Engineer"
    assert jobs[0].region == "Global"
    assert jobs[0].is_remote is True
    assert str(jobs[0].source_url).startswith("https://weworkremotely.com/")


# --- parse_jobicy ---

def _jobicy_item(**overrides):
    item = {
        "id": 1,
        "url": "https://jobicy.com/jobs/1-ml-engineer-acme",
        "jobSlug": "1-ml-engineer-acme",
        "jobTitle": "ML Engineer (Remote)",
        "companyName": "Acme AI",
        "jobIndustry": ["AI"],
        "jobType": ["Full-Time"],
        "jobGeo": "Worldwide",
        "jobLevel": "Senior",
        "jobExcerpt": "Build ML systems.",
        "jobDescription": "<p>Deep learning role.</p>",
        "pubDate": iso(FRESH),
    }
    item.update(overrides)
    return item


def test_parse_jobicy_dict_shape():
    data = {"jobs": [
        _jobicy_item(),
        _jobicy_item(id=2, pubDate=iso(STALE)),  # stale -> dropped
        _jobicy_item(id=3, jobTitle="Office Manager",  # non-AI -> dropped
                     jobIndustry=["Admin"],
                     jobExcerpt="Run the office.",
                     jobDescription="<p>Admin role.</p>"),
        _jobicy_item(id=4, pubDate=""),  # no date -> dropped
    ], "jobCount": 4}
    jobs = parse_jobicy(data, now=NOW)
    assert len(jobs) == 1
    assert jobs[0].company == "Acme AI"
    assert jobs[0].title == "ML Engineer (Remote)"
    assert jobs[0].is_remote is True
    assert str(jobs[0].source_url).startswith("https://jobicy.com/jobs/")


def test_parse_jobicy_list_shape_and_india():
    jobs = parse_jobicy(
        [_jobicy_item(jobGeo="India", jobTitle="Data Scientist")], now=NOW)
    assert len(jobs) == 1
    assert jobs[0].region == "India"


def test_parse_jobicy_bad_shape():
    assert parse_jobicy(None, now=NOW) == []
    assert parse_jobicy({"jobs": None}, now=NOW) == []


# --- parse_working_nomads ---

def _wn_item(**overrides):
    item = {
        "url": "https://www.workingnomads.com/job/go/1/",
        "title": "Machine Learning Engineer",
        "description": "<p>LLM work.</p>",
        "company_name": "Acme AI",
        "category_name": "Development",
        "tags": ["machine learning"],
        "location": "Worldwide",
        "pub_date": iso(FRESH),
    }
    item.update(overrides)
    return item


def test_parse_working_nomads():
    jobs = parse_working_nomads([
        _wn_item(),
        _wn_item(url="https://www.workingnomads.com/job/go/2/",
                 pub_date=iso(STALE)),  # stale -> dropped
        _wn_item(url="https://www.workingnomads.com/job/go/3/",
                 title="Customer Support Agent",
                 description="<p>Support role.</p>", tags=["support"]),
    ], now=NOW)
    assert len(jobs) == 1
    assert jobs[0].company == "Acme AI"
    assert jobs[0].region == "Global"
    assert jobs[0].is_remote is True


def test_parse_working_nomads_india_location():
    jobs = parse_working_nomads([_wn_item(location="India")], now=NOW)
    assert jobs[0].region == "India"


# --- parse_hirist ---

HIRIST_HTML = """
<html><body>
<div class="job-card">
  <a href="/j/ml-engineer-acme-ai-1001">ML Engineer</a>
  <div class="company-name">Acme AI</div>
  <div class="job-location">Bengaluru, India</div>
  <span class="posted-date">Posted 3 hours ago</span>
  <p>Build LLM-powered features.</p>
</div>
<div class="job-card">
  <a href="/j/data-scientist-old-1002">Data Scientist</a>
  <div class="company-name">Old Corp</div>
  <div class="job-location">Pune, India</div>
  <span class="posted-date">Posted 2 days ago</span>
  <p>ML modelling role.</p>
</div>
<div class="job-card">
  <a href="/j/accountant-1003">Accountant</a>
  <div class="company-name">Biz Ltd</div>
  <div class="job-location">Mumbai, India</div>
  <span class="posted-date">Posted 3 hours ago</span>
  <p>Bookkeeping and payroll.</p>
</div>
<div class="job-card">
  <a href="/j/ai-engineer-remote-1004">AI Engineer</a>
  <div class="company-name">Remote AI Labs</div>
  <div class="job-location">Remote</div>
  <span class="posted-date">Posted 1 day ago</span>
  <p>Generative AI role.</p>
</div>
</body></html>
"""


def test_parse_hirist_relative_dates_and_india():
    jobs = parse_hirist(HIRIST_HTML, now=NOW)
    by_title = {j.title: j for j in jobs}
    # Fresh AI job with an Indian location -> kept, region India.
    assert "ML Engineer" in by_title
    ml = by_title["ML Engineer"]
    assert ml.company == "Acme AI"
    assert ml.region == "India"
    assert ml.date == NOW - timedelta(hours=3)
    assert str(ml.source_url) == (
        "https://hirist.tech/j/ml-engineer-acme-ai-1001")
    # Stale (2 days ago) and non-AI jobs are excluded.
    assert "Data Scientist" not in by_title
    assert "Accountant" not in by_title
    # Exactly 24h old is still fresh (inclusive gate); empty/generic
    # location on the India-only board defaults to India.
    assert "AI Engineer" in by_title
    assert by_title["AI Engineer"].region == "India"
    assert by_title["AI Engineer"].is_remote is True


def test_parse_hirist_js_shell_has_no_jobs():
    shell = ("<html><body><div id='root'></div>"
             "You need to enable JavaScript to run this app.</body></html>")
    assert parse_hirist(shell, now=NOW) == []
    assert parse_hirist("", now=NOW) == []


# --- parse_feed_xml ---

NEWS_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>TechCrunch AI</title>
<item><title>Fresh AI story</title>
<link>https://techcrunch.com/2026/11/21/fresh-ai-story/</link>
<pubDate>{rfc822(FRESH)}</pubDate></item>
<item><title>Stale AI story</title>
<link>https://techcrunch.com/2026/11/20/stale-ai-story/</link>
<pubDate>{rfc822(STALE)}</pubDate></item>
<item><title>Undated AI story</title>
<link>https://techcrunch.com/2026/11/21/undated-ai-story/</link></item>
</channel></rss>"""

ARTICLE_HTML = ("<html><body><article><p>Full article body text about "
                "the fresh AI story, with enough words to extract.</p>"
                "<p>Second paragraph of the article body.</p>"
                "</article></body></html>")


def test_parse_feed_xml_fresh_only_with_full_text():
    items = parse_feed_xml(
        "TechCrunch AI", NEWS_XML,
        {"https://techcrunch.com/2026/11/21/fresh-ai-story/": ARTICLE_HTML},
        now=NOW)
    # Stale excluded; undated kept via the first-seen heuristic
    # (published_date == now) — it was previously dropped outright.
    assert len(items) == 2
    item = items[0]
    assert item.title == "Fresh AI story"
    assert item.source.name == "TechCrunch AI"
    assert item.published_date == FRESH
    assert "Full article body text" in item.full_text
    assert str(item.url) == "https://techcrunch.com/2026/11/21/fresh-ai-story/"
    assert items[1].title == "Undated AI story"
    assert items[1].published_date == NOW


def test_parse_feed_xml_undated_uses_meta_date_not_first_seen():
    html = ('<html><head><meta property="article:published_time" '
            'content="2026-11-21T10:00:00Z"></head><body><p>body</p>'
            '</body></html>')
    items = parse_feed_xml(
        "TechCrunch AI", NEWS_XML,
        {"https://techcrunch.com/2026/11/21/undated-ai-story/": html},
        now=NOW)
    undated = [i for i in items if i.title == "Undated AI story"]
    assert len(undated) == 1
    assert undated[0].published_date == FRESH.replace(hour=10, minute=0)


def test_parse_feed_xml_stale_meta_date_still_dropped():
    html = ('<html><head><meta property="article:published_time" '
            'content="2026-11-19T10:00:00Z"></head><body><p>body</p>'
            '</body></html>')
    items = parse_feed_xml(
        "TechCrunch AI", NEWS_XML,
        {"https://techcrunch.com/2026/11/21/undated-ai-story/": html},
        now=NOW)
    assert [i.title for i in items] == ["Fresh AI story"]


def test_parse_feed_xml_missing_html_keeps_item_with_empty_text():
    items = parse_feed_xml("The Decoder", NEWS_XML, {}, now=NOW)
    assert len(items) == 2  # fresh + undated (first-seen); stale excluded
    assert items[0].full_text == ""


def test_parse_feed_xml_empty_feed():
    assert parse_feed_xml("TechCrunch AI", "", {}, now=NOW) == []

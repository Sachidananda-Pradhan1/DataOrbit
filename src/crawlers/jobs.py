"""Jobs (Phase II): the 5 user-approved job boards, 24h freshness,
precise published timestamp, role_family + is_remote per the Job
schema, and India vs Global tagging on every job.

Approved boards (user, 21 Nov 2026):
- RemoteOK: JSON API; element [0] is a legal notice, not a job —
  skip it. Fields: position, company, date (ISO), epoch, location,
  tags, description (HTML), url (job page), apply_url.
- We Work Remotely: RSS; titles are "Company: Role" — split on the
  first colon. NOTE (verified live 21 Nov 2026 ~17:30 UTC): the
  programming category feed's newest item was 3 Nov 2026 — stale.
  fetch_all_jobs therefore falls back to the full feed
  (https://weworkremotely.com/remote-jobs.rss, newest item
  21 Nov 17:02 UTC) whenever the category feed yields no fresh jobs.
- Jobicy: JSON API; jobs live under the "jobs" key of the response
  dict (also handles a bare list defensively). Fields: jobTitle,
  companyName, pubDate (ISO), url, jobGeo, jobType, jobDescription.
- Working Nomads: keyless JSON API, full descriptions + exact
  pub_date. NOTE: at the 21 Nov check its newest posting was
  19 Nov, so the 24h gate can legitimately yield 0 from this board
  on a quiet day — that is data, not a bug.
- Hirist (India anchor): India tech-only board, no feed/API.
  Listing pages are a React SPA — the raw HTML is only a JS shell
  ("You need to enable JavaScript to run this app."), so a plain
  fetch finds no jobs; fetch_all_jobs tries the plain fetch first,
  then the Playwright render fallback (fetcher.fetch_with_playwright)
  when Playwright is installed. Job detail pages are also
  client-rendered (empty __NEXT_DATA__ mandate info), so the
  sitemap route does not help either. Dates on rendered cards are
  relative ("Posted 1 day ago") and are normalised via
  date_utils.parse_date after stripping the "Posted" prefix.
  Hirist jobs default to region "India" (India-only board).
  robots.txt asks crawl-delay 10 — Hirist pages are fetched
  sequentially with HIRIST_DELAY_SECS between them.

India vs Global: every job is tagged by classify_region() from its
location field (India / an Indian city or state / "Remote - India"
-> "India", else "Global").

Deliberately NOT implemented (stretch goals / fallbacks only):
- Wellfound: no public API; plain HTTP 403s (DataDome). Stretch
  goal only — do not implement a bypass.
- Hacker News "Who is hiring?": fallback note only, not a board.
- YC Work at a Startup: no posting dates on public listings, so it
  cannot prove 24h freshness — excluded on purpose.

Freshness rule: a job with no usable published date is EXCLUDED,
never guessed — an unprovable 24h claim is worse than a missing row.
The one fallback is the first-seen heuristic (crawlers/first_seen.py):
when a first-seen store is wired in, a dateless job is dated by the
time its URL was first observed, and the same 24h gate applies to
that timestamp exactly as to a published date. Same for a job with
no traceable source URL — excluded, never faked.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

import feedparser
from bs4 import BeautifulSoup
from pydantic import ValidationError

from src.crawlers.date_utils import is_fresh, parse_date
from src.crawlers.fetcher import fetch_many
from src.crawlers.first_seen import FirstSeenStore
from src.models import Job

logger = logging.getLogger(__name__)

JOB_BOARDS: dict[str, str] = {
    # JSON API — skip element [0] (legal notice).
    "RemoteOK": "https://remoteok.com/api",
    # RSS — category feed; stale at last check, see module docstring.
    "We Work Remotely": "https://weworkremotely.com/categories/remote-programming-jobs.rss",
    # JSON API — jobs under the "jobs" key.
    "Jobicy": "https://jobicy.com/api/jobs",
    # JSON API — list of dicts with pub_date.
    "Working Nomads": "https://www.workingnomads.com/api/exposed_jobs/",
    # No feed/API — React SPA listing pages (see module docstring).
    "Hirist": "https://www.hirist.tech",
}

# WWR fallback when the category feed yields no fresh jobs.
WWR_FULL_FEED = "https://weworkremotely.com/remote-jobs.rss"

# Hirist listing pages to fetch — small number, polite delays.
# (hirist.tech 301-redirects to www.hirist.tech; use the www host.)
HIRIST_PAGES = [
    "https://www.hirist.tech/jobs/ai",
    "https://www.hirist.tech/jobs/machine-learning",
    "https://www.hirist.tech/jobs/data-science",
]
# Seconds between Hirist page fetches (robots.txt: crawl-delay 10).
HIRIST_DELAY_SECS = 10.0

# Keywords to keep a job AI-relevant (case-insensitive match on
# TITLE/TAGS ONLY — see the is_ai_job note below). Short keywords
# ("ai", "ml") are matched on word boundaries only — a plain
# substring "ai" would match "training", "domain", "campaign", etc.
AI_KEYWORDS = ["ai", "artificial intelligence", "machine learning", "ml",
               "llm", "data scientist", "data science", "data engineer",
               "ml engineer", "deep learning", "nlp",
               "generative ai", "computer vision"]

# Tokens that mark a location as India (word-boundary matched).
_INDIA_TOKENS = [
    "india",
    # Major tech cities / metros.
    "bengaluru", "bangalore", "mumbai", "bombay", "delhi", "new delhi",
    "hyderabad", "pune", "chennai", "madras", "gurugram", "gurgaon",
    "noida", "kolkata", "calcutta", "ahmedabad", "jaipur", "kochi",
    "cochin", "thiruvananthapuram", "trivandrum", "indore", "bhopal",
    "lucknow", "kanpur", "nagpur", "patna", "surat", "vadodara",
    "coimbatore", "mysuru", "mysore", "visakhapatnam", "vizag",
    "bhubaneswar", "chandigarh", "dehradun", "guwahati", "goa",
    "mangaluru", "mangalore", "lucknow", "ranchi", "raipur",
    # States / UTs.
    "karnataka", "maharashtra", "tamil nadu", "telangana",
    "andhra pradesh", "gujarat", "rajasthan", "uttar pradesh",
    "madhya pradesh", "west bengal", "punjab", "haryana", "bihar",
    "odisha", "assam", "kerala", "himachal pradesh", "uttarakhand",
    "jharkhand", "chhattisgarh", "delhi ncr",
]
_INDIA_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _INDIA_TOKENS) + r")\b",
    re.IGNORECASE,
)

# Locations that say nothing about geography — a board-level region
# default (Hirist -> India) may apply to these, and only these.
_GENERIC_LOCATIONS = {"", "remote", "work from home", "wfh", "anywhere",
                      "remote-first", "remote friendly"}

def _norm_key(company: str, title: str) -> str:
    """Dedup key: normalised company|title, whitespace-collapsed."""
    return " ".join(f"{company}|{title}".lower().split())


def is_ai_job(*texts) -> bool:
    """True if any AI keyword matches the given texts. Short
    keywords need a word boundary. Accepts strings, lists of
    strings (some boards return list fields), and None.

    Deliberate scope: callers pass TITLE and TAGS (or industry /
    category) only — NOT the description. Descriptions are company
    boilerplate ("we use AI..."), so description matching admits
    non-AI roles at AI companies: a live check on 21 Nov 2026
    showed 58 of 69 WWR matches and 65 of 80 Jobicy matches were
    description-only, including nurses, sales managers and an
    "Influencer Executive". A job is an AI job when the ROLE is
    AI, and the role is in the title/tags."""
    parts: list[str] = []
    for t in texts:
        if isinstance(t, (list, tuple)):
            parts.extend(str(x) for x in t if x)
        elif t:
            parts.append(str(t))
    haystack = " ".join(parts).lower()
    if not haystack.strip():
        return False
    for kw in AI_KEYWORDS:
        if len(kw) <= 2:
            if re.search(rf"\b{re.escape(kw)}\b", haystack):
                return True
        elif kw in haystack:
            return True
    return False


def classify_region(location: str) -> str:
    """"India" if the location mentions India or an Indian city/state
    (incl. "Remote - India" / "India (Remote)"), else "Global"."""
    if location and _INDIA_RE.search(location):
        return "India"
    return "Global"


def _role_family(title: str) -> str:
    t = title.lower()
    if "research" in t:
        return "Research"
    if "data" in t or re.search(r"\b(ml|ai)\b", t) or "scientist" in t:
        return "Data & AI"
    if any(w in t for w in ("engineer", "developer", "devops", "sre",
                            "architect", "programmer")):
        return "Engineering"
    if "product" in t:
        return "Product"
    if "design" in t:
        return "Design"
    return "Other"


def _is_remote(location: str, *, default: bool = False) -> bool:
    loc = (location or "").lower()
    if "remote" in loc or "work from home" in loc or "anywhere" in loc:
        return True
    return default


def _make_job(*, company: str, title: str, date: datetime | None,
              location: str, source_url: str, default_remote: bool = False,
              default_region: str | None = None,
              remote_text: str = "") -> Job | None:
    """Build a Job, or None if the record is untraceable/unprovable:
    no date (cannot prove 24h freshness), no company, or no valid URL.
    Invalid URLs are dropped, never faked."""
    if not company or not company.strip():
        return None
    if date is None or not source_url:
        return None
    region = classify_region(location)
    if (region == "Global" and default_region
            and location.strip().lower() in _GENERIC_LOCATIONS):
        # Board default (e.g. Hirist = India-only) applies only when
        # the location is empty or generic-remote — a location that
        # names somewhere else ("Remote (US)") still wins.
        region = default_region
    try:
        return Job(
            company=company.strip(),
            title=title.strip(),
            date=date,
            # Titles often carry the remote signal, e.g. "ML Engineer
            # (Remote, Ireland)" — check them as well as the location.
            is_remote=_is_remote(f"{location} {remote_text}",
                                 default=default_remote),
            role_family=_role_family(title),
            region=region,
            source_url=source_url,
        )
    except ValidationError as exc:
        logger.debug("dropping job with invalid URL %r: %s", source_url, exc)
        return None


def _parse_epoch(value) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _date_or_first_seen(published: datetime | None, url: str,
                        first_seen: FirstSeenStore | None,
                        now: datetime) -> datetime | None:
    """First-seen fallback for a job with no parseable date: when a
    store is wired in and the job has a URL, date it by the URL's
    first observation. The caller still runs the result through
    is_fresh, so a first-seen timestamp older than 24h drops the
    job. With no store wired in, dateless jobs are excluded."""
    if published is None and first_seen is not None and url:
        return first_seen.observe(url, now)
    return published


# --- per-board parsers (fresh-only, AI-filtered, region-tagged) ---

def parse_remoteok(data: list[dict], *, now: datetime | None = None,
                   first_seen: FirstSeenStore | None = None) -> list[Job]:
    """RemoteOK JSON array. Element [0] is a legal notice — skip it."""
    now = now or datetime.now(timezone.utc)
    jobs: list[Job] = []
    for item in (data or [])[1:]:
        if not isinstance(item, dict) or "position" not in item:
            continue
        title = item.get("position") or ""
        tags = " ".join(item.get("tags") or [])
        if not is_ai_job(title, tags):  # title/tags only, see is_ai_job
            continue
        url = item.get("url") or item.get("apply_url") or ""
        published = parse_date(item.get("date") or "", now=now)
        if published is None and item.get("epoch"):
            published = _parse_epoch(item.get("epoch"))
        published = _date_or_first_seen(published, url, first_seen, now)
        if published is None or not is_fresh(published, now=now):
            continue
        job = _make_job(
            company=item.get("company") or "", title=title,
            date=published, location=item.get("location") or "",
            source_url=url, default_remote=True, remote_text=title,
        )
        if job is not None:
            jobs.append(job)
    return jobs


def parse_wwr(entries, *, now: datetime | None = None,
              first_seen: FirstSeenStore | None = None) -> list[Job]:
    """We Work Remotely RSS entries. Titles are "Company: Role" —
    split on the first colon. Entries without a colon carry no
    company and are skipped (never invent one)."""
    now = now or datetime.now(timezone.utc)
    jobs: list[Job] = []
    for entry in entries or []:
        raw_title = getattr(entry, "title", "") or ""
        if ":" not in raw_title:
            continue
        company, title = raw_title.split(":", 1)
        if not is_ai_job(title):  # title only, see is_ai_job
            continue
        url = getattr(entry, "link", "") or ""
        published = parse_date(
            getattr(entry, "published", "") or getattr(entry, "updated", "")
            or "", now=now)
        published = _date_or_first_seen(published, url, first_seen, now)
        if published is None or not is_fresh(published, now=now):
            continue
        job = _make_job(
            company=company, title=title, date=published, location="Remote",
            source_url=url,
            default_remote=True, remote_text=title,
        )
        if job is not None:
            jobs.append(job)
    return jobs


def parse_jobicy(data, *, now: datetime | None = None,
                 first_seen: FirstSeenStore | None = None) -> list[Job]:
    """Jobicy JSON API. The response is a dict with jobs under the
    "jobs" key; a bare list is also handled defensively."""
    if isinstance(data, dict):
        items = data.get("jobs") or []
    elif isinstance(data, list):
        items = data
    else:
        return []
    now = now or datetime.now(timezone.utc)
    jobs: list[Job] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("jobTitle") or ""
        # Title + industry only, see is_ai_job (no description).
        if not is_ai_job(title, item.get("jobIndustry") or ""):
            continue
        url = item.get("url") or ""
        published = parse_date(item.get("pubDate") or "", now=now)
        published = _date_or_first_seen(published, url, first_seen, now)
        if published is None or not is_fresh(published, now=now):
            continue
        job_type = item.get("jobType") or []
        if isinstance(job_type, list):
            job_type = " ".join(str(t) for t in job_type)
        location = item.get("jobGeo") or ""
        job = _make_job(
            company=item.get("companyName") or "", title=title,
            date=published,
            location=f"{location} {job_type}".strip(),
            source_url=url,
            default_remote=False, remote_text=title,
        )
        if job is not None:
            jobs.append(job)
    return jobs


def parse_working_nomads(data, *, now: datetime | None = None,
                         first_seen: FirstSeenStore | None = None) -> list[Job]:
    """Working Nomads JSON API — a list of dicts with pub_date."""
    now = now or datetime.now(timezone.utc)
    jobs: list[Job] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or ""
        tags = " ".join(item.get("tags") or [])
        # Title + tags + category only, see is_ai_job (no description).
        if not is_ai_job(title, tags, item.get("category_name") or ""):
            continue
        url = item.get("url") or ""
        published = parse_date(item.get("pub_date") or "", now=now)
        published = _date_or_first_seen(published, url, first_seen, now)
        if published is None or not is_fresh(published, now=now):
            continue
        job = _make_job(
            company=item.get("company_name") or "", title=title,
            date=published, location=item.get("location") or "",
            source_url=url, default_remote=True,
            remote_text=title,
        )
        if job is not None:
            jobs.append(job)
    return jobs


_RELATIVE_RE = re.compile(
    r"(\d+\s+(?:minute|hour|day|week|month)s?\s+ago|just now)",
    re.IGNORECASE,
)


def _hirist_field(card, *needles: str) -> str:
    """Read a field (company / location) off a Hirist job card,
    trying strategies in order — most structured first, so a
    renamed CSS class alone cannot break extraction:

    1. Structured selectors: schema.org itemprop (hiringOrganization
       / jobLocation) and data-<field> attributes.
    2. Exact class-token match: a class token equal to the needle,
       or hyphen-joined with it ("company-name", "job-location").
    3. Substring class match (legacy fallback — Hirist's rendered
       class names are not part of any contract).
    4. Label text pattern: "Company: Acme" / "Location: Remote"
       in the card text.

    Returns "" when no strategy finds the field; the caller logs
    the unparseable card instead of inventing a value."""
    needles = [n.lower() for n in needles]
    # 1. Structured selectors.
    itemprop = {"company": "hiringOrganization", "location": "jobLocation"}
    for needle in needles:
        selectors = [f"[data-{needle}]"]
        if needle in itemprop:
            selectors.append(f"[itemprop='{itemprop[needle]}']")
        for selector in selectors:
            el = card.select_one(selector)
            if el is None:
                continue
            text = (el.get(f"data-{needle}")
                    or el.get_text(" ", strip=True) or "").strip()
            if text:
                return text
    # 2. Exact class-token match (token == needle, or needle joined
    # by a hyphen on either side).
    for el in card.find_all(True, class_=True):
        tokens = [c.lower() for c in (el.get("class") or [])]
        if any(t == n or t.startswith(n + "-") or t.endswith("-" + n)
               for t in tokens for n in needles):
            text = el.get_text(" ", strip=True)
            if text:
                return text
    # 3. Substring class match (legacy fallback).
    for el in card.find_all(True, class_=True):
        classes = " ".join(el.get("class") or []).lower()
        if any(n in classes for n in needles):
            text = el.get_text(" ", strip=True)
            if text:
                return text
    # 4. Label text pattern ("Company: Acme"). Prefer a leaf
    # element's own text — bounded, unlike the space-joined card
    # text, where "Company: X Location: Y" would over-capture.
    for el in card.find_all(True):
        if el.find(True) is not None:
            continue  # only leaf elements have bounded text
        text = el.get_text(" ", strip=True)
        for needle in needles:
            match = re.match(rf"{re.escape(needle)}\s*[:\-]\s*(.+)$",
                             text, re.IGNORECASE)
            if match and match.group(1).strip():
                return match.group(1).strip()[:120]
    # Last resort: the whole card text, cut at the next "Label:" or
    # "Posted ..." and capped at four words.
    card_text = card.get_text(" ", strip=True)
    for needle in needles:
        match = re.search(rf"{re.escape(needle)}\s*[:\-]\s*([^,;|]+)",
                          card_text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            text = re.split(r"\s+[A-Za-z]+\s*:", text, maxsplit=1)[0]
            text = re.split(r"\s+Posted\b", text, maxsplit=1)[0]
            words = text.split()
            if words:
                return " ".join(words[:4])
    return ""


def parse_hirist(html: str, base_url: str = "https://hirist.tech",
                 *, now: datetime | None = None,
                 first_seen: FirstSeenStore | None = None) -> list[Job]:
    """Hirist listing page HTML (rendered DOM — see module docstring:
    the raw page is a JS shell, so callers render it with Playwright
    when the plain fetch finds no job cards).

    Each job card links to /j/<slug>-<id>. Company/location/posted
    date are read from the card; the posted date is relative
    ("Posted 1 day ago") and normalised via parse_date, falling back
    to the first-seen store when a store is wired in. Cards with no
    company, no parseable date, or no detail link are skipped — and
    counted, with a debug sample logged, instead of vanishing
    silently. Region defaults to "India" (India-only board)."""
    from urllib.parse import urljoin

    if not html:
        return []
    now = now or datetime.now(timezone.utc)
    soup = BeautifulSoup(html, "html.parser")
    jobs: list[Job] = []
    seen_urls: set[str] = set()
    unparseable = 0
    unparseable_sample = ""
    for link in soup.find_all("a", href=re.compile(r"^/j/")):
        href = link.get("href") or ""
        url = urljoin(base_url, href)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        title = link.get_text(" ", strip=True)
        if not title:
            continue
        # Climb to the card container: the smallest ancestor whose
        # text mentions a posted date (or a few levels up as fallback).
        card = link
        for _ in range(5):
            parent = card.parent
            if parent is None:
                break
            card = parent
            if _RELATIVE_RE.search(card.get_text(" ", strip=True)):
                break
        card_text = card.get_text(" ", strip=True)
        if not is_ai_job(title):  # title only, see is_ai_job
            continue
        company = _hirist_field(card, "company")
        location = _hirist_field(card, "location")
        match = _RELATIVE_RE.search(card_text)
        published = parse_date(match.group(1), now=now) if match else None
        published = _date_or_first_seen(published, url, first_seen, now)
        if not company or published is None:
            # Unparseable card: never invent a company or a date —
            # count it and keep a sample for debugging.
            unparseable += 1
            if not unparseable_sample:
                unparseable_sample = card_text[:200]
            logger.debug(
                "Hirist: dropping unparseable card %s (company=%r, "
                "date found=%s)", url, company, published is not None)
            continue
        if not is_fresh(published, now=now):
            continue
        job = _make_job(
            company=company, title=title, date=published,
            location=location, source_url=url,
            default_region="India", remote_text=title,
        )
        if job is not None:
            jobs.append(job)
    if unparseable:
        logger.warning(
            "Hirist: %d card(s) unparseable (missing company or posted "
            "date); sample card text: %r", unparseable, unparseable_sample)
    return jobs


# --- orchestration ---

async def _hirist_via_playwright(
        url: str, first_seen: FirstSeenStore | None = None) -> list[Job]:
    """Render a Hirist listing page with Playwright (JS shell) and
    parse the rendered DOM. Returns [] when Playwright is not
    installed or the render fails — Hirist must not kill the run."""
    try:
        from src.crawlers.fetcher import fetch_with_playwright

        html = await fetch_with_playwright(url)
    except ImportError:
        logger.warning(
            "Hirist %s is JS-rendered and Playwright is not installed; "
            "skipping Hirist (install playwright + chromium to enable it).",
            url)
        return []
    except Exception as exc:
        logger.warning("Hirist Playwright render failed for %s: %s", url, exc)
        return []
    return parse_hirist(html, base_url="https://www.hirist.tech",
                        first_seen=first_seen)


async def fetch_all_jobs(limit_per_board: int | None = None,
                         first_seen: FirstSeenStore | None = None) -> list[Job]:
    """Fetch all five boards, parse fresh AI jobs, dedup across boards
    on normalised company|title. One dead board logs a warning and the
    others still run. limit_per_board caps each board's contribution
    (None = all fresh jobs found). Dateless jobs fall back to the
    first-seen store (default: the persistent
    data/processed/first_seen.json); the 24h gate still applies."""
    jobs: list[Job] = []
    seen: set[str] = set()
    store = first_seen if first_seen is not None else FirstSeenStore.default()

    def collect(parsed: list[Job], board: str) -> None:
        kept = parsed[:limit_per_board] if limit_per_board else parsed
        for job in kept:
            key = _norm_key(job.company, job.title)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)
        logger.info("%s: %d fresh AI jobs", board, len(kept))

    # --- API / RSS boards in one bulk fetch ---
    api_urls = [JOB_BOARDS[b] for b in
                ("RemoteOK", "We Work Remotely", "Jobicy", "Working Nomads")]
    try:
        bodies = await fetch_many(api_urls)
    except Exception as exc:
        logger.warning("bulk job-board fetch failed: %s", exc)
        bodies = {}

    try:
        raw = bodies.get(JOB_BOARDS["RemoteOK"], "")
        collect(parse_remoteok(json.loads(raw), first_seen=store)
                if raw else [], "RemoteOK")
    except Exception as exc:
        logger.warning("RemoteOK failed: %s", exc)

    try:
        raw = bodies.get(JOB_BOARDS["We Work Remotely"], "")
        parsed = (parse_wwr(feedparser.parse(raw).entries, first_seen=store)
                  if raw else [])
        if not parsed:
            # Category feed is stale at last check — try the full feed.
            logger.info("WWR category feed yielded no fresh jobs; "
                        "trying the full feed")
            full = await fetch_many([WWR_FULL_FEED])
            raw_full = full.get(WWR_FULL_FEED, "")
            if raw_full:
                parsed = parse_wwr(feedparser.parse(raw_full).entries,
                                   first_seen=store)
        collect(parsed, "We Work Remotely")
    except Exception as exc:
        logger.warning("We Work Remotely failed: %s", exc)

    try:
        raw = bodies.get(JOB_BOARDS["Jobicy"], "")
        collect(parse_jobicy(json.loads(raw), first_seen=store)
                if raw else [], "Jobicy")
    except Exception as exc:
        logger.warning("Jobicy failed: %s", exc)

    try:
        raw = bodies.get(JOB_BOARDS["Working Nomads"], "")
        collect(parse_working_nomads(json.loads(raw), first_seen=store)
                if raw else [], "Working Nomads")
    except Exception as exc:
        logger.warning("Working Nomads failed: %s", exc)

    # --- Hirist: sequential, polite delays (robots.txt crawl-delay 10) ---
    try:
        hirist_jobs: list[Job] = []
        for page_url in HIRIST_PAGES:
            try:
                body = (await fetch_many([page_url])).get(page_url, "")
            except Exception as exc:
                logger.warning("Hirist page %s fetch failed: %s", page_url, exc)
                body = ""
            found: list[Job] = []
            if body:
                if "/j/" in body:
                    found = parse_hirist(body,
                                         base_url="https://www.hirist.tech",
                                         first_seen=store)
                else:
                    # JS shell — no job links in the raw HTML.
                    found = await _hirist_via_playwright(page_url, store)
            hirist_jobs.extend(found)
            await asyncio.sleep(HIRIST_DELAY_SECS)
        collect(hirist_jobs, "Hirist")
    except Exception as exc:
        logger.warning("Hirist failed: %s", exc)

    return jobs

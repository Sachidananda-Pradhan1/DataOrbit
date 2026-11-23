"""Research papers (Phase I): arXiv API + Hugging Face Papers API,
correlated with GitHub repos for live star counts.

ARXIV — VERIFIED LIVE 2026-11-21 (no key, no login):
- https://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=2
  HTTP 200, 2 entries. Pagination verified on cs.LG (start=100 returns
  entries; totalResults 300,288 for cs.LG alone). fetch_papers() pages
  4 categories (cs.AI, cs.LG, cs.CL, cs.CV) to the 1,000 target,
  deduping by paper_url (papers are cross-listed across categories).
  arXiv asks for at most 1 request per 3 seconds — respected via
  ARXIV_DELAY_SECS between page requests.

PAPERS WITH CODE — VERDICT 2026-11-21: NOT ACCESSIBLE (dead API).
- https://paperswithcode.com/api/v1/papers/?page=1&page_size=5 returns
  HTTP 302 redirecting to https://huggingface.co/papers/trending —
  an HTML page, not JSON. Papers With Code was acquired by Hugging
  Face and the standalone API is gone; no auth can fix a redirect.
  Documented here and skipped — NOT faked.
- SUCCESSOR: the Hugging Face Papers Search API
  (https://huggingface.co/api/papers/search?q=...&limit=100), verified
  2026-11-21: HTTP 200, 100 results for q="large language models",
  45 of which carry githubRepo (full https://github.com/... URL) AND
  githubStars. fetch_hf_papers() uses it to attach GitHub repo links
  + star counts to arXiv papers (matched on arXiv id) and to top up
  the list when arXiv alone is short of target.

GITHUB STARS — RATE LIMITS: unauthenticated GitHub API allows 60
requests/hour, so WITHOUT a GITHUB_TOKEN only the first
MAX_UNAUTH_STAR_LOOKUPS (50) papers with a repo link get a live
lookup and the rest keep github_stars=None; WITH a token all are
looked up. 0.5s delay between calls either way. Papers enriched by
Hugging Face already carry stars and need no GitHub call. Repo
links come from Hugging Face AND from a scan of each arXiv abstract
for github.com links (parse_arxiv_atom / github_url_from_text);
the HF merge in fetch_papers() is unchanged — it fills github_url
only when the abstract scan found none, and attaches HF stars
whenever the paper has none yet.
Rate-limited responses (403 with X-RateLimit-Remaining: 0 or a
"rate limit" message, or 429) are retried with backoff — Retry-After
or X-RateLimit-Reset when present, else exponential — up to
GITHUB_MAX_RETRIES attempts; if the limit persists, github_stars()
raises GitHubRateLimited and the enricher stops further lookups
instead of burning quota. Every paper records whether its stars
were fetched (ResearchPaper.github_stars_fetched); papers that
were not fetched keep github_stars=None and the flag False, and
_enrich_github_stars() prints a fetched/not-fetched summary.

LIVE VERIFICATION (needs the user's GITHUB_TOKEN — not available yet):
- export GITHUB_TOKEN=<token>  (public repo metadata needs no scopes)
- .venv/bin/python -c "from src.crawlers.papers import github_stars;
  print(github_stars('huggingface/transformers'))"  -> live star count
- Then run the bulk papers crawl and check the Research Papers sheet:
  every row with a GitHub URL should have github_stars filled.
  Without a token the same check shows the first 50 lookups filled
  and the rest None — the documented unauthenticated cap.

Every paper keeps a real paper_url (arXiv abs URL). No invented
titles, authors or URLs.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from datetime import datetime, timezone

import feedparser
import httpx
from pydantic import ValidationError

from src.config import settings
from src.crawlers.date_utils import parse_date
from src.models import ResearchPaper
from src.utils import http as http_util

# arXiv API base endpoint (was a single fixed query; now built per
# category/page by arxiv_query_url).
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL", "cs.CV")
ARXIV_PAGE_SIZE = 100
# arXiv usage policy: at most 1 request per 3 seconds.
ARXIV_DELAY_SECS = 3.0

# Hugging Face Papers Search API (successor of Papers With Code).
HF_PAPERS_API = "https://huggingface.co/api/papers/search"
HF_QUERIES = (
    "large language models",
    "computer vision",
    "natural language processing",
    "reinforcement learning",
    "diffusion models",
    "artificial intelligence",
)
HF_LIMIT_PER_QUERY = 100
HF_DELAY_SECS = 1.0

# GitHub stars: unauthenticated limit is 60 req/hour — bound lookups.
MAX_UNAUTH_STAR_LOOKUPS = 50
GITHUB_DELAY_SECS = 0.5
# Rate-limit retries: back off (Retry-After / X-RateLimit-Reset when
# present, else exponential from GITHUB_BACKOFF_BASE_SECS) and give
# up after GITHUB_MAX_RETRIES attempts rather than stalling the run.
# Never wait longer than GITHUB_MAX_WAIT_SECS for a reset window —
# a reset further out means stopping, not sleeping for an hour.
GITHUB_MAX_RETRIES = 3
GITHUB_BACKOFF_BASE_SECS = 1.0
GITHUB_MAX_WAIT_SECS = 300.0

USER_AGENT = "graphone-pipeline/0.1 (AI Engineer demo task)"

_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+/\d{7})(v\d+)?$")

# GitHub repo links authors put in the abstract ("Code available at
# https://github.com/owner/repo"). Only the owner/repo part is kept —
# deeper paths (/issues/1, /tree/main) are dropped, as are trailing
# punctuation and ".git" suffixes.
_GITHUB_LINK_RE = re.compile(
    r"https://(?:www\.)?github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)")
# First path segments that are GitHub site pages, not repos — a link
# to github.com/topics/machine-learning is not a repo.
_NON_REPO_SEGMENTS = frozenset({
    "topics", "explore", "features", "marketplace", "orgs", "sponsors",
    "search", "trending", "collections", "about", "blog", "pricing",
    "login", "signup", "join", "new", "notifications", "settings",
    "site", "enterprise", "readme",
})
_TRAILING_PUNCT = ".,;:!?)]}'\""


def github_url_from_text(text: str) -> str | None:
    """First valid github.com owner/repo link in `text` (an arXiv
    abstract), normalised to https://github.com/<owner>/<repo>.
    Trailing punctuation and a ".git" suffix are stripped; site
    pages (topics/, explore/, ...) never count. None when no repo
    link is present — never guessed."""
    for match in _GITHUB_LINK_RE.finditer(text or ""):
        owner = match.group(1).strip(_TRAILING_PUNCT)
        repo = match.group(2).rstrip(_TRAILING_PUNCT)
        if repo.lower().endswith(".git"):
            repo = repo[:-4].rstrip(_TRAILING_PUNCT)
        if not owner or not repo:
            continue
        if owner.lower() in _NON_REPO_SEGMENTS:
            continue
        if (owner.startswith((".", "-")) or owner.endswith((".", "-"))
                or repo.startswith((".", "-")) or repo.endswith((".", "-"))):
            continue
        return f"https://github.com/{owner}/{repo}"
    return None


class GitHubRateLimited(Exception):
    """GitHub API rate limit persisted after all backoff retries.

    Raised by github_stars() so the enricher can stop further lookups
    instead of burning quota; never raised for other failures."""


def arxiv_query_url(category: str, start: int, max_results: int) -> str:
    """Pure helper: arXiv API URL for one page of one category,
    newest submissions first."""
    return (
        f"{ARXIV_API}?search_query=cat:{category}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&start={start}&max_results={max_results}"
    )


def _rate_limit_wait(resp, attempt: int) -> float | None:
    """Seconds to wait before retrying a rate-limited GitHub response,
    or None if the response is NOT a rate-limit response.

    Rate limit = 429, or 403 with X-RateLimit-Remaining: 0 or a
    "rate limit" message. Prefers Retry-After, then X-RateLimit-Reset,
    else exponential backoff (base * 2**attempt). Raises
    GitHubRateLimited when the reset window is further out than
    GITHUB_MAX_WAIT_SECS — waiting that long would stall the run."""
    headers = getattr(resp, "headers", None) or {}
    if resp.status_code == 429:
        pass
    elif resp.status_code == 403:
        try:
            message = (resp.json() or {}).get("message", "") or ""
        except Exception:
            message = ""
        if (headers.get("X-RateLimit-Remaining") != "0"
                and "rate limit" not in message.lower()):
            return None  # 403 for another reason (e.g. blocked repo)
    else:
        return None
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), GITHUB_MAX_WAIT_SECS))
        except ValueError:
            pass
    reset = headers.get("X-RateLimit-Reset")
    if reset is not None:
        try:
            wait = int(reset) - int(time.time())
        except ValueError:
            wait = None
        if wait is not None:
            if wait > GITHUB_MAX_WAIT_SECS:
                raise GitHubRateLimited(
                    f"rate limit resets in {wait}s (> {GITHUB_MAX_WAIT_SECS:.0f}s max wait)")
            return max(0.0, wait)
    return GITHUB_BACKOFF_BASE_SECS * (2 ** attempt)


def github_stars(owner_repo: str) -> int | None:
    """Live GitHub stars via the GitHub API (dynamic metric, Phase I).

    Uses GITHUB_TOKEN (settings.github_token) when set — no lookup cap
    then; without a token the enricher caps lookups (see
    MAX_UNAUTH_STAR_LOOKUPS). Rate-limited responses are retried with
    backoff; if the limit persists after GITHUB_MAX_RETRIES attempts,
    raises GitHubRateLimited so the caller stops further lookups.
    Any other failure (404, network error, bad payload) returns None
    and never raises."""
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    for attempt in range(GITHUB_MAX_RETRIES + 1):
        try:
            r = http_util.get(f"https://api.github.com/repos/{owner_repo}",
                              headers=headers, timeout=30)
        except Exception:
            return None
        if r.status_code == 200:
            try:
                stars = r.json().get("stargazers_count")
            except Exception:
                return None
            return stars if isinstance(stars, int) else None
        wait = _rate_limit_wait(r, attempt)
        if wait is None:
            return None
        if attempt == GITHUB_MAX_RETRIES:
            raise GitHubRateLimited(
                f"rate limit persisted after {GITHUB_MAX_RETRIES} retries")
        time.sleep(wait)
    raise GitHubRateLimited("rate limit persisted")  # unreachable


def parse_arxiv_atom(xml: str) -> list[ResearchPaper]:
    """Parse an arXiv Atom feed (feedparser). Only papers with a real
    title, link and published date are kept. Each entry's abstract
    is scanned for a github.com repo link (see github_url_from_text);
    the Hugging Face enrichment in fetch_papers() is unchanged and
    only fills github_url when this scan found none."""
    if not xml or not xml.strip():
        return []
    papers: list[ResearchPaper] = []
    for entry in feedparser.parse(xml).entries:
        title = " ".join((entry.get("title") or "").split())
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        parsed = entry.get("published_parsed")
        if parsed:
            dt = datetime(*parsed[:6], tzinfo=timezone.utc)
        else:
            dt = parse_date(entry.get("published") or "")
        if dt is None:
            continue
        authors = [a.get("name", "") for a in entry.get("authors", [])
                   if a.get("name")]
        try:
            papers.append(ResearchPaper(
                title=title,
                authors=authors,
                paper_url=link,
                github_url=github_url_from_text(entry.get("summary") or ""),
                published_date=dt,
            ))
        except ValidationError:
            continue  # invalid URL — never keep a paper without a real link
    return papers


def _arxiv_id(paper_url: str) -> str | None:
    """Normalised arXiv id from an abs URL (version suffix stripped)."""
    path = paper_url.rstrip("/").rsplit("/abs/", 1)[-1]
    m = _ARXIV_ID_RE.match(path)
    return m.group(1) if m else None


async def _arxiv_page(client: httpx.AsyncClient, category: str,
                      start: int, max_results: int) -> list[ResearchPaper]:
    resp = await client.get(arxiv_query_url(category, start, max_results))
    resp.raise_for_status()
    return parse_arxiv_atom(resp.text)


async def _fetch_arxiv_papers(client: httpx.AsyncClient,
                              target: int) -> list[ResearchPaper]:
    """Page the arXiv categories to `target` papers, deduped by URL."""
    per_category = math.ceil(target / len(ARXIV_CATEGORIES))
    papers: list[ResearchPaper] = []
    seen: set[str] = set()
    first_request = True
    for category in ARXIV_CATEGORIES:
        if len(papers) >= target:
            break
        start = 0
        fetched = 0
        while fetched < per_category and len(papers) < target:
            if not first_request:
                await asyncio.sleep(ARXIV_DELAY_SECS)
            first_request = False
            try:
                page = await _arxiv_page(client, category, start, ARXIV_PAGE_SIZE)
            except Exception as exc:
                print(f"arXiv {category} start={start} failed: {exc}")
                break
            if not page:
                break
            fetched += len(page)
            start += ARXIV_PAGE_SIZE
            for paper in page:
                url = str(paper.paper_url)
                if url not in seen:
                    seen.add(url)
                    papers.append(paper)
            if len(page) < ARXIV_PAGE_SIZE:
                break
    return papers


def _hf_item_to_paper(item: dict) -> ResearchPaper | None:
    """One Hugging Face search hit -> ResearchPaper (or None if the
    hit lacks a usable arXiv id / title / date)."""
    paper = item.get("paper") or {}
    arxiv_id = paper.get("id") or ""
    title = " ".join((paper.get("title") or "").split())
    dt = parse_date(paper.get("publishedAt") or "")
    if not _ARXIV_ID_RE.match(arxiv_id) or not title or dt is None:
        return None
    repo = paper.get("githubRepo") or ""
    github_url = repo if repo.startswith("https://github.com/") else None
    stars = paper.get("githubStars")
    try:
        return ResearchPaper(
            title=title,
            authors=[a.get("name", "") for a in paper.get("authors", [])
                     if a.get("name")],
            paper_url=f"https://arxiv.org/abs/{arxiv_id}",
            github_url=github_url,
            github_stars=stars if isinstance(stars, int) else None,
            github_stars_fetched=isinstance(stars, int),
            published_date=dt,
        )
    except ValidationError:
        return None


async def fetch_hf_papers(
    queries: tuple[str, ...] | None = None,
    limit_per_query: int = HF_LIMIT_PER_QUERY,
    client: httpx.AsyncClient | None = None,
) -> list[ResearchPaper]:
    """Hugging Face Papers search: AI papers with GitHub repo links
    and star counts where present. Deduped by paper_url."""
    queries = HF_QUERIES if queries is None else queries
    papers: list[ResearchPaper] = []
    seen: set[str] = set()

    async def _run(c: httpx.AsyncClient) -> None:
        for i, query in enumerate(queries):
            if i:
                await asyncio.sleep(HF_DELAY_SECS)
            try:
                resp = await c.get(HF_PAPERS_API,
                                   params={"q": query, "limit": limit_per_query})
                resp.raise_for_status()
                items = resp.json()
            except Exception as exc:
                print(f"Hugging Face papers query {query!r} failed: {exc}")
                continue
            for item in items if isinstance(items, list) else []:
                paper = _hf_item_to_paper(item)
                if paper is None:
                    continue
                url = str(paper.paper_url)
                if url not in seen:
                    seen.add(url)
                    papers.append(paper)

    if client is not None:
        await _run(client)
    else:
        async with http_util.async_client(timeout=60,
                                          headers={"User-Agent": USER_AGENT}) as c:
            await _run(c)
    return papers


async def _enrich_github_stars(papers: list[ResearchPaper]) -> dict[str, object]:
    """Live star lookups for papers with a repo link but no stars yet.

    With a GITHUB_TOKEN every such paper gets a lookup; without one
    only the first MAX_UNAUTH_STAR_LOOKUPS are looked up
    (unauthenticated GitHub limit: 60/hr) and the rest keep
    github_stars=None. If GitHub rate-limits us, lookups stop
    immediately — the remaining papers keep github_stars=None and
    github_stars_fetched=False instead of burning quota. A lookup
    that fails for any other reason leaves that paper unfetched and
    the loop continues; this function never raises.

    Returns a summary dict (attempted / fetched / not_fetched /
    rate_limited) and prints it, so the run log records exactly
    which papers got stars and which did not."""
    pending = [p for p in papers
               if p.github_url and p.github_stars is None]
    budget = len(pending) if settings.github_token else MAX_UNAUTH_STAR_LOOKUPS
    attempted = fetched = 0
    rate_limited = False
    for paper in pending[:budget]:
        owner_repo = str(paper.github_url).removeprefix(
            "https://github.com/").strip("/")
        if owner_repo.count("/") != 1:  # not a plain owner/repo link
            continue
        attempted += 1
        try:
            stars = await asyncio.to_thread(github_stars, owner_repo)
        except GitHubRateLimited as exc:
            print(f"GitHub rate limit hit, stopping star lookups: {exc}")
            rate_limited = True
            break
        except Exception as exc:  # one bad lookup never kills the run
            print(f"GitHub stars lookup for {owner_repo} failed: {exc}")
            continue
        if stars is not None:
            paper.github_stars = stars
            paper.github_stars_fetched = True
            fetched += 1
        await asyncio.sleep(GITHUB_DELAY_SECS)
    summary = {"attempted": attempted, "fetched": fetched,
               "not_fetched": len(pending) - fetched,
               "rate_limited": rate_limited}
    print(f"GitHub stars: attempted={attempted} fetched={fetched} "
          f"not_fetched={summary['not_fetched']} "
          f"rate_limited={rate_limited}")
    return summary


async def fetch_papers(target: int = 1000) -> list[ResearchPaper]:
    """Fetch `target` AI research papers: arXiv first (4 categories,
    paged), then Hugging Face papers to attach GitHub repos/stars and
    top up any shortfall, then live GitHub star lookups for the rest.

    Deduped by paper_url throughout; every paper has a real arXiv URL."""
    papers: list[ResearchPaper] = []
    seen: set[str] = set()
    async with http_util.async_client(timeout=60,
                                       headers={"User-Agent": USER_AGENT}) as client:
        arxiv_papers = await _fetch_arxiv_papers(client, target)
        for paper in arxiv_papers:
            url = str(paper.paper_url)
            if url not in seen:
                seen.add(url)
                papers.append(paper)
        # HF: enrich arXiv papers with GitHub data, top up if short.
        by_arxiv_id = {_arxiv_id(str(p.paper_url)): p for p in papers}
        for hf_paper in await fetch_hf_papers(client=client):
            arxiv_id = _arxiv_id(str(hf_paper.paper_url))
            existing = by_arxiv_id.get(arxiv_id) if arxiv_id else None
            if existing is not None:
                if existing.github_url is None:
                    existing.github_url = hf_paper.github_url
                if existing.github_stars is None and hf_paper.github_stars is not None:
                    existing.github_stars = hf_paper.github_stars
                    existing.github_stars_fetched = True
            elif len(papers) < target:
                url = str(hf_paper.paper_url)
                if url not in seen:
                    seen.add(url)
                    papers.append(hf_paper)
    await _enrich_github_stars(papers)
    return papers[:target]

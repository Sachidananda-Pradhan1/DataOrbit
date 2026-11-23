"""Issue A — GitHub stars coverage (papers.py).

With a GITHUB_TOKEN every paper with a github_url must get a
star-count attempt (no 50-repo cap); without one the cap stays and
is documented. Rate limits (403/429) back off and never crash the
run; each paper records fetched/not-fetched via
ResearchPaper.github_stars_fetched.

All GitHub calls are mocked — no live lookups, no secrets.
"""
import asyncio
import time
from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.crawlers import papers
from src.models import ResearchPaper

DT = datetime(2024, 1, 2, tzinfo=timezone.utc)


async def _no_sleep(*args, **kwargs):
    return None


def _paper(i, stars=None, repo=True):
    return ResearchPaper(
        title=f"Paper {i}",
        authors=["A"],
        paper_url=f"https://arxiv.org/abs/2401.{i:05d}",
        github_url=f"https://github.com/owner{i}/repo{i}" if repo else None,
        github_stars=stars,
        github_stars_fetched=stars is not None,
        published_date=DT,
    )


class _FakeResp:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


def _with_token(monkeypatch, token="test-token"):
    monkeypatch.setattr(papers, "settings", Settings(github_token=token))


def _without_token(monkeypatch):
    monkeypatch.setattr(papers, "settings", Settings(github_token=""))


def _record_sleeps(monkeypatch):
    sleeps = []
    monkeypatch.setattr(papers.time, "sleep", sleeps.append)
    return sleeps


# --- token path: every paper with a repo gets an attempt ---

def test_token_path_attempts_every_paper_with_repo(monkeypatch):
    _with_token(monkeypatch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("headers", {})))
        n = int(url.rsplit("owner", 1)[1].split("/")[0])
        return _FakeResp(200, {"stargazers_count": 100 + n})

    monkeypatch.setattr(papers.http_util, "get", fake_get)
    # 60 papers: over the old unauthenticated cap of 50.
    batch = [_paper(i) for i in range(60)]
    batch.append(_paper(60, repo=False))  # no repo: no attempt, no stars
    summary = asyncio.run(papers._enrich_github_stars(batch))

    assert len(calls) == 60
    assert summary == {"attempted": 60, "fetched": 60,
                       "not_fetched": 0, "rate_limited": False}
    for i, paper in enumerate(batch[:60]):
        assert paper.github_stars == 100 + i
        assert paper.github_stars_fetched is True
    assert batch[60].github_stars is None
    assert batch[60].github_stars_fetched is False
    # Token sent as a Bearer auth header on every call.
    assert all(h.get("Authorization") == "Bearer test-token"
               for _, h in calls)


def test_papers_with_stars_already_need_no_lookup(monkeypatch):
    _with_token(monkeypatch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = []
    monkeypatch.setattr(papers.http_util, "get",
                        lambda *a, **k: calls.append(1) or _FakeResp(200, {"stargazers_count": 1}))
    batch = [_paper(0, stars=7), _paper(1, stars=9)]
    summary = asyncio.run(papers._enrich_github_stars(batch))
    assert calls == []
    assert summary["attempted"] == 0
    assert summary["fetched"] == 0
    assert batch[0].github_stars == 7
    assert batch[0].github_stars_fetched is True


def test_malformed_github_url_skipped_without_call(monkeypatch):
    _with_token(monkeypatch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = []
    monkeypatch.setattr(papers.http_util, "get",
                        lambda *a, **k: calls.append(1) or _FakeResp(200, {"stargazers_count": 1}))
    bad = _paper(0)
    bad.github_url = "https://github.com/just-an-owner"  # no owner/repo
    summary = asyncio.run(papers._enrich_github_stars([bad, _paper(1)]))
    assert len(calls) == 1  # only the well-formed repo was looked up
    assert bad.github_stars is None
    assert bad.github_stars_fetched is False
    assert summary["attempted"] == 1
    assert summary["fetched"] == 1
    assert summary["not_fetched"] == 1


# --- no-token path: safe cap stays ---

def test_no_token_cap_kept(monkeypatch):
    _without_token(monkeypatch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = []
    monkeypatch.setattr(
        papers.http_util, "get",
        lambda *a, **k: calls.append(1) or _FakeResp(200, {"stargazers_count": 5}))
    batch = [_paper(i) for i in range(60)]
    summary = asyncio.run(papers._enrich_github_stars(batch))

    assert len(calls) == papers.MAX_UNAUTH_STAR_LOOKUPS == 50
    assert summary == {"attempted": 50, "fetched": 50,
                       "not_fetched": 10, "rate_limited": False}
    for paper in batch[:50]:
        assert paper.github_stars == 5
        assert paper.github_stars_fetched is True
    for paper in batch[50:]:  # over the cap: untouched, recorded as not fetched
        assert paper.github_stars is None
        assert paper.github_stars_fetched is False


def test_no_token_sends_no_auth_header(monkeypatch):
    _without_token(monkeypatch)
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(kwargs.get("headers", {}))
        return _FakeResp(200, {"stargazers_count": 3})

    monkeypatch.setattr(papers.http_util, "get", fake_get)
    assert papers.github_stars("owner/repo") == 3
    assert "Authorization" not in seen


# --- rate limits: backoff, Retry-After / X-RateLimit-Reset, give up ---

def test_403_rate_limit_retries_after_retry_after(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    responses = [
        _FakeResp(403, {"message": "API rate limit exceeded"},
                  {"X-RateLimit-Remaining": "0", "Retry-After": "2"}),
        _FakeResp(200, {"stargazers_count": 42}),
    ]
    monkeypatch.setattr(papers.http_util, "get",
                        lambda *a, **k: responses.pop(0))
    assert papers.github_stars("owner/repo") == 42
    assert sleeps == [2.0]  # Retry-After honoured, then success


def test_429_retries_with_backoff(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    responses = [
        _FakeResp(429, {}, {"Retry-After": "1"}),
        _FakeResp(200, {"stargazers_count": 11}),
    ]
    monkeypatch.setattr(papers.http_util, "get",
                        lambda *a, **k: responses.pop(0))
    assert papers.github_stars("owner/repo") == 11
    assert sleeps == [1.0]


def test_rate_limit_reset_header_used(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    reset = str(int(time.time()) + 5)
    responses = [
        _FakeResp(403, {"message": "API rate limit exceeded"},
                  {"X-RateLimit-Reset": reset}),
        _FakeResp(200, {"stargazers_count": 9}),
    ]
    monkeypatch.setattr(papers.http_util, "get",
                        lambda *a, **k: responses.pop(0))
    assert papers.github_stars("owner/repo") == 9
    assert len(sleeps) == 1
    assert 4 <= sleeps[0] <= 5  # until the reset window (1s clock slack)


def test_rate_limit_without_headers_uses_exponential_backoff(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    responses = [_FakeResp(403, {"message": "API rate limit exceeded"})] * 4
    monkeypatch.setattr(papers.http_util, "get",
                        lambda *a, **k: responses.pop(0))
    with pytest.raises(papers.GitHubRateLimited):
        papers.github_stars("owner/repo")
    # 1 initial + 3 retries = 4 calls; sleeps 1s, 2s, 4s between them.
    assert responses == []
    assert sleeps == [1.0, 2.0, 4.0]


def test_rate_limit_reset_far_away_gives_up_immediately(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    reset = str(int(time.time()) + 3600)  # an hour away: not worth waiting
    calls = []
    monkeypatch.setattr(
        papers.http_util, "get",
        lambda *a, **k: calls.append(1) or _FakeResp(
            403, {"message": "API rate limit exceeded"},
            {"X-RateLimit-Reset": reset}))
    with pytest.raises(papers.GitHubRateLimited):
        papers.github_stars("owner/repo")
    assert len(calls) == 1  # no retries, no sleeping
    assert sleeps == []


def test_403_without_rate_limit_signal_returns_none_no_retry(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    calls = []
    monkeypatch.setattr(
        papers.http_util, "get",
        lambda *a, **k: calls.append(1) or _FakeResp(403, {"message": "Forbidden"}))
    assert papers.github_stars("owner/repo") is None
    assert len(calls) == 1
    assert sleeps == []


# --- enricher: rate limit stops the run gracefully, never crashes ---

def test_enricher_stops_on_rate_limit_and_records_status(monkeypatch):
    _with_token(monkeypatch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    _record_sleeps(monkeypatch)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "owner0" in url:
            return _FakeResp(200, {"stargazers_count": 5})
        return _FakeResp(403, {"message": "API rate limit exceeded"},
                         {"X-RateLimit-Remaining": "0"})

    monkeypatch.setattr(papers.http_util, "get", fake_get)
    batch = [_paper(i) for i in range(5)]
    summary = asyncio.run(papers._enrich_github_stars(batch))  # must not raise

    assert summary == {"attempted": 2, "fetched": 1,
                       "not_fetched": 4, "rate_limited": True}
    assert batch[0].github_stars == 5
    assert batch[0].github_stars_fetched is True
    for paper in batch[1:]:  # rate-limited paper + all after it: untouched
        assert paper.github_stars is None
        assert paper.github_stars_fetched is False
    # 1 success + (1 initial + 3 retries) for the rate-limited repo, then stop.
    assert len(calls) == 5


def test_enricher_continues_after_non_rate_limit_failure(monkeypatch):
    _with_token(monkeypatch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    def fake_get(url, **kwargs):
        if "owner0" in url:
            return _FakeResp(404, {})  # repo gone: skip, keep going
        return _FakeResp(200, {"stargazers_count": 8})

    monkeypatch.setattr(papers.http_util, "get", fake_get)
    batch = [_paper(i) for i in range(3)]
    summary = asyncio.run(papers._enrich_github_stars(batch))
    assert summary == {"attempted": 3, "fetched": 2,
                       "not_fetched": 1, "rate_limited": False}
    assert batch[0].github_stars is None
    assert batch[0].github_stars_fetched is False
    assert batch[1].github_stars == 8
    assert batch[2].github_stars_fetched is True


# --- fetched flag through the Hugging Face path ---

def test_hf_item_marks_stars_fetched():
    with_stars = papers._hf_item_to_paper({"paper": {
        "id": "2401.00001", "title": "T",
        "publishedAt": "2024-01-02T00:00:00.000Z",
        "githubRepo": "https://github.com/owner/repo",
        "githubStars": 123}})
    assert with_stars.github_stars == 123
    assert with_stars.github_stars_fetched is True

    without_stars = papers._hf_item_to_paper({"paper": {
        "id": "2401.00002", "title": "T",
        "publishedAt": "2024-01-02T00:00:00.000Z",
        "githubRepo": "https://github.com/owner/repo2",
        "githubStars": None}})
    assert without_stars.github_stars is None
    assert without_stars.github_stars_fetched is False


def test_fetch_papers_marks_hf_enriched_flag(monkeypatch):
    """End to end (mocked HTTP): an arXiv paper enriched with HF stars
    gets the fetched flag; an HF-only paper with a repo but no stars
    gets a live lookup attempt."""
    _with_token(monkeypatch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(papers, "github_stars", lambda owner_repo: 77)

    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>ArXiv Paper</title>
      <link href="https://arxiv.org/abs/2401.00001"/>
      <published>2024-01-02T00:00:00Z</published>
      <author><name>A</name></author></entry>
    </feed>"""
    hf_items = [
        {"paper": {"id": "2401.00001", "title": "ArXiv Paper",
                   "publishedAt": "2024-01-02T00:00:00.000Z",
                   "githubRepo": "https://github.com/owner/repo",
                   "githubStars": 55}},
        {"paper": {"id": "2401.00002", "title": "HF Only Paper",
                   "publishedAt": "2024-01-02T00:00:00.000Z",
                   "githubRepo": "https://github.com/owner/repo2",
                   "githubStars": None}},
    ]

    class _Resp:
        def __init__(self, text="", json_data=None):
            self.text = text
            self._json = json_data

        def json(self):
            return self._json

        def raise_for_status(self):
            pass

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            if "export.arxiv.org" in url:
                return _Resp(text=arxiv_xml)
            return _Resp(json_data=hf_items)

    monkeypatch.setattr(papers.http_util, "async_client", _Client)
    result = asyncio.run(papers.fetch_papers(target=2))
    assert len(result) == 2
    arxiv_paper = next(p for p in result if "2401.00001" in str(p.paper_url))
    assert arxiv_paper.github_stars == 55  # from HF, no live call needed
    assert arxiv_paper.github_stars_fetched is True
    hf_paper = next(p for p in result if "2401.00002" in str(p.paper_url))
    assert hf_paper.github_stars == 77  # live lookup via mocked github_stars
    assert hf_paper.github_stars_fetched is True

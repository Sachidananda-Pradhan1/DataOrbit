"""Tests for the runners + fetcher + dedup workstream: JSON merge
on re-run, fetch retry / failure-vs-non-AI, to_thread LLM
extraction behind a semaphore, website-first resolution,
heartbeat wiring, run_common sharing.

ALL MOCKED — no live network, no real DB, no real LLM. Scripts are
loaded from their file paths via importlib (scripts/ is not a
package), as in tests/test_scripts.py. run_common is loaded under
its real name first so both runners import the same module object
the tests hold.
"""
import asyncio
import importlib.util
import json
import logging
import sys
import threading
import types
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import pytest
import yarl
from tenacity import wait_fixed

from src.crawlers import fetcher
from src.models import Job, NewsItem, Source, Startup
from src.storage import dedup

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_common = _load_script("run_common", "run_common.py")
run_bulk = _load_script("run_bulk_merge_retry", "run_bulk.py")
run_fresh = _load_script("run_fresh_merge_retry", "run_fresh.py")


@pytest.fixture(autouse=True)
def _reset_dedup():
    dedup.reset()
    yield
    dedup.reset()


def _now():
    return datetime.now(UTC)


def _stats():
    return {"duplicates": 0, "parse_errors": 0, "llm_extracted": 0,
            "llm_skipped_no_key": 0, "llm_errors": 0}


# --- record_url / merge_json_records ---

def test_record_url_field_order():
    assert run_common.record_url({"url": "https://n.example/1"}) == "https://n.example/1"
    assert run_common.record_url({"source": {"url": "https://s.example/1"}}) == "https://s.example/1"
    assert run_common.record_url({"paper_url": "https://arxiv.org/abs/1"}) == "https://arxiv.org/abs/1"
    assert run_common.record_url({"source_url": "https://j.example/1"}) == "https://j.example/1"
    # url wins over source.url when both are present.
    assert run_common.record_url(
        {"url": "https://u", "source": {"url": "https://s"}}) == "https://u"
    # Jobs without a source URL fall back to company+date.
    assert run_common.record_url(
        {"company": "Acme", "date": "2026-11-22"}) == "job:Acme:2026-11-22"
    assert run_common.record_url({"company": "Acme"}) is None
    assert run_common.record_url({}) is None


def test_merge_json_records_new_wins_first_seen_order():
    existing = [
        {"source": {"url": "https://example.com/a"}, "entityName": "Acme OLD"},
        {"source": {"url": "https://example.com/b"}, "entityName": "Beta"},
    ]
    new = [
        {"source": {"url": "https://example.com/a"}, "entityName": "Acme NEW"},
        {"source": {"url": "https://example.com/c"}, "entityName": "Gamma"},
    ]
    merged = run_common.merge_json_records(existing, new)
    assert [r["entityName"] for r in merged] == ["Acme NEW", "Beta", "Gamma"]
    assert [run_common.record_url(r) for r in merged] == [
        "https://example.com/a", "https://example.com/b",
        "https://example.com/c"]


def test_merge_json_records_second_run_keeps_first_run():
    first = [{"url": "https://example.com/n1", "title": "One"},
             {"url": "https://example.com/n2", "title": "Two"}]
    # A second run claims nothing new (dedup): the merge must keep
    # run 1's records, not wipe the tab.
    assert run_common.merge_json_records(first, []) == first
    # A run that re-sees one URL plus one fresh one adds only the
    # fresh one — no duplicate URLs.
    second = [{"url": "https://example.com/n2", "title": "Two"},
              {"url": "https://example.com/n3", "title": "Three"}]
    merged = run_common.merge_json_records(first, second)
    assert [r["url"] for r in merged] == [
        "https://example.com/n1", "https://example.com/n2",
        "https://example.com/n3"]


def test_merge_json_records_keeps_url_less_records():
    existing = [{"note": "no url here"}]
    new = [{"note": "also no url"}]
    assert run_common.merge_json_records(existing, new) == existing + new


def test_runners_share_run_common_helpers():
    assert run_bulk.merge_json_records is run_fresh.merge_json_records
    assert run_bulk.extract_domain is run_fresh.extract_domain
    assert run_bulk.resolve_entity is run_fresh.resolve_entity
    assert run_bulk.merge_mapping_log is run_fresh.merge_mapping_log
    assert run_bulk.write_json is run_fresh.write_json
    assert run_bulk.load_json_list is run_fresh.load_json_list
    assert run_bulk.persist_to_db is run_fresh.persist_to_db
    assert run_bulk.import_optional is run_fresh.import_optional
    assert run_bulk.merge_json_records is run_common.merge_json_records


def test_llm_concurrency_env(monkeypatch):
    monkeypatch.delenv("MAX_LLM_CONCURRENCY", raising=False)
    assert run_common.llm_concurrency() == 5
    monkeypatch.setenv("MAX_LLM_CONCURRENCY", "3")
    assert run_common.llm_concurrency() == 3
    monkeypatch.setenv("MAX_LLM_CONCURRENCY", "0")
    assert run_common.llm_concurrency() == 1
    monkeypatch.setenv("MAX_LLM_CONCURRENCY", "junk")
    assert run_common.llm_concurrency() == 5


# --- fetch retry ---

def _http_error(status):
    url = yarl.URL("https://example.com/page")
    info = aiohttp.RequestInfo(url=url, method="GET", headers={},
                               real_url=url)
    return aiohttp.ClientResponseError(info, (), status=status)


class _FakeResponse:
    """Minimal aiohttp response stand-in for fetcher.fetch tests."""

    def __init__(self, outcome):
        # str (body) or Exception (raised by raise_for_status).
        self._outcome = outcome

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome

    async def text(self):
        return self._outcome


class _ScriptedSession:
    """session.get returns the next scripted outcome per call."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        return _FakeResponse(self._outcomes.pop(0))


@pytest.fixture
def _no_retry_sleep(monkeypatch):
    # Tenacity's jittered backoff would make retry tests sleep
    # seconds; the retry *count* is what these tests assert.
    monkeypatch.setattr(fetcher.fetch.retry, "wait", wait_fixed(0))


def test_fetch_retries_429_then_succeeds(_no_retry_sleep):
    session = _ScriptedSession(
        [_http_error(429), _http_error(429), "<html>ok</html>"])
    assert asyncio.run(fetcher.fetch("https://example.com/page",
                                     session)) == "<html>ok</html>"
    assert session.calls == 3


def test_fetch_retries_5xx_then_raises(_no_retry_sleep):
    session = _ScriptedSession([_http_error(503)] * 5)
    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(fetcher.fetch("https://example.com/page", session))
    assert session.calls == fetcher.FETCH_MAX_ATTEMPTS


def test_fetch_retries_connection_errors(_no_retry_sleep):
    session = _ScriptedSession(
        [aiohttp.ClientError("connection reset"), "<html>ok</html>"])
    assert asyncio.run(fetcher.fetch("https://example.com/page",
                                     session)) == "<html>ok</html>"
    assert session.calls == 2


def test_fetch_does_not_retry_404(_no_retry_sleep):
    session = _ScriptedSession([_http_error(404)] * 3)
    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(fetcher.fetch("https://example.com/page", session))
    assert session.calls == 1  # other 4xx fail fast


# --- fetch_many failures side channel + streaming ---

def test_fetch_many_reports_failures_without_changing_return_type(monkeypatch):
    async def fake_fetch(url, session):
        if "bad" in url:
            raise aiohttp.ClientError("boom")
        return f"<html>{url}</html>"

    monkeypatch.setattr(fetcher, "fetch", fake_fetch)
    failures = set()
    out = asyncio.run(fetcher.fetch_many(
        ["https://good.example/a", "https://bad.example/x"],
        failures=failures))
    assert out == {"https://good.example/a":
                   "<html>https://good.example/a</html>",
                   "https://bad.example/x": ""}
    assert failures == {"https://bad.example/x"}


def test_fetch_many_stream_yields_in_batches_and_stops_early(monkeypatch):
    seen = []

    async def fake_fetch(url, session):
        seen.append(url)
        return f"<html>{url}</html>"

    monkeypatch.setattr(fetcher, "fetch", fake_fetch)
    urls = [f"https://example.com/{i}" for i in range(5)]

    async def consume_all():
        return [pair async for pair in
                fetcher.fetch_many_stream(urls, batch_size=2)]

    pairs = asyncio.run(consume_all())
    assert [u for u, _ in pairs] == urls
    assert len(seen) == 5

    seen.clear()

    async def consume_two_then_break():
        got = []
        async for pair in fetcher.fetch_many_stream(urls, batch_size=2):
            got.append(pair)
            if len(got) == 2:
                break
        return got

    got = asyncio.run(consume_two_then_break())
    assert len(got) == 2
    assert len(seen) == 2  # only the first batch was fetched


# --- collect_entities ---

class _Crawler:
    """Fake startup/product crawler: discover returns a fixed URL
    list; parse returns prebuilt records; a page is AI-relevant
    when its text carries the ZZAI marker."""

    def __init__(self, urls, parse_by_url):
        self._urls = list(urls)
        self._parse_by_url = parse_by_url
        self.discover_limits = []

    async def discover(self, limit=1000):
        self.discover_limits.append(limit)
        return list(self._urls)[:limit]

    def is_ai_relevant_page(self, raw):
        return "ZZAI" in raw

    async def parse(self, url, raw):
        return list(self._parse_by_url.get(url, []))


@pytest.fixture
def _stub_stream(monkeypatch):
    """Stub fetch_many_stream + save_raw for collect_entities tests.
    Returns (pages, stream_calls, fail_urls): pages maps url -> text
    yielded by the stream; fail_urls are yielded as "" and reported
    in the failures set, like the real fetcher."""
    pages = {}
    fail_urls = set()
    stream_calls = []

    async def fake_stream(urls, batch_size=100, failures=None):
        stream_calls.append(list(urls))
        for u in urls:
            if u in fail_urls:
                if failures is not None:
                    failures.add(u)
                yield u, ""
            elif u in pages:
                yield u, pages[u]

    monkeypatch.setattr("src.crawlers.fetcher.fetch_many_stream", fake_stream)
    monkeypatch.setattr(run_bulk, "save_raw", lambda *a, **k: None)
    return pages, stream_calls, fail_urls


@pytest.fixture
def _fake_matcher(monkeypatch):
    """Swap src.resolver.matcher for one that records the domain it
    was given. Returns the (name, domain) call list."""
    calls = []

    def rwc(name, domain=None):
        calls.append((name, domain))
        return (name, 1.0, "exact")

    fake = types.SimpleNamespace(resolve_with_confidence=rwc,
                                 resolve=lambda name: name)
    monkeypatch.setitem(sys.modules, "src.resolver.matcher", fake)
    return calls


def _startup(url, name="Acme"):
    return Startup(source=Source(name="Y Combinator", url=url),
                   entityName=name, collectedAt=_now())


def _record_with_website(url, name, website):
    """A record carrying a website attribute (the Startup model gains
    one in another stream; a namespace stands in until then)."""
    record = types.SimpleNamespace(
        source=Source(name="Y Combinator", url=url),
        entityName=name, website=website)
    record.model_dump = lambda mode="json": {
        "entityName": record.entityName,
        "source": {"name": "Y Combinator", "url": url},
        "website": website,
    }
    return record


def test_fetch_failure_not_non_ai_and_requeued(_stub_stream):
    pages, stream_calls, fail_urls = _stub_stream
    good = "https://www.ycombinator.com/companies/good-ai"
    bad = "https://www.ycombinator.com/companies/bad-ai"
    pages[good] = "ZZAI page"
    fail_urls.add(bad)
    crawler = _Crawler([good, bad], {good: [_startup(good, "Good AI")]})
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, None, limit=2, llm_enabled=False,
        name_field="entityName", stats=stats))
    assert [r["entityName"] for r in records] == ["Good AI"]
    assert stats.get("startups_fetch_failed") == 2  # tried in two rounds
    assert stats.get("startups_filtered_non_ai", 0) == 0
    assert stats.get("startups_fetched") == 1
    assert stats["startups_shortfall"] == 1
    # The failed URL was offered to the fetcher in two rounds...
    assert [u for call in stream_calls for u in call].count(bad) == 2
    # ...then dropped (no third attempt).
    assert crawler.discover_limits == [2, 4, 8]
    # Failed URLs are never dedup-claimed: still pending.
    assert dedup.state(bad) == "pending"


def test_llm_extraction_runs_off_event_loop(_stub_stream):
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/nameless-ai"
    pages[url] = "ZZAI page"
    threads = []

    def spy_extractor(raw, domain, page_url, collected_at):
        threads.append(threading.current_thread() is threading.main_thread())
        return Startup(source=Source(name="Y Combinator", url=page_url),
                       entityName="Nameless AI", collectedAt=collected_at)

    crawler = _Crawler([url], {})  # parse yields [] -> LLM fallback
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, spy_extractor, limit=1, llm_enabled=True,
        name_field="entityName", stats=stats))
    assert [r["entityName"] for r in records] == ["Nameless AI"]
    assert stats["llm_extracted"] == 1
    assert threads == [False]  # ran in a worker thread, not the event loop


def test_llm_semaphore_uses_max_llm_concurrency(_stub_stream, monkeypatch):
    monkeypatch.setenv("MAX_LLM_CONCURRENCY", "3")
    created = []
    real_semaphore = asyncio.Semaphore

    def spy_semaphore(value=1):
        created.append(value)
        return real_semaphore(value)

    monkeypatch.setattr(asyncio, "Semaphore", spy_semaphore)
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/nameless-ai"
    pages[url] = "ZZAI page"

    def spy_extractor(raw, domain, page_url, collected_at):
        return Startup(source=Source(name="Y Combinator", url=page_url),
                       entityName="Nameless AI", collectedAt=collected_at)

    crawler = _Crawler([url], {})
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, spy_extractor, limit=1, llm_enabled=True,
        name_field="entityName", stats=stats))
    assert [r["entityName"] for r in records] == ["Nameless AI"]
    assert created == [3]


def test_resolve_prefers_record_website_domain(_stub_stream, _fake_matcher):
    calls = _fake_matcher
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/acme"
    pages[url] = "ZZAI page"
    record = _record_with_website(url, "Acme Inc",
                                  "https://acme.example.com/about")
    crawler = _Crawler([url], {url: [record]})
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, None, limit=1, llm_enabled=False,
        name_field="entityName", stats=stats))
    assert len(records) == 1
    assert calls == [("Acme Inc", "acme.example.com")]


def test_resolve_falls_back_to_source_domain(_stub_stream, _fake_matcher):
    calls = _fake_matcher
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/acme"
    pages[url] = "ZZAI page"
    crawler = _Crawler([url], {url: [_startup(url, "Acme Inc")]})
    stats = _stats()
    asyncio.run(run_bulk.collect_entities(
        "startups", crawler, None, limit=1, llm_enabled=False,
        name_field="entityName", stats=stats))
    assert calls == [("Acme Inc", "ycombinator.com")]


def test_heartbeat_called_before_llm_extraction(_stub_stream, monkeypatch):
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/nameless-ai"
    pages[url] = "ZZAI page"
    order = []
    renewed = []
    real_heartbeat = run_bulk.heartbeat

    def spy_heartbeat(claim_url, owner, lease_secs=600):
        order.append(("heartbeat", claim_url))
        renewed.append(real_heartbeat(claim_url, owner, lease_secs))

    def spy_extractor(raw, domain, page_url, collected_at):
        order.append(("extract", page_url))
        return Startup(source=Source(name="Y Combinator", url=page_url),
                       entityName="Nameless AI", collectedAt=collected_at)

    monkeypatch.setattr(run_bulk, "heartbeat", spy_heartbeat)
    crawler = _Crawler([url], {})
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, spy_extractor, limit=1, llm_enabled=True,
        name_field="entityName", stats=stats))
    assert [r["entityName"] for r in records] == ["Nameless AI"]
    assert order == [("heartbeat", url), ("extract", url)]
    assert renewed == [True]  # the live lease was actually renewed
    assert dedup.state(url) == "completed"


def test_second_collect_skips_claimed_pages_before_llm(_stub_stream):
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/nameless-ai"
    pages[url] = "ZZAI page"
    calls = []

    def spy_extractor(raw, domain, page_url, collected_at):
        calls.append(page_url)
        return Startup(source=Source(name="Y Combinator", url=page_url),
                       entityName="Nameless AI", collectedAt=collected_at)

    crawler = _Crawler([url], {})
    first, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, spy_extractor, limit=1, llm_enabled=True,
        name_field="entityName", stats=_stats()))
    assert len(first) == 1
    stats2 = _stats()
    second, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, spy_extractor, limit=1, llm_enabled=True,
        name_field="entityName", stats=stats2))
    assert second == []
    assert stats2["duplicates"] == 1
    assert calls == [url]  # LLM ran once, on the first collect only


def test_llm_error_releases_claim_with_backoff(_stub_stream):
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/broken-ai"
    pages[url] = "ZZAI page"

    def failing_extractor(*args):
        raise RuntimeError("all tiers failed: boom")

    crawler = _Crawler([url], {})
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, failing_extractor, limit=1, llm_enabled=True,
        name_field="entityName", stats=stats))
    assert records == []
    assert stats["llm_errors"] == 1
    assert dedup.state(url) == "failed"  # retryable after backoff, not lost


def test_llm_no_key_leaves_lease_for_later_keyed_run(_stub_stream):
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/nameless-ai"
    pages[url] = "ZZAI page"

    def no_key_extractor(*args):
        raise RuntimeError("all LLM tiers failed: not configured")

    crawler = _Crawler([url], {})
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, no_key_extractor, limit=1, llm_enabled=True,
        name_field="entityName", stats=stats))
    assert records == []
    assert stats["llm_skipped_no_key"] == 1
    # Not completed: a later keyed run retries once the lease expires.
    assert dedup.state(url) == "processing"


def test_no_llm_run_leaves_lease_for_later_keyed_run(_stub_stream):
    pages, _, _ = _stub_stream
    url = "https://www.ycombinator.com/companies/nameless-ai"
    pages[url] = "ZZAI page"
    crawler = _Crawler([url], {})  # parse yields nothing, no LLM
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, None, limit=1, llm_enabled=False,
        name_field="entityName", stats=stats))
    assert records == []
    assert dedup.state(url) == "processing"


# --- collect_fresh + merge (re-run keeps first run's records) ---

def _news(url, source="TechCrunch AI"):
    return NewsItem(title="AI news", source=Source(name=source, url=url),
                    published_date=_now(), full_text="body", url=url)


def _job(company, url=None):
    return Job(company=company, date=_now(), is_remote=True,
               role_family="Engineering", source_url=url)


def test_fresh_second_collect_merges_without_wiping():
    news = [_news("https://example.com/n1")]
    jobs = [_job("Acme", "https://remoteok.com/jobs/1")]
    first = run_fresh.collect_fresh(news, jobs)
    assert first["stats"]["news_fresh"] == 1
    assert first["stats"]["jobs_fresh"] == 1
    second = run_fresh.collect_fresh(news, jobs)
    assert second["news"] == [] and second["jobs"] == []
    assert run_common.merge_json_records(
        first["news"], second["news"]) == first["news"]
    assert run_common.merge_json_records(
        first["jobs"], second["jobs"]) == first["jobs"]


def test_fresh_resolve_prefers_job_website_domain(_fake_matcher):
    calls = _fake_matcher
    # Job has no website field yet; a namespace record stands in.
    record = types.SimpleNamespace(
        company="Acme Inc", date=_now(), is_remote=True,
        role_family="Engineering", source_url="https://remoteok.com/jobs/1",
        website="https://acme.example.com/careers",
        model_dump=lambda mode="json": {
            "company": record.company,
            "source_url": "https://remoteok.com/jobs/1",
        })
    result = run_fresh.collect_fresh([], [record])
    assert result["stats"]["jobs_fresh"] == 1
    assert calls == [("Acme Inc", "acme.example.com")]
    # Board stats still use the board, not the company website.
    assert result["stats"]["jobs_by_board"] == {"remoteok.com": 1}


# --- main_async end-to-end (fetchers stubbed out): re-run merge + events ---

def test_fresh_rerun_keeps_existing_json(tmp_path, monkeypatch, caplog):
    first_jobs = [{"company": "Acme", "date": "2026-11-22T00:00:00Z",
                   "source_url": "https://remoteok.com/jobs/1"}]
    first_news = [{"url": "https://example.com/n1", "title": "One"}]
    (tmp_path / "Jobs.json").write_text(json.dumps(first_jobs))
    (tmp_path / "News.json").write_text(json.dumps(first_news))
    monkeypatch.setattr(run_fresh, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(run_fresh, "MAPPING_LOG",
                        tmp_path / "Entity Mapping Log.json")
    monkeypatch.setattr(run_fresh, "import_optional", lambda *a: None)
    with caplog.at_level(logging.INFO, logger="graphone"):
        stats = asyncio.run(run_fresh.main_async(
            run_fresh.parse_args(["--no-db"])))
    assert json.loads((tmp_path / "Jobs.json").read_text()) == first_jobs
    assert json.loads((tmp_path / "News.json").read_text()) == first_news
    assert stats["jobs_fresh"] == 0
    assert '"event": "fresh_run_start"' in caplog.text
    assert '"event": "fresh_run_end"' in caplog.text


def test_bulk_rerun_keeps_existing_json(tmp_path, monkeypatch, caplog):
    first = [{"paper_url": "https://arxiv.org/abs/1", "title": "Paper"}]
    (tmp_path / "Research Papers.json").write_text(json.dumps(first))
    monkeypatch.setattr(run_bulk, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(run_bulk, "import_optional", lambda *a: None)
    with caplog.at_level(logging.INFO, logger="graphone"):
        stats = asyncio.run(run_bulk.main_async(run_bulk.parse_args(
            ["--no-db", "--no-llm", "--limit", "1"])))
    assert json.loads(
        (tmp_path / "Research Papers.json").read_text()) == first
    assert stats["papers_skipped_missing"] is True
    assert '"event": "bulk_run_start"' in caplog.text
    assert '"event": "bulk_run_end"' in caplog.text

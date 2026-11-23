"""Tests for scripts/run_bulk.py and scripts/run_fresh.py — mocked
only: no live network, no real DB, and no crawler imports at module
level (crawler modules are mid-edit by other agents).

The scripts are loaded from their file paths via importlib
(scripts/ is not a package); their pure helpers are tested
directly, and run_fresh.collect_fresh is tested with real
Pydantic models but fake items.
"""
import asyncio
import contextlib
import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.models import Job, NewsItem, Source
from src.storage import dedup

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_bulk = _load_script("run_bulk")
run_fresh = _load_script("run_fresh")


@pytest.fixture(autouse=True)
def _reset_dedup():
    dedup._seen.clear()
    yield
    dedup._seen.clear()


def _now():
    return datetime.now(UTC)


# --- domain extraction ---

@pytest.mark.parametrize("url,expected", [
    ("https://www.example.com/path?q=1", "example.com"),
    ("https://sub.example.co.uk/", "sub.example.co.uk"),
    ("http://WWW.Example.COM:8080/x", "example.com:8080"),
    ("", ""),
    ("not a url", ""),
])
def test_extract_domain(url, expected):
    assert run_bulk.extract_domain(url) == expected
    assert run_fresh.extract_domain(url) == expected


# --- mapping-log merge ---

def test_merge_mapping_log_dedups_on_raw_name_and_source_url():
    existing = [{"raw_name": "OpenAI, Inc.", "canonical_name": "OpenAI",
                 "confidence": 1.0, "match_method": "exact",
                 "source_url": "https://example.com/a"}]
    new = [
        # Same raw_name + source_url: duplicate, dropped.
        {"raw_name": "OpenAI, Inc.", "canonical_name": "OpenAI",
         "confidence": 0.9, "match_method": "fuzzy",
         "source_url": "https://example.com/a"},
        # Same raw_name, different URL: a distinct sighting, kept.
        {"raw_name": "OpenAI, Inc.", "canonical_name": "OpenAI",
         "confidence": 1.0, "match_method": "exact",
         "source_url": "https://example.com/b"},
        {"raw_name": "Acme", "canonical_name": "Acme",
         "confidence": 0.0, "match_method": "unmatched",
         "source_url": "https://example.com/c"},
    ]
    merged = run_bulk.merge_mapping_log(existing, new)
    assert len(merged) == 3
    assert merged[0] is existing[0]  # existing entry wins on conflict
    assert merged[1]["source_url"] == "https://example.com/b"
    assert merged[2]["raw_name"] == "Acme"
    # Same behaviour in run_fresh (it merges into the shared log).
    assert run_fresh.merge_mapping_log(existing, new) == merged


def test_merge_mapping_log_empty_existing():
    new = [{"raw_name": "Acme", "canonical_name": "Acme",
            "confidence": 0.0, "match_method": "unmatched",
            "source_url": "https://example.com/c"}]
    assert run_bulk.merge_mapping_log([], new) == new


def test_merge_records_keeps_existing_on_rerun():
    # Regression: a re-run overwrote the JSON with only the newly
    # claimed records, leaving empty tabs on the second run.
    existing = [{"source": {"url": "https://example.com/a"},
                 "entityName": "Acme"}]
    new = [{"source": {"url": "https://example.com/a"},
            "entityName": "Acme"},
           {"source": {"url": "https://example.com/b"},
            "entityName": "Beta"}]
    merged = run_bulk.merge_records(
        existing, new, run_bulk._record_key("Startups"))
    assert [r["entityName"] for r in merged] == ["Acme", "Beta"]
    assert run_fresh.merge_records([], new, lambda r: r["source"]["url"]) == new


# --- JSON read/write ---

def test_write_json_roundtrip(tmp_path):
    path = tmp_path / "sub" / "Jobs.json"
    records = [{"company": "Acme", "date": "2026-11-21T00:00:00Z"}]
    run_bulk.write_json(path, records)
    assert run_bulk.load_json_list(path) == records
    assert run_fresh.load_json_list(path) == records


def test_load_json_list_missing_file(tmp_path):
    assert run_bulk.load_json_list(tmp_path / "nope.json") == []


# --- dedup ---

def test_already_seen_skips_second_processing():
    url = "https://example.com/article/1"
    assert dedup.already_seen(url) is False  # first sighting: process it
    assert dedup.already_seen(url) is True   # second sighting: skip it
    assert dedup.already_seen("https://example.com/article/2") is False


# --- entity resolution (defensive fallback) ---

@contextlib.contextmanager
def _fake_matcher(monkeypatch, module):
    """Swap src.resolver.matcher in sys.modules so resolve_entity's
    importlib.import_module picks up the fake."""
    real = sys.modules.get("src.resolver.matcher")
    monkeypatch.setitem(sys.modules, "src.resolver.matcher", module)
    yield
    if real is not None:
        monkeypatch.setitem(sys.modules, "src.resolver.matcher", real)


def test_resolve_entity_falls_back_to_resolve(monkeypatch):
    fake = types.SimpleNamespace(
        resolve=lambda name: "OpenAI" if name.startswith("OpenAI") else name.strip())
    with _fake_matcher(monkeypatch, fake):
        assert run_bulk.resolve_entity("OpenAI, Inc.") == ("OpenAI", 1.0, "exact")
        assert run_bulk.resolve_entity("Zqxwv Corp") == ("Zqxwv Corp", 0.0, "unmatched")


def test_resolve_entity_prefers_resolve_with_confidence(monkeypatch):
    calls = []

    def rwc(name, domain=None, canonical=None):
        calls.append((name, domain))
        return ("Canonical Co", 0.85, "domain")

    fake = types.SimpleNamespace(resolve_with_confidence=rwc,
                                 resolve=lambda name: name)
    with _fake_matcher(monkeypatch, fake):
        assert run_fresh.resolve_entity("Canonical Co Ltd",
                                        "canonical.example.com") == (
            "Canonical Co", 0.85, "domain")
        assert calls == [("Canonical Co Ltd", "canonical.example.com")]


def test_resolve_entity_uses_real_matcher():
    # Real matcher today has only resolve(); "OpenAI, Inc." is in the
    # seed canonical list, so it resolves; an unknown name passes
    # through unmatched. (If resolve_with_confidence lands, the
    # contract still requires these outcomes.)
    canonical, confidence, method = run_bulk.resolve_entity("OpenAI, Inc.")
    assert canonical == "OpenAI"
    assert confidence > 0
    assert method in ("domain", "exact", "fuzzy")
    assert run_bulk.resolve_entity("Zqxwv Nonexistent Corp") == (
        "Zqxwv Nonexistent Corp", 0.0, "unmatched")


# --- import_optional ---

def test_import_optional_missing_module(capsys):
    assert run_bulk.import_optional("src.does_not_exist", "x") is None
    assert "ERROR" in capsys.readouterr().out


def test_import_optional_missing_attr(capsys):
    assert run_bulk.import_optional("src.storage.dedup", "nope") is None
    assert "ERROR" in capsys.readouterr().out


def test_import_optional_present():
    assert callable(run_bulk.import_optional("src.storage.dedup", "already_seen"))


# --- CLI parsing ---

def test_bulk_parse_args_defaults():
    args = run_bulk.parse_args([])
    assert args.limit == 1000
    assert args.no_db is False
    assert args.no_llm is False


def test_bulk_parse_args_flags():
    args = run_bulk.parse_args(["--limit", "5", "--no-db", "--no-llm"])
    assert (args.limit, args.no_db, args.no_llm) == (5, True, True)


def test_fresh_parse_args_defaults():
    args = run_fresh.parse_args([])
    assert args.limit is None
    assert args.no_db is False
    assert args.smoke is False


def test_fresh_smoke_implies_limit_5_and_json_only():
    args = run_fresh.parse_args(["--smoke"])
    assert run_fresh.resolve_smoke(args) == (5, False)
    args = run_fresh.parse_args(["--limit", "3"])
    assert run_fresh.resolve_smoke(args) == (3, True)
    args = run_fresh.parse_args(["--no-db"])
    assert run_fresh.resolve_smoke(args) == (None, False)


# --- collect_fresh (fake items, real models) ---

def _news(url, source="TechCrunch AI"):
    return NewsItem(title="AI news", source=Source(name=source, url=url),
                    published_date=_now(), full_text="body", url=url)


def _job(company, url=None):
    return Job(company=company, date=_now(), is_remote=True,
               role_family="Engineering", source_url=url)


def test_collect_fresh_dedups_and_resolves_companies():
    news = [_news("https://example.com/n1"),
            _news("https://example.com/n2", "The Verge AI")]
    jobs = [_job("OpenAI, Inc.", "https://remoteok.com/jobs/1"),
            _job("Zqxwv Corp", "https://remoteok.com/jobs/2")]
    result = run_fresh.collect_fresh(news, jobs)
    stats = result["stats"]
    assert stats["news_fresh"] == 2
    assert stats["jobs_fresh"] == 2
    assert stats["duplicates"] == 0
    assert stats["news_by_source"] == {"TechCrunch AI": 1, "The Verge AI": 1}
    assert stats["jobs_by_board"] == {"remoteok.com": 2}
    # Company canonicalised; news title kept as-is.
    assert result["jobs"][0]["company"] == "OpenAI"
    assert result["news"][0]["title"] == "AI news"
    # One mapping entry per job company, including the unmatched one.
    assert len(result["mapping"]) == 2
    assert result["mapping"][0]["raw_name"] == "OpenAI, Inc."
    assert result["mapping"][0]["canonical_name"] == "OpenAI"
    assert result["mapping"][1]["match_method"] == "unmatched"
    assert result["mapping"][1]["confidence"] == 0.0


def test_collect_fresh_never_reprocesses_seen_url():
    news = [_news("https://example.com/n1")]
    jobs = [_job("Acme", "https://remoteok.com/jobs/1")]
    first = run_fresh.collect_fresh(news, jobs)
    assert first["stats"]["duplicates"] == 0
    second = run_fresh.collect_fresh(news, jobs)
    assert second["news"] == [] and second["jobs"] == []
    assert second["mapping"] == []
    assert second["stats"]["duplicates"] == 2


def test_collect_fresh_marks_urls_completed():
    """Processed URLs are completed (permanent), not just leased:
    a later run — even after the 10-minute lease would have expired
    — still never reprocesses them."""
    news = [_news("https://example.com/n1")]
    jobs = [_job("Acme", "https://remoteok.com/jobs/1")]
    run_fresh.collect_fresh(news, jobs)
    assert dedup.state("https://example.com/n1") == "completed"
    assert dedup.state("https://remoteok.com/jobs/1") == "completed"


def test_collect_fresh_job_without_source_url():
    job = _job("Acme")  # no source_url: falls back to company+date key
    result = run_fresh.collect_fresh([], [job])
    assert result["stats"]["jobs_fresh"] == 1
    assert result["stats"]["jobs_by_board"] == {"unknown": 1}
    assert result["mapping"][0]["source_url"] == ""
    again = run_fresh.collect_fresh([], [job])
    assert again["stats"]["duplicates"] == 1


# --- Postgres graceful degradation ---

def test_persist_to_db_warns_and_continues_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(run_bulk.db_module, "init_db",
                        lambda url: (_ for _ in ()).throw(
                            ConnectionError("db down")))
    assert run_bulk.persist_to_db({"jobs": []}) is False
    assert "WARNING" in capsys.readouterr().out


def test_persist_to_db_calls_store_per_table(monkeypatch):
    calls = []
    monkeypatch.setattr(run_bulk.db_module, "init_db", lambda url: None)
    monkeypatch.setattr(
        run_bulk.db_module, "store_records",
        lambda url, table, records: calls.append((table, len(records))))
    payloads = {"jobs": [{"unique_url": "u1", "payload": {}}], "news": []}
    assert run_bulk.persist_to_db(payloads) is True
    assert calls == [("jobs", 1)]  # empty tables are not stored


# --- discover_urls tolerates both signatures ---

class _CrawlerWithLimit:
    async def discover(self, limit=1000):
        return [f"https://example.com/{i}" for i in range(limit)]


class _CrawlerWithoutLimit:
    async def discover(self):
        return ["https://example.com/a", "https://example.com/b",
                "https://example.com/c"]


def test_discover_urls_with_limit_param():
    urls = asyncio.run(run_bulk.discover_urls(_CrawlerWithLimit(), 2))
    assert urls == ["https://example.com/0", "https://example.com/1"]


def test_discover_urls_without_limit_param_slices():
    urls = asyncio.run(run_bulk.discover_urls(_CrawlerWithoutLimit(), 2))
    assert urls == ["https://example.com/a", "https://example.com/b"]

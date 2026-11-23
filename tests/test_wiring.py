"""Wiring tests: LLM fallback, Sheets export and Postgres DDL —
all verified without keys, credentials or a live database, by
monkeypatching the provider / SDK boundaries."""
import sys
import types

import pytest

from src.export import sheets
from src.llm import orchestrator
from src.storage import db
from src.utils.retry import retry_on_rate_limit

# --- LLM fallback ---

def test_fallback_skips_unconfigured_tier(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    calls = []

    def fake_call(provider, prompt):
        calls.append(provider)
        return "groq-answer"

    monkeypatch.setattr(orchestrator, "_call_provider", fake_call)
    assert orchestrator.extract_with_fallback("hi") == "groq-answer"
    assert calls == ["groq"]
    # gemini skipped (no key), gemini_pro skipped too (shares GEMINI_API_KEY),
    # deepseek never reached


def test_fallback_moves_on_after_failure(monkeypatch):
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(key, "test-key")

    def fake_call(provider, prompt):
        if provider == "gemini":
            raise ValueError("boom")
        return "groq-answer"

    monkeypatch.setattr(orchestrator, "_call_provider", fake_call)
    assert orchestrator.extract_with_fallback("hi") == "groq-answer"


def test_fallback_tries_gemini_pro_last(monkeypatch):
    # gemini_pro shares GEMINI_API_KEY, so all four tiers are
    # configured with just the three keys.
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    calls = []

    def fake_call(provider, prompt):
        calls.append(provider)
        if provider == "gemini_pro":
            return "pro-answer"
        raise ValueError("boom")

    monkeypatch.setattr(orchestrator, "_call_provider", fake_call)
    assert orchestrator.extract_with_fallback("hi") == "pro-answer"
    assert calls == ["gemini", "groq", "deepseek", "gemini_pro"]


def test_fallback_raises_when_all_tiers_fail(monkeypatch):
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    monkeypatch.setattr(
        orchestrator, "_call_provider",
        lambda provider, prompt: (_ for _ in ()).throw(ValueError("boom")),
    )
    with pytest.raises(RuntimeError, match="all LLM tiers failed"):
        orchestrator.extract_with_fallback("hi")


def test_fallback_raises_when_nothing_configured(monkeypatch):
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        orchestrator.extract_with_fallback("hi")


# --- Retry: only 429s are retried ---

def test_retry_raises_immediately_on_non_429(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.utils.retry.time.sleep", sleeps.append)

    @retry_on_rate_limit
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        boom()
    assert sleeps == []


def test_retry_retries_429_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.utils.retry.time.sleep", sleeps.append)
    attempts = []

    @retry_on_rate_limit
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    assert flaky() == "ok"
    assert len(sleeps) == 2


# --- Sheets export (fake gspread) ---

class _FakeWorksheet:
    def __init__(self, title):
        self.title = title
        self.values = None
        self.cleared = False

    def clear(self):
        self.cleared = True

    def update(self, range_name, values):
        self.values = values


class _FakeSpreadsheet:
    def __init__(self):
        self.tabs = {}

    def worksheet(self, title):
        return self.tabs[title]  # KeyError if missing, like WorksheetNotFound

    def add_worksheet(self, title, rows, cols):
        ws = _FakeWorksheet(title)
        self.tabs[title] = ws
        return ws


def test_sheets_export_writes_all_six_tabs(monkeypatch):
    spreadsheet = _FakeSpreadsheet()
    # Pre-create one tab so the existing-tab path is exercised too.
    spreadsheet.tabs["Startups"] = _FakeWorksheet("Startups")
    fake_client = types.SimpleNamespace(open_by_key=lambda sheet_id: spreadsheet)
    fake_gspread = types.SimpleNamespace(service_account=lambda filename: fake_client)
    monkeypatch.setitem(sys.modules, "gspread", fake_gspread)

    sheets.export_all("sheet-id", {
        "Startups": [{"entityName": "Acme AI", "employeeCount": 12,
                      "source": {"name": "TechCrunch", "url": "https://example.com/a"},
                      "collectedAt": "2026-11-21T00:00:00Z"}],
        "News": [],
        "Entity Mapping Log": [{"raw_name": "openai", "canonical_name": "OpenAI",
                                "confidence": 1.0, "match_method": "domain",
                                "source_url": "https://openai.com"}],
    })
    tabs = spreadsheet.tabs
    assert set(tabs) == set(sheets.SHEET_TABS)
    assert len(sheets.SHEET_TABS) == 6
    assert tabs["Startups"].cleared
    assert tabs["Startups"].values[0] == sheets.STARTUP_HEADERS
    # Headers are now [schemaVersion, recordType, content.entityName,
    # content.data.employeeCount, source.name, ...] with version
    # defaults filled for records that predate those fields.
    assert tabs["Startups"].values[1][2] == "Acme AI"  # content.entityName
    assert tabs["Startups"].values[1][4] == "TechCrunch"  # dotted source.name
    assert tabs["News"].values == [sheets.NEWS_HEADERS]  # empty tab: headers only
    mapping = tabs["Entity Mapping Log"]
    assert mapping.values[0] == sheets.MAPPING_HEADERS
    assert mapping.values[1] == ["1.0.0", "ENTITY_MAPPING", "openai",
                                 "OpenAI", 1.0, "domain",
                                 "https://openai.com"]


# --- Postgres DDL ---

def test_db_ddl_covers_all_tables():
    for table in db.TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in db.DDL
    assert "JSONB" in db.DDL
    assert "unique_url TEXT UNIQUE" in db.DDL


def test_db_upsert_sql():
    sql = db.upsert_sql("jobs")
    assert "INSERT INTO jobs" in sql
    assert "ON CONFLICT (unique_url)" in sql
    with pytest.raises(ValueError):
        db.upsert_sql("nope")

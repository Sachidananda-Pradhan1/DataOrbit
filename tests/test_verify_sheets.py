"""Tests for scripts/verify_sheets.py (issue E) — mocked only: no
real Google Sheets calls, no credentials. The gspread client is
faked via sys.modules; the pure checkers run against in-memory
get_all_values() output.

The script is loaded from its file path via importlib (scripts/ is
not a package), as in test_scripts.py.
"""
import importlib.util
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.export.sheets import HEADERS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_sheets = _load_script("verify_sheets")

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def _valid_tabs(now=NOW):
    """A fully valid Sheet: 1,000 startups / products / papers, fresh
    jobs + news, a mapping log with one legitimately URL-less row."""
    fresh = _iso(now - timedelta(hours=2))
    # Rows lead with schemaVersion + recordType, matching the live
    # HEADERS (Startups/Products rows also align to the content.*
    # columns — the values are positional, so keep them in order).
    return {
        "Startups": [HEADERS["Startups"]] + [
            ["1.0.0", "STARTUP", f"Startup {i}", "10", "Y Combinator",
             f"https://example.com/startup/{i}", fresh]
            for i in range(1000)],
        "Products": [HEADERS["Products"]] + [
            ["1.0.0", "PRODUCT", f"Startup {i}", f"Product {i}", "FREE",
             "BetaList", f"https://example.com/product/{i}", fresh]
            for i in range(1000)],
        "Research Papers": [HEADERS["Research Papers"]] + [
            ["1.0.0", "RESEARCH_PAPER", f"Paper {i}", "A. Author",
             f"https://arxiv.org/abs/2611.{i:05d}",
             f"https://github.com/org/repo{i}" if i % 2 == 0 else "",
             str(i) if i % 2 == 0 else "", fresh]
            for i in range(1000)],
        "Jobs": [HEADERS["Jobs"]] + [
            ["1.0.0", "JOB", "Acme", "ML Engineer",
             _iso(now - timedelta(hours=2)),
             "TRUE", "Engineering", "Global", f"https://example.com/job/{i}"]
            for i in range(5)],
        "News": [HEADERS["News"]] + [
            ["1.0.0", "NEWS", f"News {i}", "TechCrunch AI",
             f"https://example.com/news/{i}",
             _iso(now - timedelta(hours=1)), "full text here"]
            for i in range(5)],
        "Entity Mapping Log": [HEADERS["Entity Mapping Log"],
            ["1.0.0", "ENTITY_MAPPING", "OpenAI, Inc.", "OpenAI", "1.0",
             "exact", "https://example.com/a"],
            # Empty source_url is legitimate here (job posted w/o one).
            ["1.0.0", "ENTITY_MAPPING", "Zqxwv", "Zqxwv", "0.0",
             "unmatched", ""]],
    }


def _check(results, name):
    return next(r for r in results if r.name == name)


def _live_tabs():
    """Valid fixture dated relative to the real clock — main() checks
    freshness against datetime.now(), not the fixed NOW above."""
    return _valid_tabs(now=datetime.now(UTC))


# --- happy path ---

def test_fully_valid_sheet_passes(capsys):
    results = verify_sheets.verify_sheet_data(_valid_tabs(), NOW)
    assert results and all(r.passed for r in results)
    assert verify_sheets.print_report(results, "sheet-id") is True
    out = capsys.readouterr().out
    assert "VERIFICATION PASSED" in out
    assert "[FAIL]" not in out


def test_valid_sheet_covers_every_check_type():
    names = {r.name for r in verify_sheets.verify_sheet_data(_valid_tabs(), NOW)}
    assert "Tabs present" in names
    assert "Startups: row count" in names
    assert "Jobs: dates within 24h" in names
    assert "News: dates within 24h" in names
    assert "Research Papers: GitHub stars" in names
    assert "Entity Mapping Log: source URLs" in names


# --- failing cases ---

def test_missing_tab_fails():
    tabs = _valid_tabs()
    tabs["News"] = None
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "Tabs present").passed is False
    assert "News" in _check(results, "Tabs present").detail
    assert not any(r.name.startswith("News:") for r in results)


def test_wrong_header_fails_and_skips_column_checks():
    tabs = _valid_tabs()
    tabs["Jobs"][0] = [h if h != "date" else "posted_date"
                       for h in HEADERS["Jobs"]]
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "Jobs: headers").passed is False
    # Row count does not depend on columns, so it still runs and passes.
    assert _check(results, "Jobs: row count").passed is True
    # Column-dependent checks are skipped, not misleadingly failed.
    assert not any(r.name.startswith("Jobs: source URLs") for r in results)
    assert not any(r.name.startswith("Jobs: dates") for r in results)


@pytest.mark.parametrize("tab,n_rows", [
    ("Startups", 999), ("Products", 999), ("Research Papers", 999),
    ("Jobs", 0), ("News", 0),
])
def test_short_row_count_fails(tab, n_rows):
    tabs = _valid_tabs()
    tabs[tab] = tabs[tab][:n_rows + 1]  # header + n_rows data rows
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, f"{tab}: row count").passed is False


def test_missing_url_fails():
    tabs = _valid_tabs()
    tabs["Startups"][3][5] = ""  # source.url blank on one row
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    check = _check(results, "Startups: source URLs")
    assert check.passed is False
    assert "row 4" in check.detail  # header + 1-based data rows


def test_invalid_url_fails():
    tabs = _valid_tabs()
    tabs["Products"][1][6] = "not a url"  # source.url
    tabs["Jobs"][2][8] = "ftp://example.com/job"  # source_url
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "Products: source URLs").passed is False
    assert _check(results, "Jobs: source URLs").passed is False


def test_duplicate_url_fails():
    tabs = _valid_tabs()
    tabs["Jobs"][2][8] = tabs["Jobs"][1][8]  # same source_url twice
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    check = _check(results, "Jobs: duplicate URLs")
    assert check.passed is False
    assert "row 3" in check.detail


def test_stale_job_date_fails():
    tabs = _valid_tabs()
    tabs["Jobs"][1][4] = _iso(NOW - timedelta(hours=25))  # date
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "Jobs: dates within 24h").passed is False


def test_stale_news_date_fails():
    tabs = _valid_tabs()
    tabs["News"][1][5] = _iso(NOW - timedelta(days=2))  # published_date
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "News: dates within 24h").passed is False


def test_unparseable_date_fails():
    tabs = _valid_tabs()
    tabs["Jobs"][1][4] = "sometime last week"  # date
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "Jobs: dates within 24h").passed is False


def test_paper_with_github_url_but_no_stars_fails():
    tabs = _valid_tabs()
    tabs["Research Papers"][1][6] = ""  # repo link, stars blanked
    assert tabs["Research Papers"][1][5]  # sanity: github_url present
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "Research Papers: GitHub stars").passed is False


def test_paper_with_zero_stars_passes():
    tabs = _valid_tabs()
    # Row 1 (paper 0) has a repo link; "0" is a real star count.
    tabs["Research Papers"][1][6] = "0"  # github_stars
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "Research Papers: GitHub stars").passed is True


def test_empty_tab_fails():
    tabs = _valid_tabs()
    tabs["Jobs"] = []
    results = verify_sheets.verify_sheet_data(tabs, NOW)
    assert _check(results, "Jobs: headers").passed is False
    assert _check(results, "Jobs: row count").passed is False


def test_mapping_log_allows_empty_source_url():
    # Covered by the valid fixture; assert the specific check passes.
    results = verify_sheets.verify_sheet_data(_valid_tabs(), NOW)
    assert _check(results, "Entity Mapping Log: source URLs").passed is True


# --- helpers ---

@pytest.mark.parametrize("value,expected", [
    ("https://example.com/a", True),
    ("http://example.com", True),
    ("  https://example.com/a  ", True),
    ("", False),
    ("   ", False),
    ("not a url", False),
    ("ftp://example.com/x", False),
    ("https://", False),
    (None, False),
    (123, False),
])
def test_is_valid_url(value, expected):
    assert verify_sheets.is_valid_url(value) is expected


@pytest.mark.parametrize("value,expected", [
    ("2026-11-22T10:00:00Z", datetime(2026, 11, 22, 10, tzinfo=UTC)),
    ("2026-11-22T10:00:00+00:00", datetime(2026, 11, 22, 10, tzinfo=UTC)),
    ("2026-11-22T12:00:00+02:00", datetime(2026, 11, 22, 10, tzinfo=UTC)),
    ("2026-11-22", datetime(2026, 11, 22, tzinfo=UTC)),
    ("Sun, 22 Nov 2026 10:00:00 GMT", datetime(2026, 11, 22, 10, tzinfo=UTC)),
    ("", None),
    ("sometime last week", None),
    (None, None),
])
def test_parse_sheet_date(value, expected):
    assert verify_sheets.parse_sheet_date(value) == expected


# --- main() with a mocked gspread client ---

class _WorksheetNotFound(Exception):
    pass


def _fake_gspread(monkeypatch, tabs, fail=None):
    """Install a fake `gspread` module; returns the calls it recorded."""
    module = types.ModuleType("gspread")
    module.exceptions = types.SimpleNamespace(
        WorksheetNotFound=_WorksheetNotFound)
    calls = {}

    class _Worksheet:
        def __init__(self, values):
            self._values = values

        def get_all_values(self):
            return self._values

    class _Spreadsheet:
        def worksheet(self, title):
            if tabs.get(title) is None:
                raise _WorksheetNotFound(title)
            return _Worksheet(tabs[title])

    class _Client:
        def open_by_key(self, key):
            calls["sheet_id"] = key
            if fail == "open":
                raise RuntimeError("sheet not found")
            return _Spreadsheet()

    def service_account(filename):
        calls["credentials"] = filename
        if fail == "auth":
            raise RuntimeError("bad credentials")
        return _Client()

    module.service_account = service_account
    monkeypatch.setitem(sys.modules, "gspread", module)
    return calls


def test_main_passes_with_valid_mock_sheet(monkeypatch, capsys):
    _fake_gspread(monkeypatch, _live_tabs())
    assert verify_sheets.main(["sheet-id-123"]) == 0
    out = capsys.readouterr().out
    assert "VERIFICATION PASSED" in out
    assert "sheet-id-123" in out


def test_main_fails_with_bad_mock_sheet(monkeypatch, capsys):
    tabs = _live_tabs()
    tabs["News"] = None
    _fake_gspread(monkeypatch, tabs)
    assert verify_sheets.main(["sheet-id-123"]) == 1
    assert "VERIFICATION FAILED" in capsys.readouterr().out


def test_main_uses_env_sheet_id(monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_SHEET_ID", "env-sheet-id")
    calls = _fake_gspread(monkeypatch, _live_tabs())
    assert verify_sheets.main([]) == 0
    assert calls["sheet_id"] == "env-sheet-id"


def test_main_requires_sheet_id(monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_SHEET_ID", raising=False)
    _fake_gspread(monkeypatch, _valid_tabs())
    assert verify_sheets.main([]) == 2
    assert "GOOGLE_SHEET_ID" in capsys.readouterr().err


def test_main_sheet_read_error_returns_2(monkeypatch, capsys):
    _fake_gspread(monkeypatch, _valid_tabs(), fail="auth")
    assert verify_sheets.main(["sheet-id-123"]) == 2
    assert "could not read sheet" in capsys.readouterr().err


def test_fetch_passes_credentials_path(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_CREDENTIALS_JSON", "/tmp/env-creds.json")
    calls = _fake_gspread(monkeypatch, _live_tabs())
    verify_sheets.main(["sheet-id-123"])
    assert calls["credentials"] == "/tmp/env-creds.json"
    # --credentials flag wins over the env var.
    calls.clear()
    verify_sheets.main(["sheet-id-123", "--credentials", "/tmp/flag-creds.json"])
    assert calls["credentials"] == "/tmp/flag-creds.json"


def test_parse_args_defaults():
    args = verify_sheets.parse_args([])
    assert args.sheet_id is None
    assert args.credentials is None
    args = verify_sheets.parse_args(["abc", "--credentials", "c.json"])
    assert (args.sheet_id, args.credentials) == ("abc", "c.json")

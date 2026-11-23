"""Verify the exported Google Sheet deliverable (issue E).

Reads the live Sheet — same service-account auth as
scripts/export_sheets.py (credentials path from
GOOGLE_SHEETS_CREDENTIALS_JSON, default credentials.json) — and
checks, per tab:

- all 6 tabs exist with their exact names (src/export/sheets.py:SHEET_TABS)
- headers match the expected schema exactly (src/export/sheets.py:HEADERS)
- row counts: startups / products / papers >= 1,000; jobs / news > 0
- every row in the 5 entity tabs has a non-empty, valid source URL
  (anti-hallucination rule: every record must trace to a real URL)
- no duplicate source URLs within a tab
- jobs / news dates are within 24h of the run
- every paper with a github_url has a github_stars count
- every news row has non-empty full_text (the crawler drops
  empty-full_text items at crawl time; this is the safety net)

The Entity Mapping Log is exempt from the non-empty-URL and
duplicate-URL checks: a mapping entry can legitimately have no
source URL (e.g. a job posted without one) and several entities can
share one source. Non-empty URLs there are still validated.

Usage:
    GOOGLE_SHEET_ID=... python scripts/verify_sheets.py [SHEET_ID]
    python scripts/verify_sheets.py --credentials path/to/creds.json

Prints a per-check pass/fail report and exits 0 only if every check
passes (exit 1 on any failure, 2 on usage / Sheet access errors).
Read-only (get_all_values only); no LLM calls.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config  # noqa: F401  (loads .env into os.environ)
from src.export.sheets import HEADERS, SHEET_TABS

MIN_ROWS = {"Startups": 1000, "Products": 1000, "Research Papers": 1000,
            "Jobs": 1, "News": 1}
# Column holding the traceable source URL per tab. The Entity Mapping
# Log is deliberately in the optional map (see module docstring).
URL_COLUMN = {"Startups": "source.url", "Products": "source.url",
              "Research Papers": "paper_url", "Jobs": "source_url",
              "News": "source.url"}
URL_COLUMN_OPTIONAL = {"Entity Mapping Log": "source_url"}
DATE_COLUMN = {"Jobs": "date", "News": "published_date"}
FRESHNESS = timedelta(hours=24)
FUTURE_TOLERANCE = timedelta(hours=1)  # clock skew on source timestamps
MAX_EXAMPLES = 5  # rows named in a failure detail before truncating


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def is_valid_url(value: object) -> bool:
    """Non-empty http(s) URL with a host — the traceability rule."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def parse_sheet_date(value: object) -> datetime | None:
    """Parse a Sheet date cell (ISO-8601, incl. trailing Z, or
    RFC-822) to a tz-aware UTC datetime, or None if unparseable.
    Naive values are assumed UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)  # 3.11+ accepts trailing Z
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_star_count(value: object) -> bool:
    """A github_stars cell: a non-negative integer ('0' counts)."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return int(value) >= 0
    except ValueError:
        pass
    try:
        number = float(value)
    except ValueError:
        return False
    return number.is_integer() and number >= 0


def _examples(label: str, bad: list[tuple[int, str]]) -> str:
    shown = ", ".join(f"row {row} ({value})" for row, value in bad[:MAX_EXAMPLES])
    extra = f" (+{len(bad) - MAX_EXAMPLES} more)" if len(bad) > MAX_EXAMPLES else ""
    return f"{len(bad)} {label}: {shown}{extra}"


def _cell(row: list, idx: int | None) -> str:
    value = row[idx] if idx is not None and idx < len(row) else ""
    return value if isinstance(value, str) else str(value)


def _column_index(headers: list, column: str) -> int | None:
    try:
        return headers.index(column)
    except ValueError:
        return None


def _check_urls(tab: str, headers: list, rows: list) -> CheckResult:
    required = tab in URL_COLUMN
    column = URL_COLUMN.get(tab) or URL_COLUMN_OPTIONAL.get(tab)
    idx = _column_index(headers, column)
    bad = []
    for i, row in enumerate(rows):
        value = _cell(row, idx).strip()
        if required and not is_valid_url(value):
            bad.append((i + 2, value or "(empty)"))
        elif not required and value and not is_valid_url(value):
            bad.append((i + 2, value))
    if bad:
        return CheckResult(f"{tab}: source URLs", False,
                           _examples("missing/invalid URL", bad))
    detail = (f"all {len(rows)} rows have a valid {column}" if required
              else "all non-empty source_url values are valid")
    return CheckResult(f"{tab}: source URLs", True, detail)


def _check_duplicates(tab: str, headers: list, rows: list) -> CheckResult:
    idx = _column_index(headers, URL_COLUMN[tab])
    seen: set[str] = set()
    dupes = []
    for i, row in enumerate(rows):
        url = _cell(row, idx).strip()
        if not url:  # empty URLs are the URL check's job, not this one
            continue
        if url in seen:
            dupes.append((i + 2, url))
        else:
            seen.add(url)
    if dupes:
        return CheckResult(f"{tab}: duplicate URLs", False,
                           _examples("duplicate URL", dupes))
    return CheckResult(f"{tab}: duplicate URLs", True,
                       f"{len(seen)} unique URLs")


def _check_dates(tab: str, headers: list, rows: list,
                 now: datetime) -> CheckResult:
    idx = _column_index(headers, DATE_COLUMN[tab])
    bad = []
    for i, row in enumerate(rows):
        raw = _cell(row, idx)
        parsed = parse_sheet_date(raw)
        if parsed is None:
            bad.append((i + 2, f"unparseable date {raw!r}"))
        elif now - parsed > FRESHNESS:
            bad.append((i + 2, f"stale: {raw}"))
        elif parsed - now > FUTURE_TOLERANCE:
            bad.append((i + 2, f"in the future: {raw}"))
    if bad:
        return CheckResult(f"{tab}: dates within 24h", False,
                           _examples("bad date", bad))
    return CheckResult(f"{tab}: dates within 24h", True,
                       f"all {len(rows)} rows within 24h")


def _check_paper_stars(headers: list, rows: list) -> CheckResult:
    gh_idx = _column_index(headers, "github_url")
    stars_idx = _column_index(headers, "github_stars")
    bad = []
    with_repo = 0
    for i, row in enumerate(rows):
        github_url = _cell(row, gh_idx).strip()
        if not github_url:
            continue  # no repo link: no star count required
        with_repo += 1
        if not _is_star_count(_cell(row, stars_idx)):
            bad.append((i + 2, f"github_url {github_url} has no star count"))
    if bad:
        return CheckResult("Research Papers: GitHub stars", False,
                           _examples("missing stars", bad))
    return CheckResult("Research Papers: GitHub stars", True,
                       f"{with_repo} papers with a repo, all have a star count")


def _check_news_full_text(headers: list, rows: list) -> CheckResult:
    """News rows must carry full text (Phase II requirement). The
    crawler drops empty-full_text items at crawl time; this is the
    safety net — any empty row in the Sheet is a failure."""
    idx = _column_index(headers, "full_text")
    title_idx = _column_index(headers, "title")
    bad = []
    for i, row in enumerate(rows):
        if not _cell(row, idx).strip():
            bad.append((i + 2, _cell(row, title_idx) or "(no title)"))
    if bad:
        return CheckResult("News: full_text non-empty", False,
                           _examples("empty full_text", bad))
    return CheckResult("News: full_text non-empty", True,
                       f"all {len(rows)} rows have full text")


def verify_tab(tab: str, values: list, now: datetime) -> list[CheckResult]:
    """Check one tab's get_all_values() output (header row + data
    rows). A header mismatch skips the column-dependent checks —
    their indices would be meaningless."""
    expected = HEADERS[tab]
    if not values:
        return [CheckResult(f"{tab}: headers", False,
                            "tab is empty (no header row)"),
                CheckResult(f"{tab}: row count", False, "0 rows")]
    headers, rows = values[0], values[1:]
    results = [CheckResult(
        f"{tab}: headers", headers == expected,
        f"{len(headers)} columns" if headers == expected
        else f"expected {expected}, got {headers}")]
    minimum = MIN_ROWS.get(tab)
    if minimum is not None:
        results.append(CheckResult(
            f"{tab}: row count", len(rows) >= minimum,
            f"{len(rows)} rows (minimum {minimum})"))
    else:
        results.append(CheckResult(f"{tab}: row count", True,
                                   f"{len(rows)} rows (no minimum)"))
    if headers != expected:
        return results
    if tab in URL_COLUMN or tab in URL_COLUMN_OPTIONAL:
        results.append(_check_urls(tab, headers, rows))
    if tab in URL_COLUMN:
        results.append(_check_duplicates(tab, headers, rows))
    if tab in DATE_COLUMN:
        results.append(_check_dates(tab, headers, rows, now))
    if tab == "Research Papers":
        results.append(_check_paper_stars(headers, rows))
    if tab == "News":
        results.append(_check_news_full_text(headers, rows))
    return results


def verify_sheet_data(tabs: dict[str, list | None],
                      now: datetime) -> list[CheckResult]:
    """Check the whole Sheet. ``tabs`` maps tab name to its
    get_all_values() output, or None if the tab is missing."""
    missing = [tab for tab in SHEET_TABS if tabs.get(tab) is None]
    results = [CheckResult(
        "Tabs present", not missing,
        "all 6 tabs present" if not missing
        else f"missing: {', '.join(missing)}")]
    for tab in SHEET_TABS:
        values = tabs.get(tab)
        if values is not None:
            results.extend(verify_tab(tab, values, now))
    return results


def fetch_sheet_data(sheet_id: str, credentials_path: str) -> dict[str, list | None]:
    """Read every tab with gspread (same service-account auth as the
    exporter). Missing tabs map to None."""
    import gspread

    client = gspread.service_account(filename=credentials_path)
    spreadsheet = client.open_by_key(sheet_id)
    data: dict[str, list | None] = {}
    for tab in SHEET_TABS:
        try:
            data[tab] = spreadsheet.worksheet(tab).get_all_values()
        except gspread.exceptions.WorksheetNotFound:
            data[tab] = None
    return data


def print_report(results: list[CheckResult], sheet_id: str) -> bool:
    """Print the per-check report + summary. Returns True iff all pass."""
    print(f"Sheet verification: {sheet_id}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        line = f"  [{status}] {result.name}"
        if result.detail:
            line += f" — {result.detail}"
        print(line)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    verdict = "VERIFICATION PASSED" if failed == 0 else "VERIFICATION FAILED"
    print(f"Summary: {passed} passed, {failed} failed — {verdict}")
    return failed == 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify the exported Google Sheet (6 tabs, schemas, "
                    "counts, URLs, freshness, GitHub stars).")
    parser.add_argument("sheet_id", nargs="?",
                        help="Google Sheet ID (or set GOOGLE_SHEET_ID)")
    parser.add_argument("--credentials", default=None,
                        help="service-account credentials JSON "
                             "(default: $GOOGLE_SHEETS_CREDENTIALS_JSON "
                             "or credentials.json)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sheet_id = args.sheet_id or os.getenv("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        print("error: no Sheet ID — pass it as an argument or set "
              "GOOGLE_SHEET_ID", file=sys.stderr)
        return 2
    credentials_path = (args.credentials
                        or os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON",
                                     "credentials.json"))
    try:
        tabs = fetch_sheet_data(sheet_id, credentials_path)
    except Exception as exc:  # auth / network / bad Sheet ID
        print(f"error: could not read sheet {sheet_id}: {exc}", file=sys.stderr)
        return 2
    results = verify_sheet_data(tabs, datetime.now(timezone.utc))
    return 0 if print_report(results, sheet_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())

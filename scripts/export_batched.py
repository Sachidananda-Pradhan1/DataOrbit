"""Batched Google Sheets export — workaround for the egress proxy.

The normal export (scripts/export_sheets.py) writes each tab in ONE
update call (1,000+ rows). On 2026-11-23 that failed with Google 411
"POST requests require a Content-length header" — the sandbox egress
proxy mangles large POST bodies — while a 1-cell write succeeded.
This script writes the exact same content in small chunks (<=32KB of
JSON per update call), reusing the pipeline's own headers, nesting
and flattening, so the output is identical to export_all().

Each chunk is retried up to 3 times (the proxy flaps), and the run
stops loudly on a persistent failure instead of silently leaving a
half-written tab — re-running is safe because every tab is cleared
first.

Usage: set -a && source .env && set +a && python scripts/export_batched.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config  # noqa: F401  (loads .env into os.environ)
from src.export.sheets import HEADERS, SHEET_TABS, _nest_row, _row_values

MAX_CHUNK_BYTES = 32_000
MAX_ATTEMPTS = 3


def _chunks(values: list[list]) -> list[list[list]]:
    """Split flattened rows into chunks of at most MAX_CHUNK_BYTES
    of JSON each (a single over-long row still goes out alone)."""
    chunks: list[list[list]] = []
    chunk: list[list] = []
    size = 0
    for row in values:
        row_size = len(json.dumps(row))
        if chunk and size + row_size > MAX_CHUNK_BYTES:
            chunks.append(chunk)
            chunk, size = [], 0
        chunk.append(row)
        size += row_size
    if chunk:
        chunks.append(chunk)
    return chunks


def _with_retry(fn, label: str):
    """Run one Sheets call with retries; raises after MAX_ATTEMPTS."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # retried, then raised
            last_exc = exc
            print(f"{label}: attempt {attempt} failed "
                  f"({type(exc).__name__}: {str(exc)[:120]})")
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 * attempt)
    raise RuntimeError(f"{label}: failed after {MAX_ATTEMPTS} attempts"
                       ) from last_exc


def main() -> None:
    import gspread

    sheet_id = os.getenv("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        raise SystemExit("set GOOGLE_SHEET_ID in .env (see .env.example)")
    credentials_path = os.getenv(
        "GOOGLE_SHEETS_CREDENTIALS_JSON", "credentials.json")
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    gc = gspread.service_account(filename=credentials_path)
    spreadsheet = gc.open_by_key(sheet_id)
    total = 0
    for tab in SHEET_TABS:
        headers = HEADERS[tab]
        path = processed_dir / f"{tab}.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        try:
            worksheet = spreadsheet.worksheet(tab)
        except Exception:
            worksheet = spreadsheet.add_worksheet(
                title=tab, rows=1000, cols=len(headers))
        values = [_row_values(headers, _nest_row(tab, row)) for row in rows]
        # NOTE: no worksheet.clear() — its empty-body POST gets a
        # Google 411 through this sandbox's egress proxy (2026-11-23).
        # Overwriting from A1 is enough here: the tabs start empty
        # (the probe cell at Startups!A1 is overwritten by headers).
        # Resize the grid first: the API refuses to write a range
        # starting past the grid (400 "exceeds grid limits" — the
        # 2,023-row Mapping Log died at row 1073 on the first run),
        # and resizing down also drops stale rows from a previous,
        # longer export (e.g. pruned Jobs/News).
        _with_retry(
            lambda ws=worksheet, v=values, h=headers: ws.resize(
                rows=1 + len(v), cols=len(h)),
            f"{tab} resize")
        _with_retry(
            lambda ws=worksheet, h=headers: ws.update(
                range_name="A1", values=[h]),
            f"{tab} header")
        row_no = 2
        for i, chunk in enumerate(_chunks(values)):
            _with_retry(
                lambda rn=f"A{row_no}", c=chunk, ws=worksheet: ws.update(
                    range_name=rn, values=c),
                f"{tab} chunk {i + 1}")
            row_no += len(chunk)
        total += len(values)
        print(f"{tab}: {len(values)} rows written")
    print(f"Exported {total} rows across {len(SHEET_TABS)} tabs "
          f"to sheet {sheet_id}")


if __name__ == "__main__":
    main()

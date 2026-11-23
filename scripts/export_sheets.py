"""Export stored records to the Google Sheet deliverable
(tabs: Startups, Products, Research Papers, Jobs, News).

Reads processed records from data/processed/<Tab>.json — one JSON
list per tab, as written by the pipeline — and calls
src/export/sheets.py:export_all().

Usage: GOOGLE_SHEET_ID=... python scripts/export_sheets.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config  # noqa: F401  (loads .env into os.environ)
from src.export.sheets import SHEET_TABS, export_all


def main():
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        raise SystemExit("set GOOGLE_SHEET_ID in .env (see .env.example)")
    credentials_path = os.getenv(
        "GOOGLE_SHEETS_CREDENTIALS_JSON", "credentials.json")
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    rows: dict[str, list[dict]] = {}
    for tab in SHEET_TABS:
        path = processed_dir / f"{tab}.json"
        rows[tab] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    export_all(sheet_id, rows, credentials_path)
    total = sum(len(v) for v in rows.values())
    print(f"Exported {total} rows across {len(SHEET_TABS)} tabs to sheet {sheet_id}")


if __name__ == "__main__":
    main()

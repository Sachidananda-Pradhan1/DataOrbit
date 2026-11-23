"""Regenerate the fallback CSVs — one file per Sheet tab, with exactly
the content the exporter writes (same headers, nesting, flattening
and 50k-char truncation, via src/export/sheets.py).

The CSVs live in ~/workspace/your_files/DataOrbit-sheets/ and are the
manual fallback: if the Sheets API is unreachable, they can be loaded
into the Sheet via File > Import, one per tab.

Usage: python scripts/export_csvs.py [output-dir]
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export.sheets import HEADERS, SHEET_TABS, _nest_row, _row_values

DEFAULT_OUT = Path.home() / "workspace" / "your_files" / "DataOrbit-sheets"


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    total = 0
    for tab in SHEET_TABS:
        path = processed_dir / f"{tab}.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        headers = HEADERS[tab]
        with open(out_dir / f"{tab}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(_row_values(headers, _nest_row(tab, row)))
        total += len(rows)
        print(f"{tab}.csv: {len(rows)} rows")
    print(f"Wrote {total} rows across {len(SHEET_TABS)} CSVs to {out_dir}")


if __name__ == "__main__":
    main()

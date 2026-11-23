"""Google Sheets export (deliverable 1). Exactly the 6 tabs the task
requires — Startups, Products, Research Papers, Jobs, News and the
Entity Mapping Log, which records raw vs canonical name +
confidence for every resolved entity (see resolver/matcher.py).

Headers use the spec field paths where the spec defines them
(Startups: content.entityName / content.data.employeeCount;
Products: content.startupName / content.productName /
content.pricingModel) and every tab carries schemaVersion +
recordType. The JSON files on disk stay flat — _nest_row() builds
the nested row at export time only.

Wired 2026-11-21 with gspread (imported lazily, so the module loads
without credentials or the SDK installed). Needs a service-account
credentials.json and GOOGLE_SHEET_ID in .env — see .env.example.
Call from scripts/export_sheets.py.
"""
from __future__ import annotations

SHEET_TABS = ["Startups", "Products", "Research Papers", "Jobs", "News",
              "Entity Mapping Log"]

# Google Sheets rejects any cell over 50,000 characters — and fails
# the WHOLE update, so one long News full_text would lose the tab.
# Over-long cells are truncated to the limit with an ellipsis.
MAX_CELL_CHARS = 50000

STARTUP_HEADERS = ["schemaVersion", "recordType", "content.entityName",
                   "content.data.employeeCount", "source.name", "source.url",
                   "collectedAt"]
PRODUCT_HEADERS = ["schemaVersion", "recordType", "content.startupName",
                   "content.productName", "content.pricingModel",
                   "source.name", "source.url", "collectedAt"]
PAPER_HEADERS = ["schemaVersion", "recordType", "title", "authors",
                 "paper_url", "github_url", "github_stars", "published_date"]
JOB_HEADERS = ["schemaVersion", "recordType", "company", "title", "date",
               "is_remote", "role_family", "region", "source_url"]
NEWS_HEADERS = ["schemaVersion", "recordType", "title", "source.name",
                "source.url", "published_date", "full_text"]
# Entity Mapping Log: one row per resolved entity. confidence is a
# 0.0-1.0 score; match_method is domain / exact / fuzzy / unmatched.
MAPPING_HEADERS = ["schemaVersion", "recordType", "raw_name",
                   "canonical_name", "confidence", "match_method",
                   "source_url"]

HEADERS = {
    "Startups": STARTUP_HEADERS,
    "Products": PRODUCT_HEADERS,
    "Research Papers": PAPER_HEADERS,
    "Jobs": JOB_HEADERS,
    "News": NEWS_HEADERS,
    "Entity Mapping Log": MAPPING_HEADERS,
}

# recordType per tab, used when a stored record predates the field.
_RECORD_TYPES = {
    "Startups": "STARTUP",
    "Products": "PRODUCT",
    "Research Papers": "RESEARCH_PAPER",
    "Jobs": "JOB",
    "News": "NEWS",
    "Entity Mapping Log": "ENTITY_MAPPING",
}


def _nest_row(tab: str, row: dict) -> dict:
    """Shape a flat stored record into the nested row the headers
    describe: schemaVersion/recordType at the top level, spec
    content.* fields under "content" (Startups/Products only).
    Export-time only — the JSON files on disk stay flat, and `row`
    is never mutated. Missing schemaVersion/recordType get the
    model defaults, so records stored before those fields existed
    still export complete rows."""
    nested = dict(row)
    nested["schemaVersion"] = row.get("schemaVersion") or "1.0.0"
    nested["recordType"] = row.get("recordType") or _RECORD_TYPES[tab]
    if tab == "Startups":
        nested["content"] = {
            "entityName": row.get("entityName"),
            "data": {"employeeCount": row.get("employeeCount")},
        }
    elif tab == "Products":
        nested["content"] = {
            "startupName": row.get("startupName"),
            "productName": row.get("productName"),
            "pricingModel": row.get("pricingModel"),
        }
    return nested


def _row_values(headers: list[str], row: dict) -> list:
    """Flatten one record to a header-ordered row. Dotted headers
    (e.g. ``content.entityName``) walk nested dicts; lists join with
    '; '. String values over MAX_CELL_CHARS are truncated to the
    limit with "..." — Google Sheets fails the whole update on a
    single over-long cell."""
    values: list = []
    for header in headers:
        value: object = row
        for part in header.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, list):
            value = "; ".join(str(v) for v in value)
        if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
            value = value[:MAX_CELL_CHARS - 3] + "..."
        values.append("" if value is None else value)
    return values


def export_all(sheet_id: str, rows: dict[str, list[dict]],
               credentials_path: str = "credentials.json") -> None:
    """Write every tab: header row + one row per record. Missing tabs
    are created; existing tabs are cleared first so a re-export never
    leaves stale rows behind. ``rows`` is keyed by tab name; each
    record is nested (see _nest_row) before flattening."""
    import gspread

    gc = gspread.service_account(filename=credentials_path)
    spreadsheet = gc.open_by_key(sheet_id)
    for tab in SHEET_TABS:
        headers = HEADERS[tab]
        try:
            worksheet = spreadsheet.worksheet(tab)
        except Exception:
            worksheet = spreadsheet.add_worksheet(title=tab, rows=1000, cols=len(headers))
        values = [headers] + [_row_values(headers, _nest_row(tab, row))
                              for row in rows.get(tab, [])]
        worksheet.clear()
        worksheet.update(range_name="A1", values=values)

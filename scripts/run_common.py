"""Shared helpers for the runner entry points (run_bulk.py,
run_fresh.py). These were duplicated verbatim between the two
scripts; they live here now so a fix lands in one place.

Pure helpers are unit-tested in tests/test_scripts.py (via the
runners, which re-export them) and tests/test_runner_merge_retry.py.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.storage import db as db_module


def extract_domain(url: str) -> str:
    """Domain of a URL, lowercased, with any leading 'www.' stripped.
    Returns "" for an empty/unparseable URL."""
    netloc = urlparse(url or "").netloc.lower()
    netloc = netloc.removeprefix("www.")
    return netloc


def resolve_entity(name: str, domain: str | None = None) -> tuple[str, float, str]:
    """Resolve a raw entity name to (canonical, confidence, method).

    Prefers matcher.resolve_with_confidence(name, domain=...) when it
    exists; falls back to matcher.resolve(name) with confidence 1.0 /
    method "exact" on a match, else 0.0 / "unmatched". Unmatched
    names pass through unchanged — never invent a canonical form.
    """
    matcher = importlib.import_module("src.resolver.matcher")
    rwc = getattr(matcher, "resolve_with_confidence", None)
    if rwc is not None:
        try:
            return rwc(name, domain=domain)
        except TypeError:
            return rwc(name)
    canonical = matcher.resolve(name)
    if canonical != name.strip():
        return canonical, 1.0, "exact"
    return canonical, 0.0, "unmatched"


def mapping_entry(raw_name: str, canonical: str, confidence: float,
                  method: str, source_url: str) -> dict:
    return {
        "raw_name": raw_name,
        "canonical_name": canonical,
        "confidence": confidence,
        "match_method": method,
        "source_url": source_url,
    }


def merge_mapping_log(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge mapping-log entries, deduping on (raw_name, source_url).
    Existing entries win on conflict; order is preserved."""
    seen = {(e.get("raw_name"), e.get("source_url")) for e in existing}
    merged = list(existing)
    for entry in new:
        key = (entry.get("raw_name"), entry.get("source_url"))
        if key not in seen:
            seen.add(key)
            merged.append(entry)
    return merged


def record_url(record: dict) -> str | None:
    """The canonical URL of a dumped record, for JSON merge dedup.
    Tries record["url"], record["source"]["url"],
    record["paper_url"], record["source_url"] in order. Jobs without
    a source URL fall back to the stable ``job:<company>:<date>``
    key (the same fallback run_fresh uses for dedup). Returns None
    when the record carries no URL at all."""
    url = record.get("url")
    if url:
        return url
    source = record.get("source")
    if isinstance(source, dict):
        url = source.get("url")
        if url:
            return url
    url = record.get("paper_url") or record.get("source_url")
    if url:
        return url
    company = record.get("company")
    date = record.get("date")
    if company and date:
        return f"job:{company}:{date}"
    return None


def merge_json_records(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge record lists, deduping on record_url().

    NEW records win on URL clash (a re-run's data is fresher), but
    the merged order keeps each URL at its first-seen position.
    Records with no URL at all cannot be deduped and are always
    kept. This is what stops a re-run wiping output: the JSON on
    disk is merged with the newly claimed records instead of
    overwritten with only the new ones (a second collect returns 0
    new items, but the first run's 1,000 must stay)."""
    merged = list(existing)
    index: dict[str, int] = {}
    for i, record in enumerate(merged):
        key = record_url(record)
        if key is not None and key not in index:
            index[key] = i
    for record in new:
        key = record_url(record)
        if key is None:
            merged.append(record)
        elif key in index:
            merged[index[key]] = record  # new wins, position kept
        else:
            index[key] = len(merged)
            merged.append(record)
    return merged


def merge_records(existing: list[dict], new: list[dict],
                   key_fn) -> list[dict]:
    """Merge record lists, deduping on key_fn(record).
    Existing records win on conflict; order is preserved.

    Kept for backwards compatibility (tests/test_scripts.py);
    new code should use merge_json_records, which derives the key
    from the record itself and lets fresher records win."""
    seen = {key_fn(e) for e in existing}
    merged = list(existing)
    for record in new:
        key = key_fn(record)
        if key not in seen:
            seen.add(key)
            merged.append(record)
    return merged


def _record_key(tab: str):
    """Dedup key per tab: source URL for startups/products,
    paper_url for papers. (Backwards compatibility for
    merge_records; merge_json_records supersedes it.)"""
    if tab == "Research Papers":
        return lambda r: r.get("paper_url")
    return lambda r: (r.get("source") or {}).get("url")


def write_json(path: Path, records: list[dict]) -> Path:
    """Write a JSON list to `path` (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json_list(path: Path) -> list[dict]:
    """Read a JSON list; [] when the file does not exist yet."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def persist_to_db(payloads: dict[str, list[dict]]) -> bool:
    """Upsert per-table records ({"unique_url", "payload"}) to Postgres.
    Returns True on success; on ANY failure prints a warning and
    returns False — JSON output is the deliverable path, so the DB
    must never crash a run (it may simply not be running)."""
    try:
        db_module.init_db(settings.database_url)
        for table, records in payloads.items():
            if records:
                db_module.store_records(settings.database_url, table, records)
    except Exception as exc:
        print(f"WARNING: Postgres unavailable ({exc}); continuing JSON-only.")
        return False
    return True


def import_optional(module_name: str, attr: str):
    """Import `attr` from `module_name`; return None (with a clear
    error printed) instead of crashing when the module or attribute
    is not implemented yet."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        print(f"ERROR: cannot import {module_name}: {exc}")
        return None
    obj = getattr(module, attr, None)
    if obj is None:
        print(f"ERROR: {module_name}.{attr} is not implemented yet — "
              "skipping this step.")
    return obj


def llm_concurrency() -> int:
    """Max concurrent LLM extractions (MAX_LLM_CONCURRENCY env var,
    default 5). Guards the asyncio.to_thread calls in the runners so
    a bulk run cannot fan out unbounded LLM work."""
    try:
        return max(1, int(os.getenv("MAX_LLM_CONCURRENCY", "5")))
    except ValueError:
        return 5

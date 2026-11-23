"""Re-extract records from saved raw payloads — no re-fetching.

Reads data/raw/<kind>/<url-hash>.json (written raw-first by
scripts/run_bulk.py via crawlers/base.py save_raw, each file holding
{"url", "fetched_at", "body"}) and re-runs the crawlers' parse() on
them. This is the cheap recovery path the architecture promises: a
failed or improved extraction is re-run from disk instead of
re-fetching the source pages.

Scope: startups and products only — those are the kinds run_bulk
saves raw. Papers, jobs and news raw payloads are not persisted by
the runners, so there is nothing to re-extract for them (their
sources are APIs/RSS and are cheap to re-fetch).

Behaviour mirrors run_bulk.collect_entities: the AI-relevance
filter is applied before parse (a page the bulk run filtered out
stays filtered out), entity names are resolved deterministically,
and results are merged into data/processed/<Tab>.json — with one
deliberate difference from run_bulk.merge_records: re-extracted
records REPLACE existing records with the same source URL, because
re-extraction exists to fix what is on disk. The mapping log is
merged the same way run_bulk does it (existing entries win).

No network, no Postgres writes: re-run export_sheets.py afterwards
to push the merged JSON to the Sheet. parse() itself may use its
LLM fallback for pages with no deterministically extractable name
— exactly as in the bulk run — so with LLM keys configured this
script can make LLM calls; without keys the fallback yields [].

Usage:
    python scripts/re_extract.py [--kind startups|products|all]
                                 [--limit N]
                                 [--raw-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.resolver.matcher import resolve_with_confidence

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# kind -> (crawler class path, Sheet tab, entity-name field)
KINDS = {
    "startups": ("src.crawlers.startups.StartupCrawler", "Startups",
                 "entityName"),
    "products": ("src.crawlers.products.ProductCrawler", "Products",
                 "startupName"),
}


def _load_class(dotted: str):
    import importlib

    module_name, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _extract_domain(url: str) -> str:
    netloc = urlparse(url or "").netloc.lower()
    return netloc.removeprefix("www.")


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_records_new_wins(existing: list[dict],
                            new: list[dict]) -> list[dict]:
    """Merge on source URL; NEW records replace existing ones with
    the same URL (re-extraction fixes what is on disk), and new URLs
    are appended. Order of surviving records is preserved."""
    by_url = {}
    order: list[str] = []
    for record in existing:
        url = (record.get("source") or {}).get("url")
        if url not in by_url:
            order.append(url)
        by_url[url] = record
    for record in new:
        url = (record.get("source") or {}).get("url")
        if url not in by_url:
            order.append(url)
        by_url[url] = record
    return [by_url[url] for url in order]


def _merge_mapping_log(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {(e.get("raw_name"), e.get("source_url")) for e in existing}
    merged = list(existing)
    for entry in new:
        key = (entry.get("raw_name"), entry.get("source_url"))
        if key not in seen:
            seen.add(key)
            merged.append(entry)
    return merged


def _load_raw_files(raw_dir: Path, kind: str) -> list[tuple[str, str]]:
    """(url, body) pairs from data/raw/<kind>/*.json. Malformed
    files are skipped with a warning, never fatal."""
    files = sorted((raw_dir / kind).glob("*.json")) if (raw_dir / kind).exists() else []
    pairs = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            url, body = data["url"], data["body"]
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            print(f"WARNING: skipping unreadable raw file {path}: {exc}")
            continue
        pairs.append((url, body))
    return pairs


async def re_extract_kind(kind: str, raw_dir: Path, out_dir: Path,
                          limit: int, stats: dict) -> None:
    dotted, tab, name_field = KINDS[kind]
    crawler = _load_class(dotted)()
    relevance = getattr(crawler, "is_ai_relevant_page", None)
    records: list[dict] = []
    mapping: list[dict] = []
    pairs = _load_raw_files(raw_dir, kind)
    stats[f"{kind}_raw_files"] = len(pairs)
    for url, body in pairs:
        if limit and len(records) >= limit:
            break
        if relevance is not None and not relevance(body):
            stats[f"{kind}_filtered_non_ai"] = (
                stats.get(f"{kind}_filtered_non_ai", 0) + 1)
            continue
        try:
            parsed = list(await crawler.parse(url, body) or [])
        except Exception as exc:
            stats["parse_errors"] += 1
            print(f"WARNING: parse failed for {url}: {exc}")
            continue
        for record in parsed:
            raw_name = getattr(record, name_field)
            canonical, confidence, method = resolve_with_confidence(
                raw_name, domain=_extract_domain(url))
            setattr(record, name_field, canonical)
            mapping.append({
                "raw_name": raw_name,
                "canonical_name": canonical,
                "confidence": confidence,
                "match_method": method,
                "source_url": url,
            })
            records.append(record.model_dump(mode="json"))
    existing = _load_json_list(out_dir / f"{tab}.json")
    merged = _merge_records_new_wins(existing, records)
    _write_json(out_dir / f"{tab}.json", merged)
    existing_mapping = _load_json_list(out_dir / "Entity Mapping Log.json")
    _write_json(out_dir / "Entity Mapping Log.json",
                _merge_mapping_log(existing_mapping, mapping))
    stats[f"{kind}_re_extracted"] = len(records)
    stats[f"{kind}_total"] = len(merged)
    print(f"{kind}: re-extracted {len(records)} records "
          f"({stats.get(f'{kind}_filtered_non_ai', 0)} filtered as non-AI); "
          f"{tab}.json now holds {len(merged)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-extract startups/products from data/raw/ "
                    "(no re-fetching).")
    parser.add_argument("--kind", choices=["startups", "products", "all"],
                        default="all")
    parser.add_argument("--limit", type=int, default=0,
                        help="max records to re-extract per kind "
                             "(default: 0 = all)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args(argv)


async def main_async(args) -> dict:
    stats: dict = {"parse_errors": 0}
    kinds = ["startups", "products"] if args.kind == "all" else [args.kind]
    for kind in kinds:
        await re_extract_kind(kind, args.raw_dir, args.out_dir,
                              args.limit, stats)
    print(f"Parse errors: {stats['parse_errors']}")
    print(f"Wrote JSON to {args.out_dir}")
    return stats


def main(argv=None):
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()

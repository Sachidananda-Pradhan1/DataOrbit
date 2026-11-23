"""Phase II entry point: fresh signal ingestion — jobs and news,
ALL published within the last 24 hours, full text.

Freshness / idempotency guarantee: every item is claimed via
storage/dedup.claim(url) and marked completed once stored — a URL
seen before is NEVER reprocessed, so no article or job is ingested
twice, even across distributed crawler nodes (the idempotency key
is url:<sha256>). JSON output is merged with what is already on
disk (run_common merge_json_records), so a re-run never wipes the
first run's records.

STALENESS WARNING: run this within 24h of submission. A run now
produces data that is stale by Monday — the real submission run
happens Sunday evening / Monday, close to the 19:00 IST deadline.

Usage: python scripts/run_fresh.py [--limit N] [--no-db] [--smoke]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_common import (
    extract_domain,
    import_optional,
    load_json_list,
    mapping_entry,
    merge_json_records,
    merge_mapping_log,
    merge_records,  # noqa: F401  (re-exported; tested via run_fresh)
    persist_to_db,
    resolve_entity,
    write_json,
)

from src.storage.dedup import claim, complete
from src.utils.logging import event

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MAPPING_LOG = PROCESSED_DIR / "Entity Mapping Log.json"


# --- collection ---

def collect_fresh(news_items, job_items) -> dict:
    """Dedup + resolve one batch of fresh items.

    News: no entity resolution (title kept as-is). Jobs: company
    names resolved via the deterministic matcher, with a mapping-log
    entry per company (including unmatched). A URL seen before is
    NEVER reprocessed — this is the freshness/idempotency guarantee.

    Pure apart from the dedup claim (claim/complete); callers/tests
    reset dedup._seen between independent runs. Returns
    {"news", "jobs", "mapping", "stats"} with dumped records.
    """
    stats: dict = {"news_fresh": 0, "jobs_fresh": 0, "duplicates": 0,
                   "news_by_source": {}, "jobs_by_board": {}}
    news_out: list[dict] = []
    jobs_out: list[dict] = []
    mapping: list[dict] = []
    for item in news_items or []:
        url = str(item.url)
        owner = claim(url)
        if owner is None:
            stats["duplicates"] += 1
            continue
        stats["news_fresh"] += 1
        source = item.source.name
        stats["news_by_source"][source] = stats["news_by_source"].get(source, 0) + 1
        news_out.append(item.model_dump(mode="json"))
        complete(url, owner)
    for job in job_items or []:
        source_url = str(job.source_url) if job.source_url else ""
        # Jobs without a source URL still need an idempotency key;
        # company + publication date is the stable fallback.
        dedup_key = source_url or f"job:{job.company}:{job.date.isoformat()}"
        owner = claim(dedup_key)
        if owner is None:
            stats["duplicates"] += 1
            continue
        board = extract_domain(source_url) if source_url else ""
        # Prefer the job's own company website domain for
        # resolution when the record carries one — the source URL
        # is the job board page, which resolves every company to
        # the board's domain. Board stats still use the board.
        website = getattr(job, "website", None)
        domain = (extract_domain(str(website)) if website else board)
        raw_company = job.company
        canonical, confidence, method = resolve_entity(
            raw_company, domain or None)
        job.company = canonical
        mapping.append(mapping_entry(raw_company, canonical, confidence,
                                     method, source_url))
        stats["jobs_fresh"] += 1
        board_key = board or "unknown"
        stats["jobs_by_board"][board_key] = stats["jobs_by_board"].get(board_key, 0) + 1
        jobs_out.append(job.model_dump(mode="json"))
        complete(dedup_key, owner)
    return {"news": news_out, "jobs": jobs_out, "mapping": mapping,
            "stats": stats}


# --- CLI / orchestration ---

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase II: ingest jobs + news from the last 24h only.")
    parser.add_argument("--limit", type=int, default=None,
                        help="per-board job limit (default: all)")
    parser.add_argument("--no-db", action="store_true",
                        help="skip Postgres; JSON only")
    parser.add_argument("--smoke", action="store_true",
                        help="small test run: --limit 5, JSON only, "
                             "per-source counts")
    return parser.parse_args(argv)


def resolve_smoke(args) -> tuple[int | None, bool]:
    """--smoke implies a 5-per-board, JSON-only run.
    Returns (limit, use_db)."""
    if args.smoke:
        return 5, False
    return args.limit, not args.no_db


async def main_async(args) -> dict:
    limit, use_db = resolve_smoke(args)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    event("fresh_run_start", limit=limit, db_enabled=use_db,
          smoke=args.smoke)

    fetch_all_news = import_optional("src.crawlers.news", "fetch_all_news")
    fetch_all_jobs = import_optional("src.crawlers.jobs", "fetch_all_jobs")
    news_items = await fetch_all_news() if fetch_all_news is not None else []
    job_items = (await fetch_all_jobs(limit_per_board=limit)
                 if fetch_all_jobs is not None else [])

    result = collect_fresh(news_items, job_items)
    stats = result["stats"]

    # Merge with the JSON already on disk — a second run claims 0
    # new URLs (dedup), so overwriting would leave empty tabs.
    existing_jobs = load_json_list(PROCESSED_DIR / "Jobs.json")
    existing_news = load_json_list(PROCESSED_DIR / "News.json")
    write_json(PROCESSED_DIR / "Jobs.json",
               merge_json_records(existing_jobs, result["jobs"]))
    write_json(PROCESSED_DIR / "News.json",
               merge_json_records(existing_news, result["news"]))
    existing = load_json_list(MAPPING_LOG)
    merged = merge_mapping_log(existing, result["mapping"])
    write_json(MAPPING_LOG, merged)
    added = len(merged) - len(existing)

    db_ok = None
    if use_db:
        payloads = {
            "jobs": [{"unique_url": r.get("source_url")
                      or f"job:{r['company']}:{r['date']}", "payload": r}
                     for r in result["jobs"]],
            "news": [{"unique_url": r["url"], "payload": r}
                     for r in result["news"]],
        }
        db_ok = persist_to_db(payloads)

    event("fresh_run_end", jobs_fresh=stats["jobs_fresh"],
          news_fresh=stats["news_fresh"],
          duplicates=stats["duplicates"],
          mapping_entries_added=added,
          mapping_entries_total=len(merged))

    print(f"Fresh jobs: {stats['jobs_fresh']} "
          f"(per board: {stats['jobs_by_board'] or 'none'})")
    print(f"Fresh news: {stats['news_fresh']} "
          f"(per source: {stats['news_by_source'] or 'none'})")
    print(f"Duplicates skipped (already seen): {stats['duplicates']}")
    print(f"Mapping-log entries added: {added} "
          f"(total in log: {len(merged)})")
    print(f"Postgres: {'skipped' if db_ok is None else ('ok' if db_ok else 'unavailable, JSON-only')}")
    print(f"Wrote JSON to {PROCESSED_DIR}")
    return stats


def main(argv=None):
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()

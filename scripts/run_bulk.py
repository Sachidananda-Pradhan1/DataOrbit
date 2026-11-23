"""Phase I entry point: bulk extraction — 1,000 startups, 1,000
products, 1,000 research papers (with GitHub stars).

Flow per entity kind: discover URLs (in growing rounds until the
target is met or the source is exhausted) -> stream-fetch raw in
bounded batches (saved to data/raw BEFORE any LLM call) ->
AI-relevance filter (crawler.is_ai_relevant_page; rejected pages
never reach parse or the LLM fallback and never count toward the
cap) -> dedup claim on the page URL (the lease covers the whole
processing of the page, including the LLM call) -> deterministic
parse -> LLM extraction only when parse yields nothing (off the
event loop via asyncio.to_thread, behind an LLM semaphore) ->
deterministic entity resolution (one mapping-log entry per resolved
entity, including unmatched) -> JSON + Postgres.

Failed fetches are NOT non-AI pages: a URL that fails after
retries was never examined, is counted separately
(<kind>_fetch_failed), is never dedup-claimed, and is re-queued
for a later discovery round (up to FETCH_RETRY_ROUNDS attempts).

Resumable by design: storage/dedup claims each source URL on a
10-minute lease (renewed with heartbeat() before the slow LLM
step) and marks it completed once stored, and Postgres upserts on
unique_url, so re-runs never duplicate records; a crashed worker's
lease expires and the URL becomes claimable again, and a failed
extraction releases the claim with backoff (fail()) instead of
losing the record. A failed extraction yields no record, never a
fabricated one — every record traces to a real source URL. JSON
output is merged with what is already on disk (run_common
merge_json_records), so a re-run never wipes the first run's
records.

Usage: python scripts/run_bulk.py [--limit N] [--no-db] [--no-llm]
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_common import (
    _record_key,  # noqa: F401  (re-exported; tested via run_bulk)
    db_module,  # noqa: F401  (re-exported; tests patch run_bulk.db_module)
    extract_domain,
    import_optional,
    llm_concurrency,
    load_json_list,
    mapping_entry,
    merge_json_records,
    merge_mapping_log,
    merge_records,  # noqa: F401  (re-exported; tested via run_bulk)
    persist_to_db,
    resolve_entity,
    write_json,
)

from src.config import TARGETS
from src.crawlers.base import save_raw
from src.storage.dedup import claim, complete, fail, heartbeat
from src.utils.logging import event

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# Total fetch attempts per URL across discovery rounds: a failed
# fetch is re-queued for a later round, but not forever.
FETCH_RETRY_ROUNDS = 2


# --- collection ---

async def discover_urls(crawler, limit: int) -> list[str]:
    """Call crawler.discover(limit=...) when it accepts a limit,
    else discover() and slice — tolerant of both signatures."""
    try:
        params = inspect.signature(crawler.discover).parameters
    except (TypeError, ValueError):
        params = {}
    if "limit" in params:
        urls = await crawler.discover(limit=limit)
    else:
        urls = await crawler.discover()
    return list(urls or [])[:limit]


async def collect_entities(kind: str, crawler, extractor, *, limit: int,
                           llm_enabled: bool, name_field: str,
                           stats: dict) -> tuple[list[dict], list[dict]]:
    """Discover/fetch/parse/resolve one entity kind (startups or
    products). Returns (dumped records, mapping-log entries).

    AI-relevance is applied BEFORE the cap: a page the crawler's
    is_ai_relevant_page() rejects never reaches parse (or the LLM
    fallback) and never counts toward `limit`. Discovery runs in
    growing rounds — doubling the discover limit each round — until
    `limit` relevant records are collected or the source is
    exhausted, in which case a shortfall warning is printed and
    stats["<kind>_shortfall"] records how many are missing.
    Crawlers without is_ai_relevant_page() are treated as
    unfiltered (backwards-compatible).

    Fetching streams in bounded batches (fetch_many_stream), so
    only one batch of pages is held in memory at a time. A URL
    whose fetch fails after retries is counted as
    <kind>_fetch_failed — never as filtered_non_ai — and is
    re-queued for a later discovery round instead of being
    dedup-claimed or discarded.

    The dedup claim is taken on the page URL BEFORE parse/LLM, so
    the lease spans the slow extraction step (renewed with
    heartbeat() before the LLM call) and a re-run never re-parses
    or re-extracts a claimed page. The claim key is the page URL,
    which is also every record's source URL (parse and the LLM
    extractors both set source.url to the fetched URL), so this is
    the same idempotency key the old per-record claim used.
    """
    fetch_stream = import_optional("src.crawlers.fetcher", "fetch_many_stream")
    if fetch_stream is None:
        return [], []
    relevance = getattr(crawler, "is_ai_relevant_page", None)
    records: list[dict] = []
    mapping: list[dict] = []
    seen_urls: set[str] = set()
    fetch_attempts: dict[str, int] = {}
    llm_sem = asyncio.Semaphore(llm_concurrency())
    discover_limit = limit
    while len(records) < limit:
        urls = await discover_urls(crawler, discover_limit)
        new_urls = [u for u in urls if u not in seen_urls]
        if not new_urls:
            break  # source exhausted (or discover stopped growing)
        seen_urls.update(new_urls)
        stats[f"{kind}_discovered"] = len(seen_urls)
        failures: set[str] = set()
        async for url, raw in fetch_stream(new_urls, failures=failures):
            if len(records) >= limit:
                break
            if url in failures:
                # Fetch failed after retries: the page was never
                # examined, so it is not non-AI and is never
                # claimed — counted and re-queued below.
                continue
            stats[f"{kind}_fetched"] = stats.get(f"{kind}_fetched", 0) + 1
            # Raw-first: persist before any LLM call, so extraction can
            # be re-run cheaply and every row traces to its source.
            save_raw(kind, url, raw)
            if relevance is not None and not relevance(raw):
                stats[f"{kind}_filtered_non_ai"] = (
                    stats.get(f"{kind}_filtered_non_ai", 0) + 1)
                continue
            owner = claim(url)
            if owner is None:
                stats["duplicates"] += 1
                continue
            try:
                parsed = list(await crawler.parse(url, raw) or [])
            except Exception as exc:
                stats["parse_errors"] += 1
                print(f"WARNING: parse failed for {url}: {exc}")
                parsed = []
            llm_errored = False
            llm_attempted = False
            llm_no_key = False
            if not parsed and llm_enabled and extractor is not None:
                llm_attempted = True
                # The lease is live while the LLM works; renew it
                # before the slow call so a long extraction cannot
                # lose the claim to another worker.
                heartbeat(url, owner)
                async with llm_sem:
                    try:
                        # Sync LLM call: off the event loop so one
                        # extraction cannot stall the whole crawl.
                        model = await asyncio.to_thread(
                            extractor, raw, extract_domain(url), url,
                            datetime.now(UTC))
                    except RuntimeError as exc:
                        # No LLM tiers configured (no keys) vs. all tiers failed.
                        if "not configured" in str(exc):
                            stats["llm_skipped_no_key"] += 1
                            llm_no_key = True
                        else:
                            stats["llm_errors"] += 1
                            llm_errored = True
                    except Exception as exc:
                        stats["llm_errors"] += 1
                        llm_errored = True
                        print(f"WARNING: LLM extraction failed for {url}: {exc}")
                    else:
                        if model is not None:
                            parsed = [model]
                            stats["llm_extracted"] += 1
            stored = len(records)
            for record in parsed:
                source_url = str(record.source.url)
                raw_name = getattr(record, name_field)
                # Prefer the record's own company website domain for
                # resolution — the source URL is the YC/BetaList
                # page, which would resolve every entity to
                # ycombinator.com / betalist.com.
                website = getattr(record, "website", None)
                domain = (extract_domain(str(website)) if website
                          else extract_domain(source_url))
                canonical, confidence, method = resolve_entity(
                    raw_name, domain)
                setattr(record, name_field, canonical)
                mapping.append(mapping_entry(raw_name, canonical, confidence,
                                             method, source_url))
                records.append(record.model_dump(mode="json"))
            if len(records) > stored:
                complete(url, owner)
            elif llm_errored:
                # Transient failure: release with backoff so a later
                # run retries instead of losing the record.
                fail(url, owner)
            elif llm_attempted and not llm_no_key:
                # Fully processed (parse + LLM) and nothing found:
                # never reprocess.
                complete(url, owner)
            # else: LLM never ran (--no-llm, no extractor, or no
            # keys) — leave the lease to expire so a later keyed
            # run retries this page.
        # Failed fetches: distinct from non-AI pages — re-queue for a
        # later discovery round (up to FETCH_RETRY_ROUNDS attempts);
        # after that they stay in seen_urls and are dropped.
        for failed_url in failures:
            stats[f"{kind}_fetch_failed"] = (
                stats.get(f"{kind}_fetch_failed", 0) + 1)
            fetch_attempts[failed_url] = fetch_attempts.get(failed_url, 0) + 1
            if fetch_attempts[failed_url] < FETCH_RETRY_ROUNDS:
                seen_urls.discard(failed_url)
        discover_limit *= 2
    if len(records) < limit:
        stats[f"{kind}_shortfall"] = limit - len(records)
        print(f"WARNING: {kind}: only {len(records)} AI-relevant records "
              f"collected (target {limit}) — source exhausted after "
              f"scanning {len(seen_urls)} URLs "
              f"({stats.get(f'{kind}_filtered_non_ai', 0)} filtered as non-AI, "
              f"{stats.get(f'{kind}_fetch_failed', 0)} fetch failures).")
    return records[:limit], mapping[:limit]


async def collect_papers(limit: int, stats: dict) -> list[dict]:
    """Fetch research papers; resolution does not apply to paper
    titles, so no mapping-log entries. Dedup by paper_url."""
    fetch_papers = import_optional("src.crawlers.papers", "fetch_papers")
    if fetch_papers is None:
        stats["papers_skipped_missing"] = True
        return []
    try:
        papers = await fetch_papers(target=limit)
    except TypeError:
        papers = await fetch_papers(limit)
    records: list[dict] = []
    for paper in papers or []:
        url = str(paper.paper_url)
        owner = claim(url)
        if owner is None:
            stats["duplicates"] += 1
            continue
        records.append(paper.model_dump(mode="json"))
        complete(url, owner)
    return records


# --- CLI / orchestration ---

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase I: bulk-extract startups, products, papers.")
    parser.add_argument("--limit", type=int, default=TARGETS["startups"],
                        help="per-entity limit (default: %(default)s)")
    parser.add_argument("--no-db", action="store_true",
                        help="skip Postgres; JSON only")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip LLM enrichment; deterministic parse only")
    return parser.parse_args(argv)


async def main_async(args) -> dict:
    stats: dict = {
        "duplicates": 0, "parse_errors": 0, "llm_extracted": 0,
        "llm_skipped_no_key": 0, "llm_errors": 0,
        "papers_skipped_missing": False,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    llm_enabled = not args.no_llm
    event("bulk_run_start", limit=args.limit, llm_enabled=llm_enabled,
          db_enabled=not args.no_db)
    extract_startup = extract_product = None
    if llm_enabled:
        extract_startup = import_optional("src.llm.extractor", "extract_startup")
        extract_product = import_optional("src.llm.extractor", "extract_product")

    all_mapping: list[dict] = []
    payloads: dict[str, list[dict]] = {}
    outputs: dict[str, list[dict]] = {}

    startup_cls = import_optional("src.crawlers.startups", "StartupCrawler")
    if startup_cls is not None:
        startups, mapping = await collect_entities(
            "startups", startup_cls(), extract_startup, limit=args.limit,
            llm_enabled=llm_enabled, name_field="entityName", stats=stats)
        outputs["Startups"] = startups
        all_mapping.extend(mapping)
        payloads["startups"] = [
            {"unique_url": r["source"]["url"], "payload": r} for r in startups]

    product_cls = import_optional("src.crawlers.products", "ProductCrawler")
    if product_cls is not None:
        products, mapping = await collect_entities(
            "products", product_cls(), extract_product, limit=args.limit,
            llm_enabled=llm_enabled, name_field="startupName", stats=stats)
        outputs["Products"] = products
        all_mapping.extend(mapping)
        payloads["products"] = [
            {"unique_url": r["source"]["url"], "payload": r} for r in products]

    papers = await collect_papers(args.limit, stats)
    outputs["Research Papers"] = papers
    payloads["papers"] = [
        {"unique_url": r["paper_url"], "payload": r} for r in papers]

    for tab, records in outputs.items():
        # Merge with the JSON already on disk — never overwrite with
        # only this run's newly claimed records (see
        # merge_json_records: new records win on URL clash).
        existing = load_json_list(PROCESSED_DIR / f"{tab}.json")
        write_json(PROCESSED_DIR / f"{tab}.json",
                   merge_json_records(existing, records))
    existing_mapping = load_json_list(PROCESSED_DIR / "Entity Mapping Log.json")
    write_json(PROCESSED_DIR / "Entity Mapping Log.json",
               merge_mapping_log(existing_mapping, all_mapping))

    db_ok = None
    if not args.no_db:
        db_ok = persist_to_db(payloads)

    fetch_failed = (stats.get("startups_fetch_failed", 0)
                    + stats.get("products_fetch_failed", 0))
    event("bulk_run_end",
          startups=len(outputs.get("Startups", [])),
          products=len(outputs.get("Products", [])),
          papers=len(papers),
          mapping_entries=len(all_mapping),
          duplicates=stats["duplicates"],
          llm_extracted=stats["llm_extracted"],
          llm_errors=stats["llm_errors"],
          parse_errors=stats["parse_errors"],
          fetch_failed=fetch_failed)

    print(f"Startups: {len(outputs.get('Startups', []))} "
          f"(discovered {stats.get('startups_discovered', 0)}, "
          f"fetched {stats.get('startups_fetched', 0)}, "
          f"filtered non-AI {stats.get('startups_filtered_non_ai', 0)}, "
          f"fetch failed {stats.get('startups_fetch_failed', 0)})")
    print(f"Products: {len(outputs.get('Products', []))} "
          f"(discovered {stats.get('products_discovered', 0)}, "
          f"fetched {stats.get('products_fetched', 0)}, "
          f"filtered non-AI {stats.get('products_filtered_non_ai', 0)}, "
          f"fetch failed {stats.get('products_fetch_failed', 0)})")
    print(f"Research Papers: {len(papers)}")
    print(f"Mapping-log entries: {len(all_mapping)}")
    print(f"Duplicates skipped: {stats['duplicates']}")
    print(f"LLM-extracted: {stats['llm_extracted']}; "
          f"skipped (no LLM key): {stats['llm_skipped_no_key']}; "
          f"LLM errors: {stats['llm_errors']}; "
          f"parse errors: {stats['parse_errors']}")
    if stats["papers_skipped_missing"]:
        print("WARNING: papers skipped — fetch_papers not implemented yet.")
    print(f"Postgres: {'skipped (--no-db)' if db_ok is None else ('ok' if db_ok else 'unavailable, JSON-only')}")
    print(f"Wrote JSON to {PROCESSED_DIR}")
    return stats


def main(argv=None):
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()

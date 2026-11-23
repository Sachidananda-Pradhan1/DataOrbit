"""Standalone GitHub star enrichment for Research Papers.json.

Fills github_stars for papers that have a github_url but no stars yet,
using the pipeline's own _enrich_github_stars (GITHUB_TOKEN from .env,
no lookup cap with a token). Idempotent: papers that already have stars
are skipped, so re-running is safe.

Written 2026-11-22 after the evening bulk run's enrichment fetched 0
(api.github.com timing out through the egress proxy) and the /tmp repair
script was lost to a /tmp cleanup — this copy lives in the repo.

Usage (from the repo root):
    NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
        .venv/bin/python scripts/enrich_stars.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401  (loads .env into os.environ)
from src.crawlers.papers import _enrich_github_stars
from src.models import ResearchPaper

PAPERS_JSON = (Path(__file__).resolve().parent.parent
               / "data" / "processed" / "Research Papers.json")


async def main() -> None:
    papers = [ResearchPaper.model_validate(row)
              for row in json.loads(PAPERS_JSON.read_text(encoding="utf-8"))]
    pending = sum(1 for p in papers
                  if p.github_url and p.github_stars is None)
    print(f"{len(papers)} papers, {pending} with a repo link "
          f"but no stars yet")
    summary = await _enrich_github_stars(papers)
    PAPERS_JSON.write_text(json.dumps(
        [p.model_dump(mode="json") for p in papers], indent=2), encoding="utf-8")
    print(f"saved {PAPERS_JSON} — {summary}")


if __name__ == "__main__":
    asyncio.run(main())

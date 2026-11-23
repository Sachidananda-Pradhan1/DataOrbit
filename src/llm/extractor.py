"""Structured extraction: LLM JSON -> Pydantic validation.

Validation is the hallucination guardrail: if the LLM output does
not match models.py, it is rejected (not patched up), and the raw
text + source URL are kept for a manual / re-run review.

413 guard: each LLM call carries at most MAX_CHARS_PER_CALL (~6,000
tokens). Long texts are split with chunker.chunk_text and extracted
per chunk; the per-chunk JSON dicts are then merged deterministically
(merge_extractions) and validated once, so facts near the end of a
long page are no longer lost to truncation. Short inputs that fit in
one chunk take exactly one call, as before.

HTML guard: crawlers hand this module raw pages, so _extract strips
HTML to text (chunker.strip_html_to_text) before chunking whenever
the input looks like markup. The LLM never sees tags.

Bad chunks are LOGGED, not silently skipped: a chunk whose reply
is not parseable JSON logs a warning (chunk index + error) and the
other chunks still contribute. If no chunk yields a usable dict,
or the merged dict fails validation, return None.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from src.llm.chunker import chunk_text, looks_like_html, strip_html_to_text
from src.llm.json_utils import parse_llm_json
from src.llm.orchestrator import extract_with_fallback
from src.llm.prompts import (
    JOB_PROMPT,
    NEWS_PROMPT,
    PAPER_PROMPT,
    PRODUCT_PROMPT,
    STARTUP_PROMPT,
)
from src.models import Job, NewsItem, Product, ResearchPaper, Startup

# Re-exported: crawlers/products.py imports parse_llm_json from here.
__all__ = [
    "extract_job",
    "extract_news",
    "extract_paper",
    "extract_product",
    "extract_startup",
    "merge_extractions",
    "parse_llm_json",
]

logger = logging.getLogger(__name__)


def _is_empty(value: Any) -> bool:
    """Empty = missing or blank. 0 and False are real values."""
    return value is None or value == "" or value == [] or value == {}


def _dedupe_key(value: Any) -> Any:
    """Deterministic dedup key for list items (dicts/lists are unhashable).

    Scalars are tagged with their type so True and 1 do not collide;
    dicts/lists are canonicalised via sorted-key JSON.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return (type(value).__name__, value)


def merge_extractions(parts: list[dict]) -> dict:
    """Merge per-chunk extraction dicts, deterministically.

    Rules:
    - keys are visited in sorted order, so the result never depends
      on dict insertion order;
    - lists are concatenated in chunk order with first-seen dedup;
    - nested dicts merge recursively;
    - scalars: the first non-empty value (in chunk order) wins;
      empty values (None, "", [], {}) never overwrite a real one.

    Same inputs -> same output: no set iteration order, no dict-order
    dependence (sets are only used for membership checks).
    """
    merged: dict = {}
    all_keys = sorted({key for part in parts for key in part})
    for key in all_keys:
        present = [part[key] for part in parts
                   if key in part and not _is_empty(part[key])]
        if not present:
            continue
        if all(isinstance(value, list) for value in present):
            seen: set = set()
            merged_list: list = []
            for value in present:
                for item in value:
                    dedupe_key = _dedupe_key(item)
                    if dedupe_key not in seen:
                        seen.add(dedupe_key)
                        merged_list.append(item)
            merged[key] = merged_list
        elif all(isinstance(value, dict) for value in present):
            merged[key] = merge_extractions(present)
        else:
            merged[key] = present[0]
    return merged


def _extract(prompt_template: str, model_cls, text: str, **fixed):
    """Strip HTML -> chunk -> extract per chunk -> merge -> validate.

    `fixed` are deterministic fields the caller already knows (source
    URL, publication date, ...). They override anything the LLM
    returns for the same key — the fetched record is authoritative,
    the LLM must never overwrite a known URL or date with a guess.

    A chunk whose response is not valid JSON logs a warning (chunk
    index + parse error) and is skipped; the other chunks still
    contribute. If no chunk yields a usable dict, or the merged dict
    fails validation, return None.
    """
    if looks_like_html(text):
        text = strip_html_to_text(text)
    parts: list[dict] = []
    for index, chunk in enumerate(chunk_text(text)):
        raw = extract_with_fallback(prompt_template.format(chunk=chunk))
        try:
            parts.append(parse_llm_json(raw))
        except ValueError as exc:
            logger.warning("chunk %d: unparseable LLM JSON, skipped: %s",
                           index, exc)
    if not parts:
        return None
    merged = merge_extractions(parts)
    merged.update(fixed)
    try:
        return model_cls(**merged)
    except ValidationError as exc:
        logger.warning("merged extraction failed %s validation: %s",
                       model_cls.__name__, exc)
        return None


def extract_startup(text: str, source_name: str, source_url: str,
                    collected_at) -> Startup | None:
    return _extract(STARTUP_PROMPT, Startup, text,
                    source={"name": source_name, "url": source_url},
                    collectedAt=collected_at)


def extract_product(text: str, source_name: str, source_url: str,
                    collected_at) -> Product | None:
    return _extract(PRODUCT_PROMPT, Product, text,
                    source={"name": source_name, "url": source_url},
                    collectedAt=collected_at)


def extract_paper(text: str, source_url: str,
                  published_date: datetime) -> ResearchPaper | None:
    """Extract a paper; the paper URL and publication date come from
    the feed (arXiv / Hugging Face) and are passed as fixed fields,
    never taken from the LLM."""
    return _extract(PAPER_PROMPT, ResearchPaper, text,
                    paper_url=source_url, published_date=published_date)


def extract_job(text: str, source_url: str) -> Job | None:
    """Extract a job posting; the listing URL is fixed (traceability),
    company/date/remote/role come from the LLM."""
    return _extract(JOB_PROMPT, Job, text, source_url=source_url)


def extract_news(text: str, source_name: str,
                 source_url: str) -> NewsItem | None:
    """Extract a news item; source + URL are fixed, title and
    published_date come from the LLM."""
    return _extract(NEWS_PROMPT, NewsItem, text,
                    source={"name": source_name, "url": source_url},
                    url=source_url)

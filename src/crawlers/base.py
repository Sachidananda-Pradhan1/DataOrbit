"""Shared crawler contract: discover -> fetch raw -> parse -> validate.

Raw-first: save the raw HTML/JSON before any LLM call, so extraction
can be re-run cheaply and every row traces to its source URL.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path("data/raw")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def save_raw(kind: str, url: str, text: str) -> Path:
    """Persist raw payload; returns its path. Filename is the URL hash,
    so re-fetching the same URL overwrites rather than duplicating."""
    d = RAW_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{content_hash(url)}.json"
    p.write_text(json.dumps({
        "url": url, "fetched_at": datetime.now(timezone.utc).isoformat(), "body": text,
    }, ensure_ascii=False), encoding="utf-8")
    return p


class BaseCrawler(ABC):
    kind: str = "base"

    @abstractmethod
    async def discover(self) -> list[str]:
        """Return seed URLs to fetch."""

    @abstractmethod
    async def parse(self, url: str, raw: str):
        """Turn raw text into validated Pydantic models (or [])."""

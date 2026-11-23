"""First-seen store: when an article or job carries no published date
anywhere — no feed date, no meta date — the honest fallback is the
time its URL was first observed by this pipeline, not a guess.

The store is a small JSON file (default
``data/processed/first_seen.json``) mapping URL -> ISO-8601 UTC
timestamp of first observation. ``observe(url, now)`` returns the
stored timestamp, recording ``now`` on first sight. Callers still
run the result through the same 24h freshness gate (``is_fresh``)
as any published date: a URL first seen more than 24h ago is stale
and its item is dropped. First-seen is a *recorded observation*,
never an invented date.

Single-process by design (the demo's runners are single-process);
concurrent writers are out of scope.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/processed/first_seen.json")


def _parse_timestamp(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        # Stored timestamps are UTC; treat a naive one as UTC rather
        # than as local time (see date_utils.parse_date).
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class FirstSeenStore:
    """JSON-backed URL -> first-observed-timestamp store.

    ``path=None`` keeps the store in memory only (used by tests and
    by callers that do not want persistence)."""

    def __init__(self, path: Path | str | None = DEFAULT_PATH):
        self.path = Path(path) if path is not None else None
        self._seen: dict[str, datetime] = {}
        if self.path is not None:
            self._load()

    @classmethod
    def default(cls) -> FirstSeenStore:
        """The production store at data/processed/first_seen.json."""
        return cls(DEFAULT_PATH)

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("first-seen store %s unreadable (%s); "
                           "starting empty", self.path, exc)
            return
        if not isinstance(raw, dict):
            logger.warning("first-seen store %s is not a JSON object; "
                           "starting empty", self.path)
            return
        for url, value in raw.items():
            parsed = _parse_timestamp(value) if isinstance(value, str) else None
            if parsed is not None:
                self._seen[url] = parsed

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {url: dt.isoformat() for url, dt in self._seen.items()},
                ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not save first-seen store %s: %s",
                           self.path, exc)

    def first_seen(self, url: str) -> datetime | None:
        """The recorded first-observation time, or None if unseen."""
        return self._seen.get(url)

    def observe(self, url: str, now: datetime | None = None) -> datetime:
        """Return the URL's first-seen timestamp, recording ``now``
        (default: current UTC time) on first sight."""
        now = now or datetime.now(timezone.utc)
        if url not in self._seen:
            self._seen[url] = now
            self.save()
        return self._seen[url]

    def __len__(self) -> int:
        return len(self._seen)

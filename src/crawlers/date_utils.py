"""Date parsing — worth 25% of the score (Data Quality: date parsing
accuracy, 24-hour freshness). Handles ISO-8601, RFC-822 (RSS) and
relative dates ('2 hours ago', 'yesterday').

All datetimes returned are timezone-aware UTC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import dateparser


def parse_date(text: str, *, now: datetime | None = None) -> datetime | None:
    if not text:
        return None
    now = now or datetime.now(timezone.utc)
    dt = dateparser.parse(
        text, settings={
            "RELATIVE_BASE": now,
            "RETURN_AS_TIMEZONE_AWARE": True,
            # Naive timestamps (no offset) are UTC, not local time:
            # without these, "2026-11-22 10:30:00" parses as 05:00 UTC
            # on an IST machine but 10:30 UTC on a UTC one — a 5.5h
            # error on the 24h freshness gate.
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        },
    )
    if dt is None:
        return None
    return dt.astimezone(timezone.utc)


def is_fresh(published: datetime, *, hours: int = 24,
             now: datetime | None = None) -> bool:
    """Phase II: published within the last `hours` (default 24)."""
    now = now or datetime.now(timezone.utc)
    return timedelta(0) <= (now - published) <= timedelta(hours=hours)

from datetime import datetime, timezone

from src.crawlers.date_utils import is_fresh, parse_date

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def test_parse_relative():
    assert parse_date("2 hours ago", now=NOW) is not None


def test_fresh_24h():
    assert is_fresh(datetime(2026, 11, 21, 0, 0, tzinfo=timezone.utc), now=NOW)
    assert not is_fresh(datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc), now=NOW)


def test_parse_naive_timestamp_is_utc_not_local():
    # Regression: "2026-11-22 10:30:00" parsed as 05:00 UTC on an
    # IST machine (local-time assumption), 10:30 UTC on a UTC one.
    assert parse_date("2026-11-22 10:30:00", now=NOW) == datetime(
        2026, 11, 22, 10, 30, tzinfo=timezone.utc)

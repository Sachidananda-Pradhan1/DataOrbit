"""Tests for src/storage/dedup.py — the lease-based dedup state
machine (pending -> processing -> completed / failed / dead).

No live Redis needed: the client is faked, and the fake implements
the same Lua-script semantics the module sends to real Redis
(dispatching on the script text). Time is faked too: tests
monkeypatch dedup._now with a controllable clock, so lease expiry
and backoff windows are exercised without sleeping.

Every state-machine test runs against BOTH backends (in-memory
fallback and fake Redis) via the `backend` fixture.
"""
import json

import pytest

from src.storage import dedup


@pytest.fixture(autouse=True)
def _clean():
    dedup.reset()
    yield
    dedup.reset()


class _Clock:
    """Controllable epoch time for dedup._now."""

    def __init__(self):
        self.now = 1_700_000_000.0

    def advance(self, secs):
        self.now += secs


class _FakeRedis:
    """Minimal Redis: get/set plus eval() dispatching on the Lua
    scripts dedup.py sends, mirroring their semantics in Python."""

    def __init__(self):
        self.keys = {}  # key -> JSON record string

    def ping(self):
        return True

    def get(self, key):
        return self.keys.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True

    def eval(self, script, numkeys, *args):
        if script == dedup._CLAIM_LUA:
            return self._eval_claim(*args)
        if script == dedup._HEARTBEAT_LUA:
            return self._eval_heartbeat(*args)
        if script == dedup._COMPLETE_LUA:
            return self._eval_complete(*args)
        if script == dedup._FAIL_LUA:
            return self._eval_fail(*args)
        raise ValueError("unknown script")

    def _record(self, key):
        raw = self.keys.get(key)
        return json.loads(raw) if raw is not None else None

    def _save(self, key, record):
        self.keys[key] = json.dumps(record)

    def _eval_claim(self, key, owner, lease_secs, now):
        now, lease_secs = float(now), float(lease_secs)
        record = self._record(key)
        attempts = 0
        if record is not None:
            if record["state"] in ("completed", "dead"):
                return None
            if (record["state"] == "processing"
                    and record["lease_until"] > now):
                return None
            if (record["state"] == "failed"
                    and record["retry_after"] > now):
                return None
            attempts = record.get("attempts", 0)
        self._save(key, {"state": "processing", "owner": owner,
                         "lease_until": now + lease_secs,
                         "attempts": attempts})
        return owner

    def _eval_heartbeat(self, key, owner, lease_secs, now):
        now, lease_secs = float(now), float(lease_secs)
        record = self._record(key)
        if record is None or record["state"] != "processing":
            return 0
        if record["owner"] != owner or record["lease_until"] <= now:
            return 0
        record["lease_until"] = now + lease_secs
        self._save(key, record)
        return 1

    def _eval_complete(self, key, owner):
        record = self._record(key)
        if record is None or record["state"] != "processing":
            return 0
        if record["owner"] != owner:
            return 0
        record["state"] = "completed"
        self._save(key, record)
        return 1

    def _eval_fail(self, key, owner, now, max_attempts, backoff_base):
        now = float(now)
        record = self._record(key)
        if record is None or record["state"] != "processing":
            return None
        if record["owner"] != owner:
            return None
        record["attempts"] = record.get("attempts", 0) + 1
        if record["attempts"] >= int(max_attempts):
            record["state"] = "dead"
        else:
            record["state"] = "failed"
            record["retry_after"] = now + float(backoff_base) * (
                2 ** (record["attempts"] - 1)
            )
        self._save(key, record)
        return record["state"]


@pytest.fixture(params=["memory", "redis"])
def backend(request, monkeypatch):
    """(fake_redis_or_None, clock) — every state-machine test runs
    once per backend so the two stay in lockstep."""
    clock = _Clock()
    monkeypatch.setattr(dedup, "_now", lambda: clock.now)
    if request.param == "redis":
        fake = _FakeRedis()
        monkeypatch.setattr(dedup, "_redis_client", lambda: fake)
        return fake, clock
    monkeypatch.setattr(dedup, "_redis_client", lambda: None)
    return None, clock


URL = "https://example.com/a"


def test_idempotency_key_format():
    key = dedup.idempotency_key(URL)
    assert key.startswith("url:")
    assert len(key) == 4 + 64  # "url:" + sha256 hex
    assert dedup.idempotency_key(URL) == key
    assert dedup.idempotency_key("https://example.com/b") != key


def test_state_pending_for_unknown_url(backend):
    assert dedup.state(URL) == dedup.PENDING


def test_claim_then_second_claim_denied(backend):
    owner = dedup.claim(URL)
    assert owner  # ownership token
    assert dedup.claim(URL) is None
    assert dedup.state(URL) == dedup.PROCESSING


def test_lease_expiry_recovery(backend):
    """A crashed worker's lease expires and the URL is claimable
    again; the old owner loses all rights over it."""
    _, clock = backend
    owner = dedup.claim(URL, lease_secs=60)
    assert dedup.claim(URL) is None  # lease live: still held
    clock.advance(61)  # worker "crashed": no heartbeat
    assert dedup.state(URL) == dedup.PENDING
    new_owner = dedup.claim(URL)
    assert new_owner and new_owner != owner
    # The crashed worker's token is now worthless.
    assert dedup.heartbeat(URL, owner) is False
    assert dedup.complete(URL, owner) is False
    assert dedup.fail(URL, owner) is None


def test_heartbeat_renews_lease(backend):
    _, clock = backend
    owner = dedup.claim(URL, lease_secs=60)
    clock.advance(50)
    assert dedup.heartbeat(URL, owner) is True  # renewed for 600s
    clock.advance(50)  # past the original 60s expiry
    assert dedup.claim(URL) is None  # still held thanks to heartbeat
    assert dedup.state(URL) == dedup.PROCESSING


def test_heartbeat_rejects_wrong_owner_and_expired_lease(backend):
    _, clock = backend
    owner = dedup.claim(URL, lease_secs=60)
    assert dedup.heartbeat(URL, "not-the-owner") is False
    clock.advance(61)
    assert dedup.heartbeat(URL, owner) is False  # too late to renew


def test_complete_never_reprocessed(backend):
    _, clock = backend
    owner = dedup.claim(URL)
    assert dedup.complete(URL, owner) is True
    assert dedup.state(URL) == dedup.COMPLETED
    clock.advance(10_000_000)  # far future: still never reprocessed
    assert dedup.claim(URL) is None
    assert dedup.state(URL) == dedup.COMPLETED


def test_complete_rejects_wrong_owner(backend):
    dedup.claim(URL)
    assert dedup.complete(URL, "not-the-owner") is False
    assert dedup.state(URL) == dedup.PROCESSING


def test_fail_retry_then_succeed(backend):
    """Fail twice (backoff between attempts), succeed on the third
    claim, complete — then never reprocessed."""
    _, clock = backend
    owner = dedup.claim(URL)
    assert dedup.fail(URL, owner) == dedup.FAILED
    assert dedup.state(URL) == dedup.FAILED
    assert dedup.claim(URL) is None  # inside the 60s backoff
    clock.advance(61)
    assert dedup.state(URL) == dedup.PENDING
    owner = dedup.claim(URL)
    assert owner
    assert dedup.fail(URL, owner) == dedup.FAILED
    assert dedup.claim(URL) is None  # inside the 120s backoff now
    clock.advance(121)
    owner = dedup.claim(URL)
    assert owner
    assert dedup.complete(URL, owner) is True
    clock.advance(10_000_000)
    assert dedup.claim(URL) is None


def test_dead_letter_after_max_attempts(backend):
    """Three failures dead-letter the URL; it never becomes
    claimable again, even after a long time."""
    _, clock = backend
    assert dedup.MAX_ATTEMPTS == 3
    for attempt in range(3):
        owner = dedup.claim(URL)
        assert owner
        result = dedup.fail(URL, owner)
        if attempt < 2:
            assert result == dedup.FAILED
            clock.advance(60 * 2 ** attempt + 1)  # past the backoff
        else:
            assert result == dedup.DEAD
    assert dedup.state(URL) == dedup.DEAD
    clock.advance(10_000_000)
    assert dedup.claim(URL) is None
    assert dedup.state(URL) == dedup.DEAD


def test_fail_rejects_wrong_owner(backend):
    dedup.claim(URL)
    assert dedup.fail(URL, "not-the-owner") is None
    assert dedup.state(URL) == dedup.PROCESSING


def test_attempts_survive_reclaim_after_lease_expiry(backend):
    """A URL that failed once, was reclaimed, then crashed again
    keeps its attempt count — it dead-letters on the third failure
    overall, not the third failure per lease."""
    _, clock = backend
    owner = dedup.claim(URL)
    assert dedup.fail(URL, owner) == dedup.FAILED
    clock.advance(61)
    owner = dedup.claim(URL)
    clock.advance(601)  # worker crashes; lease expires
    owner = dedup.claim(URL)  # reclaimed by a new worker
    assert owner
    assert dedup.fail(URL, owner) == dedup.FAILED  # attempt 2, not 1
    clock.advance(121)
    owner = dedup.claim(URL)
    assert dedup.fail(URL, owner) == dedup.DEAD  # attempt 3: dead


def test_already_seen_compat_shim(backend):
    """The runners' old API still works: first sighting processes,
    a repeat sighting skips — and a crashed claim recovers."""
    _, clock = backend
    assert dedup.already_seen(URL) is False  # first sighting
    assert dedup.already_seen(URL) is True   # second sighting: skip
    assert dedup.already_seen("https://example.com/b") is False
    clock.advance(601)  # lease expired: crashed claim recovers
    assert dedup.already_seen(URL) is False


def test_redis_claim_stores_lease_record(monkeypatch):
    """Redis-only: the stored value is a JSON lease record with an
    owner token and a lease deadline, not a bare flag."""
    clock = _Clock()
    monkeypatch.setattr(dedup, "_now", lambda: clock.now)
    fake = _FakeRedis()
    monkeypatch.setattr(dedup, "_redis_client", lambda: fake)
    owner = dedup.claim(URL)
    key = dedup.idempotency_key(URL)
    record = json.loads(fake.keys[key])
    assert record["state"] == dedup.PROCESSING
    assert record["owner"] == owner
    assert record["lease_until"] == clock.now + dedup.LEASE_TTL_SECS
    assert record["attempts"] == 0


def test_redis_failure_falls_back_to_memory(monkeypatch):
    class _BrokenRedis:
        def get(self, *args, **kwargs):
            raise ConnectionError("redis down")

        def eval(self, *args, **kwargs):
            raise ConnectionError("redis down")

    monkeypatch.setattr(dedup, "_redis_client", lambda: _BrokenRedis())
    owner = dedup.claim(URL)
    assert owner
    assert dedup.claim(URL) is None
    assert dedup.state(URL) == dedup.PROCESSING
    assert dedup.complete(URL, owner) is True
    assert dedup.claim(URL) is None


def test_reset_clears_memory_claims(monkeypatch):
    monkeypatch.setattr(dedup, "_redis_client", lambda: None)
    assert dedup.already_seen(URL) is False
    dedup.reset()
    assert dedup.already_seen(URL) is False
    assert dedup.state(URL) == dedup.PROCESSING

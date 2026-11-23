"""Dedup / idempotency: URL hash + a lease-based state machine.

Every URL maps to the idempotency key ``url:<sha256(url)>`` and moves
through:

    pending -> processing -> completed
                    |  \
                    |   -> failed -> processing ... (retry with backoff)
                    |         \
                    |          -> dead (dead-letter, after MAX_ATTEMPTS)
                    +-> pending (lease expired: crashed worker recovery)

* ``claim()`` moves a URL to ``processing`` and hands the caller an
  ownership token plus a short lease (default 10 minutes). The worker
  renews the lease with ``heartbeat()`` while it works.
* ``complete()`` moves it to ``completed`` — a permanent record, never
  reprocessed. ``fail()`` moves it to ``failed`` with exponential
  backoff (60s, 120s, ...); after ``MAX_ATTEMPTS`` failures it is
  dead-lettered and never retried automatically.
* A crashed worker never renews its lease, so the URL becomes
  claimable again as soon as the lease expires — no more 7-day block.

Redis is the claim store when available. Every transition runs as one
atomic Lua script, and every operation that mutates a record requires
the caller's ownership token, so only the lease holder can renew,
complete or fail a URL. When Redis is unreachable or the ``redis``
package is missing, the module falls back to a process-local
in-memory store with identical semantics (guarded by a lock), so
single-worker runs and tests still work without infra. Postgres
``unique_url`` upserts remain the durable backstop.
"""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
import time

logger = logging.getLogger(__name__)

# Short lease: a crashed worker's URLs are reclaimable minutes later,
# while a healthy worker renews via heartbeat().
LEASE_TTL_SECS = 600
# Failed URLs retry with backoff BASE, 2*BASE, 4*BASE... then dead-letter.
MAX_ATTEMPTS = 3
FAILED_BACKOFF_BASE_SECS = 60

PENDING = "pending"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"
DEAD = "dead"

# --- Atomic Redis transitions (Lua: check + write in one step) ---

# KEYS[1] = key; ARGV[1] = owner, ARGV[2] = lease_secs, ARGV[3] = now.
# Returns the owner token on a successful claim, nil otherwise.
# Claimable: no record, an expired processing lease, or a failed
# record whose backoff has elapsed. Attempt counts survive reclaim.
_CLAIM_LUA = """
local raw = redis.call('GET', KEYS[1])
local attempts = 0
if raw then
  local rec = cjson.decode(raw)
  if rec.state == 'completed' or rec.state == 'dead' then
    return nil
  end
  if rec.state == 'processing'
      and tonumber(rec.lease_until) > tonumber(ARGV[3]) then
    return nil
  end
  if rec.state == 'failed'
      and tonumber(rec.retry_after) > tonumber(ARGV[3]) then
    return nil
  end
  attempts = rec.attempts or 0
end
local val = cjson.encode({state = 'processing', owner = ARGV[1],
  lease_until = tonumber(ARGV[3]) + tonumber(ARGV[2]),
  attempts = attempts})
redis.call('SET', KEYS[1], val)
return ARGV[1]
"""

# KEYS[1] = key; ARGV[1] = owner, ARGV[2] = lease_secs, ARGV[3] = now.
# Returns 1 if the lease was renewed, 0 otherwise (wrong owner,
# not processing, or lease already expired).
_HEARTBEAT_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local rec = cjson.decode(raw)
if rec.state ~= 'processing' or rec.owner ~= ARGV[1] then return 0 end
if tonumber(rec.lease_until) <= tonumber(ARGV[3]) then return 0 end
rec.lease_until = tonumber(ARGV[3]) + tonumber(ARGV[2])
redis.call('SET', KEYS[1], cjson.encode(rec))
return 1
"""

# KEYS[1] = key; ARGV[1] = owner. Returns 1 on success, 0 otherwise.
_COMPLETE_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local rec = cjson.decode(raw)
if rec.state ~= 'processing' or rec.owner ~= ARGV[1] then return 0 end
rec.state = 'completed'
redis.call('SET', KEYS[1], cjson.encode(rec))
return 1
"""

# KEYS[1] = key; ARGV[1] = owner, ARGV[2] = now, ARGV[3] = max_attempts,
# ARGV[4] = backoff base secs. Returns the new state ('failed'/'dead'),
# or nil when the caller does not hold the lease.
_FAIL_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then return nil end
local rec = cjson.decode(raw)
if rec.state ~= 'processing' or rec.owner ~= ARGV[1] then return nil end
rec.attempts = (rec.attempts or 0) + 1
if rec.attempts >= tonumber(ARGV[3]) then
  rec.state = 'dead'
else
  rec.state = 'failed'
  rec.retry_after = tonumber(ARGV[2])
    + tonumber(ARGV[4]) * (2 ^ (rec.attempts - 1))
end
redis.call('SET', KEYS[1], cjson.encode(rec))
return rec.state
"""

# In-memory fallback store: key -> record dict, same shape as the
# JSON stored in Redis. Kept under the name `_seen` for compatibility
# with existing callers/tests that clear it between runs.
_seen: dict[str, dict] = {}
_seen_lock = threading.Lock()
_client = None  # cached Redis client (None = unavailable)
_client_tried = False


def _now() -> float:
    """Current epoch time; a module-level indirection so tests can
    control the clock (lease expiry, backoff) without sleeping."""
    return time.time()


def idempotency_key(url: str) -> str:
    return f"url:{hashlib.sha256(url.encode()).hexdigest()}"


def _redis_client():
    """Lazily connect once; return None when Redis is unavailable.

    The connection decision is cached for the process lifetime: a
    run that starts without Redis stays on the in-memory store rather
    than paying a connect timeout per URL."""
    global _client, _client_tried
    if _client_tried:
        return _client
    _client_tried = True
    try:
        import redis

        from src.config import settings

        client = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=2, socket_timeout=2
        )
        client.ping()
        _client = client
    except Exception:
        _client = None
    return _client


def _decode(value):
    """redis-py returns bytes unless decode_responses is set."""
    if isinstance(value, bytes):
        return value.decode()
    return value


def _logical_state(record: dict | None, now: float) -> str:
    """The state a record is *effectively* in: an expired processing
    lease and an elapsed failed backoff both read as pending."""
    if record is None:
        return PENDING
    state = record.get("state")
    if state == PROCESSING and record.get("lease_until", 0) <= now:
        return PENDING
    if state == FAILED and record.get("retry_after", 0) <= now:
        return PENDING
    return state


# --- claim ---

def _claim_memory(key: str, owner: str, lease_secs: int) -> str | None:
    now = _now()
    with _seen_lock:
        record = _seen.get(key)
        if _logical_state(record, now) != PENDING:
            return None
        attempts = (record or {}).get("attempts", 0)
        _seen[key] = {
            "state": PROCESSING,
            "owner": owner,
            "lease_until": now + lease_secs,
            "attempts": attempts,
        }
        return owner


def claim(url: str, lease_secs: int = LEASE_TTL_SECS) -> str | None:
    """Claim a URL for processing. Returns an ownership token (the
    lease holder's proof, required by heartbeat/complete/fail), or
    None when the URL is not claimable right now — already being
    processed under a live lease, completed, dead-lettered, or inside
    its failed backoff window. A crashed worker's lease expires and
    the URL becomes claimable again."""
    key = idempotency_key(url)
    owner = secrets.token_hex(8)
    client = _redis_client()
    if client is not None:
        try:
            result = client.eval(
                _CLAIM_LUA, 1, key, owner, lease_secs, _now()
            )
            return _decode(result)
        except Exception as exc:
            logger.warning(
                "Redis claim failed for %s (%s); using memory", key, exc
            )
    return _claim_memory(key, owner, lease_secs)


# --- heartbeat ---

def _heartbeat_memory(key: str, owner: str, lease_secs: int) -> bool:
    now = _now()
    with _seen_lock:
        record = _seen.get(key)
        if record is None or record.get("state") != PROCESSING:
            return False
        if record.get("owner") != owner:
            return False
        if record.get("lease_until", 0) <= now:
            return False  # lease already expired; too late to renew
        record["lease_until"] = now + lease_secs
        return True


def heartbeat(url: str, owner: str, lease_secs: int = LEASE_TTL_SECS) -> bool:
    """Renew the caller's lease on a URL. Only the lease holder can
    renew, and only while the lease is still live. Returns False for
    a wrong owner, an expired lease, or a URL that is not processing."""
    key = idempotency_key(url)
    client = _redis_client()
    if client is not None:
        try:
            return bool(
                client.eval(_HEARTBEAT_LUA, 1, key, owner, lease_secs, _now())
            )
        except Exception as exc:
            logger.warning(
                "Redis heartbeat failed for %s (%s); using memory", key, exc
            )
    return _heartbeat_memory(key, owner, lease_secs)


# --- complete ---

def _complete_memory(key: str, owner: str) -> bool:
    with _seen_lock:
        record = _seen.get(key)
        if record is None or record.get("state") != PROCESSING:
            return False
        if record.get("owner") != owner:
            return False
        record["state"] = COMPLETED
        return True


def complete(url: str, owner: str) -> bool:
    """Mark a URL permanently processed — it is never reprocessed.
    Only the lease holder can complete it. Returns False for a wrong
    owner or a URL that is not processing."""
    key = idempotency_key(url)
    client = _redis_client()
    if client is not None:
        try:
            return bool(client.eval(_COMPLETE_LUA, 1, key, owner))
        except Exception as exc:
            logger.warning(
                "Redis complete failed for %s (%s); using memory", key, exc
            )
    return _complete_memory(key, owner)


# --- fail ---

def _fail_memory(key: str, owner: str) -> str | None:
    now = _now()
    with _seen_lock:
        record = _seen.get(key)
        if record is None or record.get("state") != PROCESSING:
            return None
        if record.get("owner") != owner:
            return None
        record["attempts"] = record.get("attempts", 0) + 1
        if record["attempts"] >= MAX_ATTEMPTS:
            record["state"] = DEAD
        else:
            record["state"] = FAILED
            record["retry_after"] = now + FAILED_BACKOFF_BASE_SECS * (
                2 ** (record["attempts"] - 1)
            )
        return record["state"]


def fail(url: str, owner: str) -> str | None:
    """Release a URL after a failed processing attempt. Returns
    ``"failed"`` (retryable once its backoff elapses) or ``"dead"``
    (dead-lettered after MAX_ATTEMPTS failures), or None when the
    caller does not hold the lease."""
    key = idempotency_key(url)
    client = _redis_client()
    if client is not None:
        try:
            result = client.eval(
                _FAIL_LUA, 1, key, owner, _now(), MAX_ATTEMPTS,
                FAILED_BACKOFF_BASE_SECS,
            )
            return _decode(result)
        except Exception as exc:
            logger.warning(
                "Redis fail failed for %s (%s); using memory", key, exc
            )
    return _fail_memory(key, owner)


# --- state / compatibility ---

def state(url: str) -> str:
    """The URL's current logical state: pending, processing,
    completed, failed or dead. Unknown URLs are pending."""
    key = idempotency_key(url)
    client = _redis_client()
    if client is not None:
        try:
            raw = client.get(key)
            record = json.loads(_decode(raw)) if raw is not None else None
            return _logical_state(record, _now())
        except Exception as exc:
            logger.warning(
                "Redis state lookup failed for %s (%s); using memory",
                key, exc,
            )
    with _seen_lock:
        return _logical_state(_seen.get(key), _now())


def already_seen(url: str) -> bool:
    """Backwards-compatible shim for the runners: True if this URL
    was already claimed (or completed / dead-lettered / backing off),
    False if it was newly claimed for processing.

    The claim carries a 10-minute lease; callers that want the
    permanent "never reprocess" guarantee should use ``claim()`` and
    call ``complete(url, owner)`` once the record is safely stored —
    a URL claimed but never completed becomes claimable again when
    its lease expires, which is exactly the crashed-worker recovery.
    """
    return claim(url) is None


def reset() -> None:
    """Clear the in-memory claim store (tests / independent runs)."""
    with _seen_lock:
        _seen.clear()

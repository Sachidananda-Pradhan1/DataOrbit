"""Exponential backoff + jitter for 429s (Phase III)."""
import functools
import random
import time


def retry_on_rate_limit(fn):
    """Retry only on 429 / rate-limit errors, with exponential
    backoff + jitter. Any other error raises immediately — retrying
    it would only stall the LLM fallback chain (each tier would
    burn ~15s of sleeps before the next tier is tried)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        delay = 1.0
        for attempt in range(5):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if "429" not in str(exc) and "rate limit" not in str(exc).lower():
                    raise
                if attempt == 4:
                    raise
                time.sleep(delay + random.uniform(0, delay))
                delay *= 2
        raise RuntimeError("unreachable")
    return wrapper

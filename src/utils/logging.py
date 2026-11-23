"""Structured logging — one JSON line per event, so failures in a
500k-record run are searchable instead of lost in stdout."""
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(message)s")
log = logging.getLogger("graphone")


def event(name: str, **fields):
    log.info(json.dumps({"event": name, **fields}))

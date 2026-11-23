"""Parse an LLM reply as a JSON object.

LLM replies are unreliable JSON: Gemini / Groq commonly wrap the
object in ```json fences, and sometimes add prose around it. Bare
json.loads fails on both, and the old extractor silently skipped
the chunk — losing data with no trace. This module is the single
parser for every LLM consumer in the repo (extractor.py and
crawlers/products.py import it).

Contract: parse_llm_json returns a dict, or raises ValueError with
a useful message. It never silently returns None / garbage — the
caller decides whether a bad chunk is skippable (log + continue)
or fatal.
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _matching_brace(text: str, start: int) -> int:
    """Index of the `}` matching the `{` at `start`, or -1.

    Brace-aware: braces inside JSON string literals (and escaped
    quotes) do not count, so nested objects and prose like
    '{"a": "} not a brace"}' resolve correctly.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_llm_json(raw: str) -> dict:
    """Parse an LLM reply as a JSON object.

    Accepts bare JSON, ```json / ``` fenced JSON, and JSON wrapped
    in prose (the first top-level {...} block is used). Raises
    ValueError on empty input, unparseable input, or a JSON value
    that is not an object.
    """
    if not raw or not raw.strip():
        raise ValueError("empty LLM response: no JSON to parse")
    text = raw.strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = _matching_brace(text, start) if start >= 0 else -1
        if start < 0 or end < 0:
            raise ValueError(
                f"no JSON object found in LLM response "
                f"(first 200 chars: {raw[:200]!r})"
            ) from None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"unparseable JSON object in LLM response: {exc} "
                f"(first 200 chars: {raw[:200]!r})"
            ) from exc
    if not isinstance(data, dict):
        # ValueError (not TypeError) by design: the contract is one
        # exception type for every unparseable reply, so callers catch
        # ValueError once.
        raise ValueError(  # noqa: TRY004
            f"LLM response parsed as {type(data).__name__}, not a JSON "
            f"object (first 200 chars: {raw[:200]!r})"
        )
    return data

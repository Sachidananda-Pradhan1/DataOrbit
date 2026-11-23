"""Tests for src/llm/extractor.py — mocked LLM, no keys needed.

Covers the 413 guard (long input is chunked, each call capped at
MAX_CHARS_PER_CALL, per-chunk results merged) and the validation
guardrail (bad JSON / bad schema -> None).
"""
import json
from datetime import datetime, timezone

import pytest

from src.llm import extractor
from src.llm.chunker import MAX_CHARS_PER_CALL

NOW = datetime(2026, 11, 21, tzinfo=timezone.utc)


def test_extract_startup_chunks_long_input(monkeypatch):
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        # First chunk yields the name, later chunks the headcount:
        # the merged result must carry both (facts at the end of a
        # long page are no longer lost to truncation).
        if len(prompts) == 1:
            return json.dumps({"entityName": "Acme AI"})
        return json.dumps({"employeeCount": 12})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    long_text = "x" * 200_000
    result = extractor.extract_startup(
        long_text, "Y Combinator", "https://example.com/acme", NOW)
    assert result is not None
    assert result.entityName == "Acme AI"
    assert result.employeeCount == 12
    # A 200 KB page is split into several calls, and no single call
    # may carry more than MAX_CHARS_PER_CALL of page text (413 guard).
    assert len(prompts) > 1
    for prompt in prompts:
        chunk = prompt.rsplit("Text:\n", 1)[1]
        assert len(chunk) <= MAX_CHARS_PER_CALL


def test_extract_startup_short_input_unchanged(monkeypatch):
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return json.dumps({"entityName": "Acme AI"})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    extractor.extract_startup(
        "short text", "Y Combinator", "https://example.com/acme", NOW)
    assert "short text" in seen["prompt"]


def test_extract_startup_bad_json_returns_none(monkeypatch):
    monkeypatch.setattr(
        extractor, "extract_with_fallback", lambda prompt: "not json{")
    assert extractor.extract_startup(
        "text", "Y Combinator", "https://example.com/acme", NOW) is None


def test_extract_startup_bad_schema_returns_none(monkeypatch):
    monkeypatch.setattr(
        extractor, "extract_with_fallback",
        lambda prompt: json.dumps({"wrongField": 1}))
    assert extractor.extract_startup(
        "text", "Y Combinator", "https://example.com/acme", NOW) is None


def test_extract_product_chunks_and_merges(monkeypatch):
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return json.dumps({"startupName": "Acme AI"})
        return json.dumps({"productName": "Acme Chat"})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    result = extractor.extract_product(
        "y" * 100_000, "BetaList", "https://example.com/acme-chat", NOW)
    assert result is not None
    assert result.startupName == "Acme AI"
    assert result.productName == "Acme Chat"
    assert len(prompts) > 1
    for prompt in prompts:
        chunk = prompt.rsplit("Text:\n", 1)[1]
        assert len(chunk) <= MAX_CHARS_PER_CALL


def test_parse_llm_json_strips_code_fences(monkeypatch):
    monkeypatch.setattr(
        extractor, "extract_with_fallback",
        lambda prompt: "```json\n{\"entityName\": \"Acme AI\"}\n```")
    result = extractor.extract_startup(
        "text", "Y Combinator", "https://example.com/acme", NOW)
    assert result is not None
    assert result.entityName == "Acme AI"


def test_parse_llm_json_tolerates_prose_around_json():
    assert extractor.parse_llm_json(
        'Here is the result:\n{"entityName": "Acme AI"}\nHope this helps.'
    ) == {"entityName": "Acme AI"}
    # Invalid input now raises ValueError (json_utils contract) instead
    # of returning None; the extractor logs and skips the chunk.
    with pytest.raises(ValueError):
        extractor.parse_llm_json("not json{")
    with pytest.raises(ValueError):
        extractor.parse_llm_json("")


def test_extract_product_pricing_model(monkeypatch):
    monkeypatch.setattr(
        extractor, "extract_with_fallback",
        lambda prompt: json.dumps({
            "startupName": "Acme AI", "productName": "Acme Chat",
            "pricingModel": "FREEMIUM"}))
    result = extractor.extract_product(
        "text", "BetaList", "https://example.com/acme-chat", NOW)
    assert result is not None
    assert result.pricingModel == "FREEMIUM"

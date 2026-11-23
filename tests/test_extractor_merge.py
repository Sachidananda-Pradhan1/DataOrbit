"""Tests for chunked extraction + deterministic merge (issue D).

Mocked LLM only — no real API calls, no keys needed.

Covers: facts near the END of a long page survive the merge;
merge determinism (same inputs -> same output, regardless of dict
key order); list dedup; scalar preference (non-empty beats empty,
first non-empty wins); single-chunk inputs behave exactly as before.
"""
import json
from datetime import datetime, timezone

from src.llm import extractor
from src.llm.chunker import MAX_CHARS_PER_CALL, chunk_text
from src.llm.extractor import merge_extractions

NOW = datetime(2026, 11, 21, tzinfo=timezone.utc)
SOURCE_URL = "https://example.com/acme"

# Filler paragraph (~18 chars incl. separator): 2,000 of them push the
# page well past one chunk, so the closing paragraph lands in a later
# chunk than the opening one.
FILLER = "filler paragraph\n\n" * 2000


def _marker_llm(marker_json: dict):
    """Fake LLM: returns the JSON for whichever marker(s) the chunk
    contains, {} for chunks with none."""
    def fake_llm(prompt):
        found = {}
        for marker, payload in marker_json.items():
            if marker in prompt:
                found.update(payload)
        return json.dumps(found)
    return fake_llm


def test_long_page_facts_at_end_survive_merge(monkeypatch):
    """Name up front, employee count at the very end of a long page:
    both must be in the merged result (truncation used to lose the
    count)."""
    text = ("NAME-MARKER Acme AI\n\n" + FILLER
            + "COUNT-MARKER the company has 42 employees")
    assert len(chunk_text(text)) > 1  # genuinely multi-chunk
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return _marker_llm({
            "NAME-MARKER": {"entityName": "Acme AI"},
            "COUNT-MARKER": {"employeeCount": 42},
        })(prompt)

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    result = extractor.extract_startup(
        text, "Y Combinator", SOURCE_URL, NOW)
    assert result is not None
    assert result.entityName == "Acme AI"
    assert result.employeeCount == 42
    # Per-call 413 guard still holds: no call exceeds the cap.
    assert len(prompts) > 1
    for prompt in prompts:
        chunk = prompt.rsplit("Text:\n", 1)[1]
        assert len(chunk) <= MAX_CHARS_PER_CALL


def test_long_product_facts_at_end_survive_merge(monkeypatch):
    text = ("NAME-MARKER Acme AI\n\n" + FILLER
            + "PRODUCT-MARKER the product is called Acme Chat")
    assert len(chunk_text(text)) > 1
    monkeypatch.setattr(extractor, "extract_with_fallback", _marker_llm({
        "NAME-MARKER": {"startupName": "Acme AI"},
        "PRODUCT-MARKER": {"productName": "Acme Chat"},
    }))
    result = extractor.extract_product(text, "BetaList", SOURCE_URL, NOW)
    assert result is not None
    assert result.startupName == "Acme AI"
    assert result.productName == "Acme Chat"


def test_bad_chunk_json_skipped_other_chunks_still_merge(monkeypatch):
    """One chunk returning garbage must not lose the other chunks:
    the middle (filler-only) chunk is corrupt, the name and count
    chunks still merge."""
    text = ("NAME-MARKER Acme AI\n\n" + FILLER * 2
            + "COUNT-MARKER the company has 42 employees")
    assert len(chunk_text(text)) == 4  # name / filler / filler / count
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            return "not json{"
        return _marker_llm({
            "NAME-MARKER": {"entityName": "Acme AI"},
            "COUNT-MARKER": {"employeeCount": 42},
        })(prompt)

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    result = extractor.extract_startup(
        text, "Y Combinator", SOURCE_URL, NOW)
    assert result is not None
    assert result.entityName == "Acme AI"
    assert result.employeeCount == 42


def test_merge_deterministic_same_input_same_output():
    parts = [
        {"entityName": "Acme AI", "tags": ["ai", "ml"],
         "meta": {"a": 1}},
        {"employeeCount": 42, "tags": ["ml", "saas"],
         "meta": {"b": 2}},
    ]
    first = merge_extractions(parts)
    second = merge_extractions(parts)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second,
                                                           sort_keys=True)


def test_merge_independent_of_dict_key_order():
    """Same facts, keys inserted in different orders -> same merge."""
    parts_a = [{"b": 1, "a": [1, 2]}, {"a": [2, 3], "c": "x"}]
    parts_b = [{"a": [1, 2], "b": 1}, {"c": "x", "a": [2, 3]}]
    assert merge_extractions(parts_a) == merge_extractions(parts_b)
    assert merge_extractions(parts_a) == {"a": [1, 2, 3], "b": 1, "c": "x"}


def test_merge_lists_dedupe_preserving_order():
    merged = merge_extractions([
        {"authors": ["Alice", "Bob"]},
        {"authors": ["Bob", "Carol"]},
        {"authors": ["Alice"]},
    ])
    assert merged == {"authors": ["Alice", "Bob", "Carol"]}


def test_merge_lists_dedupe_dict_items():
    merged = merge_extractions([
        {"links": [{"url": "https://a.example"}, {"url": "https://b.example"}]},
        {"links": [{"url": "https://a.example"}]},
    ])
    assert merged == {"links": [{"url": "https://a.example"},
                                {"url": "https://b.example"}]}


def test_merge_scalar_non_empty_beats_empty():
    assert merge_extractions(
        [{"entityName": ""}, {"entityName": "Acme AI"}]
    ) == {"entityName": "Acme AI"}
    assert merge_extractions(
        [{"entityName": None}, {"entityName": "Acme AI"}]
    ) == {"entityName": "Acme AI"}
    assert merge_extractions(
        [{"tags": []}, {"tags": ["ai"]}]
    ) == {"tags": ["ai"]}
    # All empty -> key dropped, not carried as an empty value.
    assert merge_extractions([{"entityName": ""}, {"entityName": None}]) == {}


def test_merge_scalar_first_non_empty_wins():
    assert merge_extractions(
        [{"entityName": "First"}, {"entityName": "Second"}]
    ) == {"entityName": "First"}
    assert merge_extractions(
        [{"employeeCount": 42}, {"employeeCount": 7}]
    ) == {"employeeCount": 42}


def test_merge_zero_and_false_are_values_not_empty():
    assert merge_extractions(
        [{"employeeCount": 0}, {"employeeCount": 5}]
    ) == {"employeeCount": 0}
    assert merge_extractions(
        [{"is_remote": False}, {"is_remote": True}]
    ) == {"is_remote": False}


def test_merge_nested_dicts_merge_recursively():
    merged = merge_extractions([
        {"source": {"name": "Y Combinator"}},
        {"source": {"url": "https://example.com"}},
    ])
    assert merged == {"source": {"name": "Y Combinator",
                                 "url": "https://example.com"}}


def test_single_chunk_input_unchanged(monkeypatch):
    """Short input: exactly one LLM call, full text in the prompt,
    no merge overhead — identical behaviour to before chunking."""
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return json.dumps({"entityName": "Acme AI", "employeeCount": 12})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    result = extractor.extract_startup(
        "short text", "Y Combinator", SOURCE_URL, NOW)
    assert result is not None
    assert result.entityName == "Acme AI"
    assert result.employeeCount == 12
    assert len(prompts) == 1
    assert "short text" in prompts[0]


def test_single_chunk_product_unchanged(monkeypatch):
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return json.dumps(
            {"startupName": "Acme AI", "productName": "Acme Chat"})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    result = extractor.extract_product(
        "short text", "BetaList", SOURCE_URL, NOW)
    assert result is not None
    assert result.productName == "Acme Chat"
    assert len(prompts) == 1
    assert "short text" in prompts[0]

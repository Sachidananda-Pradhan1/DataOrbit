"""Tests for the LLM-layer fixes: JSON fence parsing (json_utils),
HTML stripping before extraction, PAPER/JOB/NEWS prompt wiring, JSON
mode on every tier, and the google-genai SDK migration.

All LLM calls are mocked; no keys, no network.
"""
import json
import logging
import sys
import types as py_types
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.llm import extractor, orchestrator
from src.llm.chunker import (
    MAX_CHARS_PER_CALL,
    looks_like_html,
    strip_html_to_text,
)
from src.llm.json_utils import parse_llm_json
from src.llm.prompts import JOB_PROMPT, NEWS_PROMPT, PAPER_PROMPT, PRODUCT_PROMPT

NOW = datetime(2026, 11, 21, tzinfo=timezone.utc)
URL = "https://example.com/acme"

HTML_PAGE = """<html><head><title>Acme AI</title>
<script>var secret = 1;</script><style>.a { color: red; }</style></head>
<body><div class="main"><h1>Acme AI</h1>
<p>Acme AI builds chatbots for support teams.</p></div></body></html>"""


# --- parse_llm_json (json_utils) -------------------------------------------

def test_parse_bare_json():
    assert parse_llm_json('{"entityName": "Acme AI"}') == {
        "entityName": "Acme AI"}


def test_parse_fenced_json():
    assert parse_llm_json(
        '```json\n{"entityName": "Acme AI"}\n```') == {
        "entityName": "Acme AI"}
    # Fence without the json tag.
    assert parse_llm_json('```\n{"entityName": "Acme AI"}\n```') == {
        "entityName": "Acme AI"}


def test_parse_prose_wrapped_json():
    assert parse_llm_json(
        'Here is the result:\n{"entityName": "Acme AI"}\nHope this helps.'
    ) == {"entityName": "Acme AI"}


def test_parse_nested_braces_and_braces_in_strings():
    # Brace-matching must not stop at a "}" inside a string value.
    assert parse_llm_json(
        'prefix {"a": {"b": 1}, "note": "} not a brace"} suffix'
    ) == {"a": {"b": 1}, "note": "} not a brace"}


@pytest.mark.parametrize("bad", [
    "", "   ", "not json{", "no braces at all", '{"unclosed": 1',
    '["a", "b"]', '"just a string"', "42",
])
def test_parse_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        parse_llm_json(bad)


def test_parse_error_message_is_useful():
    with pytest.raises(ValueError, match="no JSON object found"):
        parse_llm_json("nothing here")


def test_parse_llm_json_reexported_from_extractor():
    # crawlers/products.py imports it from extractor; both names must
    # be the same function with the same raise-on-failure contract.
    assert extractor.parse_llm_json is parse_llm_json


# --- extractor: bad chunks are logged, not silently skipped ----------------

def test_bad_chunk_logged_and_other_chunks_contribute(monkeypatch, caplog):
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "not json{"
        return json.dumps({"entityName": "Acme AI"})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    with caplog.at_level(logging.WARNING, logger="src.llm.extractor"):
        result = extractor.extract_startup(
            "x" * 30_000, "Y Combinator", URL, NOW)  # 2 chunks
    assert result is not None
    assert result.entityName == "Acme AI"
    assert len(calls) == 2
    assert any("chunk 0" in record.message and "unparseable" in record.message
               for record in caplog.records)


def test_all_bad_chunks_returns_none_and_logs(monkeypatch, caplog):
    monkeypatch.setattr(
        extractor, "extract_with_fallback", lambda prompt: "garbage{")
    with caplog.at_level(logging.WARNING, logger="src.llm.extractor"):
        result = extractor.extract_startup("text", "Y Combinator", URL, NOW)
    assert result is None
    assert any("chunk 0" in record.message for record in caplog.records)


# --- HTML stripping ---------------------------------------------------------

def test_looks_like_html():
    assert looks_like_html(HTML_PAGE)
    assert looks_like_html("<p>hello</p>")
    assert not looks_like_html("plain text, no tags")
    assert not looks_like_html("a < b and c > d")  # bare < is not HTML
    assert not looks_like_html('{"entityName": "Acme AI"}')


def test_strip_html_to_text_drops_tags_scripts_styles():
    text = strip_html_to_text(HTML_PAGE)
    assert "<" not in text
    assert "Acme AI builds chatbots" in text
    assert "secret" not in text  # script content dropped
    assert "color: red" not in text  # style content dropped
    assert strip_html_to_text("") == ""


def test_html_input_stripped_before_llm(monkeypatch):
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return json.dumps({"entityName": "Acme AI"})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    result = extractor.extract_startup(HTML_PAGE, "Y Combinator", URL, NOW)
    assert result is not None
    prompt = seen["prompt"]
    assert "<div" not in prompt and "<script" not in prompt
    assert "<html" not in prompt and "<p>" not in prompt
    assert "Acme AI builds chatbots" in prompt


def test_html_stripped_before_chunking(monkeypatch):
    """The chunker itself receives tag-free text, not raw HTML."""
    from src.llm import chunker as chunker_mod

    received = []
    real_chunk_text = chunker_mod.chunk_text

    def spy_chunk_text(text, max_tokens=chunker_mod.MAX_TOKENS_PER_CHUNK):
        received.append(text)
        return real_chunk_text(text, max_tokens)

    monkeypatch.setattr(extractor, "chunk_text", spy_chunk_text)
    monkeypatch.setattr(
        extractor, "extract_with_fallback",
        lambda prompt: json.dumps({"entityName": "Acme AI"}))
    extractor.extract_startup(HTML_PAGE, "Y Combinator", URL, NOW)
    assert len(received) == 1
    assert "<" not in received[0]
    assert "Acme AI builds chatbots" in received[0]


def test_plain_text_passes_through_unchanged(monkeypatch):
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return json.dumps({"entityName": "Acme AI"})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    extractor.extract_startup(
        "Acme AI builds chatbots.", "Y Combinator", URL, NOW)
    assert "Acme AI builds chatbots." in seen["prompt"]


def test_long_html_page_stripped_then_chunked_within_cap(monkeypatch):
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return json.dumps({"entityName": "Acme AI"})

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    long_html = "<html><body>" + "<p>" + "word " * 20_000 + "</p></body></html>"
    extractor.extract_startup(long_html, "Y Combinator", URL, NOW)
    assert len(prompts) > 1
    for prompt in prompts:
        chunk = prompt.rsplit("Text:\n", 1)[1]
        assert len(chunk) <= MAX_CHARS_PER_CALL
        assert "<p>" not in chunk


# --- PAPER / JOB / NEWS prompt wiring ---------------------------------------

def test_extract_paper_uses_paper_prompt(monkeypatch):
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return json.dumps({
            "title": "Attention Is All You Need",
            "authors": ["Vaswani", "Shazeer"],
            # LLM-guessed URL must NOT override the fixed feed URL.
            "paper_url": "https://evil.example.com/fake",
            "github_url": "https://github.com/tensorflow/tensor2tensor",
        })

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    monkeypatch.setattr(
        extractor, "PAPER_PROMPT", "UNIQUE-PAPER-MARKER {chunk}")
    result = extractor.extract_paper(
        "paper abstract text", "https://arxiv.org/abs/1706.03762", NOW)
    assert "UNIQUE-PAPER-MARKER" in seen["prompt"]
    assert result is not None
    assert result.title == "Attention Is All You Need"
    assert result.authors == ["Vaswani", "Shazeer"]
    assert str(result.paper_url) == "https://arxiv.org/abs/1706.03762"
    assert str(result.github_url) == (
        "https://github.com/tensorflow/tensor2tensor")
    assert result.published_date == NOW


def test_extract_job_uses_job_prompt(monkeypatch):
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return json.dumps({
            "company": "Acme AI", "title": "ML Engineer",
            "date": "2026-11-22T10:00:00Z", "is_remote": True,
            "role_family": "Engineering",
        })

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    monkeypatch.setattr(extractor, "JOB_PROMPT", "UNIQUE-JOB-MARKER {chunk}")
    result = extractor.extract_job(
        "job posting text", "https://example.com/jobs/1")
    assert "UNIQUE-JOB-MARKER" in seen["prompt"]
    assert result is not None
    assert result.company == "Acme AI"
    assert result.title == "ML Engineer"
    assert result.is_remote is True
    assert result.role_family == "Engineering"
    assert result.date == datetime(2026, 11, 22, 10, tzinfo=timezone.utc)
    assert str(result.source_url) == "https://example.com/jobs/1"


def test_extract_news_uses_news_prompt(monkeypatch):
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return json.dumps({
            "title": "Acme AI raises $10M",
            "published_date": "2026-11-22T09:00:00Z",
        })

    monkeypatch.setattr(extractor, "extract_with_fallback", fake_llm)
    monkeypatch.setattr(
        extractor, "NEWS_PROMPT", "UNIQUE-NEWS-MARKER {chunk}")
    result = extractor.extract_news(
        "news article text", "TechCrunch", "https://example.com/news/1")
    assert "UNIQUE-NEWS-MARKER" in seen["prompt"]
    assert result is not None
    assert result.title == "Acme AI raises $10M"
    assert result.source.name == "TechCrunch"
    assert str(result.url) == "https://example.com/news/1"
    assert result.published_date == datetime(
        2026, 11, 22, 9, tzinfo=timezone.utc)


def test_prompts_request_the_model_fields():
    assert "title" in PAPER_PROMPT and "authors" in PAPER_PROMPT
    assert "company" in JOB_PROMPT and "title" in JOB_PROMPT
    assert "published_date" in NEWS_PROMPT
    # Contract with the products stream: exact key + enum.
    assert "pricingModel" in PRODUCT_PROMPT
    for value in ("FREE", "FREEMIUM", "PAID", "ENTERPRISE"):
        assert value in PRODUCT_PROMPT


# --- JSON mode + google-genai SDK migration ---------------------------------

def _install_fake_google_genai(monkeypatch):
    """Fake `google.genai` in sys.modules; the real google-genai SDK
    is NOT imported. If the orchestrator still imported the dead
    `google.generativeai` SDK, these calls would raise ImportError.
    """
    fake_types = py_types.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = MagicMock(
        side_effect=lambda **kwargs: ("CONFIG", kwargs))
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(
        text='{"entityName": "Acme AI"}')
    fake_genai = py_types.ModuleType("google.genai")
    fake_genai.Client = MagicMock(return_value=client)
    fake_genai.types = fake_types
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return fake_genai, client


def test_gemini_uses_google_genai_client_and_json_mode(monkeypatch):
    fake_genai, client = _install_fake_google_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert orchestrator._call_gemini("prompt") == '{"entityName": "Acme AI"}'
    fake_genai.Client.assert_called_once_with(api_key="test-key")
    fake_genai.types.GenerateContentConfig.assert_called_once_with(
        response_mime_type="application/json")
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-3.6-flash"
    assert kwargs["contents"] == "prompt"
    assert kwargs["config"] == (
        "CONFIG", {"response_mime_type": "application/json"})


def test_gemini_pro_uses_google_genai_client_and_json_mode(monkeypatch):
    _fake_genai, client = _install_fake_google_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    orchestrator._call_gemini_pro("prompt")
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-4-hydrogen"
    assert kwargs["config"] == (
        "CONFIG", {"response_mime_type": "application/json"})


def test_fallback_chain_reaches_gemini_via_google_genai(monkeypatch):
    _install_fake_google_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert orchestrator.extract_with_fallback("hi") == (
        '{"entityName": "Acme AI"}')


def _install_fake_chat_client(monkeypatch, module_name, attr, content):
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = response
    fake_module = py_types.ModuleType(module_name)
    fake_cls = MagicMock(return_value=client)
    setattr(fake_module, attr, fake_cls)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    return fake_cls, client


def test_groq_json_mode(monkeypatch):
    fake_groq, client = _install_fake_chat_client(
        monkeypatch, "groq", "Groq", '{"ok": true}')
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert orchestrator._call_groq("prompt") == '{"ok": true}'
    fake_groq.assert_called_once_with(api_key="test-key")
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "openai/gpt-oss-120b"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]


def test_deepseek_json_mode(monkeypatch):
    fake_openai, client = _install_fake_chat_client(
        monkeypatch, "openai", "OpenAI", '{"ok": true}')
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert orchestrator._call_deepseek("prompt") == '{"ok": true}'
    fake_openai.assert_called_once_with(
        api_key="test-key", base_url="https://api.deepseek.com/v1")
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["response_format"] == {"type": "json_object"}

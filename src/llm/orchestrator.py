"""Phase III — multi-tier LLM fallback: Gemini Flash -> Groq
-> DeepSeek -> Gemini Pro (last resort). 25% of the score (LLM Orchestration).

Real SDK calls, wired 2026-11-21. Each provider is imported lazily
inside its own function, so the module imports (and tests run) with
no LLM SDK installed and no keys set — a tier is only attempted when
its API key is present in the environment, and a missing key skips
that tier instead of failing the chain. Gemini Pro shares the
GEMINI_API_KEY with Gemini Flash but is a separate, last-resort tier
for complex reasoning: it is only reached when Flash, Groq and
DeepSeek have all failed.

Gemini uses the `google-genai` SDK (`from google import genai`,
client-based API). The old `google-generativeai` SDK reached end
of support on 2025-11-30 and must not be reintroduced.

Model names are overridable via GEMINI_MODEL / GROQ_MODEL /
DEEPSEEK_MODEL / GEMINI_PRO_MODEL; the defaults below are the
current Flash / Groq / DeepSeek chat / Pro models. If a provider
rejects a model name, that tier fails and the fallback chain moves
on — it never loses the raw text, which is already saved
(crawlers/base.py), so extraction can be retried / re-run with
another provider.

Each provider call is wrapped in exponential backoff + jitter for
429s (see utils/retry.py — only 429s are retried). All four tiers
request JSON mode (Gemini: response_mime_type, Groq/DeepSeek:
response_format json_object) so replies parse without fences.
"""
from __future__ import annotations

import os

from src.llm.chunker import chunk_text, looks_like_html, strip_html_to_text
from src.utils.retry import retry_on_rate_limit

PROVIDERS = ("gemini", "groq", "deepseek", "gemini_pro")

# Env key + default model per tier. DeepSeek is OpenAI-compatible,
# so it goes through the `openai` client (see pyproject [llm] extra).
# gemini_pro shares GEMINI_API_KEY with gemini (Flash) but has its
# own model override; it is the last-resort / complex-reasoning tier.
# Defaults verified live 2026-11-22 (each answered a JSON ping):
# gemini-2.5-flash/pro are retired for new users (404), Groq removed
# llama-3.3-70b-versatile, and gemini-3.8/3.7-flash were 503-overloaded.
_PROVIDER_CONFIG = {
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.6-flash"),
    "groq": ("GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat"),
    "gemini_pro": ("GEMINI_API_KEY", "GEMINI_PRO_MODEL", "gemini-4-hydrogen"),
}


def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    key_env, model_env, default_model = _PROVIDER_CONFIG["gemini"]
    client = genai.Client(api_key=os.environ[key_env])
    response = client.models.generate_content(
        model=os.getenv(model_env, default_model),
        contents=prompt,
        # JSON mode: ask for a JSON object, not fenced prose.
        config=types.GenerateContentConfig(
            response_mime_type="application/json"),
    )
    return response.text


def _call_gemini_pro(prompt: str) -> str:
    from google import genai
    from google.genai import types

    key_env, model_env, default_model = _PROVIDER_CONFIG["gemini_pro"]
    client = genai.Client(api_key=os.environ[key_env])
    response = client.models.generate_content(
        model=os.getenv(model_env, default_model),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"),
    )
    return response.text


def _call_groq(prompt: str) -> str:
    from groq import Groq

    key_env, model_env, default_model = _PROVIDER_CONFIG["groq"]
    client = Groq(api_key=os.environ[key_env])
    response = client.chat.completions.create(
        model=os.getenv(model_env, default_model),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _call_deepseek(prompt: str) -> str:
    from openai import OpenAI

    key_env, model_env, default_model = _PROVIDER_CONFIG["deepseek"]
    client = OpenAI(
        api_key=os.environ[key_env], base_url="https://api.deepseek.com/v1"
    )
    response = client.chat.completions.create(
        model=os.getenv(model_env, default_model),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


_CALLERS = {"gemini": _call_gemini, "groq": _call_groq,
            "deepseek": _call_deepseek, "gemini_pro": _call_gemini_pro}


def _call_provider(provider: str, prompt: str) -> str:
    """One provider call. Raises on failure so the fallback chain
    can move on."""
    return _CALLERS[provider](prompt)


@retry_on_rate_limit
def _try_provider(provider: str, prompt: str) -> str:
    return _call_provider(provider, prompt)


def extract_with_fallback(prompt: str) -> str:
    """Try providers in order; return the first successful response.
    A tier with no API key is skipped (recorded, not attempted).
    Raises RuntimeError only if every configured tier fails."""
    errors: list[str] = []
    for provider in PROVIDERS:
        key_env, _, _ = _PROVIDER_CONFIG[provider]
        if not os.getenv(key_env):
            errors.append(f"{provider}: not configured (no {key_env})")
            continue
        try:
            return _try_provider(provider, prompt)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
    raise RuntimeError("all LLM tiers failed: " + "; ".join(errors))


def extract_long_text(text: str, prompt_template: str) -> list[str]:
    """Strip HTML, chunk long payloads first (413 guard) and extract
    per chunk."""
    if looks_like_html(text):
        text = strip_html_to_text(text)
    return [extract_with_fallback(prompt_template.format(chunk=c))
            for c in chunk_text(text)]

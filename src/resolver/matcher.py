"""Phase IV — deterministic entity resolution (10% of score).

Resolution is deterministic and domain-first — never name-alone.
Domain evidence outranks name similarity: a known domain (or the
domain of a source URL) maps to its canonical name with confidence
1.0. Only when no domain is known does the name get consulted: a
curated alias table (exact normalised-string mappings, confidence
1.0), then an exact normalised match against the seed list
(confidence 1.0), then a difflib similarity match above a high 0.9
threshold (confidence = the ratio). Unknown names pass through
unchanged (never invent a canonical form) with confidence 0.0.

'OpenAI', 'OpenAI, Inc.' and 'Open AI' must all resolve to 'OpenAI'.
Runners call resolve_with_confidence(name, domain) for every
startup / product / job company and write (raw, canonical,
confidence, method, source_url) rows to the Entity Mapping Log tab.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

SEED_PATH = Path(__file__).with_name("seed_canonical.json")
SEED_DOMAINS_PATH = Path(__file__).with_name("seed_canonical_domains.json")
# Legal / generic suffixes stripped only as a trailing run, after all
# punctuation and spaces are removed — so "Open AI" -> "openai" and
# "OpenAI, Inc." -> "openai", while "OpenAI" itself is untouched.
# ("ai" is deliberately NOT a suffix: stripping it would break OpenAI.)
SUFFIXES = ("corporation", "company", "labs", "inc", "llc", "ltd",
            "corp", "lab", "pbc")

# Exact normalised-name aliases: keys are normalize()d forms, values
# are canonical seed names. These are hand-curated, deterministic
# mappings — never fuzzy guessing — for the short / legal forms
# sources actually use ("Mistral" for Mistral AI, "DeepMind" for
# Google DeepMind, "Runway ML" for Runway, "Stability" for Stability
# AI, "wandb" for Weights & Biases, "Inflection" for Inflection AI).
# "Anthropic PBC" needs no entry: the "pbc" legal suffix is stripped
# by normalize(), so it exact-matches "Anthropic".
ALIASES = {
    "mistral": "Mistral AI",
    "deepmind": "Google DeepMind",
    "runwayml": "Runway",
    "stability": "Stability AI",
    "wandb": "Weights & Biases",
    "inflection": "Inflection AI",
}


def normalize(name: str) -> str:
    n = re.sub(r"[^a-z0-9]", "", name.strip().lower())
    for suffix in SUFFIXES:
        if n.endswith(suffix) and len(n) > len(suffix):
            n = n[: -len(suffix)]
            break
    return n


def load_canonical() -> list[str]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


def load_canonical_domains() -> dict[str, str]:
    """Seed domain -> canonical name. Domains are normalised to
    lowercase with any leading www. stripped."""
    raw = json.loads(SEED_DOMAINS_PATH.read_text(encoding="utf-8"))
    return {_normalize_domain(domain): name for domain, name in raw.items()}


def _normalize_domain(domain: str) -> str:
    """Lowercase a domain and strip a leading www."""
    host = domain.strip().lower()
    host = host.removeprefix("www.")
    return host


def _host_from(domain_or_url: str) -> str:
    """Extract the host from a bare domain or a full URL
    (``https://www.openai.com/about`` -> ``openai.com``)."""
    value = domain_or_url.strip()
    if "://" in value:
        host = urlparse(value).hostname or ""
    else:
        host = value.split("/")[0].split(":")[0]
    return _normalize_domain(host)


def _registrable_domain(host: str) -> str:
    """Strip subdomains: ``blog.openai.com`` -> ``openai.com``.
    Naive last-two-labels rule — fine for the seed set, which has
    no multi-part public suffixes (no *.co.uk-style domains)."""
    labels = host.split(".")
    return ".".join(labels[-2:]) if len(labels) > 2 else host


def resolve_with_confidence(name: str, domain: str | None = None,
                            canonical: list[str] | None = None
                            ) -> tuple[str, float, str]:
    """Resolve ``name`` to its canonical form with a confidence score.

    Returns ``(canonical_name, confidence, method)`` where method is
    one of ``domain`` / ``alias`` / ``exact`` / ``fuzzy`` /
    ``unmatched``:

    - ``domain`` (1.0): ``domain`` — a bare domain or a full URL —
      matches a seed domain, exactly or after stripping subdomains.
    - ``alias`` (1.0): normalised name matches a curated entry in
      ALIASES exactly (deterministic, never fuzzy).
    - ``exact`` (1.0): normalised name matches a seed name exactly.
    - ``fuzzy`` (0.9–1.0): difflib close match at cutoff 0.9;
      confidence is the difflib ratio, rounded to 2 decimals.
    - ``unmatched`` (0.0): no evidence; the name passes through
      unchanged — never invent a canonical form.
    """
    if domain:
        domains = load_canonical_domains()
        host = _host_from(domain)
        for candidate in (host, _registrable_domain(host)):
            if candidate in domains:
                return domains[candidate], 1.0, "domain"
    canonical = canonical or load_canonical()
    by_norm = {normalize(c): c for c in canonical}
    key = normalize(name)
    if key in ALIASES and ALIASES[key] in canonical:
        return ALIASES[key], 1.0, "alias"
    if key in by_norm:
        return by_norm[key], 1.0, "exact"
    # Fuzzy fallback — high threshold on purpose: a wrong merge is
    # worse than leaving a name unresolved.
    match = difflib.get_close_matches(key, list(by_norm), n=1, cutoff=0.9)
    if match:
        ratio = difflib.SequenceMatcher(None, key, match[0]).ratio()
        return by_norm[match[0]], round(ratio, 2), "fuzzy"
    return name.strip(), 0.0, "unmatched"


def resolve(name: str, canonical: list[str] | None = None) -> str:
    """Name-only resolution (backwards compatible): canonical name
    or the input unchanged. See resolve_with_confidence for the
    domain-first version with confidence + method."""
    return resolve_with_confidence(name, canonical=canonical)[0]

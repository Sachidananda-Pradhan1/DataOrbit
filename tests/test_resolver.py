from src.resolver.matcher import (
    load_canonical,
    load_canonical_domains,
    resolve,
    resolve_with_confidence,
)


def test_openai_variants():
    for variant in ["OpenAI", "OpenAI, Inc.", "Open AI"]:
        assert resolve(variant) == "OpenAI"


def test_seed_has_50():
    assert len(load_canonical()) == 50


def test_unknown_passes_through():
    assert resolve("Some Brand New Startup XYZ") == "Some Brand New Startup XYZ"


# --- resolve_with_confidence ---

def test_domain_match_beats_name():
    # Domain evidence outranks the name: even a mangled name resolves
    # via its known domain.
    name, confidence, method = resolve_with_confidence("openai", "openai.com")
    assert (name, confidence, method) == ("OpenAI", 1.0, "domain")


def test_domain_match_from_full_url():
    name, confidence, method = resolve_with_confidence(
        "whatever", "https://www.anthropic.com/news")
    assert (name, confidence, method) == ("Anthropic", 1.0, "domain")


def test_domain_match_strips_subdomains():
    name, confidence, method = resolve_with_confidence(
        "x", "blog.anthropic.com")
    assert (name, confidence, method) == ("Anthropic", 1.0, "domain")


def test_exact_match():
    name, confidence, method = resolve_with_confidence("Anthropic")
    assert (name, confidence, method) == ("Anthropic", 1.0, "exact")


def test_fuzzy_match_confidence_is_ratio():
    name, confidence, method = resolve_with_confidence("Open AII")
    assert name == "OpenAI"
    assert method == "fuzzy"
    assert 0.9 <= confidence < 1.0


def test_unmatched_passes_through_with_zero_confidence():
    name, confidence, method = resolve_with_confidence(
        "Some Brand New Startup XYZ")
    assert (name, confidence, method) == (
        "Some Brand New Startup XYZ", 0.0, "unmatched")


def test_unknown_domain_does_not_block_name_match():
    # An unknown domain is ignored; the name can still match exactly.
    name, confidence, method = resolve_with_confidence(
        "OpenAI", "some-unknown-domain.example")
    assert (name, confidence, method) == ("OpenAI", 1.0, "exact")


def test_seed_domains_cover_seed_names():
    domains = load_canonical_domains()
    assert len(domains) == 50
    assert set(domains.values()) == set(load_canonical())

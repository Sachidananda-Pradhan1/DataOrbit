"""Issue B tests: deterministic AI-relevance filtering for startups
(YC) and products (BetaList), applied BEFORE the 1,000 cap.

ALL MOCKED — no live network in this file. Async crawlers are driven
with asyncio.run() (pytest-asyncio is not installed), and
scripts/run_bulk.py is loaded from its file path (scripts/ is not a
package), as in tests/test_scripts.py. The keyless live smoke test
(first ~100 YC sitemap entries) is a manual check, not a pytest case.
"""
import asyncio
import html
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.crawlers import products, startups
from src.crawlers.products import ProductCrawler
from src.crawlers.startups import StartupCrawler
from src.models import Source, Startup
from src.storage import dedup

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "run_bulk.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_bulk = _load_script("run_bulk_ai_relevance")


@pytest.fixture(autouse=True)
def _reset_dedup():
    dedup.reset()
    yield
    dedup.reset()


def _raising_llm(*args, **kwargs):
    raise RuntimeError("all LLM tiers failed: not configured")


def _stats():
    return {"duplicates": 0, "parse_errors": 0, "llm_extracted": 0,
            "llm_skipped_no_key": 0, "llm_errors": 0}


# --- fixtures: hand-built source snippets (real page shapes) ---

def _yc_page(slug, name, one_liner, tags, long_description=None,
             with_name=True, with_title=True):
    """YC company page: embedded JSON blob + meta/og tags, mirroring
    the real layout (verified live 2026-11-22)."""
    data = {"slug": slug}
    if with_name:
        data["name"] = name
    data["one_liner"] = one_liner
    data["tags"] = tags
    if long_description is not None:
        data["long_description"] = long_description
    blob = html.escape(json.dumps(data), quote=True)
    head = (
        f'<meta content="{one_liner} Founded in 2024 by A B, {name} has '
        f'10 employees based in San Francisco." name="description" />'
    )
    if with_title:
        head += (
            f'<meta content="{name}: {one_liner} | Y Combinator" '
            'property="og:title" />'
            f"<title>{name}: {one_liner} | Y Combinator</title>"
        )
    return (f"<html><head>{head}</head><body>"
            f'<div data-props="{blob}"></div></body></html>')


def _bl_page(name, tagline, topics=()):
    """BetaList product page: og:title, meta description, <h1> and
    an optional Topics section, mirroring the real layout."""
    topics_html = ""
    if topics:
        links = "".join(
            f'<a href="/browse/{slug}">{label}</a>'
            for slug, label in topics)
        topics_html = f"<dl><dt>Topics</dt><dd>{links}</dd></dl>"
    return (
        "<html><head>"
        f'<meta content="{name}: {tagline}" property="og:title" />'
        f'<meta content="{tagline}" name="description" />'
        f"<title>{name}: {tagline} | BetaList</title>"
        "</head><body>"
        f"<h1>{name}</h1>{topics_html}"
        "</body></html>"
    )


# --- startups: pure relevance verdict ---

def test_ai_startup_passes_on_tag_and_tagline():
    # Real shape: COACH (YC S26), tags include "AI".
    assert startups.is_ai_relevant(
        name="COACH",
        one_liner="The AI Coach for in person sales reps.",
        tags=["SaaS", "B2B", "Sales", "Sales Enablement", "AI"])


def test_ai_startup_passes_on_keyword_only():
    assert startups.is_ai_relevant(
        name="FraudShield",
        one_liner="Machine learning fraud detection for banks.",
        tags=["SaaS", "Fintech"])


def test_ai_startup_passes_on_description_only():
    assert startups.is_ai_relevant(
        name="WarehouseCo",
        one_liner="Software for warehouses.",
        description="Our computer vision system tracks every pallet.",
        tags=["Logistics"])


def test_ai_tag_alone_passes():
    assert startups.is_ai_relevant(
        name="QuietCo", one_liner="We make software.", tags=["AI"])


def test_non_ai_startup_fails():
    # Real shape: Airbnb (travel marketplace).
    assert not startups.is_ai_relevant(
        name="Airbnb",
        one_liner="Book accommodations around the world.",
        tags=["Marketplace", "Travel"])
    # Food delivery.
    assert not startups.is_ai_relevant(
        name="QuickBite",
        one_liner="Hot meals delivered to your door in 30 minutes.",
        tags=["Food", "Delivery"])
    # Fintech.
    assert not startups.is_ai_relevant(
        name="PayBridge",
        one_liner="Payments infrastructure for online businesses.",
        tags=["Fintech", "Payments"])


def test_startup_keyword_matching_is_case_insensitive():
    assert startups.is_ai_relevant(
        one_liner="ARTIFICIAL INTELLIGENCE for customer support.")
    assert startups.is_ai_relevant(
        one_liner="Ai-PoWeReD analytics for sales teams.")
    assert startups.is_ai_relevant(tags=["ai"])
    assert startups.is_ai_relevant(tags=["Machine Learning"])


def test_startup_substring_matches_do_not_count():
    # "said", "brain", "training", "domain" all contain "ai"/"ml"
    # as substrings — whole-word matching must not fire on them.
    assert not startups.is_ai_relevant(
        name="Said Corp",
        one_liner="We said it first: training for the brain.",
        description="Domain expertise in logistics.",
        tags=["Consulting"])
    assert not startups.is_ai_relevant(
        one_liner="Chairman of the board hires domain experts.")


def test_startup_empty_fields_fail():
    assert not startups.is_ai_relevant()
    assert not startups.is_ai_relevant(name="", one_liner=None, tags=[])


# --- startups: page-level relevance ---

def test_yc_page_relevance():
    crawler = StartupCrawler()
    ai_page = _yc_page("coach", "COACH",
                       "The AI Coach for in person sales reps.",
                       ["SaaS", "B2B", "Sales", "AI"])
    assert crawler.is_ai_relevant_page(ai_page)
    travel_page = _yc_page("airbnb", "Airbnb",
                           "Book accommodations around the world.",
                           ["Marketplace", "Travel"])
    assert not crawler.is_ai_relevant_page(travel_page)


def test_yc_page_relevance_uses_long_description():
    page = _yc_page("warehouseco", "WarehouseCo",
                    "Software for warehouses.", ["Logistics"],
                    long_description="Our computer vision system "
                    "tracks every pallet in real time.")
    assert StartupCrawler().is_ai_relevant_page(page)


def test_yc_page_relevance_falls_back_to_title():
    # No embedded JSON: og:title carries the one-liner.
    page = ("<html><head>"
            '<meta content="OpenAI: AI research and deployment. '
            '| Y Combinator" property="og:title" />'
            "<title>OpenAI: AI research and deployment. "
            "| Y Combinator</title>"
            "</head><body><p>no embedded json here</p></body></html>")
    assert StartupCrawler().is_ai_relevant_page(page)


def test_yc_page_title_without_one_liner_does_not_crash():
    # Live pages exist with an empty one-liner ("Name: | Y
    # Combinator"); stripping the suffix must not break the split.
    page = ("<html><head>"
            '<meta content="PlainCo: | Y Combinator" property="og:title" />'
            "<title>PlainCo: | Y Combinator</title>"
            "</head><body><p>nothing else here</p></body></html>")
    assert not StartupCrawler().is_ai_relevant_page(page)


def test_yc_page_relevance_empty_and_unstructured_fail():
    crawler = StartupCrawler()
    assert not crawler.is_ai_relevant_page("")
    assert not crawler.is_ai_relevant_page("   ")
    assert not crawler.is_ai_relevant_page(
        "<html><body>no name or metadata anywhere</body></html>")


# --- products: pure relevance verdict ---

def test_ai_product_passes_on_topic():
    # Real shape: Mova on BetaList — ai/ topic, neutral tagline.
    assert products.is_ai_relevant(
        name="Mova",
        tagline="Find daily matching roles and tailor applications.",
        topics=[("hr-recruiting/jobsearch", "Job search"),
                ("ai/artificial-intelligence", "Artificial Intelligence")])


def test_ai_product_passes_on_keyword_only():
    assert products.is_ai_relevant(
        name="Tiquiz",
        tagline="Build AI-powered quizzes that segment leads.",
        topics=[])
    assert products.is_ai_relevant(
        name="NoteMind",
        tagline="MACHINE LEARNING notes that organise themselves.")


def test_non_ai_product_fails():
    assert not products.is_ai_relevant(
        name="TheWijn.org",
        tagline="Explore curated wine facts, terms, events, and producers.",
        topics=[("commerce/wine-and-spirits", "Wine And Spirits"),
                ("commerce/food-and-beverages", "Food & Beverages")])
    assert not products.is_ai_relevant(
        name="QuestPlanner",
        tagline="A cozy planner that turns daily tasks into quests.")


def test_product_keyword_matching_is_case_insensitive():
    assert products.is_ai_relevant(tagline="Your AI ASSISTANT for email.")
    assert products.is_ai_relevant(tagline="ChatGPT-powered writing help.")


def test_product_substring_matches_do_not_count():
    assert not products.is_ai_relevant(
        tagline="Training plans for the brain, said our coaches.")


def test_product_empty_fields_fail():
    assert not products.is_ai_relevant()
    assert not products.is_ai_relevant(name="", tagline=None, topics=[])


# --- products: page-level relevance ---

def test_betalist_page_relevance():
    crawler = ProductCrawler()
    ai_page = _bl_page("Mova",
                       "AI job search that finds daily matching roles.",
                       topics=[("ai/artificial-intelligence",
                                "Artificial Intelligence")])
    assert crawler.is_ai_relevant_page(ai_page)
    wine_page = _bl_page("TheWijn.org",
                         "Explore curated wine facts, terms, and producers.",
                         topics=[("commerce/wine-and-spirits",
                                  "Wine And Spirits")])
    assert not crawler.is_ai_relevant_page(wine_page)


def test_betalist_page_without_topics_uses_tagline():
    assert ProductCrawler().is_ai_relevant_page(
        _bl_page("Tiquiz", "Build AI-powered quizzes for lead gen."))
    assert not ProductCrawler().is_ai_relevant_page(
        _bl_page("QuestPlanner", "A cozy planner for daily quests."))
    assert not ProductCrawler().is_ai_relevant_page("")


# --- runner: filter before the cap ---

class _FakeStartupCrawler(StartupCrawler):
    """Real parse + real relevance filter; fake discovery over a
    fixed URL list (slices to the requested limit, like the real
    discover). Records the limits it was asked for."""

    def __init__(self, urls):
        self._urls = urls
        self.limits_asked = []

    async def discover(self, limit=1000):
        self.limits_asked.append(limit)
        return list(self._urls)[:limit]


class _FakeProductCrawler(ProductCrawler):
    def __init__(self, urls):
        self._urls = urls
        self.limits_asked = []

    async def discover(self, limit=1000):
        self.limits_asked.append(limit)
        return list(self._urls)[:limit]


class _UnfilteredCrawler:
    """A crawler with no is_ai_relevant_page: treated as unfiltered."""

    def __init__(self, urls):
        self._urls = urls

    async def discover(self, limit=1000):
        return list(self._urls)[:limit]

    async def parse(self, url, raw):
        return [Startup(source=Source(name="Test", url=url),
                        entityName="Anything Corp",
                        collectedAt=datetime.now(UTC))]


@pytest.fixture
def _stub_fetch(monkeypatch):
    """Stub the fetcher + raw-save for collect_entities tests, and
    make the crawlers' internal LLM fallbacks raise so no test can
    ever reach a real LLM. Returns the URL -> HTML page map."""
    pages = {}

    async def fake_fetch_many(urls):
        return {u: pages[u] for u in urls if u in pages}

    async def fake_fetch_many_stream(urls, batch_size=100, failures=None):
        # collect_entities streams in bounded batches; the fake yields
        # the same pages in URL order (failures never occur here).
        for u in urls:
            if u in pages:
                yield u, pages[u]

    monkeypatch.setattr("src.crawlers.fetcher.fetch_many", fake_fetch_many)
    monkeypatch.setattr("src.crawlers.fetcher.fetch_many_stream",
                        fake_fetch_many_stream)
    monkeypatch.setattr(run_bulk, "save_raw", lambda *a, **k: None)
    monkeypatch.setattr(startups, "extract_startup", _raising_llm)
    monkeypatch.setattr(products, "extract_with_fallback", _raising_llm)
    return pages


def test_filter_before_cap_startups(_stub_fetch):
    pages = _stub_fetch
    urls = [
        "https://www.ycombinator.com/companies/bite",
        "https://www.ycombinator.com/companies/paybridge",
        "https://www.ycombinator.com/companies/coach",
        "https://www.ycombinator.com/companies/adalat-ai",
        "https://www.ycombinator.com/companies/neural-warehouse",
    ]
    pages.update({
        urls[0]: _yc_page("bite", "QuickBite",
                          "Hot meals delivered to your door in 30 minutes.",
                          ["Food", "Delivery"]),
        urls[1]: _yc_page("paybridge", "PayBridge",
                          "Payments infrastructure for online businesses.",
                          ["Fintech", "Payments"]),
        urls[2]: _yc_page("coach", "COACH",
                          "The AI Coach for in person sales reps.",
                          ["SaaS", "AI"]),
        urls[3]: _yc_page("adalat-ai", "Adalat AI",
                          "AI OS transforming courts, accelerating justice.",
                          ["GovTech", "Legal", "NLP", "AI"]),
        urls[4]: _yc_page("neural-warehouse", "NeuralWarehouse",
                          "Software for warehouses.",
                          ["Logistics"],
                          long_description="Computer vision tracks pallets."),
    })
    crawler = _FakeStartupCrawler(urls)
    stats = _stats()
    records, mapping = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, None, limit=2, llm_enabled=False,
        name_field="entityName", stats=stats))
    # The first two discovered companies are non-AI: without
    # filter-before-cap the run would stop at 0 AI records.
    assert [r["entityName"] for r in records] == ["COACH", "Adalat AI"]
    assert len(mapping) == 2
    assert stats["startups_filtered_non_ai"] == 2
    assert stats["startups_discovered"] == 4
    assert crawler.limits_asked == [2, 4]  # kept scanning past the cap
    assert "startups_shortfall" not in stats


def test_filter_before_cap_products(_stub_fetch):
    pages = _stub_fetch
    urls = [
        "https://betalist.com/startups/thewijn-org",
        "https://betalist.com/startups/questplanner",
        "https://betalist.com/startups/mova",
        "https://betalist.com/startups/tiquiz",
    ]
    pages.update({
        urls[0]: _bl_page("TheWijn.org",
                          "Explore curated wine facts and producers.",
                          topics=[("commerce/wine-and-spirits",
                                   "Wine And Spirits")]),
        urls[1]: _bl_page("QuestPlanner",
                          "A cozy planner that turns tasks into quests."),
        urls[2]: _bl_page("Mova",
                          "AI job search that finds daily matching roles.",
                          topics=[("ai/artificial-intelligence",
                                   "Artificial Intelligence")]),
        urls[3]: _bl_page("Tiquiz",
                          "Build AI-powered quizzes that segment leads."),
    })
    crawler = _FakeProductCrawler(urls)
    stats = _stats()
    records, mapping = asyncio.run(run_bulk.collect_entities(
        "products", crawler, None, limit=2, llm_enabled=False,
        name_field="startupName", stats=stats))
    assert [r["productName"] for r in records] == ["Mova", "Tiquiz"]
    assert len(mapping) == 2
    assert stats["products_filtered_non_ai"] == 2
    assert crawler.limits_asked == [2, 4]


def test_shortfall_warning_when_source_exhausted(_stub_fetch, capsys):
    pages = _stub_fetch
    urls = [
        "https://www.ycombinator.com/companies/coach",
        "https://www.ycombinator.com/companies/bite",
        "https://www.ycombinator.com/companies/paybridge",
        "https://www.ycombinator.com/companies/airbnb",
    ]
    pages.update({
        urls[0]: _yc_page("coach", "COACH",
                          "The AI Coach for in person sales reps.", ["AI"]),
        urls[1]: _yc_page("bite", "QuickBite",
                          "Hot meals delivered in 30 minutes.", ["Food"]),
        urls[2]: _yc_page("paybridge", "PayBridge",
                          "Payments infrastructure for online businesses.",
                          ["Fintech"]),
        urls[3]: _yc_page("airbnb", "Airbnb",
                          "Book accommodations around the world.",
                          ["Marketplace", "Travel"]),
    })
    crawler = _FakeStartupCrawler(urls)
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", crawler, None, limit=3,
        llm_enabled=False, name_field="entityName", stats=stats))
    assert [r["entityName"] for r in records] == ["COACH"]
    assert stats["startups_shortfall"] == 2
    assert stats["startups_filtered_non_ai"] == 3
    assert crawler.limits_asked == [3, 6, 12]  # scanned to exhaustion
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "only 1 AI-relevant records" in out
    assert "target 3" in out


def test_filtered_pages_never_reach_llm_fallback(_stub_fetch):
    pages = _stub_fetch
    ai_url = "https://www.ycombinator.com/companies/nameless-ai"
    food_url = "https://www.ycombinator.com/companies/nameless-food"
    # No name anywhere (no "name" in the JSON blob, no title), so
    # deterministic parse yields [] and the LLM fallback is the only
    # way to get a record — but only for the AI page.
    pages[ai_url] = _yc_page("nameless-ai", "Nameless AI",
                             "AI platform for support teams.", ["AI"],
                             with_name=False, with_title=False)
    pages[food_url] = _yc_page("nameless-food", "Nameless Food",
                               "Hot meals delivered in 30 minutes.",
                               ["Food"], with_name=False, with_title=False)
    calls = []

    def spy_extractor(raw, domain, url, collected_at):
        calls.append(url)
        return Startup(source=Source(name="Y Combinator", url=url),
                       entityName="Nameless AI", collectedAt=collected_at)

    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", _FakeStartupCrawler([food_url, ai_url]), spy_extractor,
        limit=1, llm_enabled=True, name_field="entityName", stats=stats))
    assert calls == [ai_url]  # the non-AI page never reached the LLM
    assert [r["entityName"] for r in records] == ["Nameless AI"]
    assert stats["startups_filtered_non_ai"] == 1


def test_crawler_without_filter_is_unfiltered(_stub_fetch):
    pages = _stub_fetch
    url = "https://example.com/companies/anything"
    pages[url] = "<html><body>food delivery, no AI here</body></html>"
    stats = _stats()
    records, _ = asyncio.run(run_bulk.collect_entities(
        "startups", _UnfilteredCrawler([url]), None, limit=1,
        llm_enabled=False, name_field="entityName", stats=stats))
    assert len(records) == 1
    assert stats.get("startups_filtered_non_ai", 0) == 0

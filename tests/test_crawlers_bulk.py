"""Bulk crawler tests: arXiv parsing, GitHub stars, Startup/Product
parse + discovery. ALL MOCKED — no live network in this file.

Async crawlers are driven with asyncio.run() (pytest-asyncio is not
installed); asyncio.sleep is monkeypatched to a no-op where a test
exercises polite delays.
"""
import asyncio
from datetime import datetime, timezone

import httpx

from src.crawlers import papers, products, startups
from src.crawlers.products import ProductCrawler
from src.crawlers.startups import StartupCrawler
from src.models import Startup


async def _no_sleep(*args, **kwargs):
    return None


def _raising_llm(*args, **kwargs):
    raise RuntimeError("all LLM tiers failed: not configured")


# --- fixtures: hand-built source snippets ---

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <title>ArXiv Query Results</title>
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>  A Test Paper
      On AI  </title>
    <published>2024-01-02T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
  </entry>
  <entry>
    <title>Paper Without A Link</title>
    <published>2024-01-03T00:00:00Z</published>
    <author><name>Nobody</name></author>
  </entry>
</feed>
"""

HF_ITEM = {"paper": {
    "id": "2401.00001",
    "title": "A Test Paper On AI",
    "authors": [{"name": "Ada Lovelace"}],
    "publishedAt": "2024-01-02T00:00:00.000Z",
    "githubRepo": "https://github.com/owner/repo",
    "githubStars": 7,
}}

HF_ITEM_NEW = {"paper": {
    "id": "2401.99999",
    "title": "A Second Paper",
    "authors": [{"name": "Grace Hopper"}],
    "publishedAt": "2024-01-05T00:00:00.000Z",
    "githubRepo": "https://github.com/other/repo2",
    "githubStars": None,
}}

YC_HTML = """<html><head>
<meta content="Book accommodations around the world. Founded in 2008 by A B, Airbnb has 6,132 employees based in San Francisco." name="description" />
<meta content="Airbnb: Book accommodations around the world. | Y Combinator" property="og:title" />
<title>Airbnb: Book accommodations around the world. | Y Combinator</title>
</head><body><div data-props="{&quot;id&quot;:271,&quot;slug&quot;:&quot;airbnb&quot;,&quot;name&quot;:&quot;Airbnb&quot;,&quot;batch&quot;:&quot;W09&quot;,&quot;one_liner&quot;:&quot;Book accommodations.&quot;,&quot;website&quot;:&quot;http://airbnb.com&quot;}"></div></body></html>"""

YC_TITLE_ONLY_HTML = """<html><head>
<meta content="OpenAI: AI research. | Y Combinator" property="og:title" />
<title>OpenAI: AI research. | Y Combinator</title>
</head><body><p>no embedded json here</p></body></html>"""

YC_WRONG_SLUG_HTML = """<html><head>
<meta content="Airbnb: Book accommodations. | Y Combinator" property="og:title" />
</head><body><div data-props="{&quot;slug&quot;:&quot;otherco&quot;,&quot;name&quot;:&quot;OtherCo&quot;}"></div></body></html>"""

SITEMAP_HTML = """
<a href="https://www.ycombinator.com/companies/airbnb">Airbnb</a>
<a href="/companies/openai">OpenAI</a>
<a href="https://www.ycombinator.com/companies/airbnb">dup</a>
<a href="https://www.ycombinator.com/companies/sitemap">Sitemap</a>
<a href="https://www.ycombinator.com/companies/women-founders">WF</a>
<url><loc>https://www.ycombinator.com/companies/stripe</loc><lastmod>2026-01-01</lastmod></url>
"""

BL_HTML = """<html><head>
<meta content="ZIV Live: Collect every guest's photos" property="og:title" />
<title>ZIV Live: Collect every guest's photos | BetaList</title>
</head><body><h1>ZIV Live</h1></body></html>"""

BL_NO_H1_HTML = """<html><head>
<meta content="Taupia: AI notes" property="og:title" />
<title>Taupia: AI notes | BetaList</title>
</head><body><p>no h1 here</p></body></html>"""

LISTING_HTML = """
<a href="/startups/alpha">A</a><a href="/startups/beta-2">B</a>
<a href="/startups/alpha">dup</a>
<a href="/startups/gamma/testflight">T</a>
<a href="/startups/delta/visit">V</a>
"""


# --- papers: parse_arxiv_atom ---

def test_parse_arxiv_atom_keeps_only_linked_entries():
    result = papers.parse_arxiv_atom(ARXIV_XML)
    assert len(result) == 1  # the entry without a link is excluded
    paper = result[0]
    assert paper.title == "A Test Paper On AI"
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert str(paper.paper_url) == "http://arxiv.org/abs/2401.00001v1"
    assert paper.published_date == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert paper.github_url is None
    assert paper.github_stars is None


def test_parse_arxiv_atom_empty_and_garbage():
    assert papers.parse_arxiv_atom("") == []
    assert papers.parse_arxiv_atom("not xml at all") == []
    assert papers.parse_arxiv_atom("<feed></feed>") == []


def test_arxiv_query_url():
    url = papers.arxiv_query_url("cs.LG", 100, 50)
    assert url.startswith("https://export.arxiv.org/api/query?")
    assert "search_query=cat:cs.LG" in url
    assert "start=100" in url
    assert "max_results=50" in url


def test_arxiv_id_normalisation():
    assert papers._arxiv_id("https://arxiv.org/abs/2401.00001v2") == "2401.00001"
    assert papers._arxiv_id("https://arxiv.org/abs/2401.00001") == "2401.00001"
    assert papers._arxiv_id("https://example.com/not-arxiv") is None


def test_hf_item_to_paper_rejects_bad_id():
    assert papers._hf_item_to_paper({"paper": {
        "id": "not an arxiv id!!", "title": "X",
        "publishedAt": "2024-01-02T00:00:00.000Z"}}) is None


# --- papers: github_stars ---

class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_github_stars_200(monkeypatch):
    monkeypatch.setattr(
        papers.http_util, "get",
        lambda *a, **k: _FakeResp(200, {"stargazers_count": 42}))
    assert papers.github_stars("owner/repo") == 42


def test_github_stars_404_returns_none(monkeypatch):
    monkeypatch.setattr(
        papers.http_util, "get", lambda *a, **k: _FakeResp(404, {}))
    assert papers.github_stars("owner/nope") is None


def test_github_stars_exception_returns_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(papers.http_util, "get", boom)
    assert papers.github_stars("owner/repo") is None


# --- papers: fetch_papers (fake HTTP client) ---

class _FakeHTTPResponse:
    def __init__(self, text="", json_data=None, status_code=200):
        self.text = text
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeAsyncClient:
    """Returns the same arXiv page for every category (cross-listed)
    and canned Hugging Face hits."""

    hf_items = (HF_ITEM,)

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params=None):
        if "export.arxiv.org" in url:
            return _FakeHTTPResponse(text=ARXIV_XML)
        if "huggingface.co" in url:
            return _FakeHTTPResponse(json_data=list(self.hf_items))
        raise AssertionError(f"unexpected URL {url}")


def test_fetch_papers_enriches_arxiv_with_hf_stars(monkeypatch):
    monkeypatch.setattr(papers.http_util, "async_client", _FakeAsyncClient)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    # Stars already present from HF: no live GitHub call must happen.
    monkeypatch.setattr(papers, "github_stars", _raising_llm)
    result = asyncio.run(papers.fetch_papers(target=1))
    assert len(result) == 1
    paper = result[0]
    assert paper.title == "A Test Paper On AI"
    assert str(paper.github_url) == "https://github.com/owner/repo"
    assert paper.github_stars == 7


def test_fetch_papers_dedups_and_tops_up_from_hf(monkeypatch):
    class _Client(_FakeAsyncClient):
        hf_items = (HF_ITEM_NEW,)

    monkeypatch.setattr(papers.http_util, "async_client", _Client)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(papers, "github_stars", lambda owner_repo: 99)
    # arXiv returns the SAME paper for every category -> deduped to 1;
    # HF tops up to the target of 2, and the new paper's stars are
    # fetched live (it has a repo link but no stars from HF).
    result = asyncio.run(papers.fetch_papers(target=2))
    assert len(result) == 2
    assert len({str(p.paper_url) for p in result}) == 2
    new_paper = next(p for p in result if "2401.99999" in str(p.paper_url))
    assert new_paper.github_stars == 99


# --- startups: parse ---

def test_startup_parse_deterministic(monkeypatch):
    # LLM fallback must not be needed (and must not be called).
    monkeypatch.setattr(startups, "extract_startup", _raising_llm)
    result = asyncio.run(StartupCrawler().parse(
        "https://www.ycombinator.com/companies/airbnb", YC_HTML))
    assert len(result) == 1
    startup = result[0]
    assert startup.entityName == "Airbnb"
    assert startup.employeeCount == 6132
    assert startup.source.name == "Y Combinator"
    assert str(startup.source.url) == "https://www.ycombinator.com/companies/airbnb"


def test_startup_parse_falls_back_to_title(monkeypatch):
    monkeypatch.setattr(startups, "extract_startup", _raising_llm)
    result = asyncio.run(StartupCrawler().parse(
        "https://www.ycombinator.com/companies/openai", YC_TITLE_ONLY_HTML))
    assert len(result) == 1
    assert result[0].entityName == "OpenAI"
    assert result[0].employeeCount is None


def test_startup_parse_never_uses_another_companys_name(monkeypatch):
    # Embedded JSON is for a different slug: title wins, not "OtherCo".
    monkeypatch.setattr(startups, "extract_startup", _raising_llm)
    result = asyncio.run(StartupCrawler().parse(
        "https://www.ycombinator.com/companies/airbnb", YC_WRONG_SLUG_HTML))
    assert len(result) == 1
    assert result[0].entityName == "Airbnb"


def test_startup_parse_empty_returns_empty_when_llm_fails(monkeypatch):
    monkeypatch.setattr(startups, "extract_startup", _raising_llm)
    crawler = StartupCrawler()
    assert asyncio.run(crawler.parse(
        "https://www.ycombinator.com/companies/x", "")) == []
    assert asyncio.run(crawler.parse(
        "https://www.ycombinator.com/companies/x",
        "<html><body>no name anywhere</body></html>")) == []


def test_startup_parse_llm_fallback(monkeypatch):
    def fake_extract(text, source_name, source_url, collected_at):
        return Startup(source={"name": source_name, "url": source_url},
                       entityName="Fallback Inc", collectedAt=collected_at)

    monkeypatch.setattr(startups, "extract_startup", fake_extract)
    result = asyncio.run(StartupCrawler().parse(
        "https://www.ycombinator.com/companies/fallback",
        "<html><body>unstructured page about Fallback Inc</body></html>"))
    assert len(result) == 1
    assert result[0].entityName == "Fallback Inc"


# --- startups: discovery ---

def test_extract_company_urls():
    # Covers both href links and XML <loc> entries (the real sitemap
    # page is XML); dupes and special directory pages excluded.
    assert startups.extract_company_urls(SITEMAP_HTML) == [
        "https://www.ycombinator.com/companies/airbnb",
        "https://www.ycombinator.com/companies/openai",
        "https://www.ycombinator.com/companies/stripe",
    ]


def test_startup_discover(monkeypatch):
    async def fake_fetch(url):
        assert url == startups.SITEMAP_URL
        return SITEMAP_HTML

    monkeypatch.setattr(startups, "_fetch_text", fake_fetch)
    assert asyncio.run(StartupCrawler().discover(limit=1)) == [
        "https://www.ycombinator.com/companies/airbnb"]
    assert len(asyncio.run(StartupCrawler().discover())) == 3


def test_startup_discover_failure_returns_empty(monkeypatch):
    async def failing_fetch(url):
        raise RuntimeError("network down")

    monkeypatch.setattr(startups, "_fetch_text", failing_fetch)
    assert asyncio.run(StartupCrawler().discover()) == []


# --- products: parse ---

def test_product_parse_deterministic(monkeypatch):
    monkeypatch.setattr(products, "extract_with_fallback", _raising_llm)
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/ziv-live", BL_HTML))
    assert len(result) == 1
    product = result[0]
    assert product.productName == "ZIV Live"
    # BetaList names no company on this page, so startupName is ""
    # — never copied from productName (review fix, item 4).
    assert product.startupName == ""
    assert product.source.name == "BetaList"
    assert str(product.source.url) == "https://betalist.com/startups/ziv-live"


def test_product_parse_falls_back_to_og_title(monkeypatch):
    monkeypatch.setattr(products, "extract_with_fallback", _raising_llm)
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/taupia", BL_NO_H1_HTML))
    assert len(result) == 1
    assert result[0].productName == "Taupia"


def test_product_parse_empty_returns_empty_when_llm_fails(monkeypatch):
    monkeypatch.setattr(products, "extract_with_fallback", _raising_llm)
    crawler = ProductCrawler()
    assert asyncio.run(crawler.parse(
        "https://betalist.com/startups/x", "")) == []
    assert asyncio.run(crawler.parse(
        "https://betalist.com/startups/x",
        "<html><body>no name anywhere</body></html>")) == []


def test_product_parse_llm_fallback(monkeypatch):
    monkeypatch.setattr(
        products, "extract_with_fallback",
        lambda prompt: '{"startupName": "Acme", "productName": "Acme AI"}')
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/acme-ai",
        "<html><body>unstructured page about Acme AI</body></html>"))
    assert len(result) == 1
    assert result[0].productName == "Acme AI"
    assert result[0].startupName == "Acme"


def test_product_llm_fallback_rejects_bad_json(monkeypatch):
    monkeypatch.setattr(products, "extract_with_fallback",
                        lambda prompt: "not json")
    assert asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/x",
        "<html><body>unstructured</body></html>")) == []


# --- products: discovery ---

def test_listing_page_url():
    assert products.listing_page_url(1) == "https://betalist.com/startups"
    assert products.listing_page_url(3) == "https://betalist.com/startups?page=3"


def test_extract_startup_urls():
    assert products.extract_startup_urls(LISTING_HTML) == [
        "https://betalist.com/startups/alpha",
        "https://betalist.com/startups/beta-2",
    ]


def test_product_discover_paginates(monkeypatch):
    pages = {
        "https://betalist.com/startups":
            '<a href="/startups/a-one">1</a><a href="/startups/a-two">2</a>',
        "https://betalist.com/startups?page=2":
            '<a href="/startups/b-three">3</a>',
        "https://betalist.com/startups?page=3":
            "<html><body>no startups here</body></html>",
    }
    fetched = []

    async def fake_fetch(url):
        fetched.append(url)
        return pages[url]

    monkeypatch.setattr(products, "_fetch_text", fake_fetch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    urls = asyncio.run(ProductCrawler().discover(limit=10))
    assert urls == [
        "https://betalist.com/startups/a-one",
        "https://betalist.com/startups/a-two",
        "https://betalist.com/startups/b-three",
    ]
    assert fetched == list(pages)  # stopped after the empty page 3


def test_product_discover_respects_limit(monkeypatch):
    async def fake_fetch(url):
        return ('<a href="/startups/a-one">1</a>'
                '<a href="/startups/a-two">2</a>')

    monkeypatch.setattr(products, "_fetch_text", fake_fetch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    urls = asyncio.run(ProductCrawler().discover(limit=1))
    assert urls == ["https://betalist.com/startups/a-one"]


def test_product_discover_stops_on_fetch_failure(monkeypatch):
    async def fake_fetch(url):
        if "page=2" in url:
            raise RuntimeError("network down")
        return '<a href="/startups/a-one">1</a>'

    monkeypatch.setattr(products, "_fetch_text", fake_fetch)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    assert asyncio.run(ProductCrawler().discover(limit=10)) == [
        "https://betalist.com/startups/a-one"]

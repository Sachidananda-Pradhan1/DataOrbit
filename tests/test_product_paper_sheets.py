"""Tests for the MODELS + PRODUCTS + PAPERS + SHEETS workstream
(review fix round 2): pricingModel/startupName, Startup.website,
arXiv abstract GitHub scan, spec sheet headers + nesting, 50k-cell
truncation, and the verify_sheets full_text check.

ALL MOCKED — no live network, no LLM calls, no real Google Sheets.
Async crawlers are driven with asyncio.run() (pytest-asyncio is not
installed). The verify_sheets script is loaded from its file path
via importlib (scripts/ is not a package), as in test_verify_sheets.py.
"""
import asyncio
import importlib.util
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.crawlers import papers, products, startups
from src.crawlers.products import ProductCrawler
from src.crawlers.startups import StartupCrawler
from src.export import sheets
from src.models import Product, ResearchPaper, Startup

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_sheets = _load_script("verify_sheets")


def _raising_llm(*args, **kwargs):
    raise RuntimeError("all LLM tiers failed: not configured")


# --- fixtures ---

BL_HTML = """<html><head>
<meta content="ZIV Live: Collect every guest's photos" property="og:title" />
<title>ZIV Live: Collect every guest's photos | BetaList</title>
</head><body><h1>ZIV Live</h1><p>Collect every guest's photos.</p></body></html>"""

BL_COMPANY_HTML = """<html><head>
<meta content="Mova: AI job search" property="og:title" />
<title>Mova: AI job search | BetaList</title>
</head><body><h1>Mova</h1>
<dl><dt>Company</dt><dd>Mova Labs Inc</dd>
<dt>Topics</dt><dd><a href="/browse/ai/artificial-intelligence">AI</a></dd></dl>
</body></html>"""


def _bl_pricing_page(body: str) -> str:
    return (f"""<html><head><title>Priced: AI thing | BetaList</title>
</head><body><h1>Priced</h1><p>{body}</p></body></html>""")


YC_WEBSITE_HTML = """<html><head>
<meta content="Airbnb: Book accommodations. | Y Combinator" property="og:title" />
</head><body><div data-props="{&quot;slug&quot;:&quot;airbnb&quot;,&quot;name&quot;:&quot;Airbnb&quot;,&quot;website&quot;:&quot;https://www.airbnb.com&quot;}"></div></body></html>"""

YC_BAD_WEBSITE_HTML = """<html><head>
<meta content="BadCo: AI thing. | Y Combinator" property="og:title" />
</head><body><div data-props="{&quot;slug&quot;:&quot;badco&quot;,&quot;name&quot;:&quot;BadCo&quot;,&quot;website&quot;:&quot;not a url&quot;}"></div></body></html>"""


def _arxiv_xml(summary: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>A Test Paper On AI</title>
    <published>2024-01-02T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
    <summary>{summary}</summary>
  </entry>
</feed>
"""


# --- models: pricingModel ---

@pytest.mark.parametrize("value", ["FREE", "FREEMIUM", "PAID", "ENTERPRISE"])
def test_product_pricing_model_accepts_enum_values(value):
    p = Product(source={"name": "BetaList", "url": "https://example.com/p"},
                startupName="Acme", productName="Acme AI",
                pricingModel=value, collectedAt=NOW)
    assert p.pricingModel == value


@pytest.mark.parametrize("value", ["SUBSCRIPTION", "free-tier", "", 123, True])
def test_product_pricing_model_invalid_becomes_none_not_error(value):
    p = Product(source={"name": "BetaList", "url": "https://example.com/p"},
                startupName="Acme", productName="Acme AI",
                pricingModel=value, collectedAt=NOW)
    assert p.pricingModel is None


def test_product_pricing_model_case_insensitive_and_none():
    kwargs = {"source": {"name": "BetaList", "url": "https://example.com/p"},
              "startupName": "Acme", "productName": "Acme AI",
              "collectedAt": NOW}
    assert Product(pricingModel="free", **kwargs).pricingModel == "FREE"
    assert Product(pricingModel="  paid ", **kwargs).pricingModel == "PAID"
    assert Product(pricingModel=None, **kwargs).pricingModel is None
    assert Product(**kwargs).pricingModel is None


def test_product_startup_name_defaults_to_empty():
    p = Product(source={"name": "BetaList", "url": "https://example.com/p"},
                productName="Acme AI", collectedAt=NOW)
    assert p.startupName == ""


# --- models: Startup.website + ResearchPaper version fields ---

def test_startup_website_field():
    s = Startup(source={"name": "Y Combinator",
                        "url": "https://example.com/companies/acme"},
                entityName="Acme", collectedAt=NOW)
    assert s.website is None
    s2 = Startup(source={"name": "Y Combinator",
                         "url": "https://example.com/companies/acme"},
                 entityName="Acme", website="https://acme.com",
                 collectedAt=NOW)
    assert s2.website == "https://acme.com"


def test_research_paper_carries_schema_version_and_record_type():
    p = ResearchPaper(title="T", authors=["A"],
                      paper_url="https://arxiv.org/abs/1234.5678",
                      published_date=NOW)
    assert p.schemaVersion == "1.0.0"
    assert p.recordType == "RESEARCH_PAPER"


# --- products: startupName never copied from productName ---

def test_product_parse_never_copies_product_name(monkeypatch):
    monkeypatch.setattr(products, "extract_with_fallback", _raising_llm)
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/ziv-live", BL_HTML))
    assert len(result) == 1
    product = result[0]
    assert product.productName == "ZIV Live"
    assert product.startupName == ""  # no company field on the page
    assert product.startupName != product.productName


def test_product_parse_uses_labelled_company_field(monkeypatch):
    monkeypatch.setattr(products, "extract_with_fallback", _raising_llm)
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/mova", BL_COMPANY_HTML))
    assert len(result) == 1
    assert result[0].productName == "Mova"
    assert result[0].startupName == "Mova Labs Inc"


# --- products: deterministic pricingModel ---

@pytest.mark.parametrize("body,expected", [
    ("Start on our free plan today.", "FREE"),
    ("Completely free, no credit card required.", "FREE"),
    ("Freemium: free tier plus paid Pro.", "FREEMIUM"),
    ("Free plan available; Pro starts at $29/month.", "FREEMIUM"),
    ("Pro is $29/month, billed annually.", "PAID"),
    ("Enterprise pricing — contact us for a quote.", "ENTERPRISE"),
    ("Custom pricing for large teams.", "ENTERPRISE"),
    ("Free up your time with AI automation.", None),  # "free" alone
    ("We raised $5M to build the future.", None),  # funding, not pricing
    ("An AI copilot for your inbox.", None),  # no pricing stated
])
def test_product_parse_deterministic_pricing(body, expected, monkeypatch):
    monkeypatch.setattr(products, "extract_with_fallback", _raising_llm)
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/priced", _bl_pricing_page(body)))
    assert len(result) == 1
    assert result[0].pricingModel == expected


# --- products: LLM fallback parses pricingModel via parse_llm_json ---

def test_product_llm_fallback_parses_fenced_json(monkeypatch):
    # ```json fences would break a bare json.loads — this proves the
    # fallback goes through parse_llm_json.
    monkeypatch.setattr(
        products, "extract_with_fallback",
        lambda prompt: 'Here is the product:\n```json\n'
                       '{"startupName": "Acme Corp", "productName": "Acme AI",'
                       ' "pricingModel": "FREEMIUM"}\n```')
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/acme-ai",
        "<html><body>unstructured page about Acme AI</body></html>"))
    assert len(result) == 1
    assert result[0].startupName == "Acme Corp"
    assert result[0].productName == "Acme AI"
    assert result[0].pricingModel == "FREEMIUM"


def test_product_llm_fallback_bad_pricing_keeps_record(monkeypatch):
    monkeypatch.setattr(
        products, "extract_with_fallback",
        lambda prompt: '{"startupName": "Acme", "productName": "Acme AI",'
                       ' "pricingModel": "PAY_PER_USE"}')
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/acme-ai",
        "<html><body>unstructured page about Acme AI</body></html>"))
    assert len(result) == 1
    assert result[0].pricingModel is None


def test_product_llm_fallback_missing_startup_name_keeps_record(monkeypatch):
    monkeypatch.setattr(products, "extract_with_fallback",
                        lambda prompt: '{"productName": "Acme AI"}')
    result = asyncio.run(ProductCrawler().parse(
        "https://betalist.com/startups/acme-ai",
        "<html><body>unstructured page about Acme AI</body></html>"))
    assert len(result) == 1
    assert result[0].startupName == ""


# --- startups: website extraction ---

def test_startup_parse_extracts_website(monkeypatch):
    monkeypatch.setattr(startups, "extract_startup", _raising_llm)
    result = asyncio.run(StartupCrawler().parse(
        "https://www.ycombinator.com/companies/airbnb", YC_WEBSITE_HTML))
    assert len(result) == 1
    assert result[0].entityName == "Airbnb"
    assert result[0].website == "https://www.airbnb.com"


def test_startup_parse_website_absent_is_none(monkeypatch):
    monkeypatch.setattr(startups, "extract_startup", _raising_llm)
    html = ("<html><head><title>OpenAI: AI research. | Y Combinator</title>"
            "</head><body><p>no embedded json here</p></body></html>")
    result = asyncio.run(StartupCrawler().parse(
        "https://www.ycombinator.com/companies/openai", html))
    assert len(result) == 1
    assert result[0].website is None


def test_startup_parse_invalid_website_is_none(monkeypatch):
    monkeypatch.setattr(startups, "extract_startup", _raising_llm)
    result = asyncio.run(StartupCrawler().parse(
        "https://www.ycombinator.com/companies/badco", YC_BAD_WEBSITE_HTML))
    assert len(result) == 1
    assert result[0].entityName == "BadCo"
    assert result[0].website is None


# --- papers: arXiv abstract GitHub scan ---

@pytest.mark.parametrize("text,expected", [
    ("Code at https://github.com/acme/transformer-x.",
     "https://github.com/acme/transformer-x"),
    ("See https://github.com/acme/repo.git for code.",
     "https://github.com/acme/repo"),
    ("https://github.com/acme/repo/issues/42",
     "https://github.com/acme/repo"),  # deep path dropped
    ("(https://github.com/acme/repo), and more text",
     "https://github.com/acme/repo"),
    ("https://www.github.com/acme/repo", "https://github.com/acme/repo"),
    ("Browse https://github.com/topics/machine-learning",
     None),  # site page, not a repo
    ("No code link in this abstract.", None),
    ("", None),
])
def test_github_url_from_text(text, expected):
    assert papers.github_url_from_text(text) == expected


def test_parse_arxiv_atom_attaches_github_url_from_abstract():
    summary = ("We present a new model. Code is available at "
               "https://github.com/acme/transformer-x.")
    result = papers.parse_arxiv_atom(_arxiv_xml(summary))
    assert len(result) == 1
    assert str(result[0].github_url) == "https://github.com/acme/transformer-x"
    assert result[0].github_stars is None  # scan attaches the URL, not stars


def test_parse_arxiv_atom_no_github_link_is_none():
    result = papers.parse_arxiv_atom(
        _arxiv_xml("We present a new model. No code link here."))
    assert len(result) == 1
    assert result[0].github_url is None


# --- sheets: headers ---

def test_every_tab_has_schema_version_and_record_type():
    for tab in sheets.SHEET_TABS:
        headers = sheets.HEADERS[tab]
        assert "schemaVersion" in headers, tab
        assert "recordType" in headers, tab


def test_startup_and_product_headers_use_spec_content_paths():
    assert "content.entityName" in sheets.STARTUP_HEADERS
    assert "content.data.employeeCount" in sheets.STARTUP_HEADERS
    assert "entityName" not in sheets.STARTUP_HEADERS
    assert "employeeCount" not in sheets.STARTUP_HEADERS
    assert "content.startupName" in sheets.PRODUCT_HEADERS
    assert "content.productName" in sheets.PRODUCT_HEADERS
    assert "content.pricingModel" in sheets.PRODUCT_HEADERS
    # Papers / Jobs / News keep their flat spec field names.
    assert "paper_url" in sheets.PAPER_HEADERS
    assert "company" in sheets.JOB_HEADERS
    assert "full_text" in sheets.NEWS_HEADERS


# --- sheets: nesting ---

def test_nest_row_startup():
    nested = sheets._nest_row("Startups", {
        "entityName": "Acme AI", "employeeCount": 12,
        "source": {"name": "Y Combinator", "url": "https://example.com/a"},
        "collectedAt": "2026-11-21T00:00:00Z"})
    assert nested["schemaVersion"] == "1.0.0"
    assert nested["recordType"] == "STARTUP"
    assert nested["content"] == {"entityName": "Acme AI",
                                 "data": {"employeeCount": 12}}
    assert nested["source"]["name"] == "Y Combinator"


def test_nest_row_product():
    nested = sheets._nest_row("Products", {
        "startupName": "Acme", "productName": "Acme AI",
        "pricingModel": "FREEMIUM",
        "source": {"name": "BetaList", "url": "https://example.com/p"},
        "collectedAt": "2026-11-21T00:00:00Z"})
    assert nested["recordType"] == "PRODUCT"
    assert nested["content"] == {"startupName": "Acme",
                                 "productName": "Acme AI",
                                 "pricingModel": "FREEMIUM"}


def test_nest_row_fills_missing_version_and_never_mutates():
    row = {"entityName": "Acme AI", "employeeCount": None}
    nested = sheets._nest_row("Startups", row)
    assert nested["schemaVersion"] == "1.0.0"
    assert nested["recordType"] == "STARTUP"
    assert nested["content"]["data"]["employeeCount"] is None
    assert "content" not in row  # input untouched
    assert "schemaVersion" not in row


def test_nest_row_preserves_stored_version_fields():
    nested = sheets._nest_row("Jobs", {"schemaVersion": "2.0.0",
                                       "recordType": "JOB", "company": "Acme"})
    assert nested["schemaVersion"] == "2.0.0"
    assert nested["recordType"] == "JOB"


def test_row_values_walks_nested_content_headers():
    row = sheets._nest_row("Products", {
        "startupName": "Acme", "productName": "Acme AI",
        "pricingModel": "PAID",
        "source": {"name": "BetaList", "url": "https://example.com/p"},
        "collectedAt": "2026-11-21T00:00:00Z"})
    values = sheets._row_values(sheets.PRODUCT_HEADERS, row)
    assert values == ["1.0.0", "PRODUCT", "Acme", "Acme AI", "PAID",
                      "BetaList", "https://example.com/p",
                      "2026-11-21T00:00:00Z"]


# --- sheets: 50k-cell truncation ---

def test_row_values_truncates_overlong_cells():
    long_text = "x" * 60_000
    values = sheets._row_values(["full_text"], {"full_text": long_text})
    assert len(values[0]) == sheets.MAX_CELL_CHARS
    assert values[0] == "x" * (sheets.MAX_CELL_CHARS - 3) + "..."


def test_row_values_leaves_short_and_nonstring_values_alone():
    at_limit = "y" * sheets.MAX_CELL_CHARS
    values = sheets._row_values(
        ["a", "b", "c"],
        {"a": at_limit, "b": 42, "c": ["one", "two"]})
    assert values[0] == at_limit  # exactly at the limit: untouched
    assert values[1] == 42
    assert values[2] == "one; two"


# --- sheets: export_all round trip with a fake gspread ---

class _FakeWorksheet:
    def __init__(self):
        self.values = None
        self.cleared = False

    def clear(self):
        self.cleared = True

    def update(self, range_name, values):
        self.values = values


class _FakeSpreadsheet:
    def __init__(self):
        self.tabs = {}

    def worksheet(self, title):
        return self.tabs[title]  # KeyError if missing

    def add_worksheet(self, title, rows, cols):
        ws = _FakeWorksheet()
        self.tabs[title] = ws
        return ws


def test_export_all_nests_and_truncates(monkeypatch):
    spreadsheet = _FakeSpreadsheet()
    fake_client = types.SimpleNamespace(
        open_by_key=lambda sheet_id: spreadsheet)
    fake_gspread = types.SimpleNamespace(
        service_account=lambda filename: fake_client)
    monkeypatch.setitem(sys.modules, "gspread", fake_gspread)

    sheets.export_all("sheet-id", {
        "Startups": [{"entityName": "Acme AI", "employeeCount": 12,
                      "source": {"name": "Y Combinator",
                                 "url": "https://example.com/a"},
                      "collectedAt": "2026-11-21T00:00:00Z"}],
        "Products": [{"startupName": "", "productName": "Acme AI",
                      "pricingModel": "FREEMIUM",
                      "source": {"name": "BetaList",
                                 "url": "https://example.com/p"},
                      "collectedAt": "2026-11-21T00:00:00Z"}],
        "News": [{"title": "Big AI news",
                  "source": {"name": "TechCrunch AI",
                             "url": "https://example.com/n"},
                  "published_date": "2026-11-22T10:00:00Z",
                  "full_text": "z" * 60_000}],
    })
    tabs = spreadsheet.tabs
    assert set(tabs) == set(sheets.SHEET_TABS)
    assert tabs["Startups"].values[0] == sheets.STARTUP_HEADERS
    assert tabs["Startups"].values[1] == [
        "1.0.0", "STARTUP", "Acme AI", 12, "Y Combinator",
        "https://example.com/a", "2026-11-21T00:00:00Z"]
    assert tabs["Products"].values[1] == [
        "1.0.0", "PRODUCT", "", "Acme AI", "FREEMIUM", "BetaList",
        "https://example.com/p", "2026-11-21T00:00:00Z"]
    news_row = tabs["News"].values[1]
    full_text_idx = sheets.NEWS_HEADERS.index("full_text")
    assert len(news_row[full_text_idx]) == sheets.MAX_CELL_CHARS
    assert news_row[full_text_idx].endswith("...")


# --- verify_sheets: News full_text check ---

def _news_values(rows):
    """get_all_values() output for a News tab built from the live
    HEADERS, so the fixture cannot drift from sheets.py."""
    headers = sheets.HEADERS["News"]
    values = [list(headers)]
    for title, url, published, full_text in rows:
        row = [""] * len(headers)
        row[headers.index("schemaVersion")] = "1.0.0"
        row[headers.index("recordType")] = "NEWS"
        row[headers.index("title")] = title
        row[headers.index("source.name")] = "TechCrunch AI"
        row[headers.index("source.url")] = url
        row[headers.index("published_date")] = published
        row[headers.index("full_text")] = full_text
        values.append(row)
    return values


def _check(results, name):
    return next(r for r in results if r.name == name)


def test_verify_flags_empty_news_full_text():
    fresh = (NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    values = _news_values([
        ("News 0", "https://example.com/news/0", fresh, "full text here"),
        ("News 1", "https://example.com/news/1", fresh, ""),  # empty
        ("News 2", "https://example.com/news/2", fresh, "   "),  # blank
    ])
    results = verify_sheets.verify_tab("News", values, NOW)
    check = _check(results, "News: full_text non-empty")
    assert check.passed is False
    assert "2 empty full_text" in check.detail
    assert "row 3" in check.detail  # header + 1-based data rows


def test_verify_passes_when_all_news_have_full_text():
    fresh = (NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    values = _news_values([
        ("News 0", "https://example.com/news/0", fresh, "full text here"),
        ("News 1", "https://example.com/news/1", fresh, "more text"),
    ])
    results = verify_sheets.verify_tab("News", values, NOW)
    assert _check(results, "News: full_text non-empty").passed is True
    assert all(r.passed for r in results)

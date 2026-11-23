"""Generate architecture.pdf (max 3 pages) from this script alone.

Usage:  python scripts/build_architecture_pdf.py
Output: architecture.pdf at the repo root.

ReportLab only — no other dependencies. Run `pdfinfo
architecture.pdf` after regenerating to confirm the page count.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "architecture.pdf"

ACCENT = colors.HexColor("#1a5276")
LIGHT = colors.HexColor("#eaf2f8")
LIGHT2 = colors.HexColor("#e8f5e9")
LIGHT3 = colors.HexColor("#fff3e0")
GREY = colors.HexColor("#666666")
BORDER = colors.HexColor("#b0bec5")

TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=17,
                       leading=20, textColor=ACCENT, spaceAfter=2)
SUBTITLE = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10,
                          leading=13, textColor=GREY, spaceAfter=8)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=11.5,
                    leading=14, textColor=ACCENT, spaceBefore=9, spaceAfter=3)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10,
                    leading=13, textColor=ACCENT, spaceBefore=7, spaceAfter=2)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9,
                      leading=12.5, spaceAfter=4)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=12,
                        bulletIndent=4, spaceAfter=2)
CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=8,
                      leading=10.5)
CELL_B = ParagraphStyle("cellB", parent=CELL, fontName="Helvetica-Bold")
CELL_W = ParagraphStyle("cellW", parent=CELL, textColor=colors.white,
                        fontName="Helvetica-Bold")
BOX = ParagraphStyle("box", fontName="Helvetica-Bold", fontSize=8.5,
                     leading=10.5, alignment=1)
BOX_SUB = ParagraphStyle("boxSub", fontName="Helvetica", fontSize=7,
                         leading=9, alignment=1, textColor=GREY)
ARROW = ParagraphStyle("arrow", fontName="Helvetica-Bold", fontSize=11,
                       leading=12, alignment=1, textColor=ACCENT)
CAPTION = ParagraphStyle("caption", fontName="Helvetica-Oblique",
                         fontSize=7.5, leading=9.5, textColor=GREY,
                         spaceBefore=2, spaceAfter=2)


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(f"•&nbsp;&nbsp;{item}", BULLET) for item in items]


def data_table(headers: list[str], rows: list[list[str]],
               widths: list[float]) -> Table:
    data = [[Paragraph(h, CELL_W) for h in headers]]
    data += [[Paragraph(c, CELL) for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f8fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def box(title: str, sub: str) -> list[Paragraph]:
    return [Paragraph(title, BOX), Paragraph(sub, BOX_SUB)]


def diagram() -> Table:
    """Pipeline flow as a styled table (reportlab cannot render Mermaid;
    the Mermaid version lives in architecture.md)."""
    arrow = Paragraph("→", ARROW)
    row1 = [
        box("Discover", "APIs / RSS / sitemaps"),
        arrow,
        box("Fetch", "async aiohttp · Playwright for Hirist"),
        arrow,
        box("Raw store", "saved by URL hash"),
        arrow,
        box("Parse", "trafilatura / BS4 / feedparser"),
        arrow,
        box("Dedup", "url:&lt;sha256&gt; claim"),
    ]
    row2 = [
        box("LLM extract", "Gemini → Groq → DeepSeek → Pro"),
        arrow,
        box("Resolve", "domain-first, deterministic"),
        arrow,
        box("Store", "Postgres JSONB upserts"),
        arrow,
        box("Export", "gspread · 6 Sheet tabs"),
        "",
        "",
    ]
    # Flow reads left-to-right, then continues on the second row.
    t = Table([row1, row2],
              colWidths=[2.6 * cm, 0.5 * cm, 2.9 * cm, 0.5 * cm, 2.9 * cm,
                         0.5 * cm, 2.9 * cm, 0.5 * cm, 2.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), LIGHT),
        ("BACKGROUND", (2, 0), (2, 0), LIGHT),
        ("BACKGROUND", (4, 0), (4, 0), LIGHT2),
        ("BACKGROUND", (6, 0), (6, 0), LIGHT),
        ("BACKGROUND", (8, 0), (8, 0), LIGHT),
        ("BACKGROUND", (0, 1), (0, 1), LIGHT3),
        ("BACKGROUND", (2, 1), (2, 1), LIGHT),
        ("BACKGROUND", (4, 1), (4, 1), LIGHT),
        ("BACKGROUND", (6, 1), (6, 1), LIGHT),
        ("BOX", (0, 0), (0, 0), 0.6, BORDER),
        ("BOX", (2, 0), (2, 0), 0.6, BORDER),
        ("BOX", (4, 0), (4, 0), 0.6, BORDER),
        ("BOX", (6, 0), (6, 0), 0.6, BORDER),
        ("BOX", (8, 0), (8, 0), 0.6, BORDER),
        ("BOX", (0, 1), (0, 1), 0.6, BORDER),
        ("BOX", (2, 1), (2, 1), 0.6, BORDER),
        ("BOX", (4, 1), (4, 1), 0.6, BORDER),
        ("BOX", (6, 1), (6, 1), 0.6, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(1.5 * cm, 1.0 * cm,
                      "DataOrbit — FrontierAtlas Ingestion Pipeline Architecture")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm,
                           f"Page {doc.page}")
    canvas.restoreState()


def build() -> Path:
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.4 * cm, bottomMargin=1.5 * cm,
        title="DataOrbit — FrontierAtlas Ingestion Pipeline Architecture",
        author="DataOrbit",
    )
    W = A4[0] - 3 * cm  # usable width
    story: list = []

    story.append(p("DataOrbit — FrontierAtlas Ingestion Pipeline Architecture",
                   TITLE))
    story.append(p("GraphOne AI Engineer demo task · Scalable ingestion for AI "
                   "startups, products, research papers, jobs and news · "
                   "November 2026", SUBTITLE))
    story.append(p(
        "DataOrbit is a hybrid, raw-first ingestion pipeline: deterministic "
        "parsing where a source is structured (APIs, RSS, sitemaps), LLM "
        "extraction only as a fallback for unstructured pages. Every record "
        "traces to a real source URL — hallucinated data is a disqualifying "
        "failure, so a failed extraction yields no record, never an invented "
        "one. The pipeline runs in two modes: <b>bulk</b> "
        "(<font face='Helvetica-Bold'>run_bulk.py</font>: 1,000 startups, "
        "1,000 products, 1,000 papers) and <b>fresh</b> "
        "(<font face='Helvetica-Bold'>run_fresh.py</font>: all jobs and news "
        "published in the last 24 hours), both writing "
        "<font face='Helvetica-Bold'>data/processed/&lt;Tab&gt;.json</font> "
        "and upserting to Postgres; <b>export_sheets.py</b> writes the six "
        "Sheet tabs."))

    story.append(p("Deliverables", H1))
    story.append(data_table(
        ["Sheet tab", "Target", "Source"],
        [["Startups", "1,000 unique", "Y Combinator directory sitemap"],
         ["Products", "1,000 unique", "BetaList directory pages"],
         ["Research Papers", "1,000, with live GitHub stars",
          "arXiv API + Hugging Face Papers API"],
         ["Jobs", "All fresh in last 24h", "5 boards (see Sources)"],
         ["News", "All fresh in last 24h, full text", "5 RSS feeds"],
         ["Entity Mapping Log", "One row per resolved entity",
          "raw → canonical + confidence + method + source URL"]],
        [3.6 * cm, 4.6 * cm, W - 8.2 * cm]))

    story.append(p("Architecture", H1))
    story.append(diagram())
    story.append(p(
        "Pipeline stages: Discover → Fetch → Raw store → Parse → Dedup → "
        "LLM extract → Resolve → Store → Export. Fetch is async aiohttp "
        "(trust_env proxy-aware, rotating user-agents, bounded concurrency); "
        "Playwright renders JS-heavy pages only (Hirist, a React SPA). "
        "Parsing is trafilatura / BeautifulSoup / feedparser, with all dates "
        "normalised to timezone-aware UTC (ISO-8601, RFC-822 and relative "
        "dates such as “4 hours ago”). The LLM is a fallback, not the "
        "default: startups and products are parsed deterministically first.",
        CAPTION))

    story.append(p("1 · Scale Strategy — 500,000 records without manual intervention", H1))
    story.append(p(
        "<b>Stateless async workers, config-driven scale.</b> Crawlers hold "
        "no state between runs; every tunable — targets, MAX_CONCURRENCY, "
        "timeouts, per-source delays — lives in <b>src/config.py</b> and "
        "<b>.env</b>. Scaling from 3,000 demo records to 500,000 is "
        "infrastructure, not code: raise MAX_CONCURRENCY, run more worker "
        "processes, grow Postgres/Redis capacity."))
    story += bullets([
        ("<b>Discovery is cheap and keyless.</b> One sitemap fetch seeds "
        "6,410 YC company URLs; BetaList pages at ~50 URLs/page; arXiv pages "
        "100 papers/request. Discovery is URL lists only — fetching and "
        "parsing fan out across workers."),
        ("<b>Bounded concurrency everywhere.</b> An asyncio semaphore caps "
        "in-flight fetches; per-source politeness delays are honoured "
        "(arXiv 1 req/3s, BetaList 1s/page, Hirist 10s/page per robots.txt, "
        "GitHub 0.5s/call). Throughput grows by adding workers, never by "
        "hammering a source."),
        ("<b>Raw-first, resumable.</b> Every payload is saved to "
        "data/raw/&lt;kind&gt;/&lt;url-hash&gt;.json before any LLM call, so "
        "a crashed or re-run job re-extracts from disk instead of "
        "re-fetching 500k pages (scripts/re_extract.py re-runs the "
        "crawlers' parse() on data/raw/), and Postgres unique_url "
        "upserts make re-runs idempotent."),
        ("<b>One dead source never kills a run.</b> Each fetcher logs a "
        "warning and the others continue (verified live: VentureBeat 429'd "
        "and the news run still returned 19 fresh items)."),
    ])

    story.append(p("2 · Handling 413s &amp; 429s — context windows and rate limits", H1))
    story.append(p(
        "<b>413s are prevented, not retried.</b> Every LLM call is capped at "
        "6,000 tokens (~24,000 chars): long pages are split on paragraph "
        "boundaries (src/llm/chunker.py), extracted per chunk, and merged "
        "deterministically (src/llm/extractor.py: merge_extractions), so "
        "facts near the end of a long page survive instead of being "
        "truncated away. The products LLM fallback additionally caps "
        "its input at 8,000 chars."))
    story += bullets([
        ("<b>429-only retry with exponential backoff + jitter.</b> "
        "src/utils/retry.py retries only 429 / rate-limit errors — 5 "
        "attempts, 1s doubling delay plus jitter. Any other error moves "
        "straight to the next tier instead of burning ~15s of sleeps."),
        ("<b>Four-tier fallback spreads the load.</b> Gemini gemini-3.6-flash "
        "→ Groq openai/gpt-oss-120b → DeepSeek deepseek-chat → Gemini "
        "gemini-4-hydrogen (last resort, shares the Gemini key). A tier with no "
        "API key is skipped, not attempted; providers are imported lazily, "
        "so the pipeline runs with no LLM SDK installed."),
        ("<b>Per-source rate notes.</b> arXiv: 1 req/3s (usage policy). "
        "GitHub: 60 req/hr unauthenticated, so without a token only the "
        "first 50 star lookups run. Hugging Face: 1s between queries. "
        "LLM concurrency is bounded, not unbounded: extraction calls "
        "run through a semaphore and asyncio.to_thread, so only a "
        "bounded number of extractions run at once and a synchronous "
        "SDK call never blocks the event loop."),
    ])

    story.append(p("3 · Freshness Tracking — never process the same article/job twice", H1))
    story.append(p(
        "<b>Lease state machine on a URL idempotency key.</b> Every URL "
        "maps to the key <b>url:&lt;sha256(url)&gt;</b> "
        "(src/storage/dedup.py) and moves through pending → processing → "
        "completed / failed / dead. A worker claims a URL with a 10-minute "
        "renewable lease (heartbeat) and an ownership token; only the "
        "lease holder can renew, complete or fail it, and atomic Lua "
        "transitions keep two workers from racing a claim. A crashed "
        "worker's lease expires within minutes and the URL becomes "
        "claimable again; failed URLs retry with exponential backoff and "
        "are dead-lettered after 3 attempts. Without Redis the module "
        "falls back to a process-local store with identical semantics. "
        "In this demo the runners are single-process, so leases and "
        "heartbeats run in-process; the Redis work queue feeding N "
        "workers is the production step (see Q4)."))
    story += bullets([
        ("<b>Postgres is the durable backstop.</b> Every table upserts on "
        "unique_url, so even if two workers race past the claim, the "
        "database stores one row."),
        ("<b>24h gate on normalised timestamps.</b> Jobs and news are kept "
        "only when their published date — parsed from ISO-8601, RFC-822 or "
        "relative text and normalised to tz-aware UTC — falls inside the "
        "last 24 hours. An item with no parseable date falls back to its "
        "first-seen timestamp (when its URL was first observed, recorded "
        "in data/processed/first_seen.json), and the same 24h gate "
        "applies to that timestamp: nothing is ever guessed."),
        ("<b>Raw-first storage decouples fetch from extract.</b> Re-runs "
        "re-extract from data/raw/ instead of re-fetching, so a reprocessed "
        "URL costs an LLM call at most — and cross-board job dedup on "
        "normalised company|title plus mapping-log merging catch the rest."),
    ])

    story.append(p("4 · Storage Strategy — primary store and the relationship graph", H1))
    story.append(p(
        "<b>Postgres with JSONB is the primary database.</b> Five tables "
        "(startups, products, papers, jobs, news), each with "
        "<b>unique_url TEXT UNIQUE</b> + <b>payload JSONB</b> + "
        "collected_at. The unique URL gives relational integrity where it "
        "matters — it is the dedup backstop and the traceability key every "
        "record hangs from — while JSONB keeps payloads flexible as schemas "
        "evolve (employeeCount appears on some startups, github_stars on "
        "some papers). Expression indexes on payload →&gt; 'entityName' "
        "and payload →&gt; 'company' serve the queries evaluators run; "
        "upserts make every run idempotent. One Postgres also keeps the "
        "demo's operational surface minimal: no separate document store to "
        "keep in sync."))
    story += bullets([
        ("<b>Redis: dedup now, queue in production.</b> Redis holds the "
        "lease state machine for idempotency keys (implemented in "
        "storage/dedup.py, with a process-local fallback) — ephemeral, "
        "high-throughput state shared across nodes. The Redis work queue "
        "feeding N distributed workers is the production step; this "
        "demo's runners are single-process. Records themselves never "
        "live only in Redis."),
        ("<b>Raw files are the audit trail.</b> data/raw/ keeps every fetched "
        "payload by URL hash: the proof behind every row and the cheap "
        "re-extraction source."),
        ("<b>Graph layer: pgvector first, Neo4j when traversal is the "
        "workload.</b> Startup–product–paper–job relationships are "
        "currently served by the deterministic resolver (domain-first "
        "matching, 50-name + 50-domain seed) and the Entity Mapping Log. "
        "When similarity search arrives (“find startups like X”), pgvector "
        "adds embeddings to the Postgres already running — no new system. "
        "When multi-hop traversal queries arrive (“which startups' products "
        "cite papers by authors who also work at…”), Neo4j is the right "
        "graph store. Neither is implemented in this demo; the JSONB "
        "payloads and the mapping log are the migration path."),
    ])

    story.append(p("Anti-bot posture (design only — no bypass tooling)", H1))
    story.append(p(
        "Challenge pages (Cloudflare / captcha interstitials), 403s and "
        "429s are detected by status code and page markers and treated "
        "as a signal to stop: back off with exponential delay, log the "
        "URL for later review, and fall back to an RSS feed, API or "
        "alternate source — never hammer the endpoint, and never solve "
        "or bypass a captcha. Robots.txt and per-source crawl delays "
        "are honoured (Hirist: 10s). Playwright renders JS-heavy pages "
        "only (Hirist, a React SPA); it is rendering, not evasion."))

    story.append(p("Sources", H1))
    story.append(data_table(
        ["Kind", "Source", "Access"],
        [["News", ("TechCrunch AI · MIT Tech Review AI · VentureBeat AI · "
          "The Verge AI · The Decoder"), "RSS/Atom; full text via trafilatura"],
         ["Jobs", ("RemoteOK · We Work Remotely · Jobicy · Working Nomads · "
          "Hirist (India)"), "JSON APIs / RSS; Hirist via Playwright"],
         ["Startups", "Y Combinator directory (6,410 company URLs)",
          "Sitemap + deterministic page parse"],
         ["Products", "BetaList directory (~50/page)", "Paginated HTML parse"],
         ["Papers", "arXiv (cs.AI/LG/CL/CV) + Hugging Face Papers",
          "APIs; live GitHub stars (50/hr unauth cap)"]],
        [1.8 * cm, 7.6 * cm, W - 9.4 * cm]))
    story.append(Spacer(1, 4))
    story.append(p(
        "Verified live 21 Nov 2026. Notes: VentureBeat was rate-limited "
        "(429) and stale (newest item 27 Aug 2026) — the run continues "
        "without it. The Papers With Code API is dead (302 to Hugging "
        "Face); the Hugging Face Papers API is its successor — documented "
        "and skipped, never faked. The We Work Remotely category feed was "
        "stale (newest 3 Nov), so the crawler falls back to the full feed. "
        "Hirist needs Playwright + Chromium (React SPA). Wellfound is a "
        "stretch goal only — no bypass tooling is implemented. Full detail: "
        "README.md and architecture.md in the repo.", CAPTION))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")

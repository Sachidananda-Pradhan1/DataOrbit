# DataOrbit

DataOrbit is an ingestion pipeline for AI-ecosystem data. It collects
information about AI startups, products, research papers, jobs, and news
from public sources; extracts structured records; resolves entities
against a canonical list; and exports the results to Google Sheets.
Each exported record retains its source URL for traceability.

## What it collects

| Dataset | Coverage |
|---|---|
| Startups | 1,000 unique AI startups |
| Products | 1,000 unique AI products |
| Research Papers | 1,000 papers, with live GitHub star counts |
| Jobs | AI-relevant jobs discovered with a publication date in the last 24 hours |
| News | AI news items discovered in the last 24 hours, with article text where available |
| Entity Mapping Log | One row per resolved entity: raw name, canonical name, confidence, match method, source URL |

The architecture is intended to support 500,000+ records through
infrastructure scaling. See `architecture.pdf` for the proposed design
and scaling approach.

## How it works

```
Discover (APIs / RSS / directories)
  -> Fetch (async HTTP; Playwright only for JS-heavy pages)
  -> Save raw first (data/raw/<kind>/<url-hash>.json, before any LLM call)
  -> Parse + date-normalise (ISO-8601, RFC-822 and relative dates,
     all normalised to tz-aware UTC)
  -> Dedup (URL hash + lease state machine — 10-min renewable leases)
  -> LLM extract, only when deterministic parsing yields nothing
     (4-tier fallback with chunking; see below)
  -> Entity resolve (deterministic, domain-first — never name alone)
  -> Store (Postgres JSONB; JSON files always written, so a missing
     database degrades gracefully)
  -> Export to Google Sheets (6 tabs)
```

## Sources

A failure in one source does not stop the entire run: the affected
fetcher logs a warning, and the remaining fetchers continue.

### News — 5 RSS/Atom feeds, full text extracted per article

| Source | Feed |
|---|---|
| TechCrunch AI | <https://techcrunch.com/category/artificial-intelligence/feed/> |
| MIT Technology Review AI | <https://www.technologyreview.com/topic/artificial-intelligence/feed/> |
| VentureBeat AI | <https://venturebeat.com/category/ai/feed/> |
| The Verge AI | <https://www.theverge.com/rss/ai-artificial-intelligence/index.xml> |
| The Decoder | <https://the-decoder.com/feed/> |

### Jobs — 5 boards, with jobs classified as India or Global by location

| Board | Access |
|---|---|
| RemoteOK | JSON API |
| We Work Remotely | RSS (falls back to the full feed when the category feed is stale) |
| Jobicy | JSON API |
| Working Nomads | JSON API, keyless |
| Hirist | Page parsing with Playwright (React SPA, no feed/API) |

AI relevance is matched against **titles and tags**, not descriptions.
Company descriptions can be generic boilerplate, so using them for
matching may incorrectly include non-AI roles at AI companies. Jobs
without a usable publication date or a traceable source URL are
excluded rather than assigned guessed values.

### Startups — Y Combinator directory

Discovery from the YC companies sitemap
(`ycombinator.com/companies/sitemap`, ~6,400 company profile URLs).
Company pages are parsed deterministically; the LLM is only a
fallback for pages where no name can be parsed. AI relevance is
enforced before the cap: pages are checked against YC tags plus
keyword matching on name/one-liner/description, and the runner scans
in rounds until 1,000 AI-relevant startups are collected or the
source is exhausted.

### Products — BetaList directory

Discovery from `betalist.com/startups?page=N` (server-rendered, ~50
product URLs per page). AI relevance uses the product page's Topics
section plus the shared keyword filter on title/tagline. Product
name comes from the `<h1>`; on BetaList the listing *is* the
startup's product, so `startupName` is the name as listed on the
source page.

### Research papers — arXiv + Hugging Face, live GitHub stars

- **arXiv API**: 4 categories (cs.AI, cs.LG, cs.CL, cs.CV), newest
  first, deduped by paper URL (papers are cross-listed). arXiv's
  1 request / 3s policy is respected.
- **Hugging Face Papers API**: attaches GitHub repo links and star
  counts to arXiv papers (matched on arXiv id) and tops up any
  shortfall.
- **GitHub stars**: live lookups via the GitHub API. With a
  `GITHUB_TOKEN` all papers with a repo link are looked up; without
  one, only the first 50 (unauthenticated limit is 60/hr).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,playwright,llm]"
cp .env.example .env   # add API keys (see Configuration)
playwright install chromium   # only needed for Hirist (JS-rendered)
```

On Windows, create and activate the environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,playwright,llm]"
playwright install chromium   # only needed for Hirist
```

Alternatively, run `run_all.bat` from PowerShell or Command Prompt;
it creates the virtual environment and installs the project dependencies.

### Google Sheets setup

The pipeline exports to a Google Sheet using a service account:

1. Create a blank Google Sheet — the exporter creates the 6 tabs
   itself, so the Sheet just needs to exist.
2. In [Google Cloud Console](https://console.cloud.google.com/),
   create a project, enable the Google Sheets API, and create a
   service account with a JSON key. Save the key as
   `credentials.json` in the project folder.
3. Share the Sheet with the service account's email address
   (Editor role) — otherwise the exporter can't write to it.
4. Copy the Sheet ID from the Sheet's URL
   (`docs.google.com/spreadsheets/d/<ID>/...`) into `.env` as
   `GOOGLE_SHEET_ID`.

Then run everything with one command:

```bash
./run_all.sh        # Mac/Linux (or Git Bash on Windows)
run_all.bat         # Windows, from PowerShell or cmd
```

That runs, in order: bulk collection, fresh jobs and news, GitHub star
enrichment, Google Sheets export, and verification. The current runtime
estimate for collecting 1,000 records of each bulk entity type is about
one hour. Actual runtime will vary with source availability, network
conditions, and API limits.

Postgres + Redis are optional — the runners default to JSON-only
mode (`--no-db`). To use them:

```bash
docker compose up -d   # Postgres + Redis
```

## Usage

```bash
python scripts/run_bulk.py --no-db
# Bulk collection: 1,000 startups, 1,000 products, 1,000 papers.
# Add --limit N to cap each entity kind; --no-llm skips LLM
# enrichment entirely (deterministic parse only).

python scripts/run_fresh.py --no-db
# Fresh jobs + news from the last 24h.
# Add --limit N to cap per board, or --smoke for a 5-per-board,
# JSON-only test run.

python scripts/enrich_stars.py     # GitHub star counts for papers
python scripts/export_batched.py   # writes all 6 tabs to the Sheet
python scripts/verify_sheets.py    # read-only verification (see below)
```

Runners write `data/processed/<Tab>.json` (one file per Sheet tab)
and upsert to Postgres when it is available; a Postgres failure
prints a warning and the run continues JSON-only. Fresh runs are
24h-gated and never reprocess a seen URL.

> Jobs and news are only fresh for 24 hours: re-run `run_fresh.py`
> (and the export) close to when the data is needed, or
> verification will flag stale rows.

## Configuration

`.env` (see `.env.example` — never commit real keys):

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini tiers (Flash + Pro) |
| `GROQ_API_KEY` | Groq tier |
| `DEEPSEEK_API_KEY` | DeepSeek tier |
| `GITHUB_TOKEN` | GitHub star lookups (lifts the 50-paper cap) |
| `GOOGLE_SHEET_ID` | Destination spreadsheet |
| `GOOGLE_SHEETS_CREDENTIALS_JSON` | Service-account JSON (default `credentials.json`) |
| `DATABASE_URL` / `REDIS_URL` | Postgres / Redis (optional) |
| `GEMINI_MODEL` / `GEMINI_PRO_MODEL` / `GROQ_MODEL` / `DEEPSEEK_MODEL` | Model overrides |
| `MAX_CONCURRENCY` | Fetcher concurrency (default 20) |
| `REQUEST_TIMEOUT_SECS` | Per-request timeout (default 30) |

An LLM API key is not required for deterministic parsing. If a provider
is not configured, its tier is skipped; the pipeline can continue without it.

## LLM fallback chain

`src/llm/orchestrator.py` — 4 tiers, tried in order, first success wins:

1. **Gemini** `gemini-3.6-flash` (high volume)
2. **Groq** `openai/gpt-oss-120b`
3. **DeepSeek** `deepseek-chat` (via the OpenAI-compatible client)
4. **Gemini Pro** `gemini-4-hydrogen` (last resort / complex reasoning;
   shares `GEMINI_API_KEY` with tier 1)

Each provider is imported lazily, so the module loads with no LLM
SDK installed; a tier with no API key is skipped, not attempted.
Only 429 / rate-limit errors are retried (exponential backoff +
jitter); any other error moves straight to the next tier. The extractor is designed to avoid oversized requests: it splits long
inputs at paragraph boundaries (`src/llm/chunker.py`), limits each call
to 6,000 tokens, and deterministically merges the per-chunk results.
This helps retain information from later sections of long pages rather
than truncating the input at the beginning. The LLM is
a fallback only: startups and products are parsed deterministically
first, and any LLM failure yields no record — never an invented one.

## Entity resolution

`src/resolver/matcher.py` — deterministic, domain-first, never
name-alone:

1. **domain** (confidence 1.0): the domain of the source URL matches
   the seed domain map (`seed_canonical_domains.json`).
2. **exact** (1.0): normalised name (lowercased, punctuation and
   legal suffixes stripped) matches a seed name exactly.
3. **fuzzy** (0.9–1.0): difflib close match at cutoff 0.9; confidence
   is the difflib ratio.
4. **unmatched** (0.0): the name passes through unchanged — a
   canonical form is never invented.

So "OpenAI", "OpenAI, Inc." and "Open AI" all resolve to "OpenAI".
Every resolution is logged to the Entity Mapping Log tab with
`raw_name`, `canonical_name`, `confidence`, `match_method` and
`source_url`.

## Dedup

`src/storage/dedup.py` is a lease state machine in Redis: each URL's
`url:<sha256>` key moves through pending -> processing -> completed
/ failed / dead, with atomic transitions and an ownership token per
claim (only the lease holder can renew, complete or fail a URL).
Leases last 10 minutes and are renewed by heartbeat, so a crashed
worker's URLs become claimable again within minutes; failed URLs
retry with exponential backoff and are dead-lettered after 3
attempts. A process-local in-memory fallback with identical
semantics covers runs without Redis, and Postgres `unique_url`
upserts are the durable backstop against duplicates across runs. See
`architecture.pdf` for the distributed design.

## Data quality

Each accepted record must include a valid, traceable source URL.
`src/models.py` is the single source of truth for schemas; the
extractor only fills fields evidenced by fetched raw text (every
prompt instructs it to leave a field null rather than guess);
Pydantic validation rejects anything that does not match the schema;
and raw saves in `data/raw/` are the proof. A failed extraction
yields no record, never a fabricated one.

## Verification

`scripts/verify_sheets.py` checks the live Sheet (read-only — it
never writes):

```bash
python scripts/verify_sheets.py            # uses GOOGLE_SHEET_ID from .env
python scripts/verify_sheets.py <SHEET_ID> # or pass the ID explicitly
```

The verifier checks that all six tabs exist and have the expected
headers; that startups, products, and papers meet the 1,000-row target;
that jobs and news contain at least one row; that each row has a
non-empty, valid source URL; and that there are no duplicate URLs
within a tab. It also checks that job and news dates fall within 24
hours of the run and that papers with a `github_url` have a
`github_stars` value. The script prints a pass/fail result for each
check and exits with a non-zero status if any check fails.

## Project structure

```
DataOrbit/
├── README.md  architecture.md  architecture.pdf
├── run_all.sh  run_all.bat     # one-command runs (Unix / Windows)
├── pyproject.toml  .env.example  docker-compose.yml
├── data/
│   ├── raw/<kind>/<url-hash>.json      # raw saves, before any LLM call
│   └── processed/<Tab>.json            # one JSON file per Sheet tab
├── src/
│   ├── config.py  models.py            # settings/targets; canonical schemas
│   ├── crawlers/
│   │   ├── base.py  fetcher.py  date_utils.py  first_seen.py
│   │   ├── startups.py  products.py  papers.py  news.py  jobs.py
│   ├── llm/
│   │   ├── orchestrator.py  chunker.py  prompts.py  extractor.py  json_utils.py
│   ├── resolver/
│   │   ├── matcher.py  seed_canonical.json  seed_canonical_domains.json
│   ├── storage/  db.py  dedup.py
│   ├── export/  sheets.py
│   └── utils/  http.py  logging.py  retry.py
├── scripts/
│   ├── run_bulk.py  run_fresh.py  run_common.py
│   ├── enrich_stars.py  export_batched.py  export_sheets.py  export_csvs.py
│   ├── re_extract.py  verify_sheets.py
│   └── build_architecture_pdf.py       # regenerates architecture.pdf
└── tests/                              # 390 tests (see below)
```

## Tests & lint

```bash
pytest       # 390 tests, all passing
ruff check   # clean
```

The test suite covers date parsing, entity resolution, chunking,
models, HTTP helpers, bulk and fresh crawler parsers, runner helpers,
deduplication claims, extractor chunk merging, LLM/Sheets/Postgres
integration points, and Sheet verification (using mocked `gspread`).
Tests use mocks and do not require live API keys.

## Limitations

- **VentureBeat is rate-limited and often stale** (HTTP 429s, and its
  newest items can be months old). The run continues without it; the
  other four news sources carry the tab.
- **Hirist needs Playwright + Chromium**: its listing pages are a
  React SPA — the raw HTML is only a JS shell, so a plain fetch
  finds no jobs. Without Playwright installed, Hirist is skipped
  with a warning and the run continues.
- **Wellfound, Product Hunt and similar are not used**: no public
  API, and their pages block plain HTTP clients. No bypass tooling
  is implemented; anti-bot strategy is outlined in
  `architecture.md`.
- **Papers With Code has no working API** (it redirects to Hugging
  Face since the acquisition); the Hugging Face Papers API is used
  instead.
- **News has no published upstream schema**: the internal `NewsItem`
  model follows the Job schema pattern, with a source, an ISO-8601
  publication timestamp, and article text.

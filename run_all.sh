#!/usr/bin/env bash
# DataOrbit — run the whole pipeline end to end with one command.
#
# Runs, in order: bulk (1,000 startups / products / papers) -> fresh
# jobs + news (last 24h) -> GitHub stars -> export to the Google Sheet
# -> verify. Stops at the first failure (set -e).
#
# Needs .venv + .env first (see README Setup). JSON-only mode (--no-db):
# no Docker / Postgres needed. On Windows, run this from Git Bash.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "No .venv found. Run the Setup steps from the README first:"
  echo "  python3.12 -m venv .venv && source .venv/bin/activate"
  echo "  pip install -e \".[dev,playwright,llm]\""
  exit 1
fi
if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and add your keys."
  exit 1
fi

source .venv/bin/activate
set -a && source .env && set +a

echo "=== 1/5 bulk run — about an hour for 1,000 of each ==="
python scripts/run_bulk.py --no-db
echo "=== 2/5 fresh jobs + news (last 24h) ==="
python scripts/run_fresh.py --no-db
echo "=== 3/5 GitHub stars ==="
python scripts/enrich_stars.py
echo "=== 4/5 export to Google Sheet ==="
python scripts/export_batched.py
echo "=== 5/5 verify ==="
python scripts/verify_sheets.py
echo "DONE — open the Sheet and check all 6 tabs."

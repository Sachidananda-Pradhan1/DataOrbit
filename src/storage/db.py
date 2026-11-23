"""Postgres storage (primary store). JSONB-friendly: models are
stored as validated JSON plus a unique URL column for the fields
evaluators query (entity name, source URL, collected date).

Wired 2026-11-21 with psycopg (imported lazily, so the module loads
with no live database). ``init_db`` creates the five tables;
``store_records`` upserts on the unique URL, so re-running a phase
never duplicates a record. Needs DATABASE_URL in .env — see
docker-compose.yml for the local Postgres.
"""
from __future__ import annotations

TABLES = ("startups", "products", "papers", "jobs", "news")

DDL = """
CREATE TABLE IF NOT EXISTS startups (
    id SERIAL PRIMARY KEY,
    unique_url TEXT UNIQUE NOT NULL,
    payload JSONB NOT NULL,
    collected_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    unique_url TEXT UNIQUE NOT NULL,
    payload JSONB NOT NULL,
    collected_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    unique_url TEXT UNIQUE NOT NULL,
    payload JSONB NOT NULL,
    collected_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    unique_url TEXT UNIQUE NOT NULL,
    payload JSONB NOT NULL,
    collected_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    unique_url TEXT UNIQUE NOT NULL,
    payload JSONB NOT NULL,
    collected_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_startups_payload_name
    ON startups ((payload ->> 'entityName'));
CREATE INDEX IF NOT EXISTS idx_jobs_payload_company
    ON jobs ((payload ->> 'company'));
"""


def upsert_sql(table: str) -> str:
    """INSERT ... ON CONFLICT for one table. Kept separate so the SQL
    is unit-testable without a database."""
    if table not in TABLES:
        raise ValueError(f"unknown table: {table}")
    return (
        f"INSERT INTO {table} (unique_url, payload) VALUES (%s, %s) "
        f"ON CONFLICT (unique_url) DO UPDATE SET payload = EXCLUDED.payload"
    )


def init_db(database_url: str) -> None:
    """Create the tables (idempotent — safe to run on every start)."""
    import psycopg

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        for statement in DDL.split(";"):
            if statement.strip():
                cur.execute(statement)
        conn.commit()


def store_records(database_url: str, table: str,
                  records: list[dict]) -> int:
    """Upsert validated records. Each record is ``{"unique_url": ...,
    "payload": {...}}``; returns the number of records written."""
    import psycopg
    from psycopg.types.json import Json

    sql = upsert_sql(table)
    params = [(r["unique_url"], Json(r["payload"])) for r in records]
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.executemany(sql, params)
        conn.commit()
    return len(records)

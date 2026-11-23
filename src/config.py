"""Central configuration. Every tunable lives here so scaling to 500k+
records is an infrastructure change (MAX_CONCURRENCY, workers, DB/Redis),
not a code change — a Phase I / Phase VI requirement."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://graphone:graphone@localhost:5432/graphone"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    google_sheet_id: str = os.getenv("GOOGLE_SHEET_ID", "")
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "20"))
    request_timeout_secs: int = int(os.getenv("REQUEST_TIMEOUT_SECS", "30"))


settings = Settings()

# Phase II freshness window (hours) — do not relax: the task requires
# news/jobs published within the last 24 hours.
FRESHNESS_WINDOW_HOURS = 24

# Phase I minimum targets from the task doc (bulk collections).
# Jobs and news are deliberately NOT in this dict: they are not capped
# at 100 — the deliverable is ALL jobs/news published in the last 24h
# (see FRESHNESS_WINDOW_HOURS). A missing key means "all fresh", not "0".
TARGETS = {"startups": 1000, "products": 1000, "papers": 1000}

# Phase II news sources — the five user-approved RSS feeds (21 Nov 2026).
# Wired AI was replaced by The Decoder; ArXiv and Hacker News AI are
# not RSS news sources for this tab (ArXiv feeds the papers tab).
NEWS_SOURCES = [
    "TechCrunch AI",
    "MIT Technology Review AI",
    "VentureBeat AI",
    "The Verge AI",
    "The Decoder",
]

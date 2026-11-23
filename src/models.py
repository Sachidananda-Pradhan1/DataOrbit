"""Single source of truth for the canonical schemas (task doc, 'Expected Schemas').

Anti-hallucination rule — disqualification risk in the task:
every record must trace back to a legitimate, valid source URL.
`source_url` / `paper_url` is therefore required on every entity that
has one in the schema, and the LLM extractor may only fill fields
evidenced by the fetched raw text (see llm/extractor.py).

Note: the task gives no separate News schema table, although News is a
required sheet tab. NewsItem below is an explicit assumption, modelled
on the Job schema (source + ISO-8601 published timestamp + full text).
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

# Spec enum for Product.pricingModel (task doc, Products schema).
PRICING_MODELS = frozenset({"FREE", "FREEMIUM", "PAID", "ENTERPRISE"})


class Source(BaseModel):
    name: str
    url: HttpUrl


class Startup(BaseModel):
    schemaVersion: str = "1.0.0"
    recordType: str = "STARTUP"
    source: Source
    entityName: str  # canonical name, after resolver
    employeeCount: int | None = None
    # Company website from the YC embedded JSON blob ("website").
    # None when the page does not state one (never guessed); the
    # resolver uses it as the domain signal for entity matching.
    website: str | None = None
    collectedAt: datetime


class Product(BaseModel):
    schemaVersion: str = "1.0.0"
    recordType: str = "PRODUCT"
    source: Source
    # Canonical name of the associated startup. "" = the source page
    # does not name a company separately from the product — NEVER
    # copied from productName (BetaList lists the product, not the
    # company, so a copied name would be invented data).
    startupName: str = ""
    productName: str
    # Spec field content.pricingModel: FREE / FREEMIUM / PAID /
    # ENTERPRISE. None = not stated on the source page (never guessed).
    pricingModel: Literal["FREE", "FREEMIUM", "PAID", "ENTERPRISE"] | None = None
    collectedAt: datetime

    @field_validator("pricingModel", mode="before")
    @classmethod
    def _coerce_pricing_model(cls, value):
        """Invalid pricing values become None — never raise, never guess.

        LLM output is untrusted: a bad enum value used to raise
        ValidationError and drop the whole record. Case-insensitive
        ("free" from the LLM is FREE); anything else is None."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().upper() in PRICING_MODELS:
            return value.strip().upper()
        return None


class ResearchPaper(BaseModel):
    schemaVersion: str = "1.0.0"
    recordType: str = "RESEARCH_PAPER"
    title: str
    authors: list[str] = Field(default_factory=list)
    paper_url: HttpUrl
    github_url: HttpUrl | None = None
    github_stars: int | None = None
    # True once github_stars holds a value fetched live from GitHub or
    # supplied by Hugging Face; False means "not fetched" (no token
    # budget left, rate-limited, lookup failed, or no repo link).
    github_stars_fetched: bool = False
    published_date: datetime


class Job(BaseModel):
    schemaVersion: str = "1.0.0"
    recordType: str = "JOB"
    company: str  # canonical company name
    title: str = ""  # job title as published (e.g. "Senior ML Engineer")
    date: datetime  # ISO-8601 publication date; must be within 24h (Phase II)
    is_remote: bool
    role_family: str  # e.g. "Engineering"
    region: str = "Global"  # "India" or "Global", from the location field
    source_url: HttpUrl | None = None  # kept for traceability (not in task table)


class NewsItem(BaseModel):
    """Assumption — see module docstring. No News table in the task doc."""
    schemaVersion: str = "1.0.0"
    recordType: str = "NEWS"
    title: str
    source: Source
    published_date: datetime  # must be within 24h (Phase II)
    full_text: str = ""  # full-text content, not just summary/meta (Phase II)
    url: HttpUrl

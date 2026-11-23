from datetime import datetime, timezone

from src.models import ResearchPaper, Startup

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def test_startup_requires_source_url():
    s = Startup(source={"name": "X", "url": "https://example.com/x"},
                entityName="OpenAI", collectedAt=NOW)
    assert s.recordType == "STARTUP" and s.schemaVersion == "1.0.0"


def test_paper_schema():
    p = ResearchPaper(title="T", authors=["A"],
                      paper_url="https://arxiv.org/abs/1234.5678",
                      published_date=NOW)
    assert p.github_stars is None

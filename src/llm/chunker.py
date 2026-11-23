"""Token-aware chunking (Phase III): never trigger a 413 by sending
a whole long article. Rough estimate: 1 token ≈ 4 chars. Chunks are
cut on paragraph boundaries where possible so each chunk stays
semantically dense.

HTML stripping lives here too: crawlers hand the extractor raw
pages (~130 KB of YC HTML), and the LLM must never see markup —
it wastes the per-call token budget and the model copies tags into
fields. strip_html_to_text runs BEFORE chunking (see extractor.py);
chunk_text itself still takes plain text.
"""
import logging
import re

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_CHUNK = 6000
CHARS_PER_TOKEN = 4
# Hard cap on characters sent to an LLM in one call (the extractor
# truncates to this; the brief calls it a "truncation strategy").
MAX_CHARS_PER_CALL = MAX_TOKENS_PER_CHUNK * CHARS_PER_TOKEN

# A `<` followed by a known tag name (open or close). Deliberately
# tag-based, not just "<", so plain text with comparisons ("a < b")
# or JSON braces never triggers the HTML path.
_HTML_TAG_RE = re.compile(
    r"</?(?:html|head|body|title|meta|link|script|style|div|span|p|a|"
    r"ul|ol|li|table|tr|td|th|section|article|header|footer|nav|"
    r"main|form|button|input|h[1-6]|img|br|hr)\b[^>]*>",
    re.IGNORECASE,
)


def looks_like_html(text: str) -> bool:
    """Heuristic: True when the text contains real HTML tags."""
    return bool(_HTML_TAG_RE.search(text))


def strip_html_to_text(html: str) -> str:
    """Extract readable text from an HTML page.

    trafilatura first (best main-content extraction, already a
    dependency); BeautifulSoup fallback for fragments trafilatura
    rejects (it returns None on short / boilerplate-heavy input).
    Script/style/noscript content is dropped in the fallback.
    Returns "" for empty input; never raises on malformed HTML.
    """
    if not html or not html.strip():
        return ""
    try:
        import trafilatura

        text = trafilatura.extract(html)
        if text:
            return text
    except Exception as exc:
        # trafilatura fails on some malformed input; the bs4 fallback
        # below still extracts the text.
        logger.debug("trafilatura.extract failed, using bs4 fallback: %s", exc)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n\n", strip=True)


def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[str]:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        # Split a single paragraph that alone exceeds the cap.
        while len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        if not para:
            continue
        sep = 2 if current else 0  # "\n\n" joiner — must count toward the cap
        if size + sep + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
            sep = 0
        current.append(para)
        size += sep + len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks

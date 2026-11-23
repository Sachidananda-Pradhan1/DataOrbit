"""Extraction prompts. The anti-hallucination instruction is in
every prompt: only use facts present in the supplied text, copy the
source URL verbatim, and leave a field null rather than guessing.
"""
SYSTEM = (
    "Extract structured data ONLY from the text below. "
    "Do not invent facts, URLs, numbers or dates. "
    "If a field is not in the text, return null for it. "
    "Return valid JSON matching the requested schema."
)

STARTUP_PROMPT = SYSTEM + "\n\nExtract a Startup: entityName, employeeCount. Text:\n{chunk}"
PRODUCT_PROMPT = SYSTEM + "\n\nExtract a Product: startupName (the company behind the product, if named separately from the product), productName, pricingModel (one of FREE, FREEMIUM, PAID, ENTERPRISE, or null if not stated). Text:\n{chunk}"
PAPER_PROMPT = SYSTEM + "\n\nExtract a Research Paper: title, authors, paper_url, github_url. Text:\n{chunk}"
JOB_PROMPT = SYSTEM + "\n\nExtract a Job: company, title, date (ISO-8601), is_remote, role_family. Text:\n{chunk}"
NEWS_PROMPT = SYSTEM + "\n\nExtract News: title, published_date (ISO-8601). Text:\n{chunk}"

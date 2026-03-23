"""
tools/parse_jd.py — Job Description Ingestion

Normalizes a job description from three possible sources into a plain string:
  - "text"  : raw pasted text (str)
  - "url"   : public URL fetched with httpx + BeautifulSoup
  - "file"  : uploaded bytes (PDF, DOCX, or TXT)

No Playwright needed — users who provide a URL already have the page open.
Simple httpx + BeautifulSoup handles the majority of public job postings.
"""

from __future__ import annotations

import re
from io import BytesIO


def from_text(text: str) -> str:
    """Clean and return pasted plain text."""
    return _clean(text)


def from_url(url: str) -> str:
    """
    Fetch a job posting URL and extract visible text.

    Raises:
        ValueError: if the URL cannot be fetched or returns no usable text.
    """
    import httpx
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=20.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP {e.response.status_code} fetching URL: {url}") from e
    except httpx.RequestError as e:
        raise ValueError(f"Network error fetching URL: {e}") from e

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    cleaned = _clean(text)

    if len(cleaned) < 100:
        raise ValueError("Fetched page contains too little text — try pasting the JD directly.")

    return cleaned


def from_file(file_bytes: bytes, filename: str) -> str:
    """
    Parse an uploaded document (PDF, DOCX, or TXT) and return plain text.

    Args:
        file_bytes: Raw bytes from the uploaded file.
        filename:   Original filename, used to determine parser.

    Raises:
        ValueError: if the file type is unsupported or parsing fails.
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        return _parse_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return _parse_docx(file_bytes)
    elif ext == "txt":
        return _clean(file_bytes.decode("utf-8", errors="replace"))
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use PDF, DOCX, or TXT.")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_pdf(data: bytes) -> str:
    try:
        import PyPDF2
    except ImportError as e:
        raise ImportError("PyPDF2 is required for PDF parsing. Run: pip install PyPDF2") from e

    reader = PyPDF2.PdfReader(BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _clean("\n".join(pages))


def _parse_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError("python-docx is required for DOCX parsing. Run: pip install python-docx") from e

    doc = Document(BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return _clean("\n".join(paragraphs))


def _clean(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs on each line
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

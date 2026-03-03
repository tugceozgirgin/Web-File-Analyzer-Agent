import io
import logging
from collections import Counter

import pdfplumber
from langchain_core.tools import tool

from src.agents.tools.download_utils import _download_file

logger = logging.getLogger(__name__)

_MAX_PAGES = 3
_MAX_CHARS = 4000


def _strip_repeated_headers(pages_text: list[list[str]]) -> list[list[str]]:
    """Remove lines that appear on every page (likely headers/footers).

    Parameters
    ----------
    pages_text:
        A list of pages, where each page is a list of stripped text lines.

    Returns
    -------
    list[list[str]]
        The same structure with repeated header/footer lines removed.
    """
    if len(pages_text) < 2:
        return pages_text

    line_counter: Counter[str] = Counter()
    for page_lines in pages_text:
        unique_lines = set(page_lines)
        for line in unique_lines:
            line_counter[line] += 1

    num_pages = len(pages_text)
    repeated = {line for line, count in line_counter.items() if count == num_pages and line}

    if not repeated:
        return pages_text

    logger.debug("Stripping %d repeated header/footer lines", len(repeated))
    return [
        [line for line in page_lines if line not in repeated]
        for page_lines in pages_text
    ]


@tool
def pdf_reader_tool(file_url: str) -> str:
    """Download a PDF from *file_url* and extract text from the first 3 pages.

    Returns a plain-text string truncated to ~4 000 characters.
    """
    data = _download_file(file_url)

    pages_text: list[list[str]] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[:_MAX_PAGES]:
            raw = page.extract_text() or ""
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            pages_text.append(lines)

    pages_text = _strip_repeated_headers(pages_text)

    full_text = "\n".join("\n".join(lines) for lines in pages_text)

    if len(full_text) > _MAX_CHARS:
        full_text = full_text[:_MAX_CHARS] + "\n…[truncated]"

    logger.info(
        "Extracted %d chars from PDF (%d pages read)", len(full_text), len(pages_text)
    )
    return full_text

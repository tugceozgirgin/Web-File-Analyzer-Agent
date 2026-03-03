import io
import logging

from docx import Document
from langchain_core.tools import tool

from src.agents.tools.download_utils import _download_file

logger = logging.getLogger(__name__)

_MAX_CHARS = 4000


@tool
def docx_reader_tool(file_url: str) -> str:
    """Download a DOCX from *file_url* and extract all paragraph text.

    Returns a plain-text string truncated to ~4 000 characters.
    """
    data = _download_file(file_url)

    doc = Document(io.BytesIO(data))

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)

    if len(full_text) > _MAX_CHARS:
        full_text = full_text[:_MAX_CHARS] + "\n…[truncated]"

    logger.info(
        "Extracted %d chars from DOCX (%d paragraphs)", len(full_text), len(paragraphs)
    )
    return full_text

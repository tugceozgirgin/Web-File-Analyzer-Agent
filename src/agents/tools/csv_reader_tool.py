import io
import logging

import pandas as pd
from langchain_core.tools import tool

from src.agents.tools.download_utils import _download_file

logger = logging.getLogger(__name__)

_SAMPLE_ROWS = 5


@tool
def csv_reader_tool(file_url: str) -> str:
    """Download a CSV from *file_url* and extract metadata with sample rows.

    The output includes:
    - Column headers
    - Total row count
    - First 5 data rows as a simple text table

    Returns a structured plain-text string.
    """
    data = _download_file(file_url)

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    headers = list(df.columns)
    total_rows = len(df)
    sample = df.head(_SAMPLE_ROWS)

    parts: list[str] = [
        f"Columns ({len(headers)}): {', '.join(str(h) for h in headers)}",
        f"Data rows: {total_rows}",
        f"Sample rows (first {min(_SAMPLE_ROWS, total_rows)}):",
    ]

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        values = " | ".join(str(v) for v in row.values)
        parts.append(f"  {i}. {values}")

    result = "\n".join(parts)
    logger.info(
        "Extracted metadata for CSV: %d columns, %d rows", len(headers), total_rows
    )
    return result

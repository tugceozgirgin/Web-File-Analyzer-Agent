"""Shared download helper used by all file-reader tools."""

import logging

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
_MAX_SIZE_BYTES = 50 * 1024 * 1024 

def _download_file(url: str) -> bytes:
    """Download a file from *url* and return its raw bytes.

    Parameters
    ----------
    url:
        Absolute HTTP(S) URL pointing to the file.

    Returns
    -------
    bytes
        The full response body.

    Raises
    ------
    httpx.HTTPStatusError
        If the server responds with a 4xx / 5xx status code.
    ValueError
        If the response exceeds ``_MAX_SIZE_BYTES``.
    """
    logger.info("Downloading %s", url)

    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > _MAX_SIZE_BYTES:
            raise ValueError(
                f"File at {url} is too large "
                f"({int(content_length)} bytes > {_MAX_SIZE_BYTES} byte limit)"
            )

        data = response.content

        if len(data) > _MAX_SIZE_BYTES:
            raise ValueError(
                f"Downloaded file from {url} is too large "
                f"({len(data)} bytes > {_MAX_SIZE_BYTES} byte limit)"
            )

    logger.info("Downloaded %d bytes from %s", len(data), url)
    return data

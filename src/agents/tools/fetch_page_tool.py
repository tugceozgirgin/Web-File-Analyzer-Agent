import os
import asyncio
import logging
import concurrent.futures
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def _build_mcp_config() -> dict:
    """Build the MultiServerMCPClient config using the npx-based MCP server.

    This matches the Firecrawl docs example:
    https://docs.firecrawl.dev/mcp-server
    """
    api_key = os.getenv("FIRECRAWL_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "FIRECRAWL_API_KEY is not set. Add it to your .env file."
        )

    return {
        "firecrawl": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "firecrawl-mcp"],
            "env": {
                "FIRECRAWL_API_KEY": api_key,
                "FIRECRAWL_RETRY_MAX_ATTEMPTS": os.getenv("FIRECRAWL_RETRY_MAX_ATTEMPTS", "3"),
                "FIRECRAWL_RETRY_INITIAL_DELAY": os.getenv("FIRECRAWL_RETRY_INITIAL_DELAY", "2000"),
                "FIRECRAWL_RETRY_MAX_DELAY": os.getenv("FIRECRAWL_RETRY_MAX_DELAY", "30000"),
                "FIRECRAWL_RETRY_BACKOFF_FACTOR": os.getenv("FIRECRAWL_RETRY_BACKOFF_FACTOR", "3"),
            },
        }
    }


def get_firecrawl_tools() -> list:
    """Get LangChain tools from the hosted Firecrawl MCP server.

    Uses ``langchain-mcp-adapters`` ``MultiServerMCPClient`` with HTTP
    transport to connect to the official Firecrawl MCP endpoint.

    Returns:
        A list of LangChain ``BaseTool`` instances from the MCP server.
    """

    async def _get():
        client = MultiServerMCPClient(_build_mcp_config())
        return await client.get_tools()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _get()).result()

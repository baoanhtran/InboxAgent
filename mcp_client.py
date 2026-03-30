"""MCP client factory for the Gmail MCP server.

Configuration is loaded from mcp.json (same directory as this file), which
follows the standard MCP server config format:

    {
      "mcpServers": {
        "gmail": {
          "url": "https://...",
          "headers": { "Authorization": "Bearer ..." }
        }
      }
    }

As of langchain-mcp-adapters 0.2+, MultiServerMCPClient is NOT an async
context manager. Usage pattern:

    client = _make_client()
    tools = await client.get_tools()
    result = await tools[0].ainvoke({})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

_MCP_JSON = Path(__file__).parent / "mcp.json"


def _load_mcp_config() -> dict:
    """Load and parse mcp.json, returning the gmail server config dict."""
    with _MCP_JSON.open() as f:
        data = json.load(f)
    return data["mcpServers"]["gmail"]


def _make_client() -> MultiServerMCPClient:
    """Create a MultiServerMCPClient configured from mcp.json."""
    cfg = _load_mcp_config()
    return MultiServerMCPClient(
        {
            "gmail": {
                "url": cfg["url"],
                "headers": cfg.get("headers", {}),
                "transport": "streamable_http",
            }
        }
    )


async def get_gmail_tools(client: MultiServerMCPClient | None = None) -> dict[str, Any]:
    """Core Gmail tools: list_emails, get_email, reply_to_email, mark_email_as_read.

    If no client is passed, a new one is created automatically.
    """
    if client is None:
        client = _make_client()
    all_tools = await client.get_tools()
    allowed = {"list_emails", "get_email", "reply_to_email", "mark_email_as_read"}
    return {tool.name: tool for tool in all_tools if tool.name in allowed}


async def get_readonly_gmail_tools(client: MultiServerMCPClient | None = None) -> dict[str, Any]:
    """Read-only subset: list_emails + get_email only. No reply or mark tools.

    Use this in every agent except mcp_executor_node to prevent accidental sends.
    If no client is passed, a new one is created automatically.
    """
    if client is None:
        client = _make_client()
    all_tools = await client.get_tools()
    allowed = {"list_emails", "get_email"}
    return {tool.name: tool for tool in all_tools if tool.name in allowed}

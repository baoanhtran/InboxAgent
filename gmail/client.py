"""MCP client factory for the Gmail MCP server.

Configuration is loaded from mcp.json at the project root, which follows
the standard MCP server config format:

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

# mcp.json lives at the project root (one level above this file's package)
_MCP_JSON = Path(__file__).parent.parent / "mcp.json"


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
    """Full Gmail tools: list_emails, get_email, reply_to_email, mark_email_as_read."""
    if client is None:
        client = _make_client()
    all_tools = await client.get_tools()
    allowed = {"list_emails", "get_email", "reply_to_email", "mark_email_as_read"}
    return {tool.name: tool for tool in all_tools if tool.name in allowed}


async def get_readonly_gmail_tools(client: MultiServerMCPClient | None = None) -> dict[str, Any]:
    """Read-only subset: list_emails + get_email only. No reply or mark tools.

    Use this in every agent except gmail.executor to prevent accidental sends.
    """
    if client is None:
        client = _make_client()
    all_tools = await client.get_tools()
    allowed = {"list_emails", "get_email"}
    return {tool.name: tool for tool in all_tools if tool.name in allowed}


async def read_mcp_resource(uri: str) -> tuple[str, str]:
    """Read a MCP resource by URI via MultiServerMCPClient.get_resources().

    Returns (base64_data, mime_type).
    Both bytes and str data are returned base64-encoded for uniform handling.
    The caller decodes based on mime_type.
    """
    import base64

    client = _make_client()
    blobs = await client.get_resources(server_name="gmail", uris=uri)

    if not blobs:
        raise RuntimeError(f"No resource returned for URI: {uri}")

    blob = blobs[0]
    mime_type = blob.mimetype or "application/octet-stream"
    data = blob.data

    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    else:
        # str — encode to bytes first
        b64 = base64.b64encode(str(data).encode("utf-8")).decode("ascii")

    return b64, mime_type

"""MCP executor node — replies to the email and marks it as read.

Uses reply_to_email (requires messageId + replyBody in HTML) so no need to
parse To/Subject from email_content — the server handles threading automatically.
Then marks the original email as read via mark_email_as_read.
"""

from __future__ import annotations

from mcp_client import get_gmail_tools
from state import AgentState


def _to_html(text: str) -> str:
    """Convert plain text reply body to minimal HTML."""
    paragraphs = text.strip().split("\n\n")
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip())


async def mcp_executor_node(state: AgentState) -> dict:
    """Reply to the original email and mark it as read."""
    message_id = state["message_id"]
    reply_body_html = _to_html(state["draft_reply"])

    tools = await get_gmail_tools()

    if "reply_to_email" not in tools:
        raise RuntimeError("reply_to_email tool not available from MCP server.")

    await tools["reply_to_email"].ainvoke({
        "messageId": message_id,
        "replyBody": reply_body_html,
        "replyAll": False,
    })

    # Mark original email as read now that we've handled it
    if "mark_email_as_read" in tools:
        await tools["mark_email_as_read"].ainvoke({"messageId": message_id})

    return {"status": "sent"}

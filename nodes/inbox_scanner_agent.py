"""Inbox scanner specialist agent.

Custom (not via factory) because finalize_node must:
1. Extract message_id from ToolMessage history.
2. Call get_email directly to set canonical email_content (never trust LLM output).
3. Store the LLM's narrative as inbox_results.

Guard: if email_content already populated, skip re-fetch to prevent corruption.

MCP server tool names: list_emails / get_email (NOT list_messages / get_message).
list_emails params: {unreadOnly: bool, limit: int}
get_email params:   {messageId: str}
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from mcp_client import get_readonly_gmail_tools
from nodes.email_utils import (
    extract_message_id_from_list_result,
    format_email_from_get_result,
    extract_text_from_result,
)
from state import InboxScannerState

_SYSTEM_PROMPT = """\
You are an inbox scanner agent. Your job is to:
1. Use list_emails to find the latest unread email in the inbox.
2. Use get_email to fetch its full content.
3. Summarize what you found: who sent it, subject, brief description of content.

Call list_emails first with unreadOnly=true and limit=1.
Then call get_email with the messageId from the result.
Finally, write a concise narrative summary (3-5 sentences) about the email you found.
"""

_UUID_RE = re.compile(
    r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", re.IGNORECASE
)

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return _llm


def _extract_message_id_from_history(messages: list) -> str:
    """Scan ToolMessage contents for a Gmail UUID message ID.

    Looks for 'ID: <uuid>' pattern first (list_emails format),
    then falls back to any UUID in the content.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # list_emails format: "ID: <uuid>\nDate: ..."
        m = re.search(r"^ID:\s*([a-f0-9\-]{36})", content, re.MULTILINE | re.IGNORECASE)
        if m:
            return m.group(1)

        # get_email format: messageId in JSON headers
        m = re.search(r'messageId:\s*"([a-f0-9\-]{36})"', content)
        if m:
            return m.group(1)

        # Any UUID as last resort
        m = _UUID_RE.search(content)
        if m:
            return m.group(0)

    return ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def scanner_agent_node(state: InboxScannerState) -> dict:
    tools = await get_readonly_gmail_tools()
    llm = _get_llm().bind_tools(list(tools.values()))

    messages = state.get("messages") or []

    if not messages:
        messages = [
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage("Please scan the inbox and fetch the latest unread email."),
        ]

    response = await llm.ainvoke(messages)
    return {"messages": [response]}


async def scanner_tool_node(state: InboxScannerState) -> dict:
    last = state["messages"][-1]
    tool_calls = last.tool_calls

    approval = interrupt({
        "type": "tool_approval",
        "agent": "inbox_scanner",
        "task": "inbox_scanner",
        "tool_name": ", ".join(tc["name"] for tc in tool_calls),
        "tool_args": tool_calls[0]["args"] if len(tool_calls) == 1 else {
            tc["name"]: tc["args"] for tc in tool_calls
        },
        "message": f"[inbox_scanner] wants to call: {', '.join(tc['name'] for tc in tool_calls)}",
    })

    tool_messages = []
    tools = await get_readonly_gmail_tools()

    for tc in tool_calls:
        if approval.get("approved", True):
            try:
                result = await tools[tc["name"]].ainvoke(tc["args"])
                content = str(result)
            except Exception as e:
                content = f"Error calling {tc['name']}: {e}"
        else:
            content = f"Tool call denied by user: {approval.get('reason', '')}"

        tool_messages.append(
            ToolMessage(content=content, tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


async def scanner_finalize_node(state: InboxScannerState) -> dict:
    """Extract message_id, fetch email_content directly, store narrative."""
    inbox_results = state["messages"][-1].content.strip()

    # Extract message_id from ToolMessage history
    message_id = state.get("message_id") or _extract_message_id_from_history(state["messages"])

    # Fetch email_content directly (never trust LLM output for this)
    email_content = state.get("email_content", "")
    if message_id and not email_content:
        try:
            tools = await get_readonly_gmail_tools()
            result = await tools["get_email"].ainvoke({"messageId": message_id})
            email_content = format_email_from_get_result(result, message_id)
        except Exception as e:
            email_content = f"(Failed to fetch email content: {e})"

    return {
        "message_id": message_id,
        "email_content": email_content,
        "inbox_results": inbox_results,
        "status": "scanned",
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route(state: InboxScannerState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "scanner_tool_node"
    return "scanner_finalize_node"


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------

def create_inbox_scanner() -> CompiledStateGraph:
    subgraph = StateGraph(InboxScannerState)

    subgraph.add_node("scanner_agent_node", scanner_agent_node)
    subgraph.add_node("scanner_tool_node", scanner_tool_node)
    subgraph.add_node("scanner_finalize_node", scanner_finalize_node)

    subgraph.set_entry_point("scanner_agent_node")
    subgraph.add_conditional_edges("scanner_agent_node", _route, {
        "scanner_tool_node": "scanner_tool_node",
        "scanner_finalize_node": "scanner_finalize_node",
    })
    subgraph.add_edge("scanner_tool_node", "scanner_agent_node")
    subgraph.add_edge("scanner_finalize_node", END)

    return subgraph.compile()

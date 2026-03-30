"""Inbox scanner specialist agent.

Custom (not via factory) because finalize_node must:
1. Extract message_id from ToolMessage history.
2. Call get_email directly to set canonical email_content (never trust LLM output).
3. Store the LLM's narrative as inbox_results.

Guard: if email_content already populated, skip re-fetch to prevent corruption.

The LLM is only given list_emails — get_email is called programmatically in
scanner_finalize_node to avoid the LLM using a LangChain tool_call_id as the messageId.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from llm_config import get_llm
from gmail.client import get_readonly_gmail_tools
from gmail.utils import extract_attachments_from_get_result, format_email_from_get_result
from state import InboxScannerState

_SYSTEM_PROMPT = """\
You are an inbox scanner agent. Your job is to:
1. Call list_emails with unreadOnly=true and limit=1 to find the latest unread email.
2. After receiving the result, write a concise summary (3-5 sentences): who sent it,
   the subject, and a brief description of the content.

Important: do NOT call get_email. The system will fetch the full email separately.
In your summary, include the exact message ID you received from list_emails.
"""

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("inbox_scanner", temperature=0)
    return _llm


_UUID_RE = re.compile(
    r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
    re.IGNORECASE,
)


def _is_lc_id(text: str, match_start: int) -> bool:
    """Return True if the UUID at match_start is a LangChain tool_call_id (lc_ prefix)."""
    prefix_start = max(0, match_start - 3)
    return "lc_" in text[prefix_start:match_start]


def _extract_message_id_from_history(messages: list) -> str:
    """Scan ToolMessage contents for a Gmail message ID.

    Tries multiple patterns in order of specificity.
    Explicitly skips LangChain tool_call_ids (lc_ prefix).
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Pattern 1: "ID: <value>" line (list_emails plain-text format)
        m = re.search(r"^ID:\s*(\S+)", content, re.MULTILINE | re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip(",")
            if not val.startswith("lc_"):
                return val

        # Pattern 2: JSON "id": "..." field
        m = re.search(r'"id"\s*:\s*"([^"]+)"', content)
        if m:
            val = m.group(1)
            if not val.startswith("lc_"):
                return val

        # Pattern 3: YAML/JSON messageId field
        m = re.search(r'messageId:\s*"?([^\s",]+)"?', content)
        if m:
            val = m.group(1)
            if not val.startswith("lc_"):
                return val

        # Pattern 4: any UUID in content, excluding lc_-prefixed ones
        for m in _UUID_RE.finditer(content):
            if not _is_lc_id(content, m.start()):
                return m.group(1)

    return ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def scanner_agent_node(state: InboxScannerState) -> dict:
    tools = await get_readonly_gmail_tools()
    # Only give list_emails — get_email is called programmatically in scanner_finalize_node.
    # Giving the LLM get_email causes it to call with the wrong ID (LangChain tool_call_id).
    list_only = {k: v for k, v in tools.items() if k == "list_emails"}
    llm = _get_llm().bind_tools(list(list_only.values()))

    messages = state.get("messages") or []

    if not messages:
        messages = [
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage("Please scan the inbox and find the latest unread email."),
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
    list_only = {k: v for k, v in tools.items() if k == "list_emails"}

    for tc in tool_calls:
        if approval.get("approved", True):
            try:
                result = await list_only[tc["name"]].ainvoke(tc["args"])
                content = str(result)
            except KeyError:
                content = f"Tool '{tc['name']}' is not available to the scanner."
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
    email_attachments: list = state.get("email_attachments") or []
    raw_result = None

    if message_id and not email_content:
        try:
            tools = await get_readonly_gmail_tools()
            raw_result = await tools["get_email"].ainvoke({"messageId": message_id})
            email_content = format_email_from_get_result(raw_result, message_id)
            email_attachments = extract_attachments_from_get_result(raw_result)
        except Exception as e:
            email_content = f"(Failed to fetch email content: {e})"

    status = "scanned" if message_id else "no_unread_emails"

    return {
        "message_id": message_id,
        "email_content": email_content,
        "email_attachments": email_attachments,
        "inbox_results": inbox_results,
        "status": status,
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

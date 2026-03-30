"""Coordinator meta-agent — the "brain" of the multi-agent system.

Reads accumulated state, decides which specialist to dispatch next, and loops
until it outputs {"next": "FINISH"}.

Key design decisions:
- `messages` is NOT in AgentState → resets to [] on every outer-graph entry.
  This prevents context blowup across coordinator iterations.
- Coordinator never touches `status` — specialists own status transitions.
- 3-layer JSON parsing for routing: strict JSON → regex → string scan → "FINISH".
- Loop guard: coordinator_iterations is incremented here; graph.py forces FINISH at >10.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from mcp_client import get_readonly_gmail_tools
from state import CoordinatorState

_SYSTEM_PROMPT = """\
You are a coordinator agent managing a team of specialist AI agents for email processing.
Your job is to decide which specialist to dispatch next based on the current state.

Available specialists:
- inbox_scanner: Lists unread emails and fetches the target email content. ALWAYS run first if email_content is empty.
- thread_researcher: Fetches the full email thread for context on the conversation history.
- sender_profiler: Researches the sender's previous emails to understand their communication style and relationship.
- composer: Drafts a professional reply using all gathered context.
- reviewer: Reviews the draft reply for quality, tone, and completeness. Outputs APPROVE or REVISE with bullets.

Routing rules:
1. If email_content is empty → ALWAYS dispatch inbox_scanner first.
2. Minimum path: inbox_scanner → composer → FINISH.
3. Do not dispatch the same specialist twice unless there is a genuine reason (e.g., composer after reviewer revision request).
4. If reviewer outputs APPROVE → output FINISH immediately.
5. Use Gmail tools (list_emails, get_email) for quick lookups if needed before deciding.
6. Be efficient — skip specialists whose output is not needed for a straightforward email.

Output your decision as JSON only:
{"next": "<specialist_name_or_FINISH>", "reason": "<brief explanation>"}

Valid values for "next": inbox_scanner, thread_researcher, sender_profiler, composer, reviewer, FINISH
"""

_SPECIALISTS = {"inbox_scanner", "thread_researcher", "sender_profiler", "composer", "reviewer", "FINISH"}

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return _llm


def _build_state_summary(state: CoordinatorState) -> str:
    """Construct the HumanMessage showing all accumulated specialist outputs."""
    iteration = state.get("coordinator_iterations", 0)
    status = state.get("status", "")

    def _preview(text: str, chars: int = 300) -> str:
        if not text:
            return "Not yet run."
        text = text.strip()
        return text[:chars] + "..." if len(text) > chars else text

    email_preview = ""
    if state.get("email_content"):
        lines = [l for l in state["email_content"].splitlines() if l.strip()][:6]
        email_preview = "\n".join(f"  {l}" for l in lines)
    else:
        email_preview = "  (not yet fetched)"

    return f"""\
=== CURRENT STATE ===
Iteration: {iteration} / Status: {status or 'none'}

EMAIL (first 6 lines):
{email_preview}

INBOX SCAN RESULTS: {_preview(state.get('inbox_results', ''))}

THREAD CONTEXT: {_preview(state.get('thread_context', ''))}

SENDER PROFILE: {_preview(state.get('sender_profile', ''))}

DRAFT REPLY: {_preview(state.get('draft_reply', ''))}

REVIEW NOTES: {_preview(state.get('review_notes', ''))}

=== DECISION REQUIRED ===
Output JSON: {{"next": "<specialist_or_FINISH>", "reason": "..."}}"""


def _parse_routing_decision(content: str) -> tuple[str, str]:
    """3-layer parse: strict JSON → regex → string scan → default FINISH."""
    content = content.strip()

    # Layer 1: strict JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "next" in data:
            action = str(data["next"]).strip()
            if action in _SPECIALISTS:
                return action, data.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 2: regex key-value scan
    m = re.search(r'"next"\s*:\s*"([^"]+)"', content)
    if m:
        action = m.group(1).strip()
        if action in _SPECIALISTS:
            reason_m = re.search(r'"reason"\s*:\s*"([^"]+)"', content)
            return action, reason_m.group(1) if reason_m else ""

    # Layer 3: substring scan
    for specialist in _SPECIALISTS:
        if specialist.lower() in content.lower():
            return specialist, ""

    return "FINISH", "Could not parse routing decision — defaulting to FINISH"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def coordinator_agent_node(state: CoordinatorState) -> dict:
    """LLM step — reasons over state summary and optionally requests tool calls."""
    tools = await get_readonly_gmail_tools()
    llm = _get_llm().bind_tools(list(tools.values()))

    messages = state.get("messages") or []

    if not messages:
        messages = [
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(_build_state_summary(state)),
        ]

    response = await llm.ainvoke(messages)
    return {"messages": [response]}


async def coordinator_tool_node(state: CoordinatorState) -> dict:
    """Optional tool execution — coordinator may do quick lookups before routing."""
    last = state["messages"][-1]
    tool_calls = last.tool_calls

    approval = interrupt({
        "type": "tool_approval",
        "agent": "coordinator",
        "task": "coordinator",
        "tool_name": ", ".join(tc["name"] for tc in tool_calls),
        "tool_args": tool_calls[0]["args"] if len(tool_calls) == 1 else {
            tc["name"]: tc["args"] for tc in tool_calls
        },
        "message": f"[coordinator] wants to call: {', '.join(tc['name'] for tc in tool_calls)}",
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


async def coordinator_finalize_node(state: CoordinatorState) -> dict:
    """Parse the routing decision and update coordinator state."""
    last_content = state["messages"][-1].content
    next_action, _reason = _parse_routing_decision(last_content)
    return {
        "next_action": next_action,
        "coordinator_iterations": state.get("coordinator_iterations", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_coordinator_internal(state: CoordinatorState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "coordinator_tool_node"
    return "coordinator_finalize_node"


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------

def create_coordinator_agent() -> CompiledStateGraph:
    subgraph = StateGraph(CoordinatorState)

    subgraph.add_node("coordinator_agent_node", coordinator_agent_node)
    subgraph.add_node("coordinator_tool_node", coordinator_tool_node)
    subgraph.add_node("coordinator_finalize_node", coordinator_finalize_node)

    subgraph.set_entry_point("coordinator_agent_node")
    subgraph.add_conditional_edges(
        "coordinator_agent_node",
        _route_coordinator_internal,
        {
            "coordinator_tool_node": "coordinator_tool_node",
            "coordinator_finalize_node": "coordinator_finalize_node",
        },
    )
    subgraph.add_edge("coordinator_tool_node", "coordinator_agent_node")
    subgraph.add_edge("coordinator_finalize_node", END)

    return subgraph.compile()

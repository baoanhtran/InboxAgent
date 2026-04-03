"""Coordinator meta-agent — the "brain" of the multi-agent system.

Uses DeepAgent (deepagents package) as the inner reasoning engine, giving the
coordinator access to:
  - write_todos / check_todo / uncheck_todo  — upfront pipeline planning
  - ls / read_file / write_file / edit_file  — context offloading to a virtual FS
  - task(name=...)                            — parallel subagent dispatch

The coordinator still maintains its original routing contract with the outer
LangGraph graph: it reads accumulated state via a HumanMessage summary and
outputs a JSON routing decision `{"next": "<specialist_or_FINISH>", "reason": "..."}`.

Architecture:
    Outer subgraph (CoordinatorState)
        └── run_agent_node  →  _inner (create_deep_agent)
        └── finalize_node   →  parse next_action + extract task results  →  END

Key design decisions (unchanged):
  - `messages` is NOT in AgentState → resets to [] on every outer-graph entry.
  - Coordinator never touches `status` — specialists own status transitions.
  - 3-layer JSON parsing for routing: strict JSON → regex → string scan → "FINISH".
  - Loop guard: coordinator_iterations is incremented here; graph.py forces FINISH at >10.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deepagents import CompiledSubAgent, create_deep_agent

from llm_config import get_llm
from gmail.client import get_readonly_gmail_tools
from state import CoordinatorState

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a coordinator agent managing a team of specialist AI agents for email processing.
Your job is to decide which specialist to dispatch next based on the current state.

Available specialists (via LangGraph routing — output their name in JSON):
- inbox_scanner: Lists unread emails and fetches the target email content. ALWAYS run first if email_content is empty.
- attachment_analyzer: Reads and summarizes email attachments using OpenAI vision/file APIs. Run if email_attachments is non-empty AND attachment_summary is empty.
- composer: Drafts a professional reply using all gathered context.
- reviewer: Reviews the draft reply for quality, tone, and completeness. Outputs APPROVE or REVISE with bullets.

Available subagents (call in parallel via the `task` tool when you need them):
- thread_researcher: Fetches the full email thread for conversation context. Returns thread_context summary.
- sender_profiler: Researches the sender's previous emails and communication style. Returns sender_profile summary.

Routing rules:
1. If email_content is empty → ALWAYS dispatch inbox_scanner first.
2. If email has attachments (ATTACHMENTS count > 0) AND attachment_summary is empty → dispatch attachment_analyzer before composer.
3. Minimum path: inbox_scanner → composer → FINISH.
4. If email has attachments: inbox_scanner → attachment_analyzer → composer → FINISH.
5. Do not dispatch the same specialist twice unless there is a genuine reason (e.g., composer after reviewer revision request).
6. If reviewer outputs APPROVE → output FINISH immediately.
7. Use Gmail tools (list_emails, get_email) for quick lookups if needed before deciding.
8. Be efficient — skip specialists whose output is not needed for a straightforward email.

Workflow:
1. FIRST: call write_todos to plan the full specialist pipeline for this email.
2. Use the `task` tool to call thread_researcher and/or sender_profiler IN PARALLEL when needed.
   Their results will come back as tool messages — you do NOT need to route to them via JSON.
3. Check off completed steps in your todo list as you go.
4. LAST: output your routing decision as a bare JSON object (no markdown, no extra text):
   {"next": "<specialist_name_or_FINISH>", "reason": "<brief explanation>"}

Valid values for "next": inbox_scanner, attachment_analyzer, composer, reviewer, FINISH

When using filesystem tools: write large context (thread summaries, attachment summaries >500 chars)
to /context/<name>.md to save tokens in future steps.
"""

_SPECIALISTS = {
    "inbox_scanner", "attachment_analyzer",
    "thread_researcher", "sender_profiler",
    "composer", "reviewer", "FINISH",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    attachments = state.get("email_attachments") or []
    att_count = len(attachments)
    att_names = ", ".join(a.get("filename", "?") for a in attachments) if attachments else "none"

    return f"""\
=== CURRENT STATE ===
Iteration: {iteration} / Status: {status or 'none'}

EMAIL (first 6 lines):
{email_preview}

ATTACHMENTS: {att_count} file(s) — {att_names}

ATTACHMENT SUMMARY: {_preview(state.get('attachment_summary', ''))}

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


def _extract_task_results(messages: list) -> dict[str, Any]:
    """Scan message history for task tool results and map them to state fields.

    When the inner DeepAgent calls task(name="thread_researcher") or
    task(name="sender_profiler"), the subagent's last message content is
    returned as a ToolMessage. We extract those here so they can be written
    to the outer AgentState fields.
    """
    # Build mapping: tool_call_id → subagent name
    tc_to_agent: dict[str, str] = {}
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if tc.get("name") == "task":
                    subagent_name = tc.get("args", {}).get("name", "")
                    tc_to_agent[tc["id"]] = subagent_name

    results: dict[str, Any] = {}
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        agent_name = tc_to_agent.get(msg.tool_call_id, "")
        if agent_name == "thread_researcher":
            results["thread_context"] = msg.content
        elif agent_name == "sender_profiler":
            results["sender_profile"] = msg.content

    return results


# ---------------------------------------------------------------------------
# Inner DeepAgent — lazy initialised (tools require an async call)
# ---------------------------------------------------------------------------

_inner_agent: CompiledStateGraph | None = None


async def _get_inner_agent() -> CompiledStateGraph:
    """Return (and cache) the compiled inner DeepAgent."""
    global _inner_agent
    if _inner_agent is not None:
        return _inner_agent

    # Resolve Gmail tools (async)
    gmail_tools = list((await get_readonly_gmail_tools()).values())

    # Build CompiledSubAgent wrappers for the two research specialists.
    # use_interrupt=False because interrupts cannot propagate through the task
    # tool back to the outer graph — these are read-only tools so it is safe.
    from agents.thread_researcher import create_thread_researcher
    from agents.sender_profiler import create_sender_profiler

    subagents = [
        CompiledSubAgent(
            name="thread_researcher",
            description=(
                "Fetches and summarises the full email thread for conversation context. "
                "Call with a description of the email. Returns a thread_context summary."
            ),
            runnable=create_thread_researcher(use_interrupt=False),
        ),
        CompiledSubAgent(
            name="sender_profiler",
            description=(
                "Researches the sender's previous emails and communication style. "
                "Call with a description of the email. Returns a sender_profile summary."
            ),
            runnable=create_sender_profiler(use_interrupt=False),
        ),
    ]

    _inner_agent = create_deep_agent(
        model=get_llm("coordinator", temperature=0),
        tools=gmail_tools,
        system_prompt=_SYSTEM_PROMPT,
        subagents=subagents,
    )
    return _inner_agent


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def run_agent_node(state: CoordinatorState) -> dict:
    """Invoke the inner DeepAgent with a summary of the current state."""
    inner = await _get_inner_agent()
    seed = HumanMessage(_build_state_summary(state))
    result = await inner.ainvoke({"messages": [seed]})
    # Propagate messages (and any task-result state updates) back
    task_results = _extract_task_results(result.get("messages", []))
    return {"messages": result.get("messages", []), **task_results}


async def finalize_node(state: CoordinatorState) -> dict:
    """Parse the routing decision from the last AI message."""
    last_content = state["messages"][-1].content
    next_action, _reason = _parse_routing_decision(last_content)
    return {
        "next_action": next_action,
        "coordinator_iterations": state.get("coordinator_iterations", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------

def create_coordinator_agent() -> CompiledStateGraph:
    subgraph = StateGraph(CoordinatorState)

    subgraph.add_node("run_agent_node", run_agent_node)
    subgraph.add_node("finalize_node", finalize_node)

    subgraph.set_entry_point("run_agent_node")
    subgraph.add_edge("run_agent_node", "finalize_node")
    subgraph.add_edge("finalize_node", END)

    return subgraph.compile()

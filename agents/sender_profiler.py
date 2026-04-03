"""Sender profiler specialist — researches the sender's communication history.

Supports persistent memory: if a profile for the sender already exists in the
store (and is less than 7 days old), the Gmail lookup is skipped entirely.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from agents.factory import create_specialist_agent
from state import SenderProfilerState

_SYSTEM_PROMPT = """\
You are a sender profiler agent. Your job is to research the sender of the email
and build a profile of their communication style and relationship with the recipient.

Use list_emails to find recent emails, then use get_email to read relevant past messages from this sender.
Note: list_emails does not support filtering by sender — use it to get recent emails, then identify ones from this sender.

Provide a concise profile including:
- How long they have been in contact and the nature of the relationship
- Their typical communication tone (formal, casual, direct, verbose, etc.)
- Any recurring topics or ongoing projects they discuss
- Key facts about them that would help craft an appropriate reply
- The appropriate tone and level of formality for a reply
"""

_CACHE_TTL_DAYS = 7
_STORE_NAMESPACE = ("sender_profiles",)


def _extract_sender_email(email_content: str) -> Optional[str]:
    """Extract the sender's email address from raw email content."""
    m = re.search(r"From:.*?([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})", email_content)
    return m.group(1).lower() if m else None


def _is_recent(updated_at: datetime) -> bool:
    now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return (now - updated_at) < timedelta(days=_CACHE_TTL_DAYS)


def _seed(state: SenderProfilerState) -> str:
    return (
        f"Please profile the sender of this email:\n\n{state['email_content']}\n\n"
        "Search for past emails from this sender and summarize their communication profile."
    )


def create_sender_profiler(
    store: Optional[BaseStore] = None,
    use_interrupt: bool = True,
) -> CompiledStateGraph:
    """Build the sender profiler subgraph.

    If *store* is provided, a cached sender profile is returned immediately
    when one exists and is less than 7 days old; the result is saved back to
    the store after each successful profiling run.
    """
    # Inner specialist (ReAct loop) — compiled once and reused
    _specialist = create_specialist_agent(
        state_class=SenderProfilerState,
        system_prompt=_SYSTEM_PROMPT,
        output_key="sender_profile",
        output_status="profiled",
        agent_name="sender_profiler",
        seed_message_fn=_seed,
        readonly=True,
        temperature=0,
        use_interrupt=use_interrupt,
    )

    # -----------------------------------------------------------------------
    # Memory nodes (only meaningful when store is set)
    # -----------------------------------------------------------------------

    async def load_profile_node(state: SenderProfilerState) -> dict:
        """Return cached profile if available and recent; otherwise pass through."""
        if store is None:
            return {}
        sender = _extract_sender_email(state.get("email_content", ""))
        if not sender:
            return {}
        item = await store.aget(_STORE_NAMESPACE, sender)
        if item and _is_recent(item.updated_at):
            return {"sender_profile": item.value["profile"], "status": "profiled"}
        return {}

    async def save_profile_node(state: SenderProfilerState) -> dict:
        """Persist the freshly built profile to the store."""
        if store is None:
            return {}
        sender = _extract_sender_email(state.get("email_content", ""))
        profile = state.get("sender_profile", "")
        if sender and profile:
            await store.aput(_STORE_NAMESPACE, sender, {"profile": profile})
        return {}

    def _route_after_load(state: SenderProfilerState) -> str:
        # If load_profile_node already populated sender_profile, skip the agent
        if state.get("sender_profile"):
            return END
        return "specialist"

    # -----------------------------------------------------------------------
    # Outer subgraph: load → (cached? → END) else → specialist → save → END
    # -----------------------------------------------------------------------
    subgraph = StateGraph(SenderProfilerState)
    subgraph.add_node("load_profile_node", load_profile_node)
    subgraph.add_node("specialist", _specialist)
    subgraph.add_node("save_profile_node", save_profile_node)

    subgraph.set_entry_point("load_profile_node")
    subgraph.add_conditional_edges(
        "load_profile_node",
        _route_after_load,
        {"specialist": "specialist", END: END},
    )
    subgraph.add_edge("specialist", "save_profile_node")
    subgraph.add_edge("save_profile_node", END)

    return subgraph.compile()

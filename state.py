"""State definitions for the Gmail Assistant — Coordinator Meta-Agent Architecture."""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Outer graph state — single source of truth shared by all nodes."""
    message_id: str
    email_content: str
    inbox_results: str           # scanner narrative summary
    thread_context: str          # thread researcher findings
    sender_profile: str          # sender profiler findings
    draft_reply: str             # composer output
    review_notes: str            # reviewer output (APPROVE or REVISE + bullets)
    next_action: str             # coordinator routing signal
    coordinator_iterations: int  # loop guard counter
    status: str
    human_feedback: Annotated[Optional[str], lambda old, new: new]


class CoordinatorState(TypedDict):
    """Coordinator subgraph state. messages resets to [] on every outer-graph entry."""
    # Shared keys with AgentState (auto passed in/out)
    message_id: str
    email_content: str
    inbox_results: str
    thread_context: str
    sender_profile: str
    draft_reply: str
    review_notes: str
    next_action: str
    coordinator_iterations: int
    status: str
    # Internal — NOT in AgentState so it resets to [] on every entry
    messages: Annotated[list, add_messages]


class InboxScannerState(TypedDict):
    """Inbox scanner subgraph state."""
    message_id: str
    email_content: str
    inbox_results: str
    status: str
    messages: Annotated[list, add_messages]


class ThreadResearcherState(TypedDict):
    """Thread researcher subgraph state."""
    email_content: str
    thread_context: str
    status: str
    messages: Annotated[list, add_messages]


class SenderProfilerState(TypedDict):
    """Sender profiler subgraph state."""
    email_content: str
    sender_profile: str
    status: str
    messages: Annotated[list, add_messages]


class ComposerState(TypedDict):
    """Composer subgraph state."""
    email_content: str
    thread_context: str
    sender_profile: str
    review_notes: str
    draft_reply: str
    status: str
    messages: Annotated[list, add_messages]


class ReviewerState(TypedDict):
    """Reviewer subgraph state."""
    email_content: str
    draft_reply: str
    review_notes: str
    status: str
    messages: Annotated[list, add_messages]

"""Outer LangGraph StateGraph — coordinator meta-agent architecture.

Graph topology:

  coordinator_agent  ← loops until "FINISH" (max 10 iterations)
      │ next_action
      ├── inbox_scanner    → back to coordinator
      ├── thread_researcher→ back to coordinator
      ├── sender_profiler  → back to coordinator
      ├── composer         → back to coordinator
      ├── reviewer         → back to coordinator
      └── FINISH / NO_EMAIL
              │
      FINISH → human_review_node  ← interrupt: approve / edit / reject draft
              │
      ┌───────┴────────────────────┐
      │ approved                   │ rejected
      ▼                            ▼
  mcp_executor_node          coordinator_agent
      │                       (re-draft loop with review_notes in state)
      END

  NO_EMAIL → END  (boîte vide, rien à faire)
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.coordinator import create_coordinator_agent
from agents.inbox_scanner import create_inbox_scanner
from agents.thread_researcher import create_thread_researcher
from agents.sender_profiler import create_sender_profiler
from agents.attachment_analyzer import attachment_analyzer_node
from agents.composer import create_composer
from agents.reviewer import create_reviewer
from agents.human_review import human_review_node
from gmail.executor import mcp_executor_node
from state import AgentState

_MAX_ITERATIONS = 10


def _route_coordinator(state: AgentState) -> str:
    if state.get("status") == "no_unread_emails":
        return "NO_EMAIL"
    if state.get("coordinator_iterations", 0) > _MAX_ITERATIONS:
        return "FINISH"
    action = state.get("next_action", "FINISH")
    valid = {"inbox_scanner", "attachment_analyzer", "thread_researcher", "sender_profiler", "composer", "reviewer", "FINISH"}
    return action if action in valid else "FINISH"


def _route_after_review(state: AgentState) -> str:
    return "mcp_executor_node" if state.get("status") == "approved" else "coordinator_agent"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("coordinator_agent",        create_coordinator_agent())
    graph.add_node("inbox_scanner_agent",      create_inbox_scanner())
    graph.add_node("attachment_analyzer_agent",attachment_analyzer_node)
    graph.add_node("thread_researcher_agent",  create_thread_researcher())
    graph.add_node("sender_profiler_agent",    create_sender_profiler())
    graph.add_node("composer_agent",           create_composer())
    graph.add_node("reviewer_agent",           create_reviewer())
    graph.add_node("human_review_node",        human_review_node)
    graph.add_node("mcp_executor_node",        mcp_executor_node)

    graph.set_entry_point("coordinator_agent")

    graph.add_conditional_edges(
        "coordinator_agent",
        _route_coordinator,
        {
            "inbox_scanner":      "inbox_scanner_agent",
            "attachment_analyzer":"attachment_analyzer_agent",
            "thread_researcher":  "thread_researcher_agent",
            "sender_profiler":    "sender_profiler_agent",
            "composer":           "composer_agent",
            "reviewer":           "reviewer_agent",
            "FINISH":             "human_review_node",
            "NO_EMAIL":           END,
        },
    )

    for specialist in [
        "inbox_scanner_agent",
        "attachment_analyzer_agent",
        "thread_researcher_agent",
        "sender_profiler_agent",
        "composer_agent",
        "reviewer_agent",
    ]:
        graph.add_edge(specialist, "coordinator_agent")

    graph.add_conditional_edges(
        "human_review_node",
        _route_after_review,
        {
            "mcp_executor_node": "mcp_executor_node",
            "coordinator_agent": "coordinator_agent",
        },
    )

    graph.add_edge("mcp_executor_node", END)

    return graph.compile(checkpointer=MemorySaver())

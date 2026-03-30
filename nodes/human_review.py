"""Human review node — pauses the graph for human approval using interrupt().

LangGraph 0.2+ pattern:
  - Call interrupt(payload) inside the node to pause execution.
  - The payload is surfaced to the caller via the graph snapshot.
  - Resume by passing Command(resume=value) to graph.astream() / graph.ainvoke().
  - The value passed to Command(resume=...) is returned by interrupt().

Expected resume value format:
  {"action": "approve"}
    — accept the current draft as-is.
  {"action": "edit", "revised_draft": "<new text>", "comment": "<optional note>"}
    — replace the draft with the human-edited version.
  {"action": "reject"}
    — send the email back to the DeepAgent for re-drafting.
"""

from __future__ import annotations

from langgraph.types import interrupt

from state import AgentState


async def human_review_node(state: AgentState) -> dict:
    """Pause for human review of the draft reply."""

    # Execution pauses here. The dict below is surfaced to the caller.
    human_input = interrupt(
        {
            "message": "Please review the draft reply and respond with an action.",
            "draft_reply": state["draft_reply"],
            "email_content": state["email_content"],
            "review_notes": state.get("review_notes", ""),
            "actions": {
                "approve": "Accept the draft and proceed to send.",
                "edit":    "Provide a revised_draft string to replace the current draft.",
                "reject":  "Send back to coordinator for a complete re-draft.",
            },
        }
    )

    # --- Process the human's response ---
    if isinstance(human_input, dict):
        action = human_input.get("action", "approve")

        if action == "edit":
            revised = human_input.get("revised_draft", state["draft_reply"])
            comment = human_input.get("comment", "Human edited the draft.")
            return {
                "draft_reply": revised,
                "human_feedback": comment,
                "status": "approved",
            }

        if action == "reject":
            return {
                "human_feedback": human_input.get("comment", "Human rejected the draft — re-drafting."),
                "status": "needs_revision",
            }

    # Default: approve the current draft
    return {
        "human_feedback": "Approved",
        "status": "approved",
    }

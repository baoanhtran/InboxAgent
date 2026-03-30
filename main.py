"""Entry point for the Gmail Executive Assistant — Coordinator Meta-Agent.

The coordinator agent reads state, dispatches specialists, and loops until done.
Every MCP tool call (list_messages, get_message, send_message) requires human
approval via interrupt() before executing.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # add your OPENAI_API_KEY
    python main.py
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv
from langgraph.types import Command

from graph import build_graph

load_dotenv()

_INITIAL_STATE = {
    "message_id": "",
    "email_content": "",
    "email_attachments": [],
    "attachment_summary": "",
    "inbox_results": "",
    "thread_context": "",
    "sender_profile": "",
    "draft_reply": "",
    "review_notes": "",
    "next_action": "",
    "coordinator_iterations": 0,
    "status": "",
    "human_feedback": None,
}


async def run() -> None:
    print("\n" + "=" * 60)
    print("  Gmail Executive Assistant")
    print("=" * 60)

    graph = build_graph()
    thread_config = {"configurable": {"thread_id": "gmail-assistant-session"}}

    graph_input: dict | Command = _INITIAL_STATE

    while True:
        print()
        async for event in graph.astream(
            graph_input,
            config=thread_config,
            stream_mode="values",
        ):
            _print_progress(event)

        snapshot = graph.get_state(thread_config)

        if not snapshot.next:
            break

        payload = _get_interrupt_payload(snapshot)

        if payload.get("type") == "tool_approval":
            resume_value = _prompt_tool_approval(payload)
        else:
            draft = payload.get("draft_reply", snapshot.values.get("draft_reply", ""))
            resume_value = _prompt_draft_review(draft)

        graph_input = Command(resume=resume_value)

    _print_final(graph.get_state(thread_config).values)


# ---------------------------------------------------------------------------
# Interrupt helpers
# ---------------------------------------------------------------------------

def _get_interrupt_payload(snapshot) -> dict:
    for task in snapshot.tasks:
        if task.interrupts:
            value = task.interrupts[0].value
            return value if isinstance(value, dict) else {"message": str(value)}
    return {}


# ---------------------------------------------------------------------------
# Human prompts
# ---------------------------------------------------------------------------

def _prompt_tool_approval(payload: dict) -> dict:
    print("\n" + "-" * 60)
    print("[Tool Approval Request]")
    print("-" * 60)
    agent = payload.get("agent", "")
    if agent:
        print(f"  Agent: {agent}")
    print(f"  Task : {payload.get('task', '')}")
    print(f"  Tool : {payload.get('tool_name', '')}")
    args = payload.get("tool_args", {})
    if args:
        print(f"  Args : {json.dumps(args, ensure_ascii=False)}")
    msg = payload.get("message", "")
    for line in msg.splitlines()[1:]:
        print(f"         {line}")

    while True:
        choice = input("\n  Allow? [y/n]: ").strip().lower()
        if choice == "y":
            return {"approved": True}
        if choice == "n":
            reason = input("  Reason (optional): ").strip()
            return {"approved": False, "reason": reason or "denied by user"}
        print("  Please enter 'y' or 'n'.")


def _prompt_draft_review(draft: str) -> dict:
    print("\n" + "-" * 60)
    print("[Draft Review]")
    print("-" * 60)
    print(f"\n{draft}")
    print("-" * 60)
    print("\n  [a] Approve and send")
    print("  [e] Edit the draft")
    print("  [r] Reject — re-draft")

    while True:
        choice = input("\nYour choice (a/e/r): ").strip().lower()

        if choice == "a":
            return {"action": "approve"}

        if choice == "e":
            print("\nPaste revised draft (finish with a line containing only '---'):")
            lines = []
            while True:
                line = input()
                if line.strip() == "---":
                    break
                lines.append(line)
            revised = "\n".join(lines).strip()
            comment = input("Comment (Enter to skip): ").strip()
            return {
                "action": "edit",
                "revised_draft": revised or draft,
                "comment": comment or "Human edited the draft.",
            }

        if choice == "r":
            comment = input("Rejection reason (Enter to skip): ").strip()
            return {
                "action": "reject",
                "comment": comment or "Human rejected — please re-draft.",
            }

        print("  Please enter 'a', 'e', or 'r'.")


# ---------------------------------------------------------------------------
# Console progress / summary
# ---------------------------------------------------------------------------

_printed: set[str] = set()


def _print_progress(event: dict) -> None:
    iterations = event.get("coordinator_iterations", 0)
    next_action = event.get("next_action", "")
    status = event.get("status", "")
    email = event.get("email_content", "")
    msg_id = event.get("message_id", "")

    if msg_id and "msg_id" not in _printed:
        print(f"  [Fetch]       Message ID: {msg_id}")
        _printed.add("msg_id")

    if email and "email" not in _printed:
        preview = [l for l in email.splitlines() if l.strip()][:3]
        for line in preview:
            print(f"  [Email]       {line}")
        _printed.add("email")

    if next_action and f"dispatch_{next_action}_{iterations}" not in _printed:
        if next_action not in ("", "FINISH"):
            print(f"  [Coordinator] iter={iterations} → dispatching {next_action}")
            _printed.add(f"dispatch_{next_action}_{iterations}")

    if status == "scanned" and "scanned" not in _printed:
        inbox = event.get("inbox_results", "")
        print(f"  [Scanner]     {inbox[:80].strip()}..." if len(inbox) > 80 else f"  [Scanner]     {inbox.strip()}")
        _printed.add("scanned")

    if status in ("attachments_analyzed", "no_attachments") and "attachments" not in _printed:
        atts = event.get("email_attachments") or []
        summary = event.get("attachment_summary", "")
        if status == "no_attachments":
            print(f"  [Analyzer]    No attachments found ({len(atts)} parsed from headers).")
        else:
            first_line = summary.splitlines()[0] if summary else "(empty)"
            print(f"  [Analyzer]    {len(atts)} attachment(s) — {first_line[:80]}")
        _printed.add("attachments")

    if status == "researched" and "researched" not in _printed:
        print("  [Researcher]  Thread context gathered.")
        _printed.add("researched")

    if status == "profiled" and "profiled" not in _printed:
        print("  [Profiler]    Sender profile built.")
        _printed.add("profiled")

    if status == "draft_ready" and "draft" not in _printed:
        draft = event.get("draft_reply", "")
        print(f"  [Composer]    {draft[:100].strip()}...")
        _printed.add("draft")

    if status == "reviewed" and "reviewed" not in _printed:
        notes = event.get("review_notes", "")
        print(f"  [Reviewer]    {notes[:100].strip()}...")
        _printed.add("reviewed")

    if status == "sent":
        print("  [Sent]        Email delivered successfully.")

    if status and status.startswith("send_denied"):
        print(f"  [Cancelled]   {status}")


def _print_final(values: dict) -> None:
    print("\n" + "=" * 60)
    status = values.get('status', 'N/A')
    if status == "no_unread_emails":
        print("  No unread emails found. Nothing to do.")
    else:
        print(f"  Status       : {status}")
        print(f"  Iterations   : {values.get('coordinator_iterations', 0)}")
        feedback = values.get("human_feedback")
        if feedback:
            print(f"  Feedback     : {feedback}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run())

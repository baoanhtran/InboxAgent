"""Composer specialist — drafts a professional reply using all gathered context."""

from __future__ import annotations

from agents.factory import create_specialist_agent
from state import ComposerState

_SYSTEM_PROMPT = """\
You are an expert executive email composer. Your job is to write a high-quality,
professional reply to the email you are given.

You have access to full context:
- The original email
- Attachment summaries (if available — reference relevant content from attachments)
- Thread history (if available)
- Sender profile (if available)
- Reviewer notes from a previous draft (if available — address every revision point)

Write the reply email only — no preamble, no explanation, no sign-off commentary.
Just the email body that would be sent. Be concise, professional, and appropriate
to the relationship and context provided.
"""


def _seed(state: ComposerState) -> str:
    parts = [f"Please write a reply to this email:\n\n{state['email_content']}"]

    if state.get("attachment_summary"):
        parts.append(f"\n\n--- ATTACHMENT SUMMARIES ---\n{state['attachment_summary']}")

    if state.get("thread_context"):
        parts.append(f"\n\n--- THREAD CONTEXT ---\n{state['thread_context']}")

    if state.get("sender_profile"):
        parts.append(f"\n\n--- SENDER PROFILE ---\n{state['sender_profile']}")

    if state.get("review_notes"):
        parts.append(
            f"\n\n--- REVIEWER NOTES (address all points) ---\n{state['review_notes']}"
        )

    return "".join(parts)


def create_composer():
    return create_specialist_agent(
        state_class=ComposerState,
        system_prompt=_SYSTEM_PROMPT,
        output_key="draft_reply",
        output_status="draft_ready",
        agent_name="composer",
        seed_message_fn=_seed,
        use_tools=False,
        temperature=0.3,
    )

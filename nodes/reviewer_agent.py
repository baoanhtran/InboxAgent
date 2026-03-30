"""Reviewer specialist — critiques the draft reply and outputs APPROVE or REVISE."""

from __future__ import annotations

from nodes.agent_factory import create_specialist_agent
from state import ReviewerState

_SYSTEM_PROMPT = """\
You are an expert email reviewer. Your job is to evaluate a draft email reply
for quality, tone, accuracy, and completeness.

Review criteria:
- Does it fully address all points raised in the original email?
- Is the tone appropriate for the relationship and context?
- Is it clear, concise, and professional?
- Are there any factual errors, awkward phrases, or missing information?
- Does it represent the sender well?

Output format — choose exactly one:

If the draft is good to send:
APPROVE
Brief explanation of why it meets the bar.

If the draft needs work:
REVISE
- Bullet point 1: specific issue and how to fix it
- Bullet point 2: specific issue and how to fix it
(list all issues that must be addressed before sending)

Do not rewrite the draft yourself — only approve or provide revision bullets.
"""


def _seed(state: ReviewerState) -> str:
    return (
        f"Please review this draft reply.\n\n"
        f"--- ORIGINAL EMAIL ---\n{state['email_content']}\n\n"
        f"--- DRAFT REPLY ---\n{state['draft_reply']}"
    )


def create_reviewer():
    return create_specialist_agent(
        state_class=ReviewerState,
        system_prompt=_SYSTEM_PROMPT,
        output_key="review_notes",
        output_status="reviewed",
        agent_name="reviewer",
        seed_message_fn=_seed,
        readonly=True,
        temperature=0,
    )

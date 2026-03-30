"""Sender profiler specialist — researches the sender's communication history."""

from __future__ import annotations

from nodes.agent_factory import create_specialist_agent
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


def _seed(state: SenderProfilerState) -> str:
    return (
        f"Please profile the sender of this email:\n\n{state['email_content']}\n\n"
        "Search for past emails from this sender and summarize their communication profile."
    )


def create_sender_profiler():
    return create_specialist_agent(
        state_class=SenderProfilerState,
        system_prompt=_SYSTEM_PROMPT,
        output_key="sender_profile",
        output_status="profiled",
        agent_name="sender_profiler",
        seed_message_fn=_seed,
        readonly=True,
        temperature=0,
    )

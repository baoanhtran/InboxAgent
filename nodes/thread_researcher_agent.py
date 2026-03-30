"""Thread researcher specialist — fetches full email thread for conversation context."""

from __future__ import annotations

from nodes.agent_factory import create_specialist_agent
from state import ThreadResearcherState

_SYSTEM_PROMPT = """\
You are a thread researcher agent. Your job is to find and summarize the full
conversation thread related to the email you are given.

Use get_email to fetch earlier messages in the thread if you have their IDs,
or use list_emails to search for related messages by subject or sender.

Provide a concise summary of:
- The history of this conversation
- Key points discussed previously
- Any commitments, decisions, or pending items from earlier messages
- The overall tone and relationship between the parties
"""


def _seed(state: ThreadResearcherState) -> str:
    return (
        f"Please research the email thread for this email:\n\n{state['email_content']}\n\n"
        "Fetch relevant prior messages and summarize the conversation history."
    )


def create_thread_researcher():
    return create_specialist_agent(
        state_class=ThreadResearcherState,
        system_prompt=_SYSTEM_PROMPT,
        output_key="thread_context",
        output_status="researched",
        agent_name="thread_researcher",
        seed_message_fn=_seed,
        readonly=True,
        temperature=0,
    )

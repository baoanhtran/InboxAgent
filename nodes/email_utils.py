"""Shared email formatting utilities for the MCP Gmail server response format.

The MCP server returns tool results as:
    [{"type": "text", "text": "<content>", "id": "<lc_id>"}]

list_emails text format (plain text):
    ID: <uuid>
    Date: <iso>
    From: <addr>
    Subject: <subj>
    Preview: <snippet>

get_email text format (JSON string):
    {
      "headers": "---\nmessageId: \"<uuid>\"\nfrom: \"...\"\nsubject: \"...\"\n...",
      "body": "<html>..."
    }
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    return s.get_text()


# ---------------------------------------------------------------------------
# Parse MCP tool result content
# ---------------------------------------------------------------------------

def extract_text_from_result(result) -> str:
    """Extract the text string from an MCP tool result.

    Handles both raw list (new API) and plain string responses.
    """
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(result)


def extract_message_id_from_list_result(result) -> str:
    """Pull the UUID from a list_emails tool result."""
    text = extract_text_from_result(result)
    m = re.search(r"^ID:\s*([a-f0-9\-]{36})", text, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: any UUID-shaped string
    m = re.search(r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", text, re.IGNORECASE)
    return m.group(1) if m else ""


def format_email_from_get_result(result, message_id: str) -> str:
    """Format a get_email result into a human-readable string for the agents."""
    text = extract_text_from_result(result)

    # Try to parse as JSON {"headers": "...", "body": "..."}
    headers_text = ""
    body_text = ""
    try:
        data = json.loads(text)
        headers_text = data.get("headers", "")
        body_raw = data.get("body", "")
        body_text = strip_html(body_raw) if body_raw else ""
    except (json.JSONDecodeError, ValueError):
        # Fallback: treat entire text as body
        body_text = strip_html(text) if "<" in text else text

    # Parse YAML-style headers into a dict
    def _parse_header(key: str) -> str:
        m = re.search(rf'^{key}:\s*"?(.+?)"?\s*$', headers_text, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip('"').strip() if m else ""

    from_ = _parse_header("from")
    subject = _parse_header("subject")
    date = _parse_header("date")
    to = _parse_header("to")
    thread_id = _parse_header("messageId") or message_id

    return (
        f"From: {from_ or 'Unknown'}\n"
        f"To: {to}\n"
        f"Subject: {subject or '(no subject)'}\n"
        f"Date: {date}\n"
        f"Message-ID: {message_id}\n"
        f"Thread-ID: {thread_id}\n"
        f"\n{body_text or '(body unavailable)'}"
    )

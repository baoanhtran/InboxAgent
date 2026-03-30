"""Email parsing utilities for MCP Gmail server responses.

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

import email.header
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
    """Pull the message ID from a list_emails tool result.

    Gmail IDs can be plain hex (e.g. 18abc4def) or UUID format.
    Match anything after 'ID: ' up to the end of the line.
    """
    text = extract_text_from_result(result)
    # Gmail IDs are hex or UUID format — never start with "lc_" (LangChain internal)
    m = re.search(r"^ID:\s*([a-f0-9][a-f0-9\-]{7,})", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_attachments_from_get_result(result) -> list[dict]:
    """Extract attachment metadata from a get_email tool result.

    Returns a list of dicts with keys: filename, mimeType, resourceUri.
    Returns empty list if the email has no attachments.

    The MCP server includes attachment info in the YAML-style headers block,
    typically in the form:
        attachments:
          - filename: "doc.pdf"
            mimeType: "application/pdf"
            resourceUri: "gmail://..."
    """
    text = extract_text_from_result(result)

    headers_text = ""
    try:
        data = json.loads(text)
        headers_text = data.get("headers", "")
    except (json.JSONDecodeError, ValueError):
        headers_text = text

    attachments = []

    # Find every resourceUri and collect associated filename/mimeType nearby
    for uri_m in re.finditer(r'resourceUri:\s*"?([^\s",\n]+)"?', headers_text):
        uri = uri_m.group(1).strip("\"'")

        # Scan a 400-char window around each uri match for metadata
        win_start = max(0, uri_m.start() - 400)
        win_end = min(len(headers_text), uri_m.end() + 100)
        window = headers_text[win_start:win_end]

        fname_m = re.search(r'filename:\s*"?([^\s",\n]+)"?', window)
        mime_m = re.search(r'mimeType:\s*"?([^\s",\n]+)"?', window)

        attachments.append({
            "resourceUri": uri,
            "filename": fname_m.group(1).strip("\"'") if fname_m else "attachment",
            "mimeType": mime_m.group(1).strip("\"'") if mime_m else "application/octet-stream",
        })

    return attachments


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
        if not m:
            return ""
        value = m.group(1).strip('"').strip()
        # Decode MIME-encoded header values (RFC 2047), e.g. =?UTF-8?Q?r=C3=A9union?=
        parts = email.header.decode_header(value)
        decoded = []
        for part, charset in parts:
            if isinstance(part, bytes):
                decoded.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                decoded.append(part)
        return "".join(decoded)

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

"""Attachment analyzer — reads email attachments via MCP resources/read and
summarizes their content using OpenAI.

Why OpenAI is hardcoded here:
  - Images: Chat Completions supports base64 image_url (gpt-4o vision).
  - PDFs:   Responses API supports input_file with base64 PDF data.
  - Other providers have inconsistent or absent support for binary file inputs.

Configure the OpenAI model via ATTACHMENT_ANALYZER_MODEL (default: gpt-4o).
OPENAI_API_KEY must be set in the environment.

Flow:
  attachment_analyzer_node(state) reads state["email_attachments"], calls
  read_mcp_resource() for each URI, dispatches to OpenAI, and writes the
  combined text summary to state["attachment_summary"].
"""

from __future__ import annotations

import base64
import os
from typing import Any

from gmail.client import read_mcp_resource
from state import AgentState

# MIME type routing helpers
_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}
_TEXT_MIMES = {
    "text/plain", "text/csv", "text/html", "text/xml",
    "application/json", "application/xml", "application/csv",
    "application/x-csv",
}


def _get_model() -> str:
    return os.getenv("ATTACHMENT_ANALYZER_MODEL", "gpt-4o")


async def _summarize_attachment(att: dict[str, Any]) -> str:
    """Read one attachment via MCP and summarize it with OpenAI."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    model = _get_model()

    uri = att.get("resourceUri", "")
    filename = att.get("filename", "attachment")
    mime_type = att.get("mimeType", "application/octet-stream")

    if not uri:
        return f"**{filename}**: No resourceUri provided — skipped."

    try:
        b64_data, actual_mime = await read_mcp_resource(uri)
    except Exception as e:
        return f"**{filename}**: Failed to read resource — {e}"

    if not b64_data:
        return f"**{filename}**: Resource returned empty content."

    # Prefer the mime type returned by the MCP server
    if actual_mime and actual_mime != "application/octet-stream":
        mime_type = actual_mime

    try:
        # --- Images: Chat Completions with base64 image_url ---
        if mime_type in _IMAGE_MIMES or mime_type.startswith("image/"):
            resp = await client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This image is an email attachment named '{filename}'. "
                                "Please describe its content clearly and concisely. "
                                "Highlight any text, data, or action items visible."
                            ),
                        },
                    ],
                }],
                max_tokens=1024,
            )
            return f"**{filename}**:\n{resp.choices[0].message.content}"

        # --- PDFs: upload via Files API then use file_id in Responses API ---
        elif mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            pdf_bytes = base64.b64decode(b64_data)
            file_id: str | None = None
            try:
                # Step 1: upload to OpenAI Files API
                file_obj = await client.files.create(
                    file=(filename, pdf_bytes, "application/pdf"),
                    purpose="user_data",
                )
                file_id = file_obj.id

                # Step 2: analyze via Responses API referencing the file_id
                resp = await client.responses.create(
                    model=model,
                    input=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": file_id,
                            },
                            {
                                "type": "input_text",
                                "text": (
                                    f"This PDF is an email attachment named '{filename}'. "
                                    "Summarize its key content, main points, and any action items."
                                ),
                            },
                        ],
                    }],
                )
                return f"**{filename}**:\n{resp.output_text}"

            except Exception as pdf_err:
                size_approx = len(pdf_bytes)
                return (
                    f"**{filename}** (PDF, ~{size_approx} bytes): "
                    f"PDF analysis failed — {pdf_err}"
                )
            finally:
                # Step 3: delete the uploaded file to avoid storage accumulation
                if file_id:
                    try:
                        await client.files.delete(file_id)
                    except Exception:
                        pass

        # --- Text-based files: decode and pass as plain text ---
        elif mime_type in _TEXT_MIMES or mime_type.startswith("text/"):
            raw_bytes = base64.b64decode(b64_data)
            text_content = raw_bytes.decode("utf-8", errors="replace")
            resp = await client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"This is the content of an email attachment named '{filename}' ({mime_type}):\n\n"
                        f"{text_content[:8000]}\n\n"
                        "Summarize the key information and any action items."
                    ),
                }],
                max_tokens=1024,
            )
            return f"**{filename}**:\n{resp.choices[0].message.content}"

        # --- Unknown binary: report metadata only ---
        else:
            size_approx = len(b64_data) * 3 // 4
            return (
                f"**{filename}** ({mime_type}, ~{size_approx} bytes): "
                "Binary format — automatic analysis not supported for this file type."
            )

    except Exception as e:
        return f"**{filename}**: Analysis error — {e}"


async def attachment_analyzer_node(state: AgentState) -> dict:
    """Outer-graph node: read and summarize all email attachments.

    Always uses OpenAI (ATTACHMENT_ANALYZER_MODEL, default gpt-4o) regardless
    of global llm_config settings, because binary file inputs require
    OpenAI's multimodal and Responses API capabilities.

    Writes:
      attachment_summary — combined human-readable summary of all attachments
      status             — "attachments_analyzed" | "no_attachments"
    """
    attachments: list = state.get("email_attachments") or []

    if not attachments:
        print("  [Analyzer]    email_attachments is empty — no attachments to read.")
        return {"attachment_summary": "", "status": "no_attachments"}

    print(f"  [Analyzer]    Reading {len(attachments)} attachment(s):")
    for att in attachments:
        print(f"                  {att.get('filename', '?')} ({att.get('mimeType', '?')}) → {att.get('resourceUri', '?')[:60]}")

    summaries = []
    for att in attachments:
        summary = await _summarize_attachment(att)
        summaries.append(summary)
        print(f"  [Analyzer]    Result for {att.get('filename', '?')}: {summary[:200]}")

    combined = "\n\n".join(summaries)
    return {
        "attachment_summary": combined,
        "status": "attachments_analyzed",
    }

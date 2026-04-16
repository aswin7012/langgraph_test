"""Agent tools: Tavily web search and SMTP email."""

from __future__ import annotations

import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx
from langchain_core.tools import tool

from retrieval_pipeline.config import PipelineConfig

_cfg = PipelineConfig()
_sent_fingerprints: set[str] = set()
MAX_TOOL_OUTPUT_CHARS = 6000


# ── Tool 1: Web Search (Tavily) ──────────────────────────────────────────────


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily and return ranked results.

    Use when the user asks about recent events, live data, or topics
    not covered by the document store.

    Args:
        query: The search query.
        max_results: Number of results to return (1-10). Default 5.
    """
    if not _cfg.tavily_api_key:
        return "Error: TAVILY_API_KEY is not configured. Add it to your .env file."

    payload = {
        "api_key": _cfg.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max(1, min(max_results, 10)),
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }

    try:
        resp = httpx.post("https://api.tavily.com/search", json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        return f"Error: Tavily returned HTTP {exc.response.status_code}."
    except httpx.RequestError as exc:
        return f"Error: Network failure reaching Tavily — {exc}."

    lines: list[str] = []
    if data.get("answer"):
        lines.append(f"[Summary] {data['answer']}\n")

    for i, r in enumerate(data.get("results", []), 1):
        snippet = r.get("content", "")[:800]
        lines.append(f"[{i}] {r.get('title', 'No title')}\n    URL: {r.get('url', '')}\n    {snippet}")

    if not lines:
        return f"No results found for: '{query}'."

    output = "\n\n".join(lines)
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n\n[Truncated]"
    return output


# ── Tool 2: Send Email (SMTP) ────────────────────────────────────────────────


@tool
def send_email(
    to_address: str,
    subject: str,
    body: str,
    content_type: str = "plain",
) -> str:
    """Send an email via SMTP using credentials from the pipeline config.

    Use when the user explicitly asks to email a result or summary.

    Args:
        to_address: Recipient email address.
        subject: Email subject line.
        body: Email body text.
        content_type: MIME subtype — 'plain' (default) or 'html'.
    """
    missing = [
        name
        for name, val in {
            "SMTP_HOST": _cfg.smtp_host,
            "SMTP_USER": _cfg.smtp_user,
            "SMTP_PASSWORD": _cfg.smtp_password,
        }.items()
        if not val
    ]
    if missing:
        return f"Error: Missing SMTP config — {', '.join(missing)}. Add to .env."

    mime_subtype = content_type if content_type in {"plain", "html"} else "plain"

    fingerprint = hashlib.sha256(f"{to_address}|{subject}|{mime_subtype}|{body}".encode()).hexdigest()
    if fingerprint in _sent_fingerprints:
        return f"Success: Email already sent to '{to_address}' with subject '{subject}'."

    msg = MIMEMultipart("alternative")
    msg["From"] = _cfg.smtp_user
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body, mime_subtype, "utf-8"))

    error_message: str | None = None

    try:
        with smtplib.SMTP(_cfg.smtp_host, _cfg.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(_cfg.smtp_user, _cfg.smtp_password)
            server.sendmail(_cfg.smtp_user, to_address, msg.as_string())
    except smtplib.SMTPAuthenticationError:
        error_message = "Error: SMTP authentication failed — check SMTP_USER and SMTP_PASSWORD."
    except smtplib.SMTPConnectError:
        error_message = f"Error: Could not connect to SMTP server '{_cfg.smtp_host}:{_cfg.smtp_port}'."
    except smtplib.SMTPRecipientsRefused:
        error_message = f"Error: Recipient '{to_address}' was refused by the server."
    except (smtplib.SMTPException, TimeoutError) as exc:
        error_message = f"Error: SMTP failure — {exc}."

    if error_message:
        return error_message

    _sent_fingerprints.add(fingerprint)
    return f"Success: Email sent to '{to_address}' with subject '{subject}'."

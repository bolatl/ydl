"""Email delivery via MailerSend (Phase 3, optional).

Sends to ADMIN_EMAIL only, on explicit user action — never in a loop. Uses the
MailerSend REST API directly via `requests` (no extra dependency).
"""
from __future__ import annotations

import requests

from core import config

_API_URL = "https://api.mailersend.com/v1/email"


def _send(subject: str, text: str) -> None:
    """POST one email to the configured ADMIN_EMAIL. Raises on misconfig or API error."""
    config.require_email_config()

    payload = {
        "from": {"email": config.MAILERSEND_FROM_EMAIL, "name": config.MAILERSEND_FROM_NAME},
        "to": [{"email": config.ADMIN_EMAIL}],
        "subject": subject,
        "text": text,
    }
    resp = requests.post(
        _API_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {config.MAILERSEND_API_KEY}",
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=30,
    )
    # MailerSend returns 202 Accepted on success.
    if resp.status_code != 202:
        raise RuntimeError(
            f"MailerSend error {resp.status_code}: {resp.text or '(empty response)'}"
        )


def send_summary_email(summary: str, subject: str = "Саммари диалога — Грант-помощник") -> None:
    """Email a chat summary to the admin."""
    _send(subject, summary)


def _format_conversation(messages: list[dict] | None) -> str:
    if not messages:
        return ""
    lines = [
        f"{'Пользователь' if m['role'] == 'user' else 'Бот'}: {m['content']}"
        for m in messages
    ]
    return "\n\n— История разговора —\n" + "\n".join(lines)


def send_contact_request(
    name: str,
    contact: str,
    request_text: str,
    conversation: list[dict] | None = None,
) -> None:
    """Email a user's contact request (lead) to the admin, with the chat transcript."""
    body = (
        "Новая заявка из чат-бота фонда Есенова.\n\n"
        f"Имя: {name or '—'}\n"
        f"Контакт: {contact}\n"
        f"Запрос: {request_text}\n"
        f"{_format_conversation(conversation)}"
    )
    _send("Новая заявка из чат-бота — Грант-помощник", body)

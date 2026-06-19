"""Per-chat conversation memory for the Telegram bot (SQLite).

Stores, per chat_id: chosen language, a rolling summary of older turns, and the
most recent messages verbatim. Older turns are folded into the summary so the
prompt stays small — but, as everywhere, memory only helps interpret the
question; grant facts always come from RAG.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from core import config
from core.llm import summarize_memory

MAX_RECENT_MESSAGES = 12          # compress once the window grows past this
KEEP_AFTER_COMPRESSION = 6        # messages kept verbatim after compression


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.BOT_MEMORY_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS conversations (
            chat_id TEXT PRIMARY KEY,
            lang TEXT DEFAULT 'ru',
            summary TEXT DEFAULT '',
            recent_json TEXT DEFAULT '[]',
            updated_at TEXT
        )"""
    )
    return conn


def load(chat_id: int | str) -> dict:
    """Return {lang, summary, recent} for a chat, creating defaults if absent."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT lang, summary, recent_json FROM conversations WHERE chat_id = ?",
            (str(chat_id),),
        ).fetchone()
    if not row:
        return {"lang": config.DEFAULT_LANG, "summary": "", "recent": []}
    return {"lang": row[0], "summary": row[1], "recent": json.loads(row[2] or "[]")}


def _save(chat_id: int | str, lang: str, summary: str, recent: list[dict]) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO conversations (chat_id, lang, summary, recent_json, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(chat_id) DO UPDATE SET
                   lang=excluded.lang, summary=excluded.summary,
                   recent_json=excluded.recent_json, updated_at=excluded.updated_at""",
            (
                str(chat_id),
                lang,
                summary,
                json.dumps(recent, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def set_lang(chat_id: int | str, lang: str) -> None:
    state = load(chat_id)
    _save(chat_id, lang, state["summary"], state["recent"])


def add_turn(chat_id: int | str, user_text: str, assistant_text: str) -> None:
    """Append a user/assistant turn, compressing older messages when the window grows."""
    state = load(chat_id)
    recent = state["recent"] + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    summary = state["summary"]
    if len(recent) > MAX_RECENT_MESSAGES:
        to_compress = recent[:-KEEP_AFTER_COMPRESSION]
        recent = recent[-KEEP_AFTER_COMPRESSION:]
        summary = summarize_memory(summary, to_compress)
    _save(chat_id, state["lang"], summary, recent)


def reset(chat_id: int | str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM conversations WHERE chat_id = ?", (str(chat_id),))


def history_for_prompt(state: dict) -> list[dict]:
    """Prior turns to pass as conversational context: rolling summary + recent."""
    history: list[dict] = []
    if state.get("summary"):
        history.append({"role": "system", "content": "Краткая память: " + state["summary"]})
    history.extend(state.get("recent", []))
    return history

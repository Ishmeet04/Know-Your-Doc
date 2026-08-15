"""
Persistent storage layer using SQLite.
Stores:
  - sessions table:      session_id, filenames (JSON), created_at
  - chat_history table:  session_id, role, content, timestamp

Also exposes SQLiteChatMessageHistory — a LangChain-compatible
BaseChatMessageHistory that reads/writes from the same DB so
conversation memory survives Flask server restarts.
"""

import sqlite3
import json
import os
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                filenames  TEXT NOT NULL DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)


def save_session(session_id: str, filenames: list):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, filenames) VALUES (?, ?)",
            (session_id, json.dumps(filenames)),
        )


def get_session(session_id: str):
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT filenames, created_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row:
        return {"filenames": json.loads(row["filenames"]), "created_at": row["created_at"]}
    return None


def get_all_sessions() -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT session_id, filenames, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "session_id": r["session_id"],
            "filenames": json.loads(r["filenames"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def delete_session(session_id: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM sessions     WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))


def save_message(session_id: str, role: str, content: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_messages(session_id: str) -> List[BaseMessage]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    msgs = []
    for row in rows:
        if row["role"] == "human":
            msgs.append(HumanMessage(content=row["content"]))
        else:
            msgs.append(AIMessage(content=row["content"]))
    return msgs


def get_messages_as_dicts(session_id: str) -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM chat_history "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """Drop-in for InMemoryChatMessageHistory — persists to SQLite."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        return get_messages(self.session_id)

    def add_message(self, message: BaseMessage):
        role = "human" if isinstance(message, HumanMessage) else "ai"
        save_message(self.session_id, role, message.content)

    def clear(self):
        with _get_conn() as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?", (self.session_id,)
            )


init_db()

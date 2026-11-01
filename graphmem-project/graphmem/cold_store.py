"""Cold storage of original texts in SQLite (plan §7.4)."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS originals (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id   TEXT,
  agent       TEXT NOT NULL,
  turn        INTEGER NOT NULL,
  text        TEXT NOT NULL,
  scenario_id TEXT,
  event_id    TEXT
);
CREATE INDEX IF NOT EXISTS idx_originals_memory ON originals(memory_id);
"""


class ColdStore:
    def __init__(self, path: str = ":memory:", scenario_id: str | None = None):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # ToolNode may run tools in worker threads, so allow cross-thread use with a lock.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.scenario_id = scenario_id

    def add(self, agent: str, turn: int, text: str, event_id: str | None = None) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO originals(agent, turn, text, scenario_id, event_id) VALUES (?,?,?,?,?)",
                (agent, turn, text, self.scenario_id, event_id))
            self.conn.commit()
            return int(cur.lastrowid)

    def link(self, cold_id: int, memory_id: str):
        with self.lock:
            self.conn.execute("UPDATE originals SET memory_id=? WHERE id=?", (memory_id, cold_id))
            self.conn.commit()

    def get(self, memory_id: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, agent, turn, text, event_id FROM originals WHERE memory_id=? ORDER BY id",
                (memory_id,)).fetchall()
        return [dict(zip(("cold_id", "agent", "turn", "text", "event_id"), r)) for r in rows]

    def count(self) -> int:
        with self.lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM originals").fetchone()[0])

    def close(self):
        self.conn.close()

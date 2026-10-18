"""
Eigen Web History Tracker

A standalone, independent audit log of every web page Eigen fetches via his
web_fetch tool. Kept separate from his episodic memory so you have a clean,
queryable history of what Eigen has been reading/studying online.

Schema: web_history.db -> web_fetches(id, timestamp, url, status, preview)

Usage:
    from web_history import log_fetch, recent_fetches, digest
    log_fetch("https://example.com", "ok", "some preview text")
    digest()  # return a human-readable summary of recent fetches
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = "/home/eigen/web_history.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS web_fetches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            url TEXT,
            status TEXT,
            preview TEXT
        )
        """
    )
    conn.commit()
    return conn


def log_fetch(url, status="ok", preview=""):
    """Record a single web fetch."""
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO web_fetches (timestamp, url, status, preview) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), url, status, (preview or "")[:500]),
        )
        conn.commit()
    finally:
        conn.close()


def recent_fetches(hours=24, limit=200):
    """Return rows from the last `hours` (oldest first)."""
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT timestamp, url, status, preview FROM web_fetches "
            "WHERE timestamp >= datetime('now', 'localtime', ?) "
            "ORDER BY id ASC LIMIT ?",
            (f"-{hours} hours", limit),
        ).fetchall()
        return rows
    finally:
        conn.close()


def digest(hours=24, limit=200):
    """Return a compact human-readable summary of recent fetches."""
    rows = recent_fetches(hours=hours, limit=limit)
    if not rows:
        return "No web fetches recorded in this window."
    lines = [f"Web history (last {hours}h, {len(rows)} fetches):"]
    for ts, url, status, preview in rows:
        short = url
        if len(short) > 90:
            short = short[:87] + "..."
        lines.append(f"- [{status}] {short}")
    return "\n".join(lines)


def clear():
    """Wipe the history table (explicit, manual only)."""
    conn = _conn()
    try:
        conn.execute("DELETE FROM web_fetches")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    print(digest(hours=24))

"""Local SQLite storage writer.

Writes lead rows to data/{client_id}/leads.db — one database file per client.
The schema matches the fixed column set used across all sources.

-- TO SWAP BACK TO GOOGLE SHEETS --
In agent/runner.py, replace:
    from storage.writer import StorageWriter
    writer = StorageWriter(client_id)
with:
    from sheets.writer import SheetsWriter
    writer = SheetsWriter(config["sheet_id"])
That's it. The rest of the codebase is unchanged.
-----------------------------------
"""

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS leads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    job_title   TEXT,
    company     TEXT,
    email       TEXT,
    phone       TEXT,
    social_media TEXT,
    source_url  TEXT UNIQUE,
    scrape_status TEXT,
    scraped_at  TEXT
);
"""


class StorageWriter:
    """Appends lead rows to a per-client SQLite database."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        db_dir = Path(f"data/{client_id}")
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / "leads.db"
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()
        logger.info("Storage ready: %s", self.db_path)

    async def append_row(
        self,
        source_url: str,
        data: dict,
        scrape_status: str = "ok",
    ) -> None:
        """Append a lead row. Skips silently if source_url already exists."""
        async with self._lock:
            await asyncio.to_thread(self._insert, source_url, data, scrape_status)

    def _insert(self, source_url: str, data: dict, scrape_status: str) -> None:
        row = (
            data.get("name"),
            data.get("job_title"),
            data.get("company"),
            data.get("email"),
            data.get("phone"),
            data.get("social_media"),
            source_url,
            scrape_status,
            datetime.now(timezone.utc).isoformat(),
        )
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO leads
                       (name, job_title, company, email, phone, social_media,
                        source_url, scrape_status, scraped_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
                conn.commit()
            logger.info("Saved lead: %s", source_url)
        except Exception as exc:
            logger.error("DB write failed for %s: %s", source_url, exc)

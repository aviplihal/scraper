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
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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
        self.saved_count = 0
        self.duplicate_count = 0
        self.saved_rows: list[dict] = []
        self._known_source_urls: set[str] = set()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE)
            rows = conn.execute("SELECT source_url FROM leads").fetchall()
            conn.commit()
        self._known_source_urls = {
            _normalize_source_url(str(row[0]))
            for row in rows
            if row and row[0]
        }
        logger.info("Storage ready: %s", self.db_path)

    async def append_row(
        self,
        source_url: str,
        data: dict,
        scrape_status: str = "ok",
    ) -> str:
        """Append a lead row and return ``saved`` or ``duplicate``."""
        async with self._lock:
            return await asyncio.to_thread(self._insert, source_url, data, scrape_status)

    def _insert(self, source_url: str, data: dict, scrape_status: str) -> str:
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
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO leads
                       (name, job_title, company, email, phone, social_media,
                        source_url, scrape_status, scraped_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
                conn.commit()
            if cursor.rowcount == 0:
                self.duplicate_count += 1
                logger.info("Skipped duplicate lead: %s", source_url)
                return "duplicate"

            self.saved_count += 1
            self._known_source_urls.add(_normalize_source_url(source_url))
            self.saved_rows.append(
                {
                    "name": data.get("name"),
                    "job_title": data.get("job_title"),
                    "company": data.get("company"),
                    "source_url": source_url,
                    "scrape_status": scrape_status,
                }
            )
            logger.info("Saved lead: %s", source_url)
            return "saved"
        except Exception as exc:
            logger.error("DB write failed for %s: %s", source_url, exc)
            raise

    def recent_rows(self, limit: int = 20) -> list[dict]:
        """Return recent saved rows for baseline comparisons."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT name, job_title, company, email, phone, social_media, source_url
                   FROM leads
                   ORDER BY id DESC
                   LIMIT ?""",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def has_source_url(self, source_url: str) -> bool:
        """Return True when a source URL already exists in storage."""
        return _normalize_source_url(source_url) in self._known_source_urls


def _normalize_source_url(source_url: str) -> str:
    """Normalize source URLs for duplicate checks."""
    parsed = urlparse(source_url if "://" in source_url else f"https://{source_url}")
    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(parse_qsl(parsed.query, keep_blank_values=True), doseq=True)
    normalized = parsed._replace(netloc=host, path=path, query=query, fragment="")
    return urlunparse(normalized)

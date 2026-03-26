"""Google Sheets writer — appends lead rows live as they are found."""

import asyncio
import logging
import os
from datetime import datetime, timezone

import gspread
from oauth2client.service_account import ServiceAccountCredentials

logger = logging.getLogger(__name__)

SHEET_COLUMNS = [
    "name",
    "job_title",
    "company",
    "email",
    "phone",
    "social_media",
    "source_url",
    "scrape_status",
    "scraped_at",
]

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


class SheetsWriter:
    """Appends rows to a Google Sheet. Thread-safe via asyncio lock."""

    def __init__(self, sheet_id: str):
        self.sheet_id = sheet_id
        self._client: gspread.Client | None = None
        self._worksheet: gspread.Worksheet | None = None
        self._lock = asyncio.Lock()

    def _connect(self) -> None:
        creds_path = os.environ["GOOGLE_SERVICE_ACCOUNT_PATH"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, SCOPES)
        self._client = gspread.authorize(creds)
        spreadsheet = self._client.open_by_key(self.sheet_id)
        try:
            self._worksheet = spreadsheet.worksheet("Leads")
        except gspread.WorksheetNotFound:
            self._worksheet = spreadsheet.add_worksheet("Leads", rows=10000, cols=len(SHEET_COLUMNS))
            self._worksheet.append_row(SHEET_COLUMNS)

    async def append_row(
        self,
        source_url: str,
        data: dict,
        scrape_status: str = "ok",
    ) -> None:
        """Append a single lead row. Runs blocking gspread calls in a thread."""
        async with self._lock:
            await asyncio.to_thread(self._append_row_sync, source_url, data, scrape_status)

    def _append_row_sync(
        self, source_url: str, data: dict, scrape_status: str
    ) -> None:
        if self._worksheet is None:
            self._connect()
        row = [
            data.get("name") or "",
            data.get("job_title") or "",
            data.get("company") or "",
            data.get("email") or "",
            data.get("phone") or "",
            data.get("social_media") or "",
            source_url,
            scrape_status,
            datetime.now(timezone.utc).isoformat(),
        ]
        self._worksheet.append_row(row, value_input_option="USER_ENTERED")
        logger.info("Wrote row to sheet: %s", source_url)

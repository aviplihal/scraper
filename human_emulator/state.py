"""State file manager for the human emulator.

Tracks:
  - profiles_queue  — ordered list of social-media profile URLs to visit
  - visited_today   — URLs already visited in today's session
  - visits_today    — count of visits today (reset each calendar day)
  - current_position — index into profiles_queue where we left off
  - restriction_count — consecutive restrictions in the current run
  - paused_until    — ISO timestamp; if set, the emulator is paused until this time
  - last_run_date   — ISO date string used to detect day rollovers
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DAILY_VISIT_LIMIT = 75


class EmulatorState:
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.path = Path(f"state/{client_id}_state.json")
        self._data: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
        else:
            self._data = self._default()
        self._maybe_reset_daily_counters()

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _default(self) -> dict:
        return {
            "profiles_queue": [],
            "visited_today": [],
            "visits_today": 0,
            "current_position": 0,
            "restriction_count": 0,
            "paused_until": None,
            "last_run_date": str(date.today()),
        }

    def _maybe_reset_daily_counters(self) -> None:
        today = str(date.today())
        if self._data.get("last_run_date") != today:
            logger.info("New day detected — resetting daily visit counters.")
            self._data["visited_today"] = []
            self._data["visits_today"] = 0
            self._data["restriction_count"] = 0
            self._data["paused_until"] = None
            self._data["last_run_date"] = today
            self._save()

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def add_profiles(self, urls: list[str]) -> int:
        """Add new profile URLs to the queue (deduplication included)."""
        existing = set(self._data["profiles_queue"]) | set(self._data["visited_today"])
        added = 0
        for url in urls:
            if url not in existing:
                self._data["profiles_queue"].append(url)
                existing.add(url)
                added += 1
        self._save()
        return added

    def next_profiles(self, batch_size: int = 1) -> list[str]:
        """Return up to batch_size unvisited profiles, respecting daily limit."""
        remaining_today = DAILY_VISIT_LIMIT - self._data["visits_today"]
        if remaining_today <= 0:
            return []
        pos = self._data["current_position"]
        queue = self._data["profiles_queue"]
        visited = set(self._data["visited_today"])
        batch: list[str] = []
        while pos < len(queue) and len(batch) < min(batch_size, remaining_today):
            url = queue[pos]
            pos += 1
            if url not in visited:
                batch.append(url)
        return batch

    def mark_visited(self, url: str) -> None:
        self._data["visited_today"].append(url)
        self._data["visits_today"] += 1
        # Advance position past this URL
        queue = self._data["profiles_queue"]
        try:
            idx = queue.index(url)
            self._data["current_position"] = max(self._data["current_position"], idx + 1)
        except ValueError:
            pass
        self._save()

    def daily_budget_exhausted(self) -> bool:
        return self._data["visits_today"] >= DAILY_VISIT_LIMIT

    def queue_exhausted(self) -> bool:
        return self._data["current_position"] >= len(self._data["profiles_queue"])

    # ------------------------------------------------------------------
    # Restriction handling
    # ------------------------------------------------------------------

    def record_restriction(self) -> int:
        """Increment restriction count and return new total."""
        self._data["restriction_count"] += 1
        self._save()
        return self._data["restriction_count"]

    def set_pause(self, until: datetime) -> None:
        self._data["paused_until"] = until.isoformat()
        self._save()

    def clear_pause(self) -> None:
        self._data["paused_until"] = None
        self._save()

    def is_paused(self) -> tuple[bool, Optional[datetime]]:
        raw = self._data.get("paused_until")
        if not raw:
            return False, None
        until = datetime.fromisoformat(raw)
        now = datetime.now(timezone.utc)
        # Make both offset-aware for comparison
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if now < until:
            return True, until
        # Pause has expired — clear it
        self.clear_pause()
        return False, None

    @property
    def restriction_count(self) -> int:
        return self._data["restriction_count"]

    @property
    def visits_today(self) -> int:
        return self._data["visits_today"]

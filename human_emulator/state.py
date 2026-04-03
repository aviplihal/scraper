"""State file manager for multi-platform human-emulator runs."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DAILY_VISIT_LIMIT = 75
DEFAULT_PLATFORMS = ("linkedin", "x")


class EmulatorState:
    """Persist per-platform queue, pause, availability, and budget state."""

    def __init__(self, client_id: str, platforms: list[str] | None = None):
        self.client_id = client_id
        self.path = Path(f"state/{client_id}_state.json")
        self._platforms = tuple(dict.fromkeys(platforms or DEFAULT_PLATFORMS))
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

        self._migrate_legacy_format()
        for platform in self._platforms:
            self.ensure_platform(platform)
        self._maybe_reset_daily_counters()
        self._save()

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _default_platform_state(self) -> dict:
        return {
            "profiles_queue": [],
            "visited_today": [],
            "visits_today": 0,
            "current_position": 0,
            "restriction_count": 0,
            "paused_until": None,
            "last_run_date": str(date.today()),
            "availability_status": "unknown",
            "availability_reason": "",
        }

    def _default(self) -> dict:
        return {
            "platforms": {
                platform: self._default_platform_state()
                for platform in self._platforms
            }
        }

    def _migrate_legacy_format(self) -> None:
        """Upgrade the old flat queue format into per-platform state."""
        if "platforms" in self._data:
            return

        legacy_queue = self._data.get("profiles_queue", [])
        legacy_visited = self._data.get("visited_today", [])
        linkedin_queue = [
            url for url in legacy_queue
            if _infer_platform_from_url(url) == "linkedin"
        ]
        linkedin_visited = [
            url for url in legacy_visited
            if _infer_platform_from_url(url) == "linkedin"
        ]

        migrated = self._default()
        migrated["platforms"]["linkedin"].update(
            {
                "profiles_queue": linkedin_queue,
                "visited_today": linkedin_visited,
                "visits_today": int(self._data.get("visits_today", len(linkedin_visited))),
                "current_position": int(self._data.get("current_position", 0)),
                "restriction_count": int(self._data.get("restriction_count", 0)),
                "paused_until": self._data.get("paused_until"),
                "last_run_date": self._data.get("last_run_date", str(date.today())),
            }
        )
        self._data = migrated

    def _maybe_reset_daily_counters(self) -> None:
        today = str(date.today())
        for platform, platform_state in self._data.get("platforms", {}).items():
            if platform_state.get("last_run_date") != today:
                logger.info("New day detected — resetting daily counters for %s.", platform)
                platform_state["visited_today"] = []
                platform_state["visits_today"] = 0
                platform_state["restriction_count"] = 0
                platform_state["paused_until"] = None
                platform_state["last_run_date"] = today

    # ------------------------------------------------------------------
    # Platform accessors
    # ------------------------------------------------------------------

    def ensure_platform(self, platform: str) -> None:
        platforms = self._data.setdefault("platforms", {})
        platforms.setdefault(platform, self._default_platform_state())

    def platform_state(self, platform: str) -> dict:
        self.ensure_platform(platform)
        return self._data["platforms"][platform]

    def enabled_platforms(self) -> list[str]:
        return list(self._data.get("platforms", {}).keys())

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def add_profiles(self, urls: list[str], platform: str = "linkedin") -> int:
        """Add new profile URLs to a platform queue with deduplication."""
        state = self.platform_state(platform)
        existing = set(state["profiles_queue"]) | set(state["visited_today"])
        added = 0
        for url in urls:
            if url not in existing:
                state["profiles_queue"].append(url)
                existing.add(url)
                added += 1
        self._save()
        return added

    def add_profiles_by_platform(self, items: dict[str, list[str]]) -> int:
        """Add profile URLs grouped by platform."""
        total = 0
        for platform, urls in items.items():
            total += self.add_profiles(urls, platform=platform)
        return total

    def next_profiles(self, platform: str = "linkedin", batch_size: int = 1) -> list[str]:
        """Return the next profile URLs for a platform without mutating state."""
        state = self.platform_state(platform)
        remaining_today = DAILY_VISIT_LIMIT - state["visits_today"]
        if remaining_today <= 0:
            return []
        pos = state["current_position"]
        queue = state["profiles_queue"]
        visited = set(state["visited_today"])
        batch: list[str] = []
        while pos < len(queue) and len(batch) < min(batch_size, remaining_today):
            url = queue[pos]
            pos += 1
            if url not in visited:
                batch.append(url)
        return batch

    def mark_visited(self, url: str, platform: str = "linkedin") -> None:
        state = self.platform_state(platform)
        if url not in state["visited_today"]:
            state["visited_today"].append(url)
        state["visits_today"] += 1
        queue = state["profiles_queue"]
        try:
            idx = queue.index(url)
            state["current_position"] = max(state["current_position"], idx + 1)
        except ValueError:
            pass
        self._save()

    def daily_budget_exhausted(self, platform: str = "linkedin") -> bool:
        return self.platform_state(platform)["visits_today"] >= DAILY_VISIT_LIMIT

    def queue_exhausted(self, platform: str = "linkedin") -> bool:
        state = self.platform_state(platform)
        return state["current_position"] >= len(state["profiles_queue"])

    def any_queue_remaining(self, platforms: list[str] | None = None) -> bool:
        targets = platforms or self.enabled_platforms()
        return any(not self.queue_exhausted(platform) for platform in targets)

    # ------------------------------------------------------------------
    # Restriction / pause handling
    # ------------------------------------------------------------------

    def record_restriction(self, platform: str = "linkedin") -> int:
        state = self.platform_state(platform)
        state["restriction_count"] += 1
        self._save()
        return state["restriction_count"]

    def set_pause(self, platform: str, until: datetime, reason: str = "") -> None:
        state = self.platform_state(platform)
        state["paused_until"] = until.isoformat()
        if reason:
            state["availability_reason"] = reason
        state["availability_status"] = "paused"
        self._save()

    def set_pause_hours(self, platform: str, hours: int, reason: str = "") -> None:
        until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.set_pause(platform, until, reason=reason)

    def clear_pause(self, platform: str = "linkedin") -> None:
        state = self.platform_state(platform)
        state["paused_until"] = None
        if state.get("availability_status") == "paused":
            state["availability_status"] = "active"
        self._save()

    def is_paused(self, platform: str = "linkedin") -> tuple[bool, Optional[datetime]]:
        state = self.platform_state(platform)
        raw = state.get("paused_until")
        if not raw:
            return False, None
        until = datetime.fromisoformat(raw)
        now = datetime.now(timezone.utc)
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if now < until:
            return True, until
        self.clear_pause(platform)
        return False, None

    # ------------------------------------------------------------------
    # Availability / reporting
    # ------------------------------------------------------------------

    def set_availability(self, platform: str, status: str, reason: str = "") -> None:
        state = self.platform_state(platform)
        state["availability_status"] = status
        state["availability_reason"] = reason
        self._save()

    def availability(self, platform: str) -> dict[str, str]:
        state = self.platform_state(platform)
        return {
            "status": state.get("availability_status", "unknown"),
            "reason": state.get("availability_reason", ""),
        }

    def platform_summary(self, platform: str) -> dict[str, object]:
        state = self.platform_state(platform)
        paused, until = self.is_paused(platform)
        return {
            "platform": platform,
            "status": "paused" if paused else state.get("availability_status", "unknown"),
            "reason": state.get("availability_reason", ""),
            "paused_until": until.isoformat() if until else None,
            "visits_today": state.get("visits_today", 0),
            "queue_remaining": max(0, len(state.get("profiles_queue", [])) - state.get("current_position", 0)),
            "restriction_count": state.get("restriction_count", 0),
        }

    @property
    def restriction_count(self) -> int:
        return sum(self.platform_state(platform)["restriction_count"] for platform in self.enabled_platforms())

    @property
    def visits_today(self) -> int:
        return sum(self.platform_state(platform)["visits_today"] for platform in self.enabled_platforms())


def _infer_platform_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if host.endswith("linkedin.com"):
        return "linkedin"
    if host.endswith("x.com") or host.endswith("twitter.com"):
        return "x"
    return "linkedin"


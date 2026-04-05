"""Tests for runner-level social startup behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from agent.runner import _preflight_social_platforms, _should_start_social_browser
from tools.registry import ToolContext


class _FakeEmulatorState:
    def __init__(self, paused_platforms: set[str] | None = None) -> None:
        self.paused_platforms = paused_platforms or set()
        self.statuses: dict[str, dict[str, str]] = {}

    def is_paused(self, platform: str) -> tuple[bool, datetime | None]:
        if platform in self.paused_platforms:
            return True, datetime.now(timezone.utc) + timedelta(hours=2)
        return False, None

    def set_availability(self, platform: str, status: str, reason: str = "") -> None:
        self.statuses[platform] = {"status": status, "reason": reason}

    def availability(self, platform: str) -> dict[str, str]:
        return self.statuses.get(platform, {"status": "unknown", "reason": ""})


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_should_start_social_browser_false_when_all_platforms_are_paused(self) -> None:
        state = _FakeEmulatorState({"linkedin", "x"})

        self.assertFalse(_should_start_social_browser(state, ["linkedin", "x"]))

    async def test_preflight_marks_paused_platforms_unavailable_without_browser(self) -> None:
        state = _FakeEmulatorState({"linkedin", "x"})
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1, "social_platforms": ["linkedin", "x"]},
            sheets_writer=object(),
            source_mode="all",
            effective_source_mode="all",
            emulator_state=state,
            emulator_browser=None,
        )

        await _preflight_social_platforms(ctx.client_config, ctx)

        self.assertEqual(ctx.effective_source_mode, "web")
        self.assertIn("linkedin", ctx.unavailable_social_platforms)
        self.assertIn("x", ctx.unavailable_social_platforms)
        self.assertIn("linkedin.com", ctx.unavailable_domains)
        self.assertIn("x.com", ctx.unavailable_domains)


if __name__ == "__main__":
    unittest.main()

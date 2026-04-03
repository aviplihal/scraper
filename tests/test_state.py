"""Tests for multi-platform emulator state."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from human_emulator.state import EmulatorState


class EmulatorStateTests(unittest.TestCase):
    def test_legacy_linkedin_queue_migrates_to_platform_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state" / "client_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "profiles_queue": [
                            "https://www.linkedin.com/in/alice",
                            "https://x.com/bob",
                        ],
                        "visited_today": ["https://www.linkedin.com/in/visited"],
                        "visits_today": 1,
                        "current_position": 0,
                        "restriction_count": 0,
                        "paused_until": None,
                        "last_run_date": "2099-01-01",
                    }
                )
            )

            with patch("human_emulator.state.Path", side_effect=lambda value: Path(temp_dir) / value):
                state = EmulatorState("client", ["linkedin", "x"])

            self.assertIn("linkedin", state.enabled_platforms())
            self.assertIn("https://www.linkedin.com/in/alice", state.platform_state("linkedin")["profiles_queue"])
            self.assertEqual(state.platform_state("x")["profiles_queue"], [])

    def test_platform_pause_and_budget_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("human_emulator.state.Path", side_effect=lambda value: Path(temp_dir) / value):
                state = EmulatorState("client", ["linkedin", "x"])
                state.mark_visited("https://www.linkedin.com/in/alice", platform="linkedin")
                state.set_availability("x", "active", "")
                state.set_pause_hours("x", hours=8, reason="checkpoint")

                paused_x, _ = state.is_paused("x")
                paused_linkedin, _ = state.is_paused("linkedin")

            self.assertTrue(paused_x)
            self.assertFalse(paused_linkedin)
            self.assertEqual(state.platform_state("linkedin")["visits_today"], 1)
            self.assertEqual(state.platform_state("x")["visits_today"], 0)


if __name__ == "__main__":
    unittest.main()


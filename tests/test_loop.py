"""Tests for agent loop stop and fallback behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.loop import (
    _auto_fail_remaining_non_actionable_pages,
    _build_follow_through_reminder,
    _try_automatic_profile_processing,
    run_agent_loop,
)
from tools.registry import ToolContext


class _DummyWriter:
    def __init__(self) -> None:
        self.saved_count = 0
        self.duplicate_count = 0
        self.saved_rows: list[dict] = []

    async def append_row(self, url: str, data: dict, scrape_status: str = "ok") -> str:  # noqa: ARG002
        self.saved_count += 1
        self.saved_rows.append({"source_url": url, **data})
        return "saved"


class LoopFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_automatic_profile_processing_saves_outstanding_profile(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "https://github.com",
                "min_leads": 1,
                "fields": {
                    "name": "Full name",
                    "job_title": "Title",
                    "social_media": "Profile URL",
                },
            },
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.page_cache["fetch-1"] = """
        <html>
          <head><title>Big-Silver (Senior Software Engineer) · GitHub</title></head>
          <body>
            <span itemprop="name">Senior Software Engineer</span>
            <div class="p-note">As a full stack developer, I have over than 11 years of web development background.</div>
          </body>
        </html>
        """
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://github.com/Big-Silver",
            "final_url": "https://github.com/Big-Silver",
            "title": "Big-Silver (Senior Software Engineer) · GitHub",
            "page_kind": "profile",
            "preview": "Big-Silver (Senior Software Engineer) · GitHub",
        }

        messages: list[dict] = []
        processed = await _try_automatic_profile_processing(ctx, messages, step=5)

        self.assertTrue(processed)
        self.assertEqual(writer.saved_count, 1)
        self.assertEqual(writer.saved_rows[0]["name"], "Big-Silver")

    async def test_automatic_profile_processing_stops_after_reaching_target(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "https://github.com",
                "min_leads": 1,
                "fields": {
                    "name": "Full name",
                    "job_title": "Title",
                    "social_media": "Profile URL",
                },
            },
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.page_cache["fetch-1"] = """
        <html>
          <head><title>alice-smith (CTO) · GitHub</title></head>
          <body>
            <span itemprop="name">Alice Smith</span>
            <div class="p-note">CTO</div>
          </body>
        </html>
        """
        ctx.page_cache["fetch-2"] = """
        <html>
          <head><title>bob-jones (Founder) · GitHub</title></head>
          <body>
            <span itemprop="name">Bob Jones</span>
            <div class="p-note">Founder</div>
          </body>
        </html>
        """
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://github.com/alice-smith",
            "final_url": "https://github.com/alice-smith",
            "title": "alice-smith (CTO) · GitHub",
            "page_kind": "profile",
            "preview": "Alice Smith",
        }
        ctx.fetch_metadata["fetch-2"] = {
            "url": "https://github.com/bob-jones",
            "final_url": "https://github.com/bob-jones",
            "title": "bob-jones (Founder) · GitHub",
            "page_kind": "profile",
            "preview": "Bob Jones",
        }

        processed = await _try_automatic_profile_processing(ctx, [], step=5)

        self.assertTrue(processed)
        self.assertEqual(writer.saved_count, 1)
        self.assertNotIn("fetch-2", ctx.processed_fetch_ids)

    def test_follow_through_reminder_includes_target_progress(self) -> None:
        writer = _DummyWriter()
        writer.saved_count = 2
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 5},
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://example.com/directory",
            "final_url": "https://example.com/directory",
            "title": "Directory",
            "page_kind": "directory",
            "preview": "Directory",
        }

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("saved 2/5 viable leads", reminder)

    def test_follow_through_reminder_requests_suggest_targets_first(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find technical decision makers we can market to",
                "job_title": "Founder",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 3,
            },
            sheets_writer=writer,
            source_mode="web",
        )

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("Call suggest_targets first", reminder)

    def test_follow_through_reminder_requests_fetch_after_suggest_targets(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find technical decision makers we can market to",
                "job_title": "Founder",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 3,
            },
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.suggest_targets_called = True
        ctx.suggested_targets = [
            {"url": "https://github.com/search?q=Founder&type=users", "priority": 100},
            {"url": "https://www.ycombinator.com/founders", "priority": 90},
        ]
        ctx.suggested_target_urls = {
            "https://github.com/search?q=Founder&type=users",
            "https://www.ycombinator.com/founders",
        }

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("Fetch 1 to 2 of the highest-priority suggested targets now", reminder)
        self.assertIn("https://github.com/search?q=Founder&type=users", reminder)

    async def test_auto_fail_remaining_non_actionable_pages_marks_pages_processed(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://example.com/jobs",
            "final_url": "https://example.com/jobs",
            "title": "Jobs",
            "page_kind": "job_board",
            "preview": "Jobs",
        }
        ctx.url_to_fetch_id["https://example.com/jobs"] = "fetch-1"

        processed = await _auto_fail_remaining_non_actionable_pages(ctx, [], step=3)

        self.assertTrue(processed)
        self.assertIn("fetch-1", ctx.processed_fetch_ids)
        self.assertEqual(ctx.failed_urls[0]["url"], "https://example.com/jobs")

    async def test_run_stops_immediately_when_target_reached(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find CTO leads",
                "job_title": "CTO",
                "area": "NA",
                "website": "NA",
                "min_leads": 1,
                "fields": {"name": "Name", "job_title": "Title"},
            },
            sheets_writer=writer,
            source_mode="web",
        )
        responses = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="save_result",
                                arguments={
                                    "url": "https://example.com/alice",
                                    "data": {"name": "Alice Smith", "job_title": "CTO"},
                                },
                            )
                        )
                    ],
                )
            )
        ]

        class _FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, **_: object) -> SimpleNamespace:
                response = responses[self.calls]
                self.calls += 1
                return response

        fake_client = _FakeClient()

        with patch("agent.loop.ollama.AsyncClient", return_value=fake_client):
            result = await run_agent_loop(ctx.client_config, "web", ctx)

        self.assertEqual(fake_client.calls, 1)
        self.assertEqual(writer.saved_count, 1)
        self.assertIn("Reached lead target", result["stop_reason"])


if __name__ == "__main__":
    unittest.main()

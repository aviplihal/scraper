"""Tests for agent loop stop and fallback behavior."""

import unittest
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from agent.loop import (
    _auto_fail_remaining_non_actionable_pages,
    _build_follow_through_reminder,
    _maybe_switch_to_discovery_phase,
    _try_automatic_profile_processing,
    run_agent_loop,
)
from source_state import SourceState
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

    def test_follow_through_reminder_requests_suggest_targets_in_human_emulator_mode(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find technical decision makers we can market to",
                "job_title": "Founder",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 3,
                "social_platforms": ["linkedin", "x"],
            },
            sheets_writer=writer,
            source_mode="human_emulator",
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
            {"url": "https://www.linkedin.com/search/results/people/?keywords=Founder", "domain": "linkedin.com"},
            {"url": "https://github.com/search?q=Founder&type=users", "domain": "github.com"},
            {"url": "https://x.com/search?q=Founder&f=user", "domain": "x.com"},
        ]
        ctx.allowed_domains = {"linkedin.com", "github.com", "x.com"}
        ctx.candidate_domains = ["linkedin.com", "github.com", "x.com"]

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("Choose 1 to 2 starter targets that best match the keyword brief", reminder)
        self.assertIn("https://www.linkedin.com/search/results/people/?keywords=Founder", reminder)

    def test_follow_through_reminder_nudges_domain_switch_when_current_domain_has_no_saves(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find public engineers",
                "job_title": "Senior Software Engineer",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 3,
            },
            sheets_writer=writer,
            source_mode="all",
        )
        ctx.suggest_targets_called = True
        ctx.keyword_brief = {
            "primary_terms": ["Senior Software Engineer"],
            "secondary_terms": ["Software Engineer", "Architect"],
            "area": "San Francisco Bay Area",
            "source_mode": "all",
        }
        ctx.suggested_targets = [
            {"url": "https://www.linkedin.com/search/results/people/?keywords=Senior+Software+Engineer", "domain": "linkedin.com"},
            {"url": "https://x.com/search?q=Senior+Software+Engineer&f=user", "domain": "x.com"},
            {"url": "https://github.com/search?q=Senior+Software+Engineer&type=users", "domain": "github.com"},
        ]
        ctx.allowed_domains = {"linkedin.com", "x.com", "github.com"}
        ctx.candidate_domains = ["linkedin.com", "x.com", "github.com"]
        ctx.domain_fetch_counts["github.com"] = 2
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://github.com/search?q=Senior+Software+Engineer&type=users",
            "final_url": "https://github.com/search?q=Senior+Software+Engineer&type=users",
            "title": "User search results · GitHub",
            "page_kind": "search_results",
            "preview": "results",
        }
        ctx.processed_fetch_ids.add("fetch-1")

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("Prefer a different domain/source now because github.com has not produced a viable lead yet", reminder)

    def test_follow_through_reminder_mentions_discovery_sampling(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find engineers",
                "job_title": "Senior Software Engineer",
                "area": "NA",
                "website": "NA",
                "min_leads": 3,
            },
            sheets_writer=writer,
            source_mode="web",
            source_phase="discovery",
        )
        ctx.suggest_targets_called = True
        ctx.keyword_brief = {"primary_terms": ["Senior Software Engineer"]}
        ctx.suggested_targets = [{"url": "https://gitlab.com/explore/users?search=Engineer", "domain": "gitlab.com"}]
        ctx.allowed_domains = {"gitlab.com"}
        ctx.candidate_domains = ["gitlab.com"]

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("sample up to 3 viable leads", reminder)

    def test_switch_to_discovery_phase_when_pass1_pool_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                writer = _DummyWriter()
                state = SourceState(
                    "test",
                    {
                        "client_id": "test",
                        "website": "NA",
                        "min_leads": 3,
                        "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                    },
                )
                ctx = ToolContext(
                    client_config={
                        "client_id": "test",
                        "website": "NA",
                        "min_leads": 3,
                        "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                    },
                    sheets_writer=writer,
                    source_mode="web",
                    source_state=state,
                    source_phase="pass1",
                )
                ctx.suggest_targets_called = True
                ctx.allowed_domains = {"github.com"}
                ctx.candidate_domains = ["github.com"]
                ctx.domain_outcomes["github.com"] = SimpleNamespace(
                    blocked_count=0,
                    irrelevant_count=2,
                    discovery_hits=0,
                    profile_hits=0,
                    saved_hits=0,
                    banned_for_run=True,
                    last_reason="job_board",
                )

                messages: list[dict] = []
                switched = _maybe_switch_to_discovery_phase(ctx, messages)

                self.assertTrue(switched)
                self.assertEqual(ctx.source_phase, "discovery")
                self.assertFalse(ctx.suggest_targets_called)
                self.assertIn("Pass 1 is exhausted", messages[0]["content"])
            finally:
                os.chdir(old_cwd)

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

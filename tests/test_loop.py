"""Tests for agent loop stop and fallback behavior."""

import unittest
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from agent.loop import (
    _auto_fail_remaining_non_actionable_pages,
    _build_follow_through_reminder,
    _candidate_preview_urls,
    _conversation_state_summary,
    _maybe_switch_to_discovery_phase,
    _maybe_compact_messages,
    _no_viable_next_actions,
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
        self.assertEqual(writer.saved_rows[0]["source_url"], "https://github.com/alice-smith")

    async def test_automatic_profile_processing_scales_batch_to_remaining_gap(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "https://github.com",
                "min_leads": 3,
                "fields": {
                    "name": "Full name",
                    "job_title": "Title",
                    "social_media": "Profile URL",
                },
            },
            sheets_writer=writer,
            source_mode="web",
        )
        for fetch_id, slug, title in [
            ("fetch-1", "alice-smith", "CTO"),
            ("fetch-2", "bob-jones", "Founder"),
            ("fetch-3", "carla-chen", "Senior Software Engineer"),
        ]:
            ctx.page_cache[fetch_id] = f"""
            <html>
              <head><title>{slug} ({title}) · GitHub</title></head>
              <body>
                <span itemprop="name">{slug.replace('-', ' ').title()}</span>
                <div class="p-note">{title}</div>
              </body>
            </html>
            """
            ctx.fetch_metadata[fetch_id] = {
                "url": f"https://github.com/{slug}",
                "final_url": f"https://github.com/{slug}",
                "title": f"{slug} ({title}) · GitHub",
                "page_kind": "profile",
                "preview": slug,
            }

        processed = await _try_automatic_profile_processing(ctx, [], step=5)

        self.assertTrue(processed)
        self.assertEqual(writer.saved_count, 3)

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
        ctx.exhausted_discovery_fetches.add("fetch-1")

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("Prefer a different domain/source now because github.com has not produced a viable lead yet", reminder)

    def test_candidate_preview_urls_are_domain_diverse_first(self) -> None:
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
        ctx.suggested_targets = [
            {"url": "https://github.com/search?q=Senior+Software+Engineer&type=users", "domain": "github.com"},
            {"url": "https://github.com/search?q=Software+Engineer&type=users", "domain": "github.com"},
            {"url": "https://www.linkedin.com/search/results/people/?keywords=Senior+Software+Engineer", "domain": "linkedin.com"},
            {"url": "https://x.com/search?q=Senior+Software+Engineer&f=user", "domain": "x.com"},
        ]
        ctx.allowed_domains = {"github.com", "linkedin.com", "x.com"}
        ctx.candidate_domains = ["github.com", "linkedin.com", "x.com"]

        preview = _candidate_preview_urls(ctx, limit=3)

        self.assertEqual(len(preview), 3)
        self.assertTrue(any("github.com" in url for url in preview))
        self.assertTrue(any("linkedin.com" in url for url in preview))
        self.assertTrue(any("x.com" in url for url in preview))

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

    def test_follow_through_reminder_prefers_discovered_profile_urls(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find engineers",
                "job_title": "Senior Software Engineer",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 10,
            },
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.suggest_targets_called = True
        ctx.fetch_metadata["fetch-search"] = {
            "url": "https://github.com/search?q=engineer&type=users",
            "final_url": "https://github.com/search?q=engineer&type=users",
            "page_kind": "search_results",
            "title": "Search",
            "preview": "results",
        }
        ctx.processed_fetch_ids.add("fetch-search")
        ctx.discovered_link_parents["https://github.com/alice-smith"] = "fetch-search"
        ctx.discovered_link_parents["https://github.com/bob-jones"] = "fetch-search"

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("Fetch the discovered profile/detail URLs", reminder)
        self.assertIn("https://github.com/alice-smith", reminder)

    def test_no_viable_next_actions_when_all_candidate_urls_are_budget_exhausted(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find engineers",
                "job_title": "Senior Software Engineer",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 10,
            },
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.suggest_targets_called = True
        ctx.allowed_domains = {"github.com"}
        ctx.candidate_domains = ["github.com"]
        ctx.suggested_targets = [
            {"url": "https://github.com/search?q=Senior+Software+Engineer&type=users", "domain": "github.com"},
            {"url": "https://github.com/search?q=Software+Engineer&type=users", "domain": "github.com"},
        ]
        ctx.fetch_budget_counts["github.com:search"] = 20

        self.assertTrue(_no_viable_next_actions(ctx))

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

    def test_pass1_does_not_switch_to_discovery_when_unfetched_target_urls_remain(self) -> None:
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
                ctx.suggested_targets = [
                    {"url": "https://github.com/search?q=Senior+Software+Engineer&type=users", "domain": "github.com"},
                    {"url": "https://github.com/search?q=Software+Engineer&type=users", "domain": "github.com"},
                ]
                ctx.fetch_metadata["fetch-1"] = {
                    "url": "https://github.com/search?q=Senior+Software+Engineer&type=users",
                    "final_url": "https://github.com/search?q=Senior+Software+Engineer&type=users",
                    "title": "User search results · GitHub",
                    "page_kind": "search_results",
                    "preview": "results",
                }
                ctx.url_to_fetch_id["https://github.com/search?q=Senior+Software+Engineer&type=users"] = "fetch-1"
                ctx.exhausted_discovery_fetches.add("fetch-1")
                ctx.exhausted_discovery_urls.add("https://github.com/search?q=Senior+Software+Engineer&type=users")
                ctx.processed_fetch_ids.add("fetch-1")

                messages: list[dict] = []
                switched = _maybe_switch_to_discovery_phase(ctx, messages)

                self.assertFalse(switched)
                self.assertEqual(ctx.source_phase, "pass1")
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

    async def test_run_tracks_ollama_prompt_and_completion_tokens(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find engineers",
                "job_title": "Engineer",
                "area": "NA",
                "website": "https://github.com",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="web",
        )

        responses = [
            SimpleNamespace(
                prompt_eval_count=123,
                eval_count=45,
                message=SimpleNamespace(content="", tool_calls=[]),
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

        self.assertEqual(result["prompt_tokens"], 123)
        self.assertEqual(result["completion_tokens"], 45)
        self.assertEqual(result["total_tokens"], 168)

    async def test_run_retries_once_after_recoverable_model_xml_error(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find engineers",
                "job_title": "Engineer",
                "area": "NA",
                "website": "https://github.com",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="web",
        )

        class _FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, **_: object) -> SimpleNamespace:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError(
                        "XML syntax error on line 9: element <function> closed by </parameter> (status code: 500)"
                    )
                return SimpleNamespace(
                    prompt_eval_count=10,
                    eval_count=1,
                    message=SimpleNamespace(content="", tool_calls=[]),
                )

        fake_client = _FakeClient()

        with patch("agent.loop.ollama.AsyncClient", return_value=fake_client):
            result = await run_agent_loop(ctx.client_config, "web", ctx)

        self.assertEqual(fake_client.calls, 2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result.get("model_error_retries"), 1)
        self.assertNotIn("Ollama call failed", result["stop_reason"])

    def test_compactor_reduces_history_and_records_summary(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find engineers",
                "job_title": "Senior Software Engineer",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 3,
            },
            sheets_writer=writer,
            source_mode="all",
            source_phase="pass1",
        )
        ctx.target_strategy = "technical_profiles"
        ctx.keyword_brief = {
            "primary_terms": ["Senior Software Engineer"],
            "secondary_terms": ["Software Engineer"],
            "area": "San Francisco Bay Area",
        }
        ctx.candidate_domains = ["github.com", "linkedin.com", "x.com"]
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://github.com/alice",
            "final_url": "https://github.com/alice",
            "page_kind": "profile",
            "title": "Alice",
        }
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "job"},
        ] + [{"role": "tool", "content": f"msg-{idx}"} for idx in range(14)]
        run_result = {"last_prompt_tokens": 3000, "compactions": 0}

        _maybe_compact_messages(messages, ctx, run_result)

        self.assertLess(len(messages), 17)
        self.assertEqual(run_result["compactions"], 1)
        self.assertIn("Context summary for the ongoing lead scrape", messages[2]["content"])
        self.assertIn("Candidate domains: github.com, linkedin.com, x.com", _conversation_state_summary(ctx))

    def test_follow_through_reminder_prefers_current_discovery_pages(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find engineers",
                "job_title": "Senior Software Engineer",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 3,
            },
            sheets_writer=writer,
            source_mode="all",
        )
        ctx.suggest_targets_called = True
        ctx.fetch_metadata["fetch-search"] = {
            "url": "https://github.com/search?q=engineer&type=users",
            "final_url": "https://github.com/search?q=engineer&type=users",
            "page_kind": "search_results",
            "title": "Search",
        }

        reminder = _build_follow_through_reminder(ctx)

        self.assertIn("You have fetched pages that are still unprocessed", reminder)
        self.assertIn("call list_links on this page", reminder)


if __name__ == "__main__":
    unittest.main()

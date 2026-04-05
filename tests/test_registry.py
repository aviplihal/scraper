"""Tests for registry guardrails and tool dispatch behavior."""

import os
import tempfile
import unittest
from unittest.mock import patch

from human_emulator.social import SocialFetchResult
from source_state import SourceState
from tools.registry import (
    DomainOutcome,
    ToolContext,
    _normalize_lead_payload,
    _normalize_url,
    _record_domain_failure,
    _record_fetch_outcome,
    dispatch_tool,
)


class _DummyWriter:
    def __init__(self) -> None:
        self.saved_count = 0
        self.duplicate_count = 0
        self.saved_rows: list[dict] = []
        self._saved_urls: set[str] = set()

    async def append_row(self, url: str, data: dict, scrape_status: str = "ok") -> str:  # noqa: ARG002
        normalized_url = _normalize_url(url)
        if normalized_url in self._saved_urls:
            self.duplicate_count += 1
            return "duplicate"
        self._saved_urls.add(normalized_url)
        self.saved_count += 1
        self.saved_rows.append({"url": url, "data": data})
        return "saved"

    def has_source_url(self, url: str) -> bool:
        return _normalize_url(url) in self._saved_urls

    def recent_rows(self, limit: int = 20) -> list[dict]:
        return [
            {
                "name": row["data"].get("name"),
                "job_title": row["data"].get("job_title"),
                "company": row["data"].get("company"),
                "email": row["data"].get("email"),
                "phone": row["data"].get("phone"),
                "social_media": row["data"].get("social_media"),
                "source_url": row["url"],
            }
            for row in self.saved_rows[:limit]
        ]


class _FakeEmulatorBrowser:
    async def get_context(self, platform: str):  # noqa: ARG002
        return object()


class _FakeScraperBrowser:
    def new_context(self):
        return object()


class _FakeEmulatorState:
    def __init__(self) -> None:
        self.platform_availability = {
            "linkedin": {"status": "active", "reason": ""},
            "x": {"status": "active", "reason": ""},
        }
        self.added_profiles: list[tuple[str, list[str]]] = []
        self.visited: list[tuple[str, str]] = []

    def availability(self, platform: str) -> dict[str, str]:
        return self.platform_availability.get(platform, {"status": "unknown", "reason": ""})

    def add_profiles(self, urls: list[str], platform: str = "linkedin") -> int:
        self.added_profiles.append((platform, urls))
        return len(urls)

    def mark_visited(self, url: str, platform: str = "linkedin") -> None:
        self.visited.append((platform, url))

    def record_restriction(self, platform: str = "linkedin") -> int:  # noqa: ARG002
        return 1

    def set_pause_hours(self, platform: str, hours: int, reason: str = "") -> None:  # noqa: ARG002
        self.platform_availability[platform] = {"status": "paused", "reason": reason}

    def set_availability(self, platform: str, status: str, reason: str = "") -> None:
        self.platform_availability[platform] = {"status": status, "reason": reason}


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_suggest_targets_same_phase_returns_unchanged_brief(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find public engineers",
                "job_title": "Senior Software Engineer",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="all",
        )

        first = await dispatch_tool("suggest_targets", {"limit": 4}, ctx)
        second = await dispatch_tool("suggest_targets", {"limit": 4}, ctx)

        self.assertEqual(first["strategy"], "technical_profiles")
        self.assertEqual(second["status"], "unchanged")
        self.assertEqual(second["phase"], "pass1")
        self.assertLessEqual(len(second["candidate_targets"]), 1)

    async def test_suggest_targets_filters_unavailable_social_domains(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find public engineers",
                "job_title": "Senior Software Engineer",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 1,
                "social_platforms": ["linkedin", "x"],
            },
            sheets_writer=writer,
            source_mode="all",
            effective_source_mode="web",
        )
        ctx.unavailable_domains.update({"linkedin.com", "x.com"})

        result = await dispatch_tool("suggest_targets", {"limit": 6}, ctx)

        self.assertNotIn("linkedin.com", result["allowed_domains"])
        self.assertNotIn("x.com", result["allowed_domains"])
        self.assertTrue(all(target["domain"] not in {"linkedin.com", "x.com"} for target in result["candidate_targets"]))

    async def test_suggest_targets_returns_curated_leadership_targets(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find technical decision makers we can market to",
                "job_title": "Founder",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="web",
        )

        result = await dispatch_tool("suggest_targets", {"limit": 4}, ctx)

        self.assertEqual(result["strategy"], "leadership_people")
        urls = [target["url"] for target in result["candidate_targets"]]
        self.assertIn("https://www.ycombinator.com/founders", urls)
        self.assertNotIn("https://www.crunchbase.com/people", urls)
        self.assertTrue(ctx.suggest_targets_called)
        self.assertEqual(ctx.keyword_brief["primary_terms"], ["Founder"])
        self.assertIn("ycombinator.com", ctx.allowed_domains)

    async def test_broad_mode_rejects_fetch_before_suggest_targets(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find technical decision makers we can market to",
                "job_title": "Founder",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="web",
        )

        result = await dispatch_tool(
            "fetch_page",
            {"url": "https://example.com/people", "needs_javascript": True},
            ctx,
        )

        self.assertIn("error", result)
        self.assertIn("suggest_targets", result["error"])

    async def test_broad_mode_rejects_domain_outside_curated_pool(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find technical decision makers we can market to",
                "job_title": "Founder",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="web",
        )

        await dispatch_tool("suggest_targets", {"limit": 4}, ctx)
        result = await dispatch_tool(
            "fetch_page",
            {"url": "https://example.com/people", "needs_javascript": True},
            ctx,
        )

        self.assertIn("error", result)
        self.assertIn("outside the candidate domain pool", result["error"])

    async def test_broad_mode_allows_model_selected_url_within_allowed_domain_pool(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "job": "find public engineers",
                "job_title": "Senior Software Engineer",
                "area": "San Francisco Bay Area",
                "website": "NA",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="web",
            scraper_browser=_FakeScraperBrowser(),
        )
        ctx.suggest_targets_called = True
        ctx.allowed_domains = {"github.com"}
        ctx.candidate_domains = ["github.com"]

        class _FakeFetchResult:
            final_url = "https://github.com/alice-smith"
            html = "<html><head><title>Alice Smith</title></head><body><h1>Alice Smith</h1></body></html>"

        with patch("tools.registry.smart_fetch", return_value=_FakeFetchResult()):
            result = await dispatch_tool(
                "fetch_page",
                {"url": "https://github.com/alice-smith", "needs_javascript": False},
                ctx,
            )

        self.assertEqual(result["page_kind"], "profile")

    async def test_web_mode_rejects_social_media_urls(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )

        result = await dispatch_tool(
            "fetch_page",
            {"url": "https://www.linkedin.com/in/test-person", "needs_javascript": True},
            ctx,
        )

        self.assertIn("error", result)
        self.assertIn("not allowed in web mode", result["error"])

    async def test_fetch_page_defaults_needs_javascript_when_model_omits_it(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "https://github.com",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="web",
            scraper_browser=_FakeScraperBrowser(),
        )

        class _FakeFetchResult:
            final_url = "https://github.com/alice-smith"
            html = "<html><body><h1>Alice Smith</h1></body></html>"

        with patch("tools.registry.smart_fetch", return_value=_FakeFetchResult()):
            result = await dispatch_tool(
                "fetch_page",
                {"url": "https://github.com/alice-smith"},
                ctx,
            )

        self.assertEqual(result["page_kind"], "profile")

    async def test_fetch_url_alias_routes_to_fetch_page(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "https://github.com",
                "min_leads": 1,
            },
            sheets_writer=writer,
            source_mode="web",
            scraper_browser=_FakeScraperBrowser(),
        )

        class _FakeFetchResult:
            final_url = "https://github.com/alice-smith"
            html = "<html><body><h1>Alice Smith</h1></body></html>"

        with patch("tools.registry.smart_fetch", return_value=_FakeFetchResult()):
            result = await dispatch_tool(
                "fetch_url",
                {"url": "https://github.com/alice-smith"},
                ctx,
            )

        self.assertEqual(result["page_kind"], "profile")

    async def test_list_links_can_resolve_fetch_id_from_url(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "https://github.com", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.page_cache["fetch-lookup"] = "<html><body><a href='https://github.com/alice'>Alice</a></body></html>"
        ctx.fetch_metadata["fetch-lookup"] = {
            "url": "https://github.com/search?q=test&type=users",
            "final_url": "https://github.com/search?q=test&type=users",
            "title": "User search results · GitHub",
            "page_kind": "search_results",
            "preview": "results",
        }
        ctx.url_to_fetch_id["https://github.com/search?q=test&type=users"] = "fetch-lookup"

        result = await dispatch_tool(
            "list_links",
            {"url": "https://github.com/search?q=test&type=users", "limit": 5},
            ctx,
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["links"][0]["url"], "https://github.com/alice")

    async def test_list_links_returns_unseen_candidates_then_exhausts(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "https://github.com", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.page_cache["fetch-1"] = """
        <html><body>
          <a href="https://github.com/alice">Alice</a>
          <a href="https://github.com/bob">Bob</a>
          <a href="https://github.com/carla">Carla</a>
        </body></html>
        """
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://github.com/search?q=engineer&type=users",
            "final_url": "https://github.com/search?q=engineer&type=users",
            "title": "User search results · GitHub",
            "page_kind": "search_results",
            "preview": "results",
        }

        first = await dispatch_tool("list_links", {"fetch_id": "fetch-1", "limit": 2}, ctx)
        second = await dispatch_tool("list_links", {"fetch_id": "fetch-1", "limit": 2}, ctx)

        self.assertEqual(len(first["links"]), 2)
        self.assertEqual(len(second["links"]), 1)
        self.assertTrue(second["exhausted"])

    async def test_exhausted_search_url_cannot_be_refetched(self) -> None:
        writer = _DummyWriter()
        search_url = "https://github.com/search?q=engineer&type=users"
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
            scraper_browser=_FakeScraperBrowser(),
        )
        ctx.suggest_targets_called = True
        ctx.allowed_domains = {"github.com"}
        ctx.candidate_domains = ["github.com"]
        ctx.exhausted_discovery_urls.add(_normalize_url(search_url))

        result = await dispatch_tool(
            "fetch_page",
            {"url": search_url, "needs_javascript": False},
            ctx,
        )

        self.assertIn("error", result)
        self.assertIn("exhausted", result["error"])

    async def test_fetch_page_skips_known_duplicate_source_url(self) -> None:
        writer = _DummyWriter()
        await writer.append_row(
            "https://github.com/alice-smith",
            {"name": "Alice Smith", "job_title": "Engineer"},
        )
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
            scraper_browser=_FakeScraperBrowser(),
        )
        ctx.suggest_targets_called = True
        ctx.allowed_domains = {"github.com"}
        ctx.candidate_domains = ["github.com"]

        result = await dispatch_tool(
            "fetch_page",
            {"url": "https://github.com/alice-smith", "needs_javascript": False},
            ctx,
        )

        self.assertIn("error", result)
        self.assertIn("already exists in storage", result["error"])

    async def test_list_links_skips_urls_already_in_storage(self) -> None:
        writer = _DummyWriter()
        await writer.append_row(
            "https://github.com/alice-smith",
            {"name": "Alice Smith", "job_title": "Engineer"},
        )
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "https://github.com", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.page_cache["fetch-1"] = """
        <html><body>
          <a href="https://github.com/alice-smith">Alice</a>
          <a href="https://github.com/bob-jones">Bob</a>
        </body></html>
        """
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://github.com/search?q=test&type=users",
            "final_url": "https://github.com/search?q=test&type=users",
            "title": "User search results · GitHub",
            "page_kind": "search_results",
            "preview": "results",
        }

        result = await dispatch_tool("list_links", {"fetch_id": "fetch-1", "limit": 5}, ctx)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["links"][0]["url"], "https://github.com/bob-jones")

    def test_normalize_lead_payload_splits_title_and_company(self) -> None:
        normalized = _normalize_lead_payload(
            "https://github.com/dabrad26",
            {
                "name": "David Bradshaw",
                "job_title": "Senior Software Engineer at@adobe",
                "company": "@adobe",
                "social_media": "https://davidbradshaw.us/",
            },
        )

        self.assertEqual(normalized["job_title"], "Senior Software Engineer")
        self.assertEqual(normalized["company"], "Adobe")

    def test_normalize_lead_payload_humanizes_handle_style_company(self) -> None:
        normalized = _normalize_lead_payload(
            "https://github.com/elliedori",
            {
                "name": "Ellie Bahadori",
                "job_title": "Senior software engineer@VantaInc",
                "company": None,
                "social_media": "https://bsky.app/profile/elliedori.bsky.social",
            },
        )

        self.assertEqual(normalized["job_title"], "Senior software engineer")
        self.assertEqual(normalized["company"], "Vanta Inc")

    def test_social_profile_urls_canonicalize_mini_profile_variants(self) -> None:
        self.assertEqual(
            _normalize_url("https://www.linkedin.com/in/test-person?miniProfileUrn=abc"),
            _normalize_url("https://www.linkedin.com/in/test-person/"),
        )

    async def test_social_url_routes_through_matching_adapter(self) -> None:
        writer = _DummyWriter()
        state = _FakeEmulatorState()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "NA",
                "min_leads": 1,
                "social_platforms": ["linkedin"],
            },
            sheets_writer=writer,
            source_mode="all",
            emulator_browser=_FakeEmulatorBrowser(),
            emulator_state=state,
        )
        ctx.suggest_targets_called = True
        ctx.allowed_domains = {"linkedin.com"}
        ctx.suggested_target_urls = {"https://www.linkedin.com/in/test-person"}

        class _FakeLinkedInAdapter:
            platform = "linkedin"

            def __init__(self, context, state_arg, client_id: str):  # noqa: ARG002
                self.context = context
                self.state = state_arg

            async def fetch(self, url: str) -> SocialFetchResult:
                return SocialFetchResult(
                    final_url=url,
                    title="Alice Smith",
                    page_kind="profile",
                    html=(
                        "<html><body><h1>Alice Smith</h1>"
                        "<div class='headline'>CTO</div>"
                        "<a class='social' href='https://www.linkedin.com/in/test-person'>profile</a>"
                        "</body></html>"
                    ),
                    extracted_data={
                        "name": "Alice Smith",
                        "job_title": "CTO",
                        "social_media": url,
                    },
                )

        with patch("tools.registry.adapter_for_url", return_value=_FakeLinkedInAdapter):
            result = await dispatch_tool(
                "fetch_page",
                {"url": "https://www.linkedin.com/in/test-person", "needs_javascript": True},
                ctx,
            )

        self.assertEqual(result["page_kind"], "profile")
        self.assertEqual(ctx.fetch_metadata[result["fetch_id"]]["platform"], "linkedin")
        self.assertEqual(state.visited[0], ("linkedin", "https://www.linkedin.com/in/test-person"))

    async def test_social_search_fetch_adds_profiles_to_platform_queue(self) -> None:
        writer = _DummyWriter()
        state = _FakeEmulatorState()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "NA",
                "min_leads": 1,
                "social_platforms": ["x"],
            },
            sheets_writer=writer,
            source_mode="human_emulator",
            emulator_browser=_FakeEmulatorBrowser(),
            emulator_state=state,
        )
        ctx.suggest_targets_called = True
        ctx.allowed_domains = {"x.com"}
        ctx.suggested_target_urls = {"https://x.com/search?q=Founder&f=user"}

        class _FakeXAdapter:
            platform = "x"

            def __init__(self, context, state_arg, client_id: str):  # noqa: ARG002
                self.context = context
                self.state = state_arg

            async def fetch(self, url: str) -> SocialFetchResult:
                return SocialFetchResult(
                    final_url=url,
                    title="X User Search",
                    page_kind="search_results",
                    html=(
                        "<html><body>"
                        "<a href='https://x.com/alice' data-hovercard-type='user'>Alice Smith</a>"
                        "</body></html>"
                    ),
                    extracted_data={
                        "results": [
                            {"url": "https://x.com/alice", "name": "Alice Smith", "headline": "Founder"}
                        ]
                    },
                )

        with patch("tools.registry.adapter_for_url", return_value=_FakeXAdapter):
            result = await dispatch_tool(
                "fetch_page",
                {"url": "https://x.com/search?q=Founder&f=user", "needs_javascript": True},
                ctx,
            )

        self.assertEqual(result["page_kind"], "search_results")
        self.assertEqual(state.added_profiles[0], ("x", ["https://x.com/alice"]))

    async def test_parse_html_rejects_search_pages(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.page_cache["fetch-1"] = "<html><body><a href='/alice'>Alice</a></body></html>"
        ctx.fetch_metadata["fetch-1"] = {
            "url": "https://github.com/search?q=test&type=users",
            "final_url": "https://github.com/search?q=test&type=users",
            "title": "User search results · GitHub",
            "page_kind": "search_results",
            "preview": "User search results",
        }

        result = await dispatch_tool(
            "parse_html",
            {"fetch_id": "fetch-1", "fields": {"name": "h1"}},
            ctx,
        )

        self.assertIn("error", result)
        self.assertIn("detail/profile pages", result["error"])

    async def test_parse_html_uses_builtin_field_names(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "https://github.com",
                "min_leads": 1,
                "fields": {
                    "name": "Full name",
                    "job_title": "Title",
                    "company": "Company",
                },
            },
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.page_cache["fetch-2"] = """
        <html>
          <head><title>alice-smith (Senior Software Engineer) · GitHub</title></head>
          <body>
            <span itemprop="name">Alice Smith</span>
            <div class="p-note">Senior Software Engineer</div>
            <span class="p-org">Example Co</span>
          </body>
        </html>
        """
        ctx.fetch_metadata["fetch-2"] = {
            "url": "https://github.com/alice-smith",
            "final_url": "https://github.com/alice-smith",
            "title": "alice-smith (Senior Software Engineer) · GitHub",
            "page_kind": "profile",
            "preview": "Alice Smith",
        }

        result = await dispatch_tool(
            "parse_html",
            {"fetch_id": "fetch-2", "field_names": ["name", "job_title", "company"]},
            ctx,
        )

        self.assertEqual(
            result["fields"],
            {
                "name": "Alice Smith",
                "job_title": "Senior Software Engineer",
                "company": "Example Co",
            },
        )

    async def test_parse_html_github_falls_back_to_username_for_name(self) -> None:
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
        ctx.page_cache["fetch-3"] = """
        <html>
          <head><title>Big-Silver (Senior Software Engineer) · GitHub</title></head>
          <body>
            <span itemprop="name">Senior Software Engineer</span>
            <div class="p-note">As a full stack developer, I have over than 11 years of web development background.</div>
          </body>
        </html>
        """
        ctx.fetch_metadata["fetch-3"] = {
            "url": "https://github.com/Big-Silver",
            "final_url": "https://github.com/Big-Silver",
            "title": "Big-Silver (Senior Software Engineer) · GitHub",
            "page_kind": "profile",
            "preview": "Big-Silver (Senior Software Engineer) · GitHub",
        }

        result = await dispatch_tool(
            "parse_html",
            {"fetch_id": "fetch-3", "field_names": ["name", "job_title", "social_media"]},
            ctx,
        )

        self.assertEqual(
            result["fields"],
            {
                "name": "Big-Silver",
                "job_title": "As a full stack developer, I have over than 11 years of web development background.",
                "social_media": "https://github.com/Big-Silver",
            },
        )

    async def test_parse_html_social_uses_search_hints_when_profile_fields_are_blank(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "NA",
                "min_leads": 1,
                "fields": {
                    "name": "Full name",
                    "job_title": "Title",
                    "social_media": "Profile URL",
                },
            },
            sheets_writer=writer,
            source_mode="all",
        )
        profile_url = "https://www.linkedin.com/in/abhiraj-gupta-86711516b/"
        ctx.page_cache["fetch-social-1"] = (
            "<html><body><h1>Unknown</h1>"
            "<a class='social' href='https://www.linkedin.com/in/abhiraj-gupta-86711516b/'>profile</a>"
            "</body></html>"
        )
        ctx.fetch_metadata["fetch-social-1"] = {
            "url": profile_url,
            "final_url": profile_url,
            "title": "LinkedIn profile",
            "page_kind": "profile",
            "preview": "profile",
            "platform": "linkedin",
            "extracted_data": {"name": "Unknown", "job_title": None, "social_media": profile_url},
        }
        ctx.social_profile_hints[profile_url] = {
            "name": "Abhiraj Gupta",
            "job_title": "Senior Software Engineer",
            "social_media": profile_url,
        }

        result = await dispatch_tool(
            "parse_html",
            {"fetch_id": "fetch-social-1", "field_names": ["name", "job_title", "social_media"]},
            ctx,
        )

        self.assertEqual(
            result["fields"],
            {
                "name": "Abhiraj Gupta",
                "job_title": "Senior Software Engineer",
                "social_media": profile_url,
            },
        )

    async def test_parse_html_blank_social_profile_becomes_terminal(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "NA",
                "min_leads": 1,
                "fields": {
                    "name": "Full name",
                    "job_title": "Title",
                    "company": "Company",
                    "email": "Email",
                    "phone": "Phone",
                    "social_media": "Profile URL",
                },
            },
            sheets_writer=writer,
            source_mode="all",
        )
        profile_url = "https://www.linkedin.com/in/test-person"
        ctx.page_cache["fetch-social-blank"] = "<html><body></body></html>"
        ctx.fetch_metadata["fetch-social-blank"] = {
            "url": profile_url,
            "final_url": profile_url,
            "title": "LinkedIn profile",
            "page_kind": "profile",
            "preview": "",
            "platform": "linkedin",
            "extracted_data": {},
        }

        result = await dispatch_tool(
            "parse_html",
            {
                "fetch_id": "fetch-social-blank",
                "field_names": ["name", "job_title", "company", "email", "phone", "social_media"],
            },
            ctx,
        )

        self.assertIsNone(result["fields"]["name"])
        self.assertEqual(ctx.fetch_metadata["fetch-social-blank"]["page_kind"], "not_found")
        self.assertEqual(ctx.terminal_url_outcomes[_normalize_url(profile_url)], "blank_profile")

    async def test_save_result_accepts_flat_arguments_with_fetch_id(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "https://github.com",
                "min_leads": 1,
                "fields": {
                    "name": "Full name",
                    "job_title": "Title",
                    "company": "Company",
                },
            },
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.fetch_metadata["fetch-4"] = {
            "url": "https://github.com/Big-Silver",
            "final_url": "https://github.com/Big-Silver",
            "title": "Big-Silver (Senior Software Engineer) · GitHub",
            "page_kind": "profile",
            "preview": "Big-Silver",
        }
        ctx.parsed_results["fetch-4"] = {
            "name": "Big-Silver",
            "job_title": "Full Stack Developer",
            "company": None,
        }

        result = await dispatch_tool(
            "save_result",
            {
                "fetch_id": "fetch-4",
                "name": "Big-Silver",
                "job_title": "Full Stack Developer",
                "company": "None",
                "email": "None",
                "phone": "None",
            },
            ctx,
        )

        self.assertEqual(result["status"], "saved")
        self.assertEqual(writer.saved_rows[0]["url"], "https://github.com/Big-Silver")
        self.assertEqual(writer.saved_rows[0]["data"]["name"], "Big-Silver")

    async def test_save_result_rejects_name_only_row(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )

        result = await dispatch_tool(
            "save_result",
            {"url": "https://example.com/alice", "data": {"name": "Alice Smith"}},
            ctx,
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(writer.saved_count, 0)
        self.assertEqual(ctx.rejected_weak_count, 1)

    async def test_save_result_accepts_name_plus_job_title(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )

        result = await dispatch_tool(
            "save_result",
            {
                "url": "https://example.com/alice",
                "data": {"name": "Alice Smith", "job_title": "CTO"},
            },
            ctx,
        )

        self.assertEqual(result["status"], "saved")
        self.assertEqual(writer.saved_count, 1)

    async def test_save_result_normalizes_noisy_job_title_and_company(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )

        result = await dispatch_tool(
            "save_result",
            {
                "url": "https://github.com/example-user",
                "data": {
                    "name": "Example User",
                    "job_title": "Senior Software Engineer\n{ Work and reside in San Francisco }",
                    "company": "Data Platform@twilio",
                    "social_media": "www.example.com",
                },
            },
            ctx,
        )

        self.assertEqual(result["status"], "saved")
        self.assertEqual(writer.saved_rows[0]["data"]["job_title"], "Senior Software Engineer")
        self.assertEqual(writer.saved_rows[0]["data"]["company"], "Twilio")
        self.assertEqual(writer.saved_rows[0]["data"]["social_media"], "https://www.example.com")

    async def test_duplicate_saved_row_does_not_increment_saved_count(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 2},
            sheets_writer=writer,
            source_mode="web",
        )

        first = await dispatch_tool(
            "save_result",
            {
                "url": "https://example.com/alice",
                "data": {"name": "Alice Smith", "job_title": "CTO"},
            },
            ctx,
        )
        second = await dispatch_tool(
            "save_result",
            {
                "url": "https://example.com/alice",
                "data": {"name": "Alice Smith", "job_title": "CTO"},
            },
            ctx,
        )

        self.assertEqual(first["status"], "saved")
        self.assertEqual(second["status"], "duplicate")
        self.assertEqual(writer.saved_count, 1)
        self.assertEqual(writer.duplicate_count, 1)

    async def test_fail_url_accepts_fetch_id_without_url(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.fetch_metadata["fetch-5"] = {
            "url": "https://www.crunchbase.com/people",
            "final_url": "https://www.crunchbase.com/people",
            "title": "Just a moment...",
            "page_kind": "blocked",
            "preview": "blocked",
        }
        ctx.url_to_fetch_id["https://www.crunchbase.com/people"] = "fetch-5"

        result = await dispatch_tool(
            "fail_url",
            {"fetch_id": "fetch-5", "reason": "blocked"},
            ctx,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["url"], "https://www.crunchbase.com/people")
        self.assertIn("fetch-5", ctx.processed_fetch_ids)

    async def test_blocked_fetch_bans_domain_for_run(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )
        ctx.suggest_targets_called = True
        ctx.allowed_domains = {"crunchbase.com"}
        _record_fetch_outcome("fetch-6", "crunchbase.com", "blocked", ctx)

        result = await dispatch_tool(
            "fetch_page",
            {"url": "https://www.crunchbase.com/people/search", "needs_javascript": True},
            ctx,
        )

        self.assertIn("error", result)
        self.assertIn("banned for this run", result["error"])

    def test_two_irrelevant_failures_ban_domain_for_run(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA", "min_leads": 1},
            sheets_writer=writer,
            source_mode="web",
        )

        _record_domain_failure("https://example.com/news", "landing page", ctx)
        _record_domain_failure("https://example.com/blog", "article or news page", ctx)

        self.assertTrue(ctx.domain_outcomes["example.com"].banned_for_run)

    async def test_discovered_source_is_promoted_to_temporary_seed_at_balanced_threshold(self) -> None:
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
                        "min_leads": 1,
                        "source_accuracy": "balanced",
                        "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                    },
                )
                ctx = ToolContext(
                    client_config={
                        "client_id": "test",
                        "website": "NA",
                        "min_leads": 1,
                        "source_accuracy": "balanced",
                        "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                        "job_title": "Senior Software Engineer",
                        "job": "find engineers",
                        "area": "NA",
                    },
                    sheets_writer=writer,
                    source_mode="web",
                    source_state=state,
                    source_phase="discovery",
                )
                ctx.current_run_saved_leads.append(
                    {
                        "url": "https://github.com/alice",
                        "data": {"name": "Alice Smith", "job_title": "Senior Software Engineer", "company": "Example"},
                        "source_status": "approved",
                    }
                )

                for idx in range(3):
                    result = await dispatch_tool(
                        "save_result",
                        {
                            "url": f"https://gitlab.com/user{idx}",
                            "data": {
                                "name": f"Engineer {idx}",
                                "job_title": "Senior Software Engineer",
                                "company": "Example Co",
                            },
                        },
                        ctx,
                    )

                self.assertEqual(result["status"], "temporary_seed")
                self.assertIn("gitlab.com", state.temporary_seed_sources()["web_domains"])
                self.assertEqual(writer.saved_count, 3)
            finally:
                os.chdir(old_cwd)

    async def test_discovered_source_is_queued_for_review_when_score_is_borderline(self) -> None:
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
                        "min_leads": 1,
                        "source_accuracy": "balanced",
                        "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                    },
                )
                ctx = ToolContext(
                    client_config={
                        "client_id": "test",
                        "website": "NA",
                        "min_leads": 1,
                        "source_accuracy": "balanced",
                        "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                        "job_title": "Senior Software Engineer",
                        "job": "find engineers",
                        "area": "NA",
                    },
                    sheets_writer=writer,
                    source_mode="web",
                    source_state=state,
                    source_phase="discovery",
                )
                ctx.current_run_saved_leads.append(
                    {
                        "url": "https://github.com/alice",
                        "data": {
                            "name": "Alice Smith",
                            "job_title": "Senior Software Engineer",
                            "company": "Example",
                        },
                        "source_status": "approved",
                    }
                )

                for idx in range(3):
                    result = await dispatch_tool(
                        "save_result",
                        {
                            "url": f"https://gitlab.com/user{idx}",
                            "data": {
                                "name": f"Engineer {idx}",
                                "job_title": "Engineer",
                                "company": "Example Co",
                            },
                        },
                        ctx,
                    )

                self.assertEqual(result["status"], "pending_review")
                self.assertIn("gitlab.com", state.pending_review_sources()["web_domains"])
                self.assertEqual(writer.saved_count, 0)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()

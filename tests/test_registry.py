"""Tests for registry guardrails and tool dispatch behavior."""

import unittest

from tools.registry import ToolContext, dispatch_tool


class _DummyWriter:
    def __init__(self) -> None:
        self.saved_count = 0
        self.duplicate_count = 0
        self.saved_rows: list[dict] = []

    async def append_row(self, url: str, data: dict, scrape_status: str = "ok") -> str:  # noqa: ARG002
        self.saved_count += 1
        self.saved_rows.append({"url": url, "data": data})
        return "saved"


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_mode_rejects_social_media_urls(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA"},
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

    async def test_parse_html_rejects_search_pages(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA"},
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

    async def test_save_result_accepts_flat_arguments_with_fetch_id(self) -> None:
        writer = _DummyWriter()
        ctx = ToolContext(
            client_config={
                "client_id": "test",
                "website": "https://github.com",
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


if __name__ == "__main__":
    unittest.main()

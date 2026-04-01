"""Tests for registry guardrails and tool dispatch behavior."""

import unittest

from tools.registry import ToolContext, dispatch_tool


class _DummyWriter:
    saved_count = 0
    duplicate_count = 0
    saved_rows = []

    async def append_row(self, url: str, data: dict, scrape_status: str = "ok") -> str:  # noqa: ARG002
        return "saved"


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_mode_rejects_social_media_urls(self) -> None:
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA"},
            sheets_writer=_DummyWriter(),
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
        ctx = ToolContext(
            client_config={"client_id": "test", "website": "NA"},
            sheets_writer=_DummyWriter(),
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


if __name__ == "__main__":
    unittest.main()

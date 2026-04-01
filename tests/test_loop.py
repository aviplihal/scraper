"""Tests for agent loop fallback behavior."""

import unittest

from agent.loop import _try_automatic_profile_processing
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


if __name__ == "__main__":
    unittest.main()

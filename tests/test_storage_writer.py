"""Tests for SQLite storage writer behavior."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest

from storage.writer import StorageWriter


class StorageWriterTests(unittest.TestCase):
    def test_duplicate_urls_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                writer = StorageWriter("client_a")
                lead = {
                    "name": "Alice Smith",
                    "job_title": "Senior Software Engineer",
                    "company": "Example Co",
                    "email": None,
                    "phone": None,
                    "social_media": None,
                }

                first = asyncio.run(writer.append_row("https://example.com/alice", lead))
                second = asyncio.run(writer.append_row("https://example.com/alice", lead))

                self.assertEqual(first, "saved")
                self.assertEqual(second, "duplicate")
                self.assertEqual(writer.saved_count, 1)
                self.assertEqual(writer.duplicate_count, 1)
            finally:
                os.chdir(old_cwd)

    def test_recent_rows_returns_latest_saved_lead(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                writer = StorageWriter("client_a")
                lead = {
                    "name": "Alice Smith",
                    "job_title": "Senior Software Engineer",
                    "company": "Example Co",
                    "email": None,
                    "phone": None,
                    "social_media": None,
                }

                asyncio.run(writer.append_row("https://example.com/alice", lead))
                rows = writer.recent_rows(limit=1)

                self.assertEqual(rows[0]["source_url"], "https://example.com/alice")
                self.assertEqual(rows[0]["name"], "Alice Smith")
            finally:
                os.chdir(old_cwd)

    def test_has_source_url_uses_normalized_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                writer = StorageWriter("client_a")
                lead = {
                    "name": "Alice Smith",
                    "job_title": "Senior Software Engineer",
                    "company": "Example Co",
                    "email": None,
                    "phone": None,
                    "social_media": None,
                }

                asyncio.run(writer.append_row("https://www.example.com/alice/", lead))

                self.assertTrue(writer.has_source_url("https://example.com/alice"))
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()

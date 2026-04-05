"""Tests for persistent source approval and temporary-seed lifecycle."""

from __future__ import annotations

import os
import tempfile
import unittest

from source_state import SourceState


class SourceStateTests(unittest.TestCase):
    def test_seeds_approved_sources_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                state = SourceState(
                    "example_client",
                    {
                        "client_id": "example_client",
                        "website": "https://github.com",
                        "social_platforms": ["linkedin", "x"],
                    },
                )

                self.assertEqual(
                    state.approved_sources(),
                    {
                        "web_domains": ["github.com"],
                        "social_platforms": ["linkedin", "x"],
                    },
                )
            finally:
                os.chdir(old_cwd)

    def test_temporary_seed_persists_then_is_removed_when_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                state = SourceState(
                    "example_client",
                    {
                        "client_id": "example_client",
                        "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                    },
                )

                state.promote_temporary_seed("web_domain", "gitlab.com", "developer_profiles", 82)
                self.assertIn("gitlab.com", state.temporary_seed_sources()["web_domains"])

                state.finalize_run(
                    {
                        "web_domain:gitlab.com": {
                            "fetch_count": 2,
                            "saved_count": 0,
                            "duplicate_count": 1,
                            "rejected_count": 0,
                        }
                    }
                )
                self.assertIn("gitlab.com", state.temporary_seed_sources()["web_domains"])

                state.finalize_run(
                    {
                        "web_domain:gitlab.com": {
                            "fetch_count": 2,
                            "saved_count": 0,
                            "duplicate_count": 1,
                            "rejected_count": 0,
                        }
                    }
                )
                self.assertNotIn("gitlab.com", state.temporary_seed_sources()["web_domains"])
                self.assertTrue(state.metadata_for("web_domain", "gitlab.com")["exhausted"])
            finally:
                os.chdir(old_cwd)

    def test_queue_for_review_writes_review_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                state = SourceState(
                    "example_client",
                    {
                        "client_id": "example_client",
                        "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                    },
                )

                state.queue_for_review(
                    "web_domain",
                    "gitlab.com",
                    "developer_profiles",
                    76,
                    [{"name": "Alice", "source_url": "https://gitlab.com/alice"}],
                    [{"name": "Bob", "source_url": "https://github.com/bob"}],
                )

                self.assertIn("gitlab.com", state.pending_review_sources()["web_domains"])
                self.assertTrue(state.review_path.exists())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()

"""Tests for curated broad-mode target selection."""

import os
import tempfile
import unittest

from source_state import SourceState
from tools.targeting import suggest_targets


class TargetingTests(unittest.TestCase):
    def test_leadership_strategy_prefers_relevant_people_targets(self) -> None:
        config = {
            "client_id": "marketing_broad_test",
            "job": "find technical decision makers we can market to",
            "job_title": "Founder",
            "area": "San Francisco Bay Area",
            "website": "NA",
            "min_leads": 3,
        }

        result = suggest_targets(config, "web", limit=4)

        self.assertEqual(result["strategy"], "leadership_people")
        self.assertEqual(result["source_mix"], "web_only")
        self.assertEqual(result["keyword_brief"]["primary_terms"], ["Founder"])
        urls = [target["url"] for target in result["candidate_targets"]]
        self.assertIn("https://www.ycombinator.com/founders", urls)
        self.assertNotIn("https://www.ycombinator.com/companies", urls)
        self.assertNotIn("https://www.crunchbase.com/people", urls)
        self.assertIn("crunchbase.com", result["avoid_domains"])

    def test_all_mode_interleaves_web_and_social_targets(self) -> None:
        config = {
            "client_id": "marketing_broad_test",
            "job": "find technical decision makers we can market to",
            "job_title": "Founder",
            "area": "San Francisco Bay Area",
            "website": "NA",
            "min_leads": 3,
            "social_platforms": ["linkedin", "x"],
        }

        result = suggest_targets(config, "all", limit=6)

        self.assertEqual(result["source_mix"], "model_decides")
        urls = [target["url"] for target in result["candidate_targets"]]
        self.assertTrue(any("github.com/search" in url for url in urls))
        self.assertTrue(any("linkedin.com/search/results/people" in url for url in urls))
        self.assertTrue(any("x.com/search" in url for url in urls))
        first_wave_domains = [target["domain"] for target in result["candidate_targets"][:3]]
        self.assertEqual(len(first_wave_domains), len(set(first_wave_domains)))
        self.assertIn("github.com", first_wave_domains)
        self.assertTrue(any(domain in {"linkedin.com", "x.com"} for domain in first_wave_domains))

    def test_human_emulator_mode_returns_social_targets_only(self) -> None:
        config = {
            "client_id": "social_test",
            "job": "find technical decision makers we can market to",
            "job_title": "Founder",
            "area": "San Francisco Bay Area",
            "website": "NA",
            "min_leads": 2,
            "social_platforms": ["linkedin", "x"],
        }

        result = suggest_targets(config, "human_emulator", limit=4)

        self.assertEqual(result["source_mix"], "social_only")
        urls = [target["url"] for target in result["candidate_targets"]]
        self.assertTrue(urls)
        self.assertTrue(all("github.com" not in url and "ycombinator.com" not in url for url in urls))
        self.assertTrue(any("linkedin.com" in url for url in urls))
        self.assertTrue(any("x.com" in url for url in urls))
        self.assertEqual(set(result["allowed_domains"]), {"linkedin.com", "x.com"})

    def test_pinned_site_targets_stay_on_domain(self) -> None:
        config = {
            "client_id": "github_public_profiles",
            "job": "find public engineers",
            "job_title": "Senior Software Engineer",
            "area": "NA",
            "website": "https://github.com",
            "min_leads": 3,
        }

        result = suggest_targets(config, "web", limit=3)

        self.assertEqual(result["strategy"], "pinned_site")
        for target in result["candidate_targets"]:
            self.assertTrue(str(target["url"]).startswith("https://github.com"))

    def test_pass1_uses_only_approved_and_temporary_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                config = {
                    "client_id": "example_client",
                    "job": "find senior software engineers open to new opportunities",
                    "job_title": "Senior Software Engineer",
                    "area": "San Francisco Bay Area",
                    "website": "NA",
                    "min_leads": 3,
                    "approved_sources": {"web_domains": ["github.com"], "social_platforms": ["linkedin"]},
                    "social_platforms": ["linkedin", "x"],
                }
                state = SourceState("example_client", config)
                state.promote_temporary_seed("web_domain", "gitlab.com", "developer_profiles", 82)

                result = suggest_targets(config, "all", limit=6, source_state=state, phase="pass1")

                domains = [target["domain"] for target in result["candidate_targets"]]
                self.assertIn("github.com", domains)
                self.assertIn("linkedin.com", domains)
                self.assertIn("gitlab.com", domains)
                self.assertNotIn("x.com", domains)
            finally:
                os.chdir(old_cwd)

    def test_discovery_excludes_approved_and_temporary_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                config = {
                    "client_id": "example_client",
                    "job": "find senior software engineers open to new opportunities",
                    "job_title": "Senior Software Engineer",
                    "area": "San Francisco Bay Area",
                    "website": "NA",
                    "min_leads": 3,
                    "approved_sources": {"web_domains": ["github.com"], "social_platforms": ["linkedin"]},
                    "social_platforms": ["linkedin", "x"],
                }
                state = SourceState("example_client", config)
                state.promote_temporary_seed("web_domain", "gitlab.com", "developer_profiles", 82)

                result = suggest_targets(config, "all", limit=6, source_state=state, phase="discovery")

                domains = [target["domain"] for target in result["candidate_targets"]]
                self.assertNotIn("github.com", domains)
                self.assertNotIn("gitlab.com", domains)
                self.assertNotIn("linkedin.com", domains)
                self.assertIn("x.com", domains)
                self.assertIn("duckduckgo.com", domains)
            finally:
                os.chdir(old_cwd)

    def test_discovery_web_mode_adds_public_search_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                config = {
                    "client_id": "example_client",
                    "job": "find senior software engineers open to new opportunities",
                    "job_title": "Senior Software Engineer",
                    "area": "San Francisco Bay Area",
                    "website": "NA",
                    "min_leads": 3,
                    "approved_sources": {"web_domains": ["github.com"], "social_platforms": []},
                }
                state = SourceState("example_client", config)

                result = suggest_targets(config, "web", limit=8, source_state=state, phase="discovery")

                domains = [target["domain"] for target in result["candidate_targets"]]
                self.assertIn("duckduckgo.com", domains)
            finally:
                os.chdir(old_cwd)

    def test_large_all_mode_technical_pass1_prefers_web_targets(self) -> None:
        config = {
            "client_id": "example_client",
            "job": "find senior software engineers open to new opportunities",
            "job_title": "Senior Software Engineer",
            "area": "San Francisco Bay Area",
            "website": "NA",
            "min_leads": 10,
            "social_platforms": ["linkedin", "x"],
        }

        result = suggest_targets(config, "all", limit=6, phase="pass1")

        domains = [target["domain"] for target in result["candidate_targets"]]
        self.assertIn("github.com", domains)
        self.assertNotIn("linkedin.com", domains)
        self.assertNotIn("x.com", domains)

    def test_large_technical_run_expands_github_seed_catalog(self) -> None:
        config = {
            "client_id": "example_client",
            "job": "find senior software engineers open to new opportunities",
            "job_title": "Senior Software Engineer",
            "area": "San Francisco Bay Area",
            "website": "NA",
            "min_leads": 100,
        }

        result = suggest_targets(config, "web", limit=80, phase="pass1")

        github_urls = [
            str(target["url"])
            for target in result["candidate_targets"]
            if target["domain"] == "github.com"
        ]
        self.assertGreaterEqual(len(github_urls), 20)
        self.assertTrue(any("Staff+Engineer" in url for url in github_urls))
        self.assertTrue(any("Principal+" in url for url in github_urls))
        self.assertTrue(any("location%3A%22San+Francisco%22" in url for url in github_urls))


if __name__ == "__main__":
    unittest.main()

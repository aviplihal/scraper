"""Tests for curated broad-mode target selection."""

import unittest

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
        urls = [target["url"] for target in result["targets"]]
        self.assertIn("https://www.ycombinator.com/founders", urls)
        self.assertNotIn("https://www.ycombinator.com/companies", urls)
        self.assertNotIn("https://www.crunchbase.com/people", urls)

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

        urls = [target["url"] for target in result["targets"]]
        self.assertTrue(any("github.com/search" in url for url in urls))
        self.assertTrue(any("linkedin.com/search/results/people" in url for url in urls))
        self.assertTrue(any("x.com/search" in url for url in urls))
        self.assertNotEqual(urls[0].split("/")[2], urls[1].split("/")[2])

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

        urls = [target["url"] for target in result["targets"]]
        self.assertTrue(urls)
        self.assertTrue(all("github.com" not in url and "ycombinator.com" not in url for url in urls))
        self.assertTrue(any("linkedin.com" in url for url in urls))
        self.assertTrue(any("x.com" in url for url in urls))

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
        for target in result["targets"]:
            self.assertTrue(str(target["url"]).startswith("https://github.com"))


if __name__ == "__main__":
    unittest.main()

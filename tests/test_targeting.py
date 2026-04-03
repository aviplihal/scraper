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

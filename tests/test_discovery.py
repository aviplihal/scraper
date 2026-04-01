"""Tests for page classification and link extraction."""

import unittest

from tools.discovery import classify_page, extract_links


class DiscoveryTests(unittest.TestCase):
    def test_classify_github_user_search(self) -> None:
        html = """
        <html>
          <head><title>User search results · GitHub</title></head>
          <body>
            <div data-testid="results-list">
              <a data-hovercard-type="user" href="/alice-smith">Alice Smith</a>
            </div>
          </body>
        </html>
        """

        page_info = classify_page(
            "https://github.com/search?q=senior+software+engineer&type=users",
            "https://github.com/search?q=senior+software+engineer&type=users",
            html,
        )

        self.assertEqual(page_info.page_kind, "search_results")

    def test_classify_blocked_and_job_board(self) -> None:
        blocked_html = """
        <html><head><title>Just a moment...</title></head><body>Verify you are human</body></html>
        """
        blocked = classify_page("https://www.crunchbase.com/people/search", "https://www.crunchbase.com/people/search", blocked_html)
        self.assertEqual(blocked.page_kind, "blocked")

        jobs_html = """
        <html><head><title>Search Jobs | Dice.com</title></head><body>Job Search Career Advice</body></html>
        """
        job_board = classify_page("https://www.dice.com/jobs/search", "https://www.dice.com/jobs/search", jobs_html)
        self.assertEqual(job_board.page_kind, "job_board")

    def test_extract_links_prefers_profile_links(self) -> None:
        html = """
        <html>
          <body>
            <a href="/pricing">Pricing</a>
            <a data-hovercard-type="user" href="/alice-smith">Alice Smith</a>
            <a data-hovercard-type="user" href="/bob-jones">Bob Jones</a>
          </body>
        </html>
        """

        links = extract_links(
            html,
            "https://github.com/search?q=senior+software+engineer&type=users",
        )

        self.assertEqual(
            links,
            [
                {"url": "https://github.com/alice-smith", "text": "Alice Smith"},
                {"url": "https://github.com/bob-jones", "text": "Bob Jones"},
            ],
        )

    def test_extract_links_same_domain_only(self) -> None:
        html = """
        <html>
          <body>
            <a href="https://example.com/team/alice">Alice</a>
            <a href="https://other.com/people/bob">Bob</a>
          </body>
        </html>
        """

        links = extract_links(
            html,
            "https://example.com/search",
            same_domain_only=True,
        )

        self.assertEqual(links, [{"url": "https://example.com/team/alice", "text": "Alice"}])


if __name__ == "__main__":
    unittest.main()

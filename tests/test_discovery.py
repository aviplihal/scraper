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

    def test_classify_company_directory_company_page_and_news(self) -> None:
        yc_directory = """
        <html><head><title>The YC Startup Directory | Y Combinator</title></head><body>Startup directory</body></html>
        """
        directory = classify_page(
            "https://www.ycombinator.com/companies",
            "https://www.ycombinator.com/companies",
            yc_directory,
        )
        self.assertEqual(directory.page_kind, "company_directory")

        company_page_html = """
        <html><head><title>Leadership Team | Example Co</title></head><body>Meet the team</body></html>
        """
        company_page = classify_page(
            "https://example.com/team",
            "https://example.com/team",
            company_page_html,
        )
        self.assertEqual(company_page.page_kind, "company_page")

        news_home_html = """
        <html><head><title>TechCrunch | Startup and Technology News</title></head><body>startup and technology news</body></html>
        """
        news_page = classify_page(
            "https://techcrunch.com",
            "https://techcrunch.com/",
            news_home_html,
        )
        self.assertEqual(news_page.page_kind, "article_or_news")

    def test_classify_landing_page(self) -> None:
        landing_html = """
        <html><head><title>Acme SaaS</title></head><body>Welcome to Acme</body></html>
        """
        landing = classify_page("https://acme.example", "https://acme.example/", landing_html)
        self.assertEqual(landing.page_kind, "landing_page")

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

    def test_extract_links_filters_github_user_search_junk_links(self) -> None:
        html = """
        <html>
          <body>
            <a href="/team">Team</a>
            <a href="/topics">Topics</a>
            <a href="/alice-smith">Alice Smith</a>
            <a href="/bob-jones">Bob Jones</a>
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

    def test_extract_links_prefers_people_links_from_company_page(self) -> None:
        html = """
        <html>
          <body>
            <a href="/pricing">Pricing</a>
            <a href="/products/platform">Platform</a>
            <a href="/leadership">Leadership</a>
            <a href="/team">Meet the team</a>
            <a href="/blog/funding-round">Funding news</a>
          </body>
        </html>
        """

        links = extract_links(html, "https://example.com/about", limit=3)

        self.assertEqual(links[2], {"url": "https://example.com/products/platform", "text": "Platform"})
        self.assertEqual(
            {links[0]["url"], links[1]["url"]},
            {"https://example.com/leadership", "https://example.com/team"},
        )


if __name__ == "__main__":
    unittest.main()

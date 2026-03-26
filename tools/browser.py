"""Playwright browser manager for the web scraper.

Policy:
  - One browser process is created per job and reused across all URLs.
  - Each URL gets a fresh BrowserContext for isolation.
  - Cookies are session-scoped and never persisted to disk.
"""

import logging

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

_VIEWPORT = {"width": 1280, "height": 800}


class ScraperBrowser:
    """Manages a single shared Chromium browser for the web-scraper source."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        logger.info("Web-scraper browser started.")

    async def new_context(self) -> BrowserContext:
        """Return a fresh, isolated browser context (no persistent cookies)."""
        if self._browser is None:
            raise RuntimeError("ScraperBrowser not started — call start() first.")
        ctx = await self._browser.new_context(
            user_agent=_USER_AGENT,
            viewport=_VIEWPORT,
            locale="en-US",
        )
        return ctx

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Web-scraper browser closed.")

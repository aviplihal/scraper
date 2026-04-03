"""Persistent browser profile manager for the human emulator.

Creates one persistent Playwright context per social platform so cookies,
cached assets, login sessions, and browser history accumulate between runs
like a real browser for that specific platform.
"""

import logging
import os
from pathlib import Path

from playwright.async_api import BrowserContext, Playwright, async_playwright
from playwright_stealth import Stealth

_stealth = Stealth()

logger = logging.getLogger(__name__)

# Timezone and geolocation matching the account's expected location (US/Eastern).
# Override via env vars if the account is registered in a different locale.
_TIMEZONE_ID  = os.environ.get("EMULATOR_TIMEZONE", "America/New_York")
_GEOLOCATION  = {
    "latitude":  float(os.environ.get("EMULATOR_GEO_LAT",  "40.7128")),
    "longitude": float(os.environ.get("EMULATOR_GEO_LON", "-74.0060")),
}

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

_VIEWPORT = {"width": 1280, "height": 800}


class EmulatorBrowser:
    """Owns Playwright and per-platform persistent browser contexts."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self._profile_root = Path("profiles/social")
        self._playwright: Playwright | None = None
        self._contexts: dict[str, BrowserContext] = {}

    async def start(self) -> "EmulatorBrowser":
        """Initialize Playwright. Contexts are launched lazily per platform."""
        self._profile_root.mkdir(parents=True, exist_ok=True)
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        logger.info("Social emulator browser started for client '%s'", self.client_id)
        return self

    async def get_context(self, platform: str) -> BrowserContext:
        """Launch or return the persistent context for a platform."""
        if self._playwright is None:
            await self.start()
        if platform in self._contexts:
            return self._contexts[platform]

        profile_dir = self._profile_root / platform
        profile_dir.mkdir(parents=True, exist_ok=True)
        context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
            user_agent=_USER_AGENT,
            viewport=_VIEWPORT,
            timezone_id=_TIMEZONE_ID,
            geolocation=_GEOLOCATION,
            permissions=["geolocation"],
            locale="en-US",
        )
        context.on("page", self._on_new_page)
        self._contexts[platform] = context
        logger.info("Persistent browser context started for social platform '%s'", platform)
        return context

    async def _on_new_page(self, page) -> None:  # noqa: ANN001
        await _stealth.apply_stealth_async(page)

    async def new_page(self, platform: str):
        """Open a new page in the platform's persistent context with stealth applied."""
        context = await self.get_context(platform)
        page = await context.new_page()
        await _stealth.apply_stealth_async(page)
        return page

    async def close(self) -> None:
        for context in self._contexts.values():
            await context.close()
        self._contexts.clear()
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Persistent browser contexts closed.")

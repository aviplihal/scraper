"""Persistent browser profile manager for the human emulator.

Creates a Playwright persistent context at profiles/{client_id}/ so that
cookies, cached assets, login sessions, and browser history accumulate
between runs exactly like a real user's browser.

playwright-stealth patches every context to suppress headless fingerprints.
"""

import logging
import os
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright
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
    """Owns the Playwright instance and the persistent browser context."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self._profile_dir = Path(f"profiles/{client_id}")
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None

    async def start(self) -> BrowserContext:
        """Launch (or resume) the persistent browser context."""
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self._profile_dir),
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
        # Apply stealth to every new page that opens in this context
        self._context.on("page", self._on_new_page)
        logger.info("Persistent browser context started for client '%s'", self.client_id)
        return self._context

    async def _on_new_page(self, page) -> None:  # noqa: ANN001
        await _stealth.apply_stealth_async(page)

    async def new_page(self):
        """Open a new page in the persistent context with stealth applied."""
        if self._context is None:
            raise RuntimeError("EmulatorBrowser not started — call start() first.")
        page = await self._context.new_page()
        await _stealth.apply_stealth_async(page)
        return page

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Persistent browser context closed.")

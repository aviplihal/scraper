"""LinkedIn-specific navigation and data extraction for the human emulator.

All generic human-behaviour primitives are imported from core.py.
Platform-specific logic (selectors, login flow, restriction detection) lives here.
"""

import asyncio
import logging
import os
import random

from playwright.async_api import BrowserContext, Page

from human_emulator.core import (
    SessionRhythmManager,
    human_mouse_move,
    human_scroll,
    human_type,
    session_delay,
    wait_reading_time,
)
from human_emulator.notifier import send_alert
from human_emulator.state import EmulatorState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

_LOGIN_EMAIL_SEL    = "input#username"
_LOGIN_PASS_SEL     = "input#password"
_LOGIN_SUBMIT_SEL   = "button[type='submit']"
_FEED_INDICATOR_SEL = "div.feed-identity-module"

_PROFILE_NAME_SEL    = "h1.text-heading-xlarge"
_PROFILE_TITLE_SEL   = ".text-body-medium.break-words"
_PROFILE_COMPANY_SEL = "div.experience-section li:first-child .pv-entity__secondary-title"
_CONTACT_INFO_BTN    = "a[href*='overlay/contact-info']"
_CONTACT_MODAL_SEL   = "div.pv-contact-info"
_MODAL_EMAIL_SEL     = "section.ci-email a"
_MODAL_PHONE_SEL     = "section.ci-phone span.t-14"
_MODAL_CLOSE_BTN     = "button[aria-label='Dismiss']"

# Restriction / security-check indicators
_RESTRICTION_URLS     = ["/checkpoint/", "/authwall", "/challenge/"]
_RESTRICTION_TEXT_CUE = "unusual activity"
_CAPTCHA_SEL          = "div#captcha-internal"

# How many times we pause before treating it as a hard stop
_MAX_RESTRICTIONS = 2
_PAUSE_HOURS      = 8


# ---------------------------------------------------------------------------
# LinkedIn emulator
# ---------------------------------------------------------------------------

class LinkedInEmulator:
    """Visits LinkedIn profile pages and extracts lead data."""

    def __init__(self, context: BrowserContext, state: EmulatorState, client_id: str):
        self._context   = context
        self._state     = state
        self._client_id = client_id
        self._rhythm    = SessionRhythmManager()
        self._logged_in = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_profiles(
        self, profile_urls: list[str], on_result
    ) -> None:
        """Visit each URL in profile_urls, extract data, call on_result(url, data).

        Respects daily limits, restriction pausing, and session rhythm.
        """
        for url in profile_urls:
            # Check pause state
            paused, until = self._state.is_paused()
            if paused:
                logger.info("Emulator is paused until %s — stopping.", until)
                await send_alert(
                    f"⏸ LinkedIn emulator paused until {until} for client <b>{self._client_id}</b>."
                )
                return

            if self._state.daily_budget_exhausted():
                logger.info("Daily visit budget exhausted (%d visits).", self._state.visits_today)
                return

            # Session rhythm / break
            await self._rhythm.maybe_take_break()
            await self._rhythm.pace_variation()

            # Ensure we are logged in
            if not self._logged_in:
                await self._login()

            try:
                data = await self._visit_profile(url)
                self._state.mark_visited(url)
                await on_result(url, data)
            except RestrictionDetected:
                count = self._state.record_restriction()
                if count >= _MAX_RESTRICTIONS:
                    msg = (
                        f"🚨 LinkedIn emulator HARD STOP — {count} restrictions hit "
                        f"for client <b>{self._client_id}</b>. Manual intervention required."
                    )
                    logger.error(msg)
                    await send_alert(msg)
                    return
                else:
                    from datetime import datetime, timedelta, timezone
                    resume_at = datetime.now(timezone.utc) + timedelta(hours=_PAUSE_HOURS)
                    self._state.set_pause(resume_at)
                    msg = (
                        f"⚠️ LinkedIn restriction detected for client <b>{self._client_id}</b>. "
                        f"Pausing for {_PAUSE_HOURS} hours (until {resume_at.strftime('%H:%M UTC')})."
                    )
                    logger.warning(msg)
                    await send_alert(msg)
                    return
            except Exception as exc:
                logger.warning("Failed to scrape %s: %s", url, exc)
                await on_result(url, {})

            await session_delay()

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    async def _login(self) -> None:
        email    = os.environ["EMULATOR_EMAIL"]
        password = os.environ["EMULATOR_PASSWORD"]

        page = await self._context.new_page()
        try:
            await page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
            await wait_reading_time(page)

            # Check if already logged in (session cookie still valid)
            if await page.locator(_FEED_INDICATOR_SEL).count() > 0 or \
               "feed" in page.url:
                logger.info("Already logged in to LinkedIn.")
                self._logged_in = True
                return

            await _check_restriction(page)

            # Type credentials
            await human_type(page, _LOGIN_EMAIL_SEL, email)
            await asyncio.sleep(random.uniform(0.4, 1.0))
            await human_type(page, _LOGIN_PASS_SEL, password)
            await asyncio.sleep(random.uniform(0.3, 0.8))

            submit = page.locator(_LOGIN_SUBMIT_SEL).first
            bbox   = await submit.bounding_box()
            if bbox:
                await human_mouse_move(page, bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
            await submit.click()

            await page.wait_for_load_state("domcontentloaded")
            await wait_reading_time(page)
            await _check_restriction(page)

            self._logged_in = True
            logger.info("LinkedIn login successful.")
        finally:
            await page.close()

    # ------------------------------------------------------------------
    # Profile visit
    # ------------------------------------------------------------------

    async def _visit_profile(self, url: str) -> dict:
        page = await self._context.new_page()
        data: dict = {}
        try:
            logger.info("Visiting LinkedIn profile: %s", url)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await wait_reading_time(page)
            await _check_restriction(page)

            # Natural scroll to read the profile
            for _ in range(random.randint(2, 4)):
                await human_scroll(page, "down")
                await asyncio.sleep(random.uniform(0.5, 1.5))

            # Extract visible fields
            data["name"]      = await _safe_text(page, _PROFILE_NAME_SEL)
            data["job_title"] = await _safe_text(page, _PROFILE_TITLE_SEL)
            data["company"]   = await _extract_current_company(page)

            # Try contact info modal
            contact = await self._open_contact_modal(page)
            data["email"]        = contact.get("email")
            data["phone"]        = contact.get("phone")
            data["social_media"] = url

            return data
        finally:
            await page.close()

    async def _open_contact_modal(self, page: Page) -> dict:
        result: dict = {}
        btn = page.locator(_CONTACT_INFO_BTN).first
        try:
            if await btn.count() == 0:
                return result

            bbox = await btn.bounding_box()
            if bbox:
                await human_mouse_move(page, bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
            await btn.click()
            await asyncio.sleep(random.uniform(1.0, 2.0))
            await _check_restriction(page)

            result["email"] = await _safe_text(page, _MODAL_EMAIL_SEL)
            result["phone"] = await _safe_text(page, _MODAL_PHONE_SEL)

            # Close the modal
            close = page.locator(_MODAL_CLOSE_BTN).first
            if await close.count() > 0:
                await close.click()
                await asyncio.sleep(random.uniform(0.3, 0.7))
        except Exception as exc:
            logger.debug("Contact modal failed: %s", exc)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RestrictionDetected(Exception):
    pass


async def _check_restriction(page: Page) -> None:
    """Raise RestrictionDetected if LinkedIn is showing a security checkpoint."""
    current_url = page.url
    for fragment in _RESTRICTION_URLS:
        if fragment in current_url:
            raise RestrictionDetected(f"Restriction URL detected: {current_url}")

    # Check for captcha element
    if await page.locator(_CAPTCHA_SEL).count() > 0:
        raise RestrictionDetected("CAPTCHA detected.")

    # Check for restriction text in page body
    try:
        body_text = await page.evaluate("document.body.innerText")
        if _RESTRICTION_TEXT_CUE in (body_text or "").lower():
            raise RestrictionDetected("Unusual-activity text detected.")
    except Exception:
        pass


async def _safe_text(page: Page, selector: str) -> str | None:
    """Return stripped text for the first matching element, or None."""
    try:
        loc = page.locator(selector).first
        if await loc.count() > 0:
            text = await loc.text_content()
            return text.strip() if text else None
    except Exception:
        pass
    return None


async def _extract_current_company(page: Page) -> str | None:
    """Extract the current employer from the experience section."""
    # Try the compact top-of-profile card first
    selectors = [
        "div.pv-text-details__right-panel .t-14",
        ".pv-top-card--experience-list-item .t-14",
        "span[aria-label*='Current company']",
    ]
    for sel in selectors:
        text = await _safe_text(page, sel)
        if text:
            return text
    return None

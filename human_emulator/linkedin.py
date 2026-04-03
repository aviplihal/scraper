"""LinkedIn adapter for the multi-platform human emulator."""

from __future__ import annotations

import asyncio
import logging
import random
from urllib.parse import urlparse

from playwright.async_api import Page

from human_emulator.core import (
    SessionRhythmManager,
    human_mouse_move,
    human_scroll,
    human_type,
    session_delay,
    wait_reading_time,
)
from human_emulator.notifier import send_alert
from human_emulator.social import (
    RestrictionDetected,
    SocialAdapter,
    SocialFetchResult,
    build_profile_html,
    build_search_html,
    first_href,
    first_text,
)

logger = logging.getLogger(__name__)

_LOGIN_EMAIL_SEL = "input#username, input[name='session_key']"
_LOGIN_PASS_SEL = "input#password, input[name='session_password']"
_LOGIN_SUBMIT_SEL = "button[type='submit']"
_FEED_INDICATORS = [
    "div.feed-identity-module",
    "a[href='/feed/']",
    "input[placeholder*='Search']",
]

_PROFILE_NAME_SELECTORS = ["h1.text-heading-xlarge", "h1"]
_PROFILE_TITLE_SELECTORS = [".text-body-medium.break-words", ".pv-text-details__left-panel div.text-body-medium"]
_PROFILE_COMPANY_SELECTORS = [
    "div.pv-text-details__right-panel .t-14",
    ".pv-top-card--experience-list-item .t-14",
    "span[aria-label*='Current company']",
]
_CONTACT_INFO_BTN = "a[href*='overlay/contact-info']"
_MODAL_EMAIL_SEL = "section.ci-email a, a[href^='mailto:']"
_MODAL_PHONE_SEL = "section.ci-phone span.t-14, a[href^='tel:']"
_MODAL_CLOSE_BTN = "button[aria-label='Dismiss'], button[aria-label='Close']"
_WEBSITE_SELECTORS = ["section.ci-websites a", "a[href*='http']"]

_SEARCH_RESULT_CARDS = [
    "li.reusable-search__result-container",
    "div.search-results-container li",
]
_SEARCH_RESULT_LINKS = [
    "a[href*='/in/']",
    "a.app-aware-link[href*='/in/']",
]
_SEARCH_RESULT_NAME = ["span[aria-hidden='true']", "span.entity-result__title-text"]
_SEARCH_RESULT_HEADLINE = [".entity-result__primary-subtitle", ".entity-result__summary"]

_RESTRICTION_URLS = ["/checkpoint/", "/authwall", "/challenge/", "/login-submit"]
_RESTRICTION_TEXT_CUES = ["unusual activity", "verify your identity", "security verification", "challenge"]
_CAPTCHA_SELECTORS = ["div#captcha-internal", "iframe[title*='captcha']", "input[name='captcha']"]
_PAUSE_HOURS = 8


class LinkedInAdapter(SocialAdapter):
    """Code-driven LinkedIn adapter."""

    platform = "linkedin"
    domains = ("linkedin.com",)

    def __init__(self, context, state, client_id: str):
        super().__init__(context, state, client_id)
        self._rhythm = SessionRhythmManager()

    async def fetch(self, url: str) -> SocialFetchResult:
        await self._rhythm.maybe_take_break()
        await self._rhythm.pace_variation()
        await self.ensure_logged_in()

        parsed = urlparse(url)
        if "/search/results/people" in parsed.path:
            result = await self._fetch_search(url)
        else:
            result = await self._fetch_profile(url)

        await session_delay()
        return result

    async def _login(self) -> None:
        creds = self.credentials()
        page = await self._context.new_page()
        try:
            await page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
            await wait_reading_time(page)
            if await self._looks_logged_in(page):
                logger.info("LinkedIn session already active.")
                return

            await _check_restriction(page)

            username = creds["username"] or ""
            password = creds["password"] or ""
            await human_type(page, _LOGIN_EMAIL_SEL, username)
            await asyncio.sleep(random.uniform(0.4, 1.0))
            await human_type(page, _LOGIN_PASS_SEL, password)
            await asyncio.sleep(random.uniform(0.3, 0.8))

            submit = page.locator(_LOGIN_SUBMIT_SEL).first
            bbox = await submit.bounding_box()
            if bbox:
                await human_mouse_move(page, bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
            await submit.click()
            await page.wait_for_load_state("domcontentloaded")
            await wait_reading_time(page)
            await _check_restriction(page)
            logger.info("LinkedIn login successful.")
        finally:
            await page.close()

    async def _looks_logged_in(self, page: Page) -> bool:
        for selector in _FEED_INDICATORS:
            try:
                if await page.locator(selector).count() > 0:
                    return True
            except Exception:
                continue
        return "feed" in page.url or "mynetwork" in page.url

    async def _fetch_search(self, url: str) -> SocialFetchResult:
        page = await self._context.new_page()
        try:
            logger.info("Visiting LinkedIn search: %s", url)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await wait_reading_time(page)
            await _check_restriction(page)
            for _ in range(random.randint(1, 2)):
                await human_scroll(page, "down")
                await asyncio.sleep(random.uniform(0.4, 1.0))

            results = await _extract_search_results(page)
            html = build_search_html(self.platform, "LinkedIn people search", results)
            return SocialFetchResult(
                final_url=page.url,
                title="LinkedIn People Search",
                page_kind="search_results",
                html=html,
                extracted_data={"results": results},
            )
        finally:
            await page.close()

    async def _fetch_profile(self, url: str) -> SocialFetchResult:
        page = await self._context.new_page()
        try:
            logger.info("Visiting LinkedIn profile: %s", url)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await wait_reading_time(page)
            await _check_restriction(page)

            for _ in range(random.randint(2, 4)):
                await human_scroll(page, "down")
                await asyncio.sleep(random.uniform(0.5, 1.5))

            name = await first_text(page, _PROFILE_NAME_SELECTORS)
            job_title = await first_text(page, _PROFILE_TITLE_SELECTORS)
            company = await first_text(page, _PROFILE_COMPANY_SELECTORS)
            contact = await self._open_contact_modal(page)

            data = {
                "name": name,
                "job_title": job_title,
                "company": company,
                "email": contact.get("email"),
                "phone": contact.get("phone"),
                "website": contact.get("website"),
                "social_media": url,
            }
            html = build_profile_html(data)
            title = name or "LinkedIn profile"
            return SocialFetchResult(
                final_url=page.url,
                title=title,
                page_kind="profile",
                html=html,
                extracted_data=data,
            )
        finally:
            await page.close()

    async def _open_contact_modal(self, page: Page) -> dict[str, str | None]:
        result: dict[str, str | None] = {"email": None, "phone": None, "website": None}
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

            result["email"] = await first_text(page, [_MODAL_EMAIL_SEL])
            result["phone"] = await first_text(page, [_MODAL_PHONE_SEL])
            result["website"] = await first_href(page, _WEBSITE_SELECTORS)

            close = page.locator(_MODAL_CLOSE_BTN).first
            if await close.count() > 0:
                await close.click()
                await asyncio.sleep(random.uniform(0.3, 0.7))
        except Exception as exc:
            logger.debug("LinkedIn contact modal failed: %s", exc)
        return result


async def _extract_search_results(page: Page) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for card_selector in _SEARCH_RESULT_CARDS:
        cards = page.locator(card_selector)
        try:
            count = await cards.count()
        except Exception:
            continue
        if count == 0:
            continue
        for idx in range(min(count, 10)):
            card = cards.nth(idx)
            url = None
            for link_selector in _SEARCH_RESULT_LINKS:
                try:
                    href = await card.locator(link_selector).first.get_attribute("href")
                    if href:
                        url = href if href.startswith("http") else f"https://www.linkedin.com{href}"
                        break
                except Exception:
                    continue
            if not url or url in seen_urls:
                continue
            name = await _text_from_card(card, _SEARCH_RESULT_NAME)
            headline = await _text_from_card(card, _SEARCH_RESULT_HEADLINE)
            results.append({"url": url, "name": name or url, "headline": headline or ""})
            seen_urls.add(url)
        if results:
            break

    return results


async def _text_from_card(card, selectors: list[str]) -> str | None:  # noqa: ANN001
    for selector in selectors:
        try:
            locator = card.locator(selector).first
            if await locator.count() > 0:
                text = await locator.text_content()
                if text and text.strip():
                    return text.strip()
        except Exception:
            continue
    return None


async def _check_restriction(page: Page) -> None:
    current_url = page.url.lower()
    for fragment in _RESTRICTION_URLS:
        if fragment in current_url:
            message = f"LinkedIn restriction detected at {current_url}"
            logger.warning(message)
            await send_alert(f"⚠️ {message}")
            raise RestrictionDetected(message)

    for selector in _CAPTCHA_SELECTORS:
        try:
            if await page.locator(selector).count() > 0:
                message = "LinkedIn captcha detected."
                logger.warning(message)
                await send_alert(f"⚠️ {message}")
                raise RestrictionDetected(message)
        except RestrictionDetected:
            raise
        except Exception:
            continue

    try:
        body_text = (await page.evaluate("document.body.innerText") or "").lower()
    except Exception:
        body_text = ""
    if any(cue in body_text for cue in _RESTRICTION_TEXT_CUES):
        message = "LinkedIn unusual-activity or verification challenge detected."
        logger.warning(message)
        await send_alert(f"⚠️ {message} Pausing platform for {_PAUSE_HOURS} hours.")
        raise RestrictionDetected(message)


# Backward-compatible alias for older references.
LinkedInEmulator = LinkedInAdapter

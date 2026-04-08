"""Instagram adapter for the multi-platform human emulator."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from urllib.parse import parse_qs, urlparse

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

_LOGIN_URL = "https://www.instagram.com/accounts/login/"
_HOME_INDICATORS = [
    "svg[aria-label='Home']",
    "a[href='/']",
    "input[placeholder='Search']",
]
_USERNAME_SELECTORS = ["input[name='username']", "input[autocomplete='username']"]
_PASSWORD_SELECTORS = ["input[name='password']"]
_LOGIN_BUTTON_SELECTORS = ["button[type='submit']", "button:has-text('Log in')"]
_PROFILE_NAME_SELECTORS = ["header h2", "header h1", "main h2", "main h1"]
_PROFILE_BIO_SELECTORS = ["header section div", "main header div"]
_PROFILE_WEBSITE_SELECTORS = ["header a[href^='http']", "main a[href^='http']"]
_PROFILE_LINK_SELECTORS = ["a[href^='/']"]
_CHALLENGE_URLS = ["/challenge/", "/accounts/suspended/", "/accounts/login/"]
_CHALLENGE_TEXT_CUES = [
    "confirm it's you",
    "suspicious login attempt",
    "help us confirm you own this account",
    "security code",
]
_CAPTCHA_SELECTORS = ["iframe[title*='captcha']", "input[name='captcha']"]
_RESERVED_PATHS = {
    "about",
    "accounts",
    "api",
    "developer",
    "direct",
    "explore",
    "legal",
    "p",
    "reel",
    "stories",
}


class InstagramAdapter(SocialAdapter):
    """Code-driven Instagram adapter."""

    platform = "instagram"
    domains = ("instagram.com",)

    def __init__(self, context, state, client_id: str):
        super().__init__(context, state, client_id)
        self._rhythm = SessionRhythmManager()

    async def fetch(self, url: str) -> SocialFetchResult:
        await self._rhythm.maybe_take_break()
        await self._rhythm.pace_variation()
        await self.ensure_logged_in()

        parsed = urlparse(url)
        if "/explore/search" in parsed.path or parse_qs(parsed.query).get("q"):
            result = await self._fetch_search(url)
        else:
            result = await self._fetch_profile(url)

        await session_delay()
        return result

    async def _login(self) -> None:
        creds = self.credentials()
        page = await self._context.new_page()
        try:
            await page.goto(_LOGIN_URL, wait_until="domcontentloaded")
            await wait_reading_time(page)
            if await self._looks_logged_in(page):
                logger.info("Instagram session already active.")
                return

            await _check_restriction(page, allow_login_page=True)
            await _type_first_available(page, _USERNAME_SELECTORS, creds["username"] or "")
            await asyncio.sleep(random.uniform(0.4, 1.0))
            await _type_first_available(page, _PASSWORD_SELECTORS, creds["password"] or "")
            await asyncio.sleep(random.uniform(0.3, 0.8))
            await _click_first_available(page, _LOGIN_BUTTON_SELECTORS)
            await page.wait_for_load_state("domcontentloaded")
            await wait_reading_time(page)
            await _check_restriction(page, allow_login_page=False)
            logger.info("Instagram login successful.")
        finally:
            await page.close()

    async def _looks_logged_in(self, page: Page) -> bool:
        for selector in _HOME_INDICATORS:
            try:
                if await page.locator(selector).count() > 0:
                    return True
            except Exception:
                continue
        return False

    async def _fetch_search(self, url: str) -> SocialFetchResult:
        page = await self._context.new_page()
        try:
            logger.info("Visiting Instagram search: %s", url)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await wait_reading_time(page)
            await _check_restriction(page, allow_login_page=False)
            for _ in range(random.randint(1, 2)):
                await human_scroll(page, "down")
                await asyncio.sleep(random.uniform(0.4, 1.0))

            results = await _extract_profile_links(page)
            query = parse_qs(urlparse(url).query).get("q", ["Instagram people search"])[0]
            html = build_search_html(self.platform, query, results)
            return SocialFetchResult(
                final_url=page.url,
                title="Instagram People Search",
                page_kind="search_results",
                html=html,
                extracted_data={"results": results},
            )
        finally:
            await page.close()

    async def _fetch_profile(self, url: str) -> SocialFetchResult:
        page = await self._context.new_page()
        try:
            logger.info("Visiting Instagram profile: %s", url)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await wait_reading_time(page)
            await _check_restriction(page, allow_login_page=False)
            for _ in range(random.randint(1, 2)):
                await human_scroll(page, "down")
                await asyncio.sleep(random.uniform(0.3, 0.9))

            handle = _handle_from_url(page.url)
            meta_title = await _meta_content(page, "meta[property='og:title']")
            name = _name_from_meta_title(meta_title) or await first_text(page, _PROFILE_NAME_SELECTORS) or handle
            job_title = await first_text(page, _PROFILE_BIO_SELECTORS)
            website = await first_href(page, _PROFILE_WEBSITE_SELECTORS)
            data = {
                "name": name,
                "job_title": job_title,
                "company": None,
                "email": None,
                "phone": None,
                "website": website,
                "social_media": page.url,
            }
            html = build_profile_html(data)
            return SocialFetchResult(
                final_url=page.url,
                title=name or "Instagram profile",
                page_kind="profile",
                html=html,
                extracted_data=data,
            )
        finally:
            await page.close()


async def _type_first_available(page: Page, selectors: list[str], value: str) -> None:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0:
                await human_type(page, selector, value)
                return
        except Exception:
            continue
    raise RuntimeError("No matching Instagram login input field was found.")


async def _click_first_available(page: Page, selectors: list[str]) -> None:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0:
                bbox = await locator.bounding_box()
                if bbox:
                    await human_mouse_move(page, bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
                await locator.click()
                return
        except Exception:
            continue
    raise RuntimeError("No matching Instagram login button was found.")


async def _extract_profile_links(page: Page) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for selector in _PROFILE_LINK_SELECTORS:
        links = page.locator(selector)
        try:
            count = await links.count()
        except Exception:
            continue
        for idx in range(min(count, 30)):
            try:
                href = await links.nth(idx).get_attribute("href")
            except Exception:
                continue
            if not href or not _looks_like_profile_path(href):
                continue
            url = href if href.startswith("http") else f"https://www.instagram.com{href}"
            if url in seen_urls:
                continue
            handle = _handle_from_url(url) or url
            results.append({"url": url, "name": handle, "headline": ""})
            seen_urls.add(url)
            if len(results) >= 10:
                return results
    return results


def _looks_like_profile_path(path_or_url: str) -> bool:
    parsed = urlparse(path_or_url)
    path = parsed.path or path_or_url
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) != 1:
        return False
    handle = segments[0].lower()
    return handle not in _RESERVED_PATHS


def _handle_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) == 1 and segments[0].lower() not in _RESERVED_PATHS:
        return segments[0]
    return None


async def _meta_content(page: Page, selector: str) -> str | None:
    try:
        value = await page.locator(selector).first.get_attribute("content")
        if value and value.strip():
            return value.strip()
    except Exception:
        return None
    return None


def _name_from_meta_title(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s*\(@[^)]+\).*", "", value).strip()
    return cleaned or None


async def _check_restriction(page: Page, allow_login_page: bool) -> None:
    current_url = page.url.lower()
    for fragment in _CHALLENGE_URLS:
        if fragment == "/accounts/login/" and allow_login_page:
            continue
        if fragment in current_url:
            message = f"Instagram restriction detected at {current_url}"
            logger.warning(message)
            await send_alert(f"Alert: {message}")
            raise RestrictionDetected(message)

    for selector in _CAPTCHA_SELECTORS:
        try:
            if await page.locator(selector).count() > 0:
                message = "Instagram captcha detected."
                logger.warning(message)
                await send_alert(f"Alert: {message}")
                raise RestrictionDetected(message)
        except RestrictionDetected:
            raise
        except Exception:
            continue

    try:
        body_text = (await page.evaluate("document.body.innerText") or "").lower()
    except Exception:
        body_text = ""
    if any(cue in body_text for cue in _CHALLENGE_TEXT_CUES):
        message = "Instagram security or identity challenge detected."
        logger.warning(message)
        await send_alert(f"Alert: {message}")
        raise RestrictionDetected(message)

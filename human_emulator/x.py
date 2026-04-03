"""X/Twitter adapter for the multi-platform human emulator."""

from __future__ import annotations

import asyncio
import logging
import random
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

_LOGIN_URL = "https://x.com/i/flow/login"
_HOME_INDICATORS = [
    "a[data-testid='AppTabBar_Home_Link']",
    "[data-testid='SideNav_NewTweet_Button']",
]
_IDENTIFIER_SELECTORS = ["input[autocomplete='username']", "input[name='text']"]
_PASSWORD_SELECTORS = ["input[name='password']"]
_NEXT_BUTTON_SELECTORS = ["button:has-text('Next')", "div[role='button']:has-text('Next')"]
_LOGIN_BUTTON_SELECTORS = ["button:has-text('Log in')", "div[role='button']:has-text('Log in')"]
_EMAIL_VERIFY_SELECTORS = ["input[data-testid='ocfEnterTextTextInput']", "input[name='text']"]

_PROFILE_NAME_SELECTORS = ["div[data-testid='UserName'] span", "h2[role='heading'] span"]
_PROFILE_BIO_SELECTORS = ["div[data-testid='UserDescription']", "div[data-testid='UserProfileHeader_Items']"]
_PROFILE_WEBSITE_SELECTORS = ["a[data-testid='UserUrl']", "a[href^='http']"]

_SEARCH_USER_CELLS = ["div[data-testid='UserCell']", "section[role='region'] div[data-testid='cellInnerDiv']"]
_SEARCH_USER_LINKS = ["a[href^='/'][role='link']", "a[href^='/']"]

_CHALLENGE_URLS = ["/account/access", "/i/flow", "/account/login_challenge"]
_CHALLENGE_TEXT_CUES = ["enter your phone number", "confirm your identity", "suspicious activity", "security challenge"]
_CAPTCHA_SELECTORS = ["iframe[title*='captcha']", "input[name='captcha_response']"]


class XAdapter(SocialAdapter):
    """Code-driven X adapter."""

    platform = "x"
    domains = ("x.com", "twitter.com")

    def __init__(self, context, state, client_id: str):
        super().__init__(context, state, client_id)
        self._rhythm = SessionRhythmManager()

    async def fetch(self, url: str) -> SocialFetchResult:
        await self._rhythm.maybe_take_break()
        await self._rhythm.pace_variation()
        await self.ensure_logged_in()

        parsed = urlparse(url)
        if "/search" in parsed.path:
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
                logger.info("X session already active.")
                return

            await _check_restriction(page)
            await _type_first_available(page, _IDENTIFIER_SELECTORS, creds["username"] or "")
            await _click_first_available(page, _NEXT_BUTTON_SELECTORS)
            await asyncio.sleep(random.uniform(1.0, 2.0))

            await self._handle_secondary_identity(page, creds.get("email"))
            await _type_first_available(page, _PASSWORD_SELECTORS, creds["password"] or "")
            await _click_first_available(page, _LOGIN_BUTTON_SELECTORS)
            await page.wait_for_load_state("domcontentloaded")
            await wait_reading_time(page)
            await _check_restriction(page)
            logger.info("X login successful.")
        finally:
            await page.close()

    async def _looks_logged_in(self, page: Page) -> bool:
        for selector in _HOME_INDICATORS:
            try:
                if await page.locator(selector).count() > 0:
                    return True
            except Exception:
                continue
        return "/home" in page.url

    async def _handle_secondary_identity(self, page: Page, email: str | None) -> None:
        if not email:
            return
        try:
            for selector in _EMAIL_VERIFY_SELECTORS:
                field = page.locator(selector).first
                if await field.count() > 0:
                    await human_type(page, selector, email)
                    await _click_first_available(page, _NEXT_BUTTON_SELECTORS)
                    await asyncio.sleep(random.uniform(0.8, 1.5))
                    break
        except Exception:
            return

    async def _fetch_search(self, url: str) -> SocialFetchResult:
        page = await self._context.new_page()
        try:
            logger.info("Visiting X search: %s", url)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await wait_reading_time(page)
            await _check_restriction(page)
            for _ in range(random.randint(1, 2)):
                await human_scroll(page, "down")
                await asyncio.sleep(random.uniform(0.4, 1.0))

            results = await _extract_search_results(page)
            query = parse_qs(urlparse(url).query).get("q", ["X people search"])[0]
            html = build_search_html(self.platform, query, results)
            return SocialFetchResult(
                final_url=page.url,
                title="X User Search",
                page_kind="search_results",
                html=html,
                extracted_data={"results": results},
            )
        finally:
            await page.close()

    async def _fetch_profile(self, url: str) -> SocialFetchResult:
        page = await self._context.new_page()
        try:
            logger.info("Visiting X profile: %s", url)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await wait_reading_time(page)
            await _check_restriction(page)
            for _ in range(random.randint(1, 3)):
                await human_scroll(page, "down")
                await asyncio.sleep(random.uniform(0.3, 0.9))

            name = await first_text(page, _PROFILE_NAME_SELECTORS)
            job_title = await first_text(page, _PROFILE_BIO_SELECTORS)
            website = await first_href(page, _PROFILE_WEBSITE_SELECTORS)
            handle = _handle_from_url(page.url)
            data = {
                "name": name or handle,
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
                title=(name or handle or "X profile"),
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
    raise RuntimeError("No matching X login input field was found.")


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
    raise RuntimeError("No matching X login button was found.")


async def _extract_search_results(page: Page) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for cell_selector in _SEARCH_USER_CELLS:
        cells = page.locator(cell_selector)
        try:
            count = await cells.count()
        except Exception:
            continue
        if count == 0:
            continue
        for idx in range(min(count, 10)):
            cell = cells.nth(idx)
            url = None
            for link_selector in _SEARCH_USER_LINKS:
                try:
                    href = await cell.locator(link_selector).first.get_attribute("href")
                    if href and _looks_like_handle_path(href):
                        url = href if href.startswith("http") else f"https://x.com{href}"
                        break
                except Exception:
                    continue
            if not url or url in seen_urls:
                continue
            name = await _search_cell_name(cell)
            headline = await _search_cell_headline(cell)
            results.append({"url": url, "name": name or url, "headline": headline or ""})
            seen_urls.add(url)
        if results:
            break
    return results


async def _search_cell_name(cell) -> str | None:  # noqa: ANN001
    try:
        name = await cell.locator("div[data-testid='UserName'] span").first.text_content()
        if name and name.strip():
            return name.strip()
    except Exception:
        return None
    return None


async def _search_cell_headline(cell) -> str | None:  # noqa: ANN001
    try:
        text = await cell.locator("div[data-testid='UserDescription']").first.text_content()
        if text and text.strip():
            return text.strip()
    except Exception:
        return None
    return None


def _looks_like_handle_path(path_or_url: str) -> bool:
    parsed = urlparse(path_or_url)
    path = parsed.path or path_or_url
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) != 1:
        return False
    return segments[0].lower() not in {"home", "explore", "messages", "notifications", "search", "login", "i"}


def _handle_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) == 1:
        return segments[0]
    return None


async def _check_restriction(page: Page) -> None:
    current_url = page.url.lower()
    if any(fragment in current_url for fragment in _CHALLENGE_URLS):
        message = f"X checkpoint detected at {current_url}"
        logger.warning(message)
        await send_alert(f"⚠️ {message}")
        raise RestrictionDetected(message)

    for selector in _CAPTCHA_SELECTORS:
        try:
            if await page.locator(selector).count() > 0:
                message = "X captcha detected."
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
    if any(cue in body_text for cue in _CHALLENGE_TEXT_CUES):
        message = "X security or identity challenge detected."
        logger.warning(message)
        await send_alert(f"⚠️ {message}")
        raise RestrictionDetected(message)


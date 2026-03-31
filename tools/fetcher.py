"""smart_fetch — tries httpx first; falls back to Playwright for JS pages.

Rules:
  - For static pages (no JS needed), httpx is tried first.
  - If the httpx response is empty or too short (< 500 chars of visible text),
    Playwright is used automatically.
  - Social-media URLs are never fetched here — the caller must route them to the
    human emulator before reaching smart_fetch.
"""

import logging
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import BrowserContext

logger = logging.getLogger(__name__)

_MIN_TEXT_LENGTH = 500
_HTTPX_TIMEOUT   = 20
_HEADERS         = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


async def smart_fetch(
    url: str,
    needs_javascript: bool,
    get_context,  # Callable[[], Awaitable[BrowserContext]]
) -> str:
    """Fetch a URL and return the full HTML string.

    Args:
        url:              Target URL.
        needs_javascript: If True, skip httpx and go straight to Playwright.
        get_context:      Async callable that returns a fresh BrowserContext.

    Returns:
        HTML string of the page.
    """
    if _should_force_browser(url):
        logger.debug("smart_fetch: forcing Playwright for %s", url)
        return await _fetch_playwright(url, get_context)

    if not needs_javascript:
        html = await _fetch_httpx(url)
        if (
            html
            and _visible_text_length(html) >= _MIN_TEXT_LENGTH
            and not _looks_like_client_shell(url, html)
        ):
            logger.debug("smart_fetch: httpx succeeded for %s", url)
            return html
        logger.debug(
            "smart_fetch: httpx result unusable (%d chars text) — falling back to Playwright for %s",
            _visible_text_length(html) if html else 0,
            url,
        )

    return await _fetch_playwright(url, get_context)


async def _fetch_httpx(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            timeout=_HTTPX_TIMEOUT,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.text
            logger.debug("httpx returned %s for %s", resp.status_code, url)
    except Exception as exc:
        logger.debug("httpx failed for %s: %s", url, exc)
    return None


async def _fetch_playwright(url: str, get_context) -> str:
    ctx = await get_context()
    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await _wait_for_page_content(page, url)
        html = await page.content()
        return html
    finally:
        await page.close()
        await ctx.close()


def _visible_text_length(html: str) -> int:
    """Rough count of visible text characters in the HTML."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        return len(soup.get_text())
    except Exception:
        return 0


def _should_force_browser(url: str) -> bool:
    """Return True for URLs that are known to require JS rendering."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parse_qs(parsed.query)

    if "github.com" in host and path == "/search":
        search_type = (query.get("type") or [""])[0].lower()
        if search_type in {"users", "repositories", "issues", "pullrequests"}:
            return True

    return False


def _looks_like_client_shell(url: str, html: str) -> bool:
    """Detect HTML that is mostly app chrome rather than usable content."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parse_qs(parsed.query)
    html_lower = html.lower()

    if "github.com" in host and path == "/search":
        search_type = (query.get("type") or [""])[0].lower()
        if search_type == "users":
            if "user search results" in html_lower and "/search?" in html_lower:
                missing_result_markers = [
                    'data-testid="results-list"',
                    'data-hovercard-type="user"',
                    '/users/',
                ]
                return not any(marker in html_lower for marker in missing_result_markers)

    return False


async def _wait_for_page_content(page, url: str) -> None:
    """Give JS-heavy pages a chance to populate useful content."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parse_qs(parsed.query)

    try:
        await page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        logger.debug("networkidle wait timed out for %s", url)

    if "github.com" in host and path == "/search":
        search_type = (query.get("type") or [""])[0].lower()
        if search_type == "users":
            selectors = [
                '[data-testid="results-list"]',
                '[data-hovercard-type="user"]',
                'a[href^="/"][data-hovercard-type="user"]',
            ]
            for selector in selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5_000)
                    return
                except Exception:
                    continue

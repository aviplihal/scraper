"""smart_fetch — tries httpx first; falls back to Playwright for JS pages.

Rules:
  - For static pages (no JS needed), httpx is tried first.
  - If the httpx response is empty or too short (< 500 chars of visible text),
    Playwright is used automatically.
  - Social-media URLs are never fetched here — the caller must route them to the
    human emulator before reaching smart_fetch.
"""

import logging

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
    if not needs_javascript:
        html = await _fetch_httpx(url)
        if html and _visible_text_length(html) >= _MIN_TEXT_LENGTH:
            logger.debug("smart_fetch: httpx succeeded for %s", url)
            return html
        logger.debug(
            "smart_fetch: httpx result too short (%d chars text) — falling back to Playwright for %s",
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

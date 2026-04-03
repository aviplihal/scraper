"""Shared social-platform adapter primitives for the human emulator."""

from __future__ import annotations

import html
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Page

from human_emulator.state import EmulatorState


@dataclass(slots=True)
class SocialFetchResult:
    """Normalized social fetch result used by the generic tool flow."""

    final_url: str
    title: str
    page_kind: str
    html: str
    extracted_data: dict[str, Any] | None = None


class RestrictionDetected(Exception):
    """Raised when a platform shows a checkpoint, captcha, or auth challenge."""


class SocialAdapter(ABC):
    """Base class for social-platform emulation adapters."""

    platform: str = ""
    domains: tuple[str, ...] = ()

    def __init__(self, context: BrowserContext, state: EmulatorState, client_id: str):
        self._context = context
        self._state = state
        self._client_id = client_id
        self._logged_in = False

    @classmethod
    def matches_url(cls, url: str) -> bool:
        host = _normalize_host(urlparse(url).netloc)
        return any(host == domain or host.endswith(f".{domain}") for domain in cls.domains)

    @classmethod
    def username_env(cls) -> str:
        return f"SOCIAL_{cls.platform.upper()}_USERNAME"

    @classmethod
    def password_env(cls) -> str:
        return f"SOCIAL_{cls.platform.upper()}_PASSWORD"

    @classmethod
    def email_env(cls) -> str:
        return f"SOCIAL_{cls.platform.upper()}_EMAIL"

    @classmethod
    def credentials_present(cls) -> bool:
        return bool(os.environ.get(cls.username_env())) and bool(os.environ.get(cls.password_env()))

    @classmethod
    def credentials(cls) -> dict[str, str | None]:
        return {
            "username": os.environ.get(cls.username_env()),
            "password": os.environ.get(cls.password_env()),
            "email": os.environ.get(cls.email_env()),
        }

    async def preflight(self) -> tuple[str, str]:
        """Validate or establish a logged-in session for this platform."""
        if not self.credentials_present():
            return "missing_credentials", "Missing required platform credentials."

        paused, until = self._state.is_paused(self.platform)
        if paused:
            return "paused", f"Paused until {until.isoformat() if until else 'unknown'}."

        try:
            await self.ensure_logged_in()
        except RestrictionDetected as exc:
            self._state.record_restriction(self.platform)
            self._state.set_pause_hours(self.platform, hours=8, reason=str(exc))
            return "paused", str(exc)
        except Exception as exc:  # pragma: no cover - defensive against platform changes
            return "unavailable", str(exc)

        return "active", "Session ready."

    async def ensure_logged_in(self) -> None:
        """Ensure the platform session is authenticated."""
        if self._logged_in:
            return
        await self._login()
        self._logged_in = True

    @abstractmethod
    async def fetch(self, url: str) -> SocialFetchResult:
        """Fetch a social URL and normalize it into synthetic HTML."""

    @abstractmethod
    async def _login(self) -> None:
        """Perform platform-specific login flow."""


def build_search_html(platform: str, query_label: str, results: list[dict[str, Any]]) -> str:
    """Return synthetic HTML for discovery/search pages."""
    cards: list[str] = []
    for item in results:
        url = html.escape(str(item.get("url", "")), quote=True)
        name = html.escape(str(item.get("name", "")).strip() or url)
        headline = html.escape(str(item.get("headline", "")).strip())
        company = html.escape(str(item.get("company", "")).strip())
        details = " ".join(part for part in (headline, company) if part).strip()
        cards.append(
            "<article class='user-card'>"
            f"<a href='{url}' data-hovercard-type='user'>{name}</a>"
            f"<div class='headline'>{details}</div>"
            "</article>"
        )

    body = "".join(cards) or "<div class='empty'>No matching social profiles were found.</div>"
    title = html.escape(query_label)
    return (
        "<html><head>"
        f"<title>{platform.title()} search results for {title}</title>"
        "</head><body>"
        f"<h1>{platform.title()} People Search</h1>"
        f"<div class='search-results'>{body}</div>"
        "</body></html>"
    )


def build_profile_html(data: dict[str, Any]) -> str:
    """Return synthetic HTML for normalized profile/detail pages."""
    name = html.escape(str(data.get("name") or "Unknown"), quote=False)
    job_title = html.escape(str(data.get("job_title") or ""), quote=False)
    company = html.escape(str(data.get("company") or ""), quote=False)
    email = html.escape(str(data.get("email") or ""), quote=True)
    phone = html.escape(str(data.get("phone") or ""), quote=True)
    social_media = html.escape(str(data.get("social_media") or ""), quote=True)
    website = html.escape(str(data.get("website") or ""), quote=True)

    parts = [
        "<html><body>",
        f"<h1>{name}</h1>",
    ]
    if job_title:
        parts.append(f"<div class='headline'>{job_title}</div>")
        parts.append(f"<div class='job-title'>{job_title}</div>")
    if company:
        parts.append(f"<div class='company'>{company}</div>")
    if email:
        parts.append(f"<a href='mailto:{email}'>{email}</a>")
    if phone:
        parts.append(f"<a href='tel:{phone}'>{phone}</a>")
    if website:
        parts.append(f"<a href='{website}' class='website'>{website}</a>")
    if social_media:
        parts.append(f"<a href='{social_media}' class='social'>{social_media}</a>")
    parts.append("</body></html>")
    return "".join(parts)


async def first_text(page: Page, selectors: list[str]) -> str | None:
    """Return the first non-empty text value among the given selectors."""
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0:
                text = await locator.text_content()
                if text and text.strip():
                    return text.strip()
        except Exception:
            continue
    return None


async def first_href(page: Page, selectors: list[str]) -> str | None:
    """Return the first non-empty href attribute among the given selectors."""
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0:
                href = await locator.get_attribute("href")
                if href and href.strip():
                    return href.strip()
        except Exception:
            continue
    return None


def _normalize_host(host: str) -> str:
    host = host.lower().split(":", 1)[0]
    return host[4:] if host.startswith("www.") else host


"""Discovery helpers for classifying fetched pages and extracting links."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

_BLOCKED_MARKERS = (
    "just a moment",
    "verify you are human",
    "enable javascript and cookies",
    "security check",
    "access denied",
    "authwall",
    "captcha",
    "protect glassdoor",
    "we must verify your session",
)

_AUTH_REQUIRED_MARKERS = (
    "sign in or sign up before continuing",
    "sign in to gitlab",
    "sign in to continue",
    "log in to continue",
    "authentication required",
)

_NOT_FOUND_MARKERS = (
    "page not found",
    "404",
    "doesn't exist",
    "does not exist",
    "not found",
)

_JOB_MARKERS = (
    "search jobs",
    "job search",
    "employment",
    "career advice",
    "startup jobs",
    "job listings",
    "best tech jobs",
)

_COMPANY_DIRECTORY_MARKERS = (
    "startup directory",
    "company directory",
    "portfolio companies",
    "browse companies",
    "startup database",
)

_COMPANY_PAGE_MARKERS = (
    "our team",
    "leadership team",
    "management team",
    "founding team",
    "meet the team",
    "about us",
)

_ARTICLE_MARKERS = (
    "technology news",
    "startup and technology news",
    "latest news",
    "editorial",
    "press release",
)

_SEARCH_MARKERS = (
    "search results",
    "results found",
    "showing results",
    "people search",
    "user search results",
)

_DIRECTORY_SEGMENTS = {
    "people",
    "users",
    "members",
    "directory",
    "experts",
    "profiles",
}

_COMPANY_DIRECTORY_SEGMENTS = {
    "companies",
    "company",
    "startups",
    "portfolio",
    "organizations",
    "organization",
    "ventures",
}

_COMPANY_PAGE_SEGMENTS = {
    "team",
    "about",
    "leadership",
    "management",
    "staff",
    "founders",
    "bios",
    "bio",
}

_PROFILE_SEGMENTS = {
    "profile",
    "profiles",
    "users",
    "people",
    "members",
    "staff",
    "team",
}

_STOP_LINK_TEXT = {
    "sign in",
    "log in",
    "login",
    "sign up",
    "signup",
    "register",
    "home",
    "pricing",
    "features",
    "docs",
    "documentation",
    "blog",
    "privacy",
    "terms",
    "about",
    "support",
    "help",
    "contact",
    "careers",
    "jobs",
}

_STOP_PATH_SEGMENTS = {
    "login",
    "signin",
    "signup",
    "register",
    "pricing",
    "features",
    "docs",
    "blog",
    "privacy",
    "terms",
    "support",
    "help",
    "careers",
    "jobs",
    "search",
}

_PEOPLE_DISCOVERY_TERMS = {
    "team",
    "leadership",
    "management",
    "staff",
    "founders",
    "speaker",
    "speakers",
    "bios",
    "bio",
    "people",
    "members",
}

_LOW_VALUE_TERMS = {
    "jobs",
    "careers",
    "pricing",
    "news",
    "blog",
    "product",
    "companies",
    "funding",
    "search",
    "login",
}

_GITHUB_RESERVED_SEGMENTS = {
    "",
    "accelerator",
    "about",
    "collections",
    "contact",
    "customer-stories",
    "enterprise",
    "events",
    "explore",
    "features",
    "issues",
    "login",
    "marketplace",
    "mcp",
    "new",
    "notifications",
    "orgs",
    "organizations",
    "partners",
    "pricing",
    "premium-support",
    "pulls",
    "resources",
    "search",
    "security",
    "sessions",
    "settings",
    "signup",
    "site",
    "solutions",
    "sponsors",
    "team",
    "trust-center",
    "topics",
    "trending",
    "why-github",
}

_DUCKDUCKGO_HOSTS = {"duckduckgo.com", "html.duckduckgo.com"}
_DISCOVERY_PROFILE_DOMAINS = {"github.com", "gitlab.com", "linkedin.com", "x.com", "twitter.com"}


@dataclass(slots=True)
class PageInfo:
    """Classification metadata for a fetched page."""

    final_url: str
    title: str
    page_kind: str


def classify_page(url: str, final_url: str, html: str) -> PageInfo:
    """Classify a fetched page so the agent can choose the right next tool."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text(separator=" ", strip=True)
    text_lower = text.lower()
    title_lower = title.lower()
    parsed = urlparse(final_url or url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    host = _normalize_host(parsed.netloc)
    path_segments = [segment for segment in path.split("/") if segment]

    if _looks_like_duckduckgo_error_page(host, path, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="blocked")

    if _contains_marker(title_lower, text_lower, _BLOCKED_MARKERS):
        return PageInfo(final_url=final_url or url, title=title, page_kind="blocked")

    if _looks_like_auth_wall(host, path, path_segments, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="blocked")

    if _contains_marker(title_lower, text_lower, _NOT_FOUND_MARKERS):
        return PageInfo(final_url=final_url or url, title=title, page_kind="not_found")

    if _looks_like_duckduckgo_results(host, path, query, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="search_results")

    if _looks_like_job_board(path, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="job_board")

    if _looks_like_search_results(path, query, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="search_results")

    if _looks_like_company_directory(path_segments, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="company_directory")

    if _looks_like_directory(path_segments, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="directory")

    if _looks_like_profile(host, path_segments, soup, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="profile")

    if _looks_like_company_page(path_segments, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="company_page")

    if _looks_like_article_or_news(host, path_segments, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="article_or_news")

    if _looks_like_landing_page(path_segments, title_lower, text_lower):
        return PageInfo(final_url=final_url or url, title=title, page_kind="landing_page")

    return PageInfo(final_url=final_url or url, title=title, page_kind="unknown")


def build_preview(html: str, limit: int = 3000) -> str:
    """Build a text preview from visible page text."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)[:limit]


def extract_links(
    html: str,
    base_url: str,
    selector: str | None = None,
    limit: int = 25,
    same_domain_only: bool = False,
) -> list[dict[str, str]]:
    """Extract and rank useful navigation links from a page."""
    soup = BeautifulSoup(html, "html.parser")
    anchors = _select_anchors(soup, base_url, selector)
    base_host = _normalize_host(urlparse(base_url).netloc)
    github_user_search = _is_github_user_search(base_url)
    preferred_domains = _preferred_profile_domains_for_search(base_url)
    ranked_links: list[tuple[int, str, str]] = []

    for anchor in anchors:
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        normalized_url = unwrap_discovery_redirect_url(urljoin(base_url, href), base_url=base_url)
        if not normalized_url:
            continue

        target_host = _normalize_host(urlparse(normalized_url).netloc)
        if same_domain_only and not _same_or_subdomain(target_host, base_host):
            continue

        text = anchor.get_text(separator=" ", strip=True)
        if github_user_search and not _is_github_profile_url(normalized_url):
            continue
        if base_host in _DUCKDUCKGO_HOSTS:
            if target_host in _DUCKDUCKGO_HOSTS:
                continue
            if preferred_domains and not any(_same_or_subdomain(target_host, domain) for domain in preferred_domains):
                continue
        score = _score_link(anchor, normalized_url, text, base_url)
        if selector is None and score <= 0:
            continue

        ranked_links.append((score, normalized_url, text))

    ranked_links.sort(key=lambda item: (-item[0], item[1]))

    links: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for _, url, text in ranked_links:
        if url in seen_urls:
            continue
        links.append({"url": url, "text": text})
        seen_urls.add(url)
        if len(links) >= limit:
            break

    return links


def _contains_marker(title_lower: str, text_lower: str, markers: tuple[str, ...]) -> bool:
    return any(marker in title_lower or marker in text_lower for marker in markers)


def unwrap_discovery_redirect_url(url: str, base_url: str | None = None) -> str:
    """Resolve known discovery-result wrapper URLs to their outbound targets."""
    normalized = _normalize_url(urljoin(base_url, url) if base_url else url)
    if not normalized:
        return ""

    parsed = urlparse(normalized)
    host = _normalize_host(parsed.netloc)
    if host not in _DUCKDUCKGO_HOSTS:
        return normalized

    query = parse_qs(parsed.query)
    uddg_values = query.get("uddg") or query.get("rut")
    if "uddg" in query and query["uddg"]:
        decoded = _normalize_url(unquote(query["uddg"][0]))
        return decoded or normalized

    if parsed.path.lower() == "/l" and not uddg_values:
        return normalized

    return normalized


def _looks_like_auth_wall(
    host: str,
    path: str,
    path_segments: list[str],
    title_lower: str,
    text_lower: str,
) -> bool:
    """Return True when a page is clearly a login/auth wall instead of usable content."""
    if _contains_marker(title_lower, text_lower, _AUTH_REQUIRED_MARKERS):
        return True
    if path in {"/users/sign_in", "/login", "/signin", "/sign_in"}:
        return True
    if any(segment in {"login", "signin", "sign_in"} for segment in path_segments):
        return True
    if host == "gitlab.com" and path.startswith("/users/sign_in"):
        return True
    return title_lower.startswith(("sign in", "log in", "login"))


def _looks_like_duckduckgo_error_page(host: str, path: str, title_lower: str, text_lower: str) -> bool:
    """Return True for DuckDuckGo anti-bot/error pages that should be treated as blocked."""
    if host not in _DUCKDUCKGO_HOSTS:
        return False
    if path.startswith("/static-pages/418"):
        return True
    return "duckduckgo" in title_lower and (
        "error getting results" in text_lower
        or "418" in title_lower
        or "if this persists, please email us" in text_lower
    )


def _looks_like_duckduckgo_results(
    host: str,
    path: str,
    query: str,
    title_lower: str,
    text_lower: str,
) -> bool:
    """Return True for DuckDuckGo HTML search result pages before generic job-board matching."""
    if host not in _DUCKDUCKGO_HOSTS:
        return False
    if path == "/html" and ("q=" in query or "search only" in text_lower):
        return True
    return "duckduckgo" in title_lower and "search results" in text_lower


def _looks_like_job_board(path: str, title_lower: str, text_lower: str) -> bool:
    if "/jobs" in path or "/job" in path or "/careers" in path or "/career" in path:
        return True
    return _contains_marker(title_lower, text_lower, _JOB_MARKERS)


def _looks_like_search_results(path: str, query: str, title_lower: str, text_lower: str) -> bool:
    if "/search" in path or "search=" in query or "q=" in query:
        return True
    return _contains_marker(title_lower, text_lower, _SEARCH_MARKERS)


def _looks_like_directory(path_segments: list[str], title_lower: str, text_lower: str) -> bool:
    if len(path_segments) == 1 and path_segments[0] in _DIRECTORY_SEGMENTS:
        return True
    if any(phrase in title_lower for phrase in ("directory", "members", "people")):
        return True
    return any(
        phrase in text_lower
        for phrase in ("people directory", "staff directory", "team directory", "browse people")
    )


def _looks_like_company_directory(path_segments: list[str], title_lower: str, text_lower: str) -> bool:
    if len(path_segments) == 1 and path_segments[0] in _COMPANY_DIRECTORY_SEGMENTS:
        return True
    if _contains_marker(title_lower, text_lower, _COMPANY_DIRECTORY_MARKERS):
        return True
    return any(
        phrase in text_lower
        for phrase in ("company directory", "startup directory", "portfolio companies", "browse startups")
    )


def _looks_like_profile(
    host: str,
    path_segments: list[str],
    soup: BeautifulSoup,
    title_lower: str,
    text_lower: str,
) -> bool:
    if host == "github.com" and len(path_segments) == 1:
        return path_segments[0] not in _GITHUB_RESERVED_SEGMENTS

    if len(path_segments) == 1 and path_segments[0] in _COMPANY_PAGE_SEGMENTS:
        return False

    if soup.select_one("meta[property='og:type'][content='profile']"):
        return True

    if soup.select_one("a[href^='mailto:']") or soup.select_one("a[href^='tel:']"):
        return True

    if soup.select_one("[itemprop='name']") or soup.select_one("[itemprop='worksFor']"):
        return True

    if (
        len(path_segments) >= 2
        and any(segment in _PROFILE_SEGMENTS for segment in path_segments)
        and path_segments[-1] not in _COMPANY_PAGE_SEGMENTS
    ):
        return True

    has_person_like_title = any(word in title_lower for word in ("founder", "engineer", "developer", "marketer"))
    has_contact_or_org_signal = "@" in text_lower or "company" in text_lower or "works at" in text_lower
    return has_person_like_title and has_contact_or_org_signal


def _looks_like_company_page(path_segments: list[str], title_lower: str, text_lower: str) -> bool:
    if len(path_segments) == 1 and path_segments[0] in _COMPANY_PAGE_SEGMENTS:
        return True
    if len(path_segments) >= 2 and path_segments[0] in _COMPANY_DIRECTORY_SEGMENTS:
        return True
    if any(segment in _COMPANY_PAGE_SEGMENTS for segment in path_segments):
        return True
    if _contains_marker(title_lower, text_lower, _COMPANY_PAGE_MARKERS):
        return True
    return any(phrase in title_lower for phrase in ("team", "leadership", "management", "about"))


def _looks_like_article_or_news(
    host: str,
    path_segments: list[str],
    title_lower: str,
    text_lower: str,
) -> bool:
    if host in {"techcrunch.com", "venturebeat.com"}:
        return True
    if any(segment in {"blog", "news", "press", "article", "articles"} for segment in path_segments):
        return True
    return _contains_marker(title_lower, text_lower, _ARTICLE_MARKERS)


def _looks_like_landing_page(path_segments: list[str], title_lower: str, text_lower: str) -> bool:
    if path_segments:
        return False
    return not any(
        phrase in title_lower or phrase in text_lower
        for phrase in (
            "directory",
            "people",
            "team",
            "leadership",
            "staff",
            "founder",
            "speaker",
        )
    )


def _select_anchors(soup: BeautifulSoup, base_url: str, selector: str | None):
    if selector:
        anchors = []
        for element in soup.select(selector):
            if getattr(element, "name", None) == "a" and element.get("href"):
                anchors.append(element)
                continue
            anchors.extend(anchor for anchor in element.select("a[href]"))
        return anchors

    parsed = urlparse(base_url)
    host = _normalize_host(parsed.netloc)
    path = parsed.path.lower()
    query = parsed.query.lower()

    if host == "github.com" and path == "/search" and "type=users" in query:
        specific = soup.select("a[data-hovercard-type='user'][href]")
        if specific:
            return specific

    if host in _DUCKDUCKGO_HOSTS and path == "/html":
        specific = soup.select("a.result__a[href], a[data-testid='result-title-a'][href]")
        if specific:
            return specific

    return soup.select("a[href]")


def _is_github_user_search(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return _normalize_host(parsed.netloc) == "github.com" and parsed.path.lower() == "/search" and "type=users" in parsed.query.lower()


def _is_github_profile_url(url: str) -> bool:
    parsed = urlparse(url)
    if _normalize_host(parsed.netloc) != "github.com":
        return False
    path_segments = [segment for segment in parsed.path.split("/") if segment]
    return len(path_segments) == 1 and path_segments[0] not in _GITHUB_RESERVED_SEGMENTS


def _score_link(anchor, normalized_url: str, text: str, base_url: str) -> int:
    parsed = urlparse(normalized_url)
    host = _normalize_host(parsed.netloc)
    path_segments = [segment for segment in parsed.path.split("/") if segment]
    text_lower = text.strip().lower()
    score = 0

    if parsed.scheme not in {"http", "https"}:
        return -100

    if text_lower in _STOP_LINK_TEXT:
        return -50

    if any(segment.lower() in _STOP_PATH_SEGMENTS for segment in path_segments):
        score -= 12

    if _same_or_subdomain(host, _normalize_host(urlparse(base_url).netloc)):
        score += 2

    if anchor.get("data-hovercard-type") == "user":
        score += 12

    if host == "github.com" and len(path_segments) == 1 and path_segments[0] not in _GITHUB_RESERVED_SEGMENTS:
        score += 10
    elif host in _DISCOVERY_PROFILE_DOMAINS:
        score += 5

    if any(segment.lower() in _PROFILE_SEGMENTS for segment in path_segments):
        score += 7

    if any(term in text_lower for term in _PEOPLE_DISCOVERY_TERMS):
        score += 6

    if any(segment.lower() in _PEOPLE_DISCOVERY_TERMS for segment in path_segments):
        score += 6

    if any(word in text_lower for word in ("founder", "engineer", "developer", "marketer", "ceo", "cto")):
        score += 2

    if any(term in text_lower for term in _LOW_VALUE_TERMS):
        score -= 4

    if any(segment.lower() in _LOW_VALUE_TERMS for segment in path_segments):
        score -= 8

    if len(path_segments) == 0:
        score -= 10

    if normalized_url == _normalize_url(base_url):
        score -= 10

    return score


def _preferred_profile_domains_for_search(base_url: str) -> set[str]:
    """Infer which outbound profile domains a discovery/search page is expected to yield."""
    parsed = urlparse(base_url)
    host = _normalize_host(parsed.netloc)
    if host not in _DUCKDUCKGO_HOSTS:
        return set()

    query = parse_qs(parsed.query).get("q", [""])[0].lower()
    preferred: set[str] = set()
    for match in re.findall(r"site:([a-z0-9.-]+)", query):
        preferred.add(match.removeprefix("www."))
    for domain in _DISCOVERY_PROFILE_DOMAINS:
        if domain in query:
            preferred.add(domain)
    return preferred


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    normalized = parsed._replace(path=path, fragment="")
    return urlunparse(normalized)


def _normalize_host(host: str) -> str:
    host = host.lower().split(":", 1)[0]
    return host[4:] if host.startswith("www.") else host


def _same_or_subdomain(host: str, target_host: str) -> bool:
    host = _normalize_host(host)
    target_host = _normalize_host(target_host)
    return host == target_host or host.endswith(f".{target_host}")

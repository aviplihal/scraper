"""Curated target selection helpers for broad-mode discovery."""

from __future__ import annotations

from urllib.parse import quote_plus, urljoin, urlparse

_LEADERSHIP_TERMS = (
    "founder",
    "cofounder",
    "cto",
    "chief technology officer",
    "ceo",
    "chief executive officer",
    "head of engineering",
    "vp engineering",
    "director of engineering",
    "technical decision maker",
)

_TECHNICAL_TERMS = (
    "engineer",
    "developer",
    "architect",
    "devops",
    "programmer",
    "software",
)

_MARKETING_SALES_TERMS = (
    "marketing",
    "sales",
    "growth",
    "revops",
    "revenue",
    "demand gen",
    "business development",
)


def suggest_targets(client_config: dict, source_mode: str, limit: int = 8) -> dict:
    """Return ranked starter URLs for the current client persona."""
    website = str(client_config.get("website", "NA")).strip()
    if website and website.upper() != "NA":
        targets = _pinned_site_targets(website)
        return {
            "strategy": "pinned_site",
            "targets": targets[:limit],
        }

    strategy = _infer_strategy(client_config)
    targets = _targets_for_strategy(strategy, client_config, source_mode)
    deduped: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    for target in targets:
        url = str(target["url"])
        if url in seen_urls:
            continue
        deduped.append(target)
        seen_urls.add(url)
        if len(deduped) >= max(1, min(limit, 20)):
            break

    return {
        "strategy": strategy,
        "targets": deduped,
    }


def _infer_strategy(client_config: dict) -> str:
    combined = " ".join(
        str(client_config.get(key, "")).lower()
        for key in ("job", "job_title")
    )
    if any(term in combined for term in _LEADERSHIP_TERMS):
        return "leadership_people"
    if any(term in combined for term in _TECHNICAL_TERMS):
        return "technical_profiles"
    if any(term in combined for term in _MARKETING_SALES_TERMS):
        return "marketing_sales_people"
    return "general_people"


def _targets_for_strategy(strategy: str, client_config: dict, source_mode: str) -> list[dict[str, object]]:
    area = str(client_config.get("area", "NA")).strip()
    requested_title = str(client_config.get("job_title", "")).strip()
    if strategy == "leadership_people":
        search_terms = _dedupe_terms(
            [requested_title, "Founder", "CTO", "Head of Engineering"],
        )
        targets = [
            _github_search_target(term, area, priority)
            for priority, term in zip((100, 95, 90), search_terms[:3], strict=False)
        ]
        targets.extend(
            [
                {
                    "url": "https://www.ycombinator.com/founders",
                    "kind": "people_directory",
                    "reason": "Public founder directory aligned to startup leadership personas.",
                    "priority": 85,
                },
                {
                    "url": "https://www.ycombinator.com/people",
                    "kind": "people_directory",
                    "reason": "Public people directory aligned to startup leadership personas.",
                    "priority": 80,
                },
            ]
        )
        return targets

    if strategy == "technical_profiles":
        search_terms = _dedupe_terms(
            [requested_title, "Senior Software Engineer", "Software Engineer", "Architect"],
        )
        return [
            _github_search_target(term, area, priority)
            for priority, term in zip((100, 95, 90, 85), search_terms[:4], strict=False)
        ]

    if strategy == "marketing_sales_people":
        search_terms = _dedupe_terms(
            [requested_title, "Head of Growth", "VP Sales", "Marketing Director"],
        )
        return [
            _github_search_target(term, area, priority)
            for priority, term in zip((100, 95, 90), search_terms[:3], strict=False)
        ]

    search_terms = _dedupe_terms([requested_title, str(client_config.get("job", "")).strip()])
    return [
        _github_search_target(term, area, priority)
        for priority, term in zip((100, 95), search_terms[:2], strict=False)
        if term
    ]


def _pinned_site_targets(website: str) -> list[dict[str, object]]:
    parsed = urlparse(website if "://" in website else f"https://{website}")
    base = f"{parsed.scheme or 'https'}://{parsed.netloc or parsed.path}".rstrip("/")
    paths = [
        "",
        "/people",
        "/team",
        "/leadership",
        "/about",
        "/staff",
    ]
    return [
        {
            "url": urljoin(base + "/", path.lstrip("/")),
            "kind": "company_page" if path else "fallback",
            "reason": "Pinned site starter page.",
            "priority": 100 - idx,
        }
        for idx, path in enumerate(paths)
    ]


def _github_search_target(term: str, area: str, priority: int) -> dict[str, object]:
    query_parts = [term.strip()]
    if area and area.upper() != "NA":
        query_parts.append(f'location:"{area}"')
    query = " ".join(part for part in query_parts if part).strip()
    return {
        "url": f"https://github.com/search?q={quote_plus(query)}&type=users",
        "kind": "search_results",
        "reason": f"Public GitHub user search for '{query}'.",
        "priority": priority,
    }


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        normalized = term.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(term.strip())
    return deduped

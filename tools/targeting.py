"""Keyword-driven target selection helpers for web and social discovery."""

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

_SUPPORTED_SOCIAL_PLATFORMS = ("linkedin", "x")
_AVOID_DOMAINS = (
    "crunchbase.com",
    "wellfound.com",
    "techcrunch.com",
    "producthunt.com",
)


def suggest_targets(client_config: dict, source_mode: str, limit: int = 8) -> dict:
    """Return a keyword brief plus domain-diverse candidate targets for the run."""
    website = str(client_config.get("website", "NA")).strip()
    strategy = _infer_strategy(client_config)
    keyword_brief = _build_keyword_brief(strategy, client_config, source_mode)

    if website and website.upper() != "NA":
        candidate_targets = _dedupe_candidate_targets(
            _pinned_site_targets(website),
            limit,
        )
        allowed_domains = _ordered_domains(candidate_targets)
        return {
            "strategy": "pinned_site",
            "keyword_brief": keyword_brief,
            "source_mix": _source_mix_for_mode(source_mode),
            "candidate_targets": candidate_targets,
            "allowed_domains": allowed_domains,
            "avoid_domains": list(_AVOID_DOMAINS),
            "selection_notes": _selection_notes(source_mode, social_enabled=False),
        }

    candidate_targets = _dedupe_candidate_targets(
        _candidate_targets_for_strategy(strategy, keyword_brief, client_config, source_mode),
        limit,
    )
    allowed_domains = _ordered_domains(candidate_targets)
    social_enabled = bool(_enabled_social_platforms(client_config))
    return {
        "strategy": strategy,
        "keyword_brief": keyword_brief,
        "source_mix": _source_mix_for_mode(source_mode),
        "candidate_targets": candidate_targets,
        "allowed_domains": allowed_domains,
        "avoid_domains": list(_AVOID_DOMAINS),
        "selection_notes": _selection_notes(source_mode, social_enabled=social_enabled),
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


def _build_keyword_brief(strategy: str, client_config: dict, source_mode: str) -> dict[str, object]:
    requested_title = str(client_config.get("job_title", "")).strip()
    requested_job = str(client_config.get("job", "")).strip()
    strategy_terms = _strategy_terms(strategy, requested_title, client_config)
    primary_terms = _dedupe_terms([requested_title or requested_job or strategy_terms[0]])
    secondary_terms = [
        term
        for term in strategy_terms
        if term.lower() not in {value.lower() for value in primary_terms}
    ]
    return {
        "primary_terms": primary_terms[:2],
        "secondary_terms": secondary_terms[:3],
        "area": str(client_config.get("area", "NA")).strip() or "NA",
        "source_mode": source_mode,
    }


def _candidate_targets_for_strategy(
    strategy: str,
    keyword_brief: dict[str, object],
    client_config: dict,
    source_mode: str,
) -> list[dict[str, object]]:
    primary_terms = list(keyword_brief.get("primary_terms", []))
    secondary_terms = list(keyword_brief.get("secondary_terms", []))
    search_terms = _dedupe_terms([*primary_terms, *secondary_terms])
    area = str(keyword_brief.get("area", "NA"))

    web_groups = _web_target_groups(strategy, search_terms, area)
    social_groups = _social_target_groups(search_terms, area, _enabled_social_platforms(client_config))

    if source_mode == "web":
        return _interleave_target_groups(list(web_groups.values()))
    if source_mode == "human_emulator":
        return _interleave_target_groups(list(social_groups.values()))
    if source_mode == "all":
        groups: list[list[dict[str, object]]] = []
        groups.extend(group for group in social_groups.values() if group)
        groups.extend(group for group in web_groups.values() if group)
        return _interleave_target_groups(groups)
    return _interleave_target_groups(list(web_groups.values()))


def _strategy_terms(strategy: str, requested_title: str, client_config: dict) -> list[str]:
    if strategy == "leadership_people":
        return _dedupe_terms([requested_title, "Founder", "CTO", "Head of Engineering"])
    if strategy == "technical_profiles":
        return _dedupe_terms([requested_title, "Senior Software Engineer", "Software Engineer", "Architect"])
    if strategy == "marketing_sales_people":
        return _dedupe_terms([requested_title, "Head of Growth", "VP Sales", "Marketing Director"])
    return _dedupe_terms([requested_title, str(client_config.get("job", "")).strip()])


def _web_target_groups(strategy: str, search_terms: list[str], area: str) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    if strategy == "leadership_people":
        groups["ycombinator.com"] = [
            _candidate_target(
                "https://www.ycombinator.com/founders",
                source="web",
                family="people_directory",
                reason="Public founder directory aligned to startup leadership personas.",
            ),
            _candidate_target(
                "https://www.ycombinator.com/people",
                source="web",
                family="people_directory",
                reason="Public people directory aligned to startup leadership personas.",
            ),
        ]
        groups["github.com"] = [_github_search_target(term, area) for term in search_terms[:3]]
        return groups

    if strategy in {"technical_profiles", "marketing_sales_people", "general_people"}:
        groups["github.com"] = [_github_search_target(term, area) for term in search_terms[:4]]
        return groups

    groups["github.com"] = [_github_search_target(term, area) for term in search_terms[:2]]
    return groups


def _social_target_groups(
    search_terms: list[str],
    area: str,
    enabled_platforms: list[str],
) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for platform in enabled_platforms:
        targets: list[dict[str, object]] = []
        for term in search_terms[:3]:
            target = _social_search_target(platform, term, area)
            if target:
                targets.append(target)
        if targets:
            groups[targets[0]["domain"]] = targets
    return groups


def _enabled_social_platforms(client_config: dict) -> list[str]:
    configured = [
        str(platform).strip().lower()
        for platform in client_config.get("social_platforms", [])
        if str(platform).strip()
    ]
    return [platform for platform in configured if platform in _SUPPORTED_SOCIAL_PLATFORMS]


def _pinned_site_targets(website: str) -> list[dict[str, object]]:
    parsed = urlparse(website if "://" in website else f"https://{website}")
    base = f"{parsed.scheme or 'https'}://{parsed.netloc or parsed.path}".rstrip("/")
    domain = _domain_for_url(base)
    source = "social" if domain in {"linkedin.com", "x.com", "twitter.com"} else "web"
    paths = [
        "",
        "/people",
        "/team",
        "/leadership",
        "/about",
        "/staff",
    ]
    return [
        _candidate_target(
            urljoin(base + "/", path.lstrip("/")),
            source=source,
            family="pinned_site",
            reason="Pinned site starter page.",
        )
        for path in paths
    ]


def _github_search_target(term: str, area: str) -> dict[str, object]:
    query_parts = [term.strip()]
    if area and area.upper() != "NA":
        query_parts.append(f'location:"{area}"')
    query = " ".join(part for part in query_parts if part).strip()
    return _candidate_target(
        f"https://github.com/search?q={quote_plus(query)}&type=users",
        source="web",
        family="developer_profiles",
        reason=f"Public GitHub user search matching '{query}'.",
    )


def _social_search_target(platform: str, term: str, area: str) -> dict[str, object] | None:
    query_parts = [term.strip()]
    if area and area.upper() != "NA":
        query_parts.append(area)
    query = " ".join(part for part in query_parts if part).strip()
    if platform == "linkedin":
        return _candidate_target(
            f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(query)}",
            source="social",
            family="people_search",
            reason=f"Enabled social people search matching '{query}'.",
        )
    if platform == "x":
        return _candidate_target(
            f"https://x.com/search?q={quote_plus(query)}&f=user",
            source="social",
            family="people_search",
            reason=f"Enabled social user search matching '{query}'.",
        )
    return None


def _candidate_target(url: str, source: str, family: str, reason: str) -> dict[str, object]:
    return {
        "url": url,
        "source": source,
        "family": family,
        "domain": _domain_for_url(url),
        "reason": reason,
    }


def _source_mix_for_mode(source_mode: str) -> str:
    if source_mode == "all":
        return "model_decides"
    if source_mode == "human_emulator":
        return "social_only"
    if source_mode == "web":
        return "web_only"
    return source_mode


def _selection_notes(source_mode: str, social_enabled: bool) -> list[str]:
    notes = [
        "Choose the next site that best matches the keyword brief.",
        "Do not assume GitHub is first.",
        "Prefer a new domain/source if the previous one produced no viable leads.",
    ]
    if source_mode == "web":
        notes.append("Stay within non-social web domains for this run.")
    if source_mode == "human_emulator":
        notes.append("Use only the enabled social platforms in this run.")
    if source_mode == "all" and social_enabled:
        notes.append("Compare social people search with web profile search before choosing the next site.")
    return notes


def _interleave_target_groups(groups: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """Interleave per-domain groups so the first wave stays domain-diverse."""
    ordered: list[dict[str, object]] = []
    working = [list(group) for group in groups if group]
    while any(working):
        for group in working:
            if group:
                ordered.append(group.pop(0))
    return ordered


def _dedupe_candidate_targets(targets: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    for target in targets:
        url = str(target.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        deduped.append(target)
        seen_urls.add(url)
        if len(deduped) >= max(1, min(limit, 20)):
            break
    return deduped


def _ordered_domains(targets: list[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        domain = str(target.get("domain", "")).strip().lower()
        if not domain or domain in seen:
            continue
        seen.add(domain)
        ordered.append(domain)
    return ordered


def _domain_for_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.netloc.lower().split(":", 1)[0].removeprefix("www.")


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

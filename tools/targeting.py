"""Keyword-driven target selection helpers for web and social discovery."""

from __future__ import annotations

from urllib.parse import quote, quote_plus, urljoin, urlparse

from source_state import SourceState, infer_source_family, source_key

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

_SUPPORTED_SOCIAL_PLATFORMS = ("linkedin", "x", "instagram", "snapchat")
_SOCIAL_DOMAINS = {"linkedin.com", "x.com", "twitter.com", "instagram.com", "snapchat.com"}
_AVOID_DOMAINS = (
    "crunchbase.com",
    "wellfound.com",
    "techcrunch.com",
    "producthunt.com",
)

_HIGH_VOLUME_TECHNICAL_TERMS = (
    "Staff Engineer",
    "Staff Software Engineer",
    "Principal Engineer",
    "Principal Software Engineer",
    "Lead Software Engineer",
    "Senior Backend Engineer",
    "Senior Frontend Engineer",
    "Senior Full Stack Engineer",
    "Platform Engineer",
    "Distributed Systems Engineer",
)


def suggest_targets(
    client_config: dict,
    source_mode: str,
    limit: int = 8,
    source_state: SourceState | None = None,
    phase: str = "pass1",
    extra_terms: list[str] | None = None,
    extra_areas: list[str] | None = None,
) -> dict:
    """Return a keyword brief plus domain-diverse candidate targets for the run."""
    website = str(client_config.get("website", "NA")).strip()
    strategy = _infer_strategy(client_config)
    keyword_brief = _build_keyword_brief(strategy, client_config, source_mode, extra_terms=extra_terms)

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
            "phase": "pass1",
            "candidate_targets": candidate_targets,
            "allowed_domains": allowed_domains,
            "avoid_domains": list(_AVOID_DOMAINS),
            "selection_notes": _selection_notes(source_mode, social_enabled=False, phase="pass1"),
        }

    candidate_targets = _dedupe_candidate_targets(
        _candidate_targets_for_strategy(
            strategy,
            keyword_brief,
            client_config,
            source_mode,
            source_state=source_state,
            phase=phase,
            extra_areas=extra_areas,
        ),
        limit,
    )
    allowed_domains = _ordered_domains(candidate_targets)
    social_enabled = bool(_enabled_social_platforms(client_config))
    return {
        "strategy": strategy,
        "keyword_brief": keyword_brief,
        "source_mix": _source_mix_for_mode(source_mode),
        "phase": phase,
        "candidate_targets": candidate_targets,
        "allowed_domains": allowed_domains,
        "avoid_domains": list(_AVOID_DOMAINS),
        "selection_notes": _selection_notes(source_mode, social_enabled=social_enabled, phase=phase),
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


def _build_keyword_brief(
    strategy: str,
    client_config: dict,
    source_mode: str,
    extra_terms: list[str] | None = None,
) -> dict[str, object]:
    requested_title = str(client_config.get("job_title", "")).strip()
    requested_job = str(client_config.get("job", "")).strip()
    strategy_terms = _strategy_terms(strategy, requested_title, client_config)
    if extra_terms:
        strategy_terms = _dedupe_terms([*strategy_terms, *extra_terms])
    secondary_limit = 10 if strategy == "technical_profiles" and _lead_target_size(client_config) >= 25 else 3
    primary_terms = _dedupe_terms([requested_title or requested_job or strategy_terms[0]])
    secondary_terms = [
        term
        for term in strategy_terms
        if term.lower() not in {value.lower() for value in primary_terms}
    ]
    return {
        "primary_terms": primary_terms[:2],
        "secondary_terms": secondary_terms[:secondary_limit],
        "area": str(client_config.get("area", "NA")).strip() or "NA",
        "source_mode": source_mode,
    }


def _candidate_targets_for_strategy(
    strategy: str,
    keyword_brief: dict[str, object],
    client_config: dict,
    source_mode: str,
    source_state: SourceState | None = None,
    phase: str = "pass1",
    extra_areas: list[str] | None = None,
) -> list[dict[str, object]]:
    primary_terms = list(keyword_brief.get("primary_terms", []))
    secondary_terms = list(keyword_brief.get("secondary_terms", []))
    search_terms = _dedupe_terms([*primary_terms, *secondary_terms])
    area = str(keyword_brief.get("area", "NA"))

    effective_source_mode = source_mode
    if (
        source_mode == "all"
        and phase == "pass1"
        and strategy == "technical_profiles"
        and int(client_config.get("min_leads", 3) or 3) >= 5
    ):
        effective_source_mode = "web"

    groups = _catalog_for_strategy(
        strategy,
        search_terms,
        area,
        client_config,
        extra_areas,
        _enabled_social_platforms(client_config),
        effective_source_mode,
    )
    if phase == "discovery" and source_mode in {"web", "all"}:
        groups.extend(_web_discovery_target_groups(strategy, search_terms, area))
    if source_state is None:
        return _interleave_target_groups([group["targets"] for group in groups])

    if phase == "pass1":
        selected_groups = _pass1_groups(groups, source_state)
    else:
        selected_groups = _discovery_groups(groups, source_state)
    return _interleave_target_groups([group["targets"] for group in selected_groups])


def _strategy_terms(strategy: str, requested_title: str, client_config: dict) -> list[str]:
    if strategy == "leadership_people":
        return _dedupe_terms([requested_title, "Founder", "CTO", "Head of Engineering"])
    if strategy == "technical_profiles":
        terms = [requested_title, "Senior Software Engineer", "Software Engineer", "Architect"]
        if _lead_target_size(client_config) >= 25:
            terms.extend(_HIGH_VOLUME_TECHNICAL_TERMS)
        return _dedupe_terms(terms)
    if strategy == "marketing_sales_people":
        return _dedupe_terms([requested_title, "Head of Growth", "VP Sales", "Marketing Director"])
    return _dedupe_terms([requested_title, str(client_config.get("job", "")).strip()])


def _lead_target_size(client_config: dict) -> int:
    """Return the configured lead target for this run."""
    return max(1, int(client_config.get("min_leads", 1) or 1))


def _area_variants(area: str, client_config: dict) -> list[str]:
    """Return a broader set of area variants for large technical runs."""
    normalized = str(area).strip()
    if not normalized or normalized.upper() == "NA":
        return ["NA"]

    variants = [normalized]
    if _lead_target_size(client_config) < 25:
        return variants

    lower = normalized.lower()
    if "san francisco bay area" in lower:
        variants.extend(["San Francisco", "Bay Area"])
    elif "san francisco" in lower:
        variants.append("Bay Area")
    elif "bay area" in lower:
        variants.append("San Francisco")
    return _dedupe_terms(variants)


def _effective_area_variants(
    area: str,
    client_config: dict,
    extra_areas: list[str] | None = None,
) -> list[str]:
    """Return the final area variants after merging config and reseed hints."""
    merged = [*_area_variants(area, client_config), *(extra_areas or [])]
    normalized = _dedupe_terms([value for value in merged if str(value).strip()])
    return normalized or ["NA"]


def _social_target_groups(
    search_terms: list[str],
    area: str,
    enabled_platforms: list[str],
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for platform in enabled_platforms:
        targets: list[dict[str, object]] = []
        for term in search_terms[:3]:
            target = _social_search_target(platform, term, area)
            if target:
                targets.append(target)
        if targets:
            groups.append(
                {
                    "kind": "social_platform",
                    "identifier": platform,
                    "domain": targets[0]["domain"],
                    "family": "people_search",
                    "targets": targets,
                }
            )
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
    source = "social" if domain in _SOCIAL_DOMAINS else "web"
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
        source_kind="web_domain",
        source_id="github.com",
    )


def _gitlab_search_target(term: str, area: str) -> dict[str, object]:
    query_parts = [term.strip()]
    if area and area.upper() != "NA":
        query_parts.append(area)
    query = " ".join(part for part in query_parts if part).strip()
    return _candidate_target(
        f"https://gitlab.com/explore/users?search={quote_plus(query)}",
        source="web",
        family="developer_profiles",
        reason=f"Public GitLab user search matching '{query}'.",
        source_kind="web_domain",
        source_id="gitlab.com",
    )


def _duckduckgo_profile_search_target(term: str, area: str, site_domain: str = "github.com") -> dict[str, object]:
    query_parts = [f"site:{site_domain}"]
    if term.strip():
        query_parts.append(f'"{term.strip()}"')
    if area and area.upper() != "NA":
        query_parts.append(f'"{area}"')
    query = " ".join(query_parts).strip()
    return _candidate_target(
        f"https://duckduckgo.com/html/?q={quote_plus(query)}",
        source="web",
        family="developer_profiles",
        reason=f"Public web search for developer profiles matching {query}.",
        source_kind="web_domain",
        source_id="duckduckgo.com",
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
            source_kind="social_platform",
            source_id="linkedin",
        )
    if platform == "x":
        return _candidate_target(
            f"https://x.com/search?q={quote_plus(query)}&f=user",
            source="social",
            family="people_search",
            reason=f"Enabled social user search matching '{query}'.",
            source_kind="social_platform",
            source_id="x",
        )
    if platform == "instagram":
        return _candidate_target(
            f"https://www.instagram.com/explore/search/keyword/?q={quote_plus(query)}",
            source="social",
            family="people_search",
            reason=f"Enabled Instagram people search matching '{query}'.",
            source_kind="social_platform",
            source_id="instagram",
        )
    if platform == "snapchat":
        return _candidate_target(
            f"https://www.snapchat.com/search/{quote(query)}",
            source="social",
            family="people_search",
            reason=f"Enabled Snapchat people search matching '{query}'.",
            source_kind="social_platform",
            source_id="snapchat",
        )
    return None


def _candidate_target(
    url: str,
    source: str,
    family: str,
    reason: str,
    source_kind: str | None = None,
    source_id: str | None = None,
) -> dict[str, object]:
    domain = _domain_for_url(url)
    resolved_kind = source_kind or ("social_platform" if source == "social" else "web_domain")
    resolved_source_id = source_id or (domain if resolved_kind == "web_domain" else "")
    return {
        "url": url,
        "source": source,
        "family": family,
        "domain": domain,
        "reason": reason,
        "source_kind": resolved_kind,
        "source_id": resolved_source_id,
        "source_key": source_key(resolved_kind, resolved_source_id),
    }


def _source_mix_for_mode(source_mode: str) -> str:
    if source_mode == "all":
        return "model_decides"
    if source_mode == "human_emulator":
        return "social_only"
    if source_mode == "web":
        return "web_only"
    return source_mode


def _selection_notes(source_mode: str, social_enabled: bool, phase: str) -> list[str]:
    notes = [
        "Choose the next site that best matches the keyword brief.",
        "Do not assume GitHub is first.",
        "Prefer a new domain/source if the previous one produced no viable leads.",
    ]
    if phase == "pass1":
        notes.append("Pass 1 uses only approved sources and active temporary seed sources.")
    else:
        notes.append(
            "Pass 2 is discovery mode: sample up to 3 viable leads from a new source before deeper scraping."
        )
    if source_mode == "web":
        notes.append("Stay within non-social web domains for this run.")
    if source_mode == "human_emulator":
        notes.append("Use only the enabled social platforms in this run.")
    if source_mode == "all" and social_enabled:
        notes.append("Compare social people search with web profile search before choosing the next site.")
    return notes


def _catalog_for_strategy(
    strategy: str,
    search_terms: list[str],
    area: str,
    client_config: dict,
    extra_areas: list[str] | None,
    enabled_social_platforms: list[str],
    source_mode: str,
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    if source_mode in {"web", "all"}:
        groups.extend(_web_target_groups(strategy, search_terms, area, client_config, extra_areas))
    if source_mode in {"human_emulator", "all"}:
        groups.extend(_social_target_groups(search_terms, area, enabled_social_platforms))
    return groups


def _web_target_groups(
    strategy: str,
    search_terms: list[str],
    area: str,
    client_config: dict,
    extra_areas: list[str] | None,
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []

    if strategy == "leadership_people":
        groups.append(
            {
                "kind": "web_domain",
                "identifier": "ycombinator.com",
                "domain": "ycombinator.com",
                "family": "people_directory",
                "targets": [
                    _candidate_target(
                        "https://www.ycombinator.com/founders",
                        source="web",
                        family="people_directory",
                        reason="Public founder directory aligned to startup leadership personas.",
                        source_kind="web_domain",
                        source_id="ycombinator.com",
                    ),
                    _candidate_target(
                        "https://www.ycombinator.com/people",
                        source="web",
                        family="people_directory",
                        reason="Public people directory aligned to startup leadership personas.",
                        source_kind="web_domain",
                        source_id="ycombinator.com",
                    ),
                ],
            }
        )
        groups.append(
            {
                "kind": "web_domain",
                "identifier": "github.com",
                "domain": "github.com",
                "family": "developer_profiles",
                "targets": [_github_search_target(term, area) for term in search_terms[:3]],
            }
        )
        return groups

    if strategy in {"technical_profiles", "marketing_sales_people", "general_people"}:
        github_targets: list[dict[str, object]]
        if strategy == "technical_profiles":
            github_targets = [
                _github_search_target(term, area_variant)
                for area_variant in _effective_area_variants(area, client_config, extra_areas)
                for term in search_terms
            ]
        else:
            github_targets = [_github_search_target(term, area) for term in search_terms[:3]]
        groups.append(
            {
                "kind": "web_domain",
                "identifier": "github.com",
                "domain": "github.com",
                "family": "developer_profiles",
                "targets": github_targets,
            }
        )
        groups.append(
            {
                "kind": "web_domain",
                "identifier": "gitlab.com",
                "domain": "gitlab.com",
                "family": "developer_profiles",
                "targets": [_gitlab_search_target(term, area) for term in search_terms[:3]],
            }
        )
        return groups

    groups.append(
        {
            "kind": "web_domain",
            "identifier": "github.com",
            "domain": "github.com",
            "family": "developer_profiles",
            "targets": [_github_search_target(term, area) for term in search_terms[:2]],
        }
    )
    return groups


def _web_discovery_target_groups(strategy: str, search_terms: list[str], area: str) -> list[dict[str, object]]:
    """Return broader public-web fallback targets for discovery mode."""
    if strategy not in {"technical_profiles", "marketing_sales_people", "general_people", "leadership_people"}:
        return []

    targets = [
        _duckduckgo_profile_search_target(term, area, site_domain=site_domain)
        for site_domain in ("github.com", "gitlab.com")
        for term in search_terms[:2]
    ]
    if not targets:
        return []
    return [
        {
            "kind": "web_domain",
            "identifier": "duckduckgo.com",
            "domain": "duckduckgo.com",
            "family": "developer_profiles",
            "targets": targets,
        }
    ]


def _pass1_groups(groups: list[dict[str, object]], source_state: SourceState) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    approved_groups: list[dict[str, object]] = []
    temporary_groups: list[dict[str, object]] = []

    for group in groups:
        kind = str(group["kind"])
        identifier = str(group["identifier"])
        status = source_state.source_status(kind, identifier)
        if status == "approved":
            approved_groups.append(group)
        elif status == "temporary_seed":
            temporary_groups.append(group)

    selected.extend(approved_groups)
    selected.extend(temporary_groups)
    return selected


def _discovery_groups(groups: list[dict[str, object]], source_state: SourceState) -> list[dict[str, object]]:
    prioritized: list[dict[str, object]] = []
    fallback: list[dict[str, object]] = []
    approved_families = source_state.family_hints(include_temporary=True)

    for group in groups:
        kind = str(group["kind"])
        identifier = str(group["identifier"])
        status = source_state.source_status(kind, identifier)
        if status in {"approved", "temporary_seed", "pending_review", "rejected"}:
            continue
        family = str(group.get("family") or infer_source_family(kind, identifier))
        if family in approved_families:
            prioritized.append(group)
        else:
            fallback.append(group)
    return prioritized + fallback


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
        if len(deduped) >= max(1, int(limit)):
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

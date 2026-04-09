"""Tool registry: tool definitions plus the async dispatcher.

Six tools are exposed to the agent:
  1. suggest_targets — return a keyword brief plus candidate targets for the current run
  2. fetch_page      — fetch a URL and classify the page
  3. list_links      — extract candidate navigation/profile links from a page
  4. parse_html      — extract fields from previously fetched HTML via CSS selectors
  5. save_result     — write a lead row to the client's SQLite database
  6. fail_url        — log a URL failure and skip it

A ToolContext object is passed to every tool so they can share state
(browser, page cache, metadata, crawler state, human emulator, storage writer)
without globals.
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

from human_emulator.platforms import adapter_for_url
from human_emulator.social import RestrictionDetected, build_profile_html
from source_state import SourceState, domain_for_platform, infer_source_family, infer_source_identity, source_key
from tools.discovery import build_preview, classify_page, extract_links, unwrap_discovery_redirect_url
from tools.fetcher import smart_fetch
from tools.parser import parse_fields
from tools.targeting import suggest_targets

logger = logging.getLogger(__name__)
_MAX_FETCHES_PER_DOMAIN = 8
_GITHUB_PROFILE_SELECTORS: dict[str, list[str]] = {
    "name": ["span[itemprop='name']", "h1.vcard-names span.p-name", "h1 span.p-name"],
    "job_title": ["div.p-note", "div.js-profile-editable-area div.p-note", "span.p-note"],
    "company": ["li[itemprop='worksFor'] span.p-org", "span.p-org", "li.vcard-detail[itemprop='worksFor'] a"],
    "email": ["li[itemprop='email'] a[href^='mailto:']", "a.u-email[href^='mailto:']"],
    "phone": ["a[href^='tel:']"],
    "social_media": [
        "li[itemprop='url'] a",
        "a.Link--primary[href*='linkedin.com']",
        "a.Link--primary[href*='twitter.com']",
        "a.Link--primary[href*='x.com']",
    ],
}
_GENERIC_PROFILE_SELECTORS: dict[str, list[str]] = {
    "name": ["h1", "[itemprop='name']", ".name", ".profile-name"],
    "job_title": [".title", ".job-title", ".headline", ".role", "[itemprop='jobTitle']"],
    "company": [".company", ".organization", "[itemprop='worksFor']", ".employer"],
    "email": ["a[href^='mailto:']"],
    "phone": ["a[href^='tel:']"],
    "website": ["a.website[href]", "a[href][class*='website']", "a[href^='http']"],
    "social_media": [
        "a[href*='linkedin.com']",
        "a[href*='twitter.com']",
        "a[href*='x.com']",
        "a[href*='instagram.com']",
        "a[href*='snapchat.com']",
        "a[href*='github.com']",
    ],
}
_COMPANY_TEAM_HINTS = {
    "team",
    "platform",
    "engineering",
    "developer",
    "software",
    "product",
    "security",
    "data",
    "devops",
    "infrastructure",
    "cloud",
    "ai",
}
_LOW_QUALITY_COMPANY_VALUES = {
    "me",
    "myself",
    "self",
    "pdf reader",
}

# Social-media domains that must be routed to the human emulator
SOCIAL_MEDIA_DOMAINS: frozenset[str] = frozenset(
    ["linkedin.com", "facebook.com", "instagram.com", "snapchat.com", "twitter.com", "x.com"]
)
_SUPPORTED_SOCIAL_PROFILE_DOMAINS: frozenset[str] = frozenset(
    ["linkedin.com", "instagram.com", "snapchat.com", "twitter.com", "x.com"]
)
_SOCIAL_PLATFORM_BY_DOMAIN: dict[str, str] = {
    "linkedin.com": "linkedin",
    "instagram.com": "instagram",
    "snapchat.com": "snapchat",
    "twitter.com": "x",
    "x.com": "x",
}


def _is_social_media(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in SOCIAL_MEDIA_DOMAINS)


def _social_platform_for_domain(domain: str) -> str | None:
    """Return the configured platform key for a supported social domain."""
    return _SOCIAL_PLATFORM_BY_DOMAIN.get(domain)


@dataclass
class DomainOutcome:
    """Per-run domain memory for broad-mode selection."""

    blocked_count: int = 0
    irrelevant_count: int = 0
    discovery_hits: int = 0
    profile_hits: int = 0
    saved_hits: int = 0
    banned_for_run: bool = False
    last_reason: str = ""


@dataclass
class SourceRunStats:
    """Per-run counters for an individual source."""

    kind: str
    identifier: str
    domain: str
    family: str = ""
    fetch_count: int = 0
    saved_count: int = 0
    duplicate_count: int = 0
    rejected_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "domain": self.domain,
            "family": self.family,
            "fetch_count": self.fetch_count,
            "saved_count": self.saved_count,
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
        }


# ---------------------------------------------------------------------------
# Tool context
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Shared state passed to every tool execution."""
    client_config:  dict
    sheets_writer:  Any                              # storage.writer.StorageWriter
    source_mode:    str             = "web"
    effective_source_mode: str | None = None
    source_phase:   str             = "pass1"
    target_domain:  str | None      = None
    scraper_browser: Any | None      = None          # tools.browser.ScraperBrowser
    emulator_browser: Any | None     = None          # human_emulator.browser.EmulatorBrowser
    emulator_state:  Any | None      = None          # human_emulator.state.EmulatorState
    page_cache:      dict[str, str]  = field(default_factory=dict)   # fetch_id → HTML
    fetch_metadata:  dict[str, dict[str, Any]] = field(default_factory=dict)  # fetch_id → metadata
    url_to_fetch_id: dict[str, str]  = field(default_factory=dict)   # normalized url → fetch_id
    failed_url_flag: bool            = False          # set by fail_url to break the step loop
    _logged_sites_chosen: list[str]  = field(default_factory=list)   # for website=NA logging
    fetch_count:     int             = 0
    fetch_error_count: int           = 0
    tool_call_count: int             = 0
    failed_urls:     list[dict[str, str]] = field(default_factory=list)
    visited_urls:    set[str]        = field(default_factory=set)
    processed_fetch_ids: set[str]    = field(default_factory=set)
    domain_fetch_counts: dict[str, int] = field(default_factory=dict)
    fetch_budget_counts: dict[str, int] = field(default_factory=dict)
    parsed_results:  dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected_weak_count: int           = 0
    domain_outcomes: dict[str, DomainOutcome] = field(default_factory=dict)
    accounted_fetch_ids: set[str]      = field(default_factory=set)
    suggest_targets_called: bool       = False
    suggested_targets: list[dict[str, Any]] = field(default_factory=list)
    suggested_target_urls: set[str]    = field(default_factory=set)
    allowed_domains: set[str]          = field(default_factory=set)
    candidate_domains: list[str]       = field(default_factory=list)
    avoid_domains: list[str]           = field(default_factory=list)
    target_strategy: str | None        = None
    keyword_brief: dict[str, Any]      = field(default_factory=dict)
    source_mix: str | None             = None
    social_adapters: dict[str, Any]    = field(default_factory=dict)
    source_state: SourceState | None   = None
    source_run_stats: dict[str, SourceRunStats] = field(default_factory=dict)
    discovery_source_samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    current_run_saved_leads: list[dict[str, Any]] = field(default_factory=list)
    social_profile_hints: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_social_platforms: set[str] = field(default_factory=set)
    unavailable_social_platforms: set[str] = field(default_factory=set)
    unavailable_domains: set[str] = field(default_factory=set)
    terminal_url_outcomes: dict[str, str] = field(default_factory=dict)
    discovery_seen_links: dict[str, set[str]] = field(default_factory=dict)
    exhausted_discovery_fetches: set[str] = field(default_factory=set)
    exhausted_discovery_urls: set[str] = field(default_factory=set)
    fetch_parent_seeds: dict[str, str] = field(default_factory=dict)
    discovered_link_parents: dict[str, str] = field(default_factory=dict)
    social_blank_seed_counts: dict[str, int] = field(default_factory=dict)
    social_platform_failures: dict[str, int] = field(default_factory=dict)
    social_platform_saves: dict[str, int] = field(default_factory=dict)
    low_yield_platforms: set[str] = field(default_factory=set)
    parse_attempt_counts: dict[str, int] = field(default_factory=dict)
    reseed_search_terms: list[str] = field(default_factory=list)
    reseed_area_variants: list[str] = field(default_factory=list)
    reseed_generation: int = 0
    last_suggest_targets_signature: tuple[Any, ...] | None = None
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    social_platform_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    source_sample_locks: dict[str, asyncio.Lock] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool definitions (Ollama / OpenAI format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "suggest_targets",
            "description": (
                "Return a keyword brief plus candidate targets for broad-mode discovery when website is NA. "
                "Use this before your first fetch so you can choose from allowed, persona-relevant domains "
                "instead of inventing domains ad hoc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of starter targets to return.",
                        "default": 8,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch a web page and return a preview of its content. "
                "Social-media URLs (LinkedIn, Facebook, Instagram, Twitter/X) are automatically "
                "routed to the human emulator. "
                "Returns a fetch_id plus page metadata: final_url, title, page_kind, and a 3000-character text preview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch.",
                    },
                    "needs_javascript": {
                        "type": "boolean",
                        "description": (
                            "Set to true if the page requires JavaScript to render "
                            "(e.g. single-page applications, React/Angular sites). "
                            "Set to false for static HTML pages."
                        ),
                    },
                },
                "required": ["url", "needs_javascript"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_links",
            "description": (
                "Extract candidate navigation or profile links from a previously fetched page. "
                "Use this on search results, directories, and list pages to discover person profile/detail pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fetch_id": {
                        "type": "string",
                        "description": "The fetch_id returned by fetch_page.",
                    },
                    "selector": {
                        "type": "string",
                        "description": (
                            "Optional CSS selector to narrow which elements to extract links from. "
                            "If omitted, the system returns filtered candidate links automatically."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of links to return.",
                        "default": 25,
                    },
                    "same_domain_only": {
                        "type": "boolean",
                        "description": "If true, only return links on the same domain as the fetched page.",
                        "default": False,
                    },
                },
                "required": ["fetch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_html",
            "description": (
                "Extract specific fields from a previously fetched page using CSS selectors. "
                "Returns a dict of field names to their extracted text values (or null if not found). "
                "Only returns data that is explicitly present in the HTML — never guesses or infers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fetch_id": {
                        "type": "string",
                        "description": "The fetch_id returned by fetch_page.",
                    },
                    "field_names": {
                        "type": "array",
                        "description": (
                            "Field names to extract from a detail/profile page. "
                            "Use the client's schema fields, for example: ['name', 'job_title', 'company', 'email']."
                        ),
                        "items": {"type": "string"},
                    },
                    "fields": {
                        "type": "object",
                        "description": (
                            "Optional advanced mapping of field_name to CSS selector string. "
                            "Prefer field_names for normal usage."
                        ),
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                    },
                },
                "required": ["fetch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_result",
            "description": (
                "Save a lead result to the client's SQLite database. "
                "Call this once per lead after extracting all available fields. "
                "Pass null for any field that could not be found on the page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The source URL this lead was found on.",
                    },
                    "data": {
                        "type": "object",
                        "description": (
                            "Dict of field names to their values. "
                            "Use null for any field that could not be found."
                        ),
                    },
                },
                "required": ["url", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fail_url",
            "description": (
                "Mark a URL as failed and skip it. "
                "Call this when a page is inaccessible, returns an error, "
                "requires a login you cannot complete, or contains no relevant data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL that failed.",
                    },
                    "fetch_id": {
                        "type": "string",
                        "description": "Optional fetch_id for a previously fetched page when you want the system to infer the URL.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for the failure.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

async def dispatch_tool(tool_name: str, arguments: dict, ctx: ToolContext) -> dict[str, Any]:
    """Route a tool call to the appropriate handler and return its result."""
    ctx.tool_call_count += 1
    if tool_name == "fetch_url":
        tool_name = "fetch_page"
    if tool_name == "suggest_targets":
        return _tool_suggest_targets(arguments.get("limit", 8), ctx)
    if tool_name == "fetch_page":
        url, needs_javascript = _coerce_fetch_page_args(arguments, ctx)
        if not url:
            return {"error": "fetch_page requires a URL.", "arguments": arguments}
        return await _tool_fetch_page(url, needs_javascript, ctx)
    elif tool_name == "list_links":
        fetch_id = _coerce_fetch_id(arguments, ctx)
        if not fetch_id:
            return {"error": "list_links requires a valid fetch_id or URL for a previously fetched page.", "arguments": arguments}
        return _tool_list_links(
            fetch_id,
            arguments.get("selector"),
            arguments.get("limit", 25),
            arguments.get("same_domain_only", False),
            ctx,
        )
    elif tool_name == "parse_html":
        fetch_id = _coerce_fetch_id(arguments, ctx)
        if not fetch_id:
            return {"error": "parse_html requires a valid fetch_id or URL for a previously fetched page.", "arguments": arguments}
        return _tool_parse_html(
            fetch_id,
            arguments.get("field_names"),
            arguments.get("fields", {}),
            ctx,
        )
    elif tool_name == "save_result":
        url, data = _coerce_save_result_args(arguments, ctx)
        if not url:
            return {"error": "save_result requires a source URL or fetch_id.", "arguments": arguments}
        return await _tool_save_result(url, data, ctx)
    elif tool_name == "fail_url":
        url = _coerce_fail_url(arguments, ctx)
        if not url:
            return {"error": "fail_url requires a source URL or fetch_id.", "arguments": arguments}
        return _tool_fail_url(url, arguments.get("reason", ""), ctx)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_suggest_targets(limit: int, ctx: ToolContext) -> dict[str, Any]:
    """Return a keyword brief and candidate targets for the current run."""
    requested_limit = max(1, min(int(limit), 20))
    signature = (
        ctx.source_phase,
        _saved_count(ctx),
        len(ctx.failed_urls),
        len(ctx.terminal_url_outcomes),
        len(ctx.low_yield_platforms),
        len(ctx.exhausted_discovery_urls),
        tuple(sorted(ctx.unavailable_domains)),
        tuple(sorted(domain for domain, outcome in ctx.domain_outcomes.items() if outcome.banned_for_run)),
        tuple(ctx.reseed_search_terms),
        tuple(ctx.reseed_area_variants),
        ctx.reseed_generation,
    )
    if (
        ctx.suggest_targets_called
        and ctx.last_suggest_targets_signature == signature
        and ctx.suggested_targets
    ):
        filtered_targets = _filter_candidate_targets(ctx.suggested_targets, ctx)
        filtered_domains = _ordered_target_domains(filtered_targets)
        ctx.suggested_targets = list(filtered_targets)
        ctx.suggested_target_urls = {
            _normalize_url(str(target["url"]))
            for target in filtered_targets
            if isinstance(target, dict) and target.get("url")
        }
        ctx.candidate_domains = list(filtered_domains)
        ctx.allowed_domains = set(filtered_domains)
        return {
            "status": "unchanged",
            "phase": ctx.source_phase,
            "strategy": ctx.target_strategy or "unknown",
            "keyword_brief": dict(ctx.keyword_brief),
            "allowed_domains": list(filtered_domains),
            "candidate_targets": list(filtered_targets[:requested_limit]),
            "message": (
                "Target brief unchanged in the current source phase. Continue the current viable seed or remaining candidate domains."
                if filtered_targets
                else "No viable starter targets remain in the current phase."
            ),
        }

    result = suggest_targets(
        ctx.client_config,
        ctx.effective_source_mode or ctx.source_mode,
        limit=_suggest_target_catalog_limit(ctx, requested_limit),
        source_state=ctx.source_state,
        phase=ctx.source_phase,
        extra_terms=ctx.reseed_search_terms,
        extra_areas=ctx.reseed_area_variants,
    )
    targets = _filter_candidate_targets(
        result.get("candidate_targets", []),
        ctx,
    )
    result["candidate_targets"] = list(targets[:requested_limit])
    result["allowed_domains"] = _ordered_target_domains(targets)
    ctx.suggest_targets_called = True
    ctx.suggested_targets = list(targets)
    ctx.target_strategy = str(result.get("strategy", "unknown"))
    ctx.keyword_brief = dict(result.get("keyword_brief", {}))
    ctx.source_mix = str(result.get("source_mix", "unknown"))
    ctx.suggested_target_urls = {
        _normalize_url(str(target["url"]))
        for target in targets
        if isinstance(target, dict) and target.get("url")
    }
    candidate_domains = [
        str(domain).strip().lower()
        for domain in result.get("allowed_domains", [])
        if str(domain).strip()
    ]
    if not candidate_domains:
        candidate_domains = [
            _domain_for_url(str(target["url"]))
            for target in targets
            if isinstance(target, dict) and target.get("url")
        ]
    candidate_domains = list(dict.fromkeys(candidate_domains))
    ctx.candidate_domains = candidate_domains
    ctx.allowed_domains = set(candidate_domains)
    ctx.avoid_domains = [
        str(domain).strip().lower()
        for domain in result.get("avoid_domains", [])
        if str(domain).strip()
    ]
    ctx.last_suggest_targets_signature = signature
    return result


def _suggest_target_catalog_limit(ctx: ToolContext, requested_limit: int) -> int:
    """Return the internal catalog size to keep on hand for repeated target requests."""
    min_leads = max(1, int(ctx.client_config.get("min_leads", 1) or 1))
    if ctx.target_strategy == "technical_profiles" or "engineer" in str(ctx.client_config.get("job", "")).lower():
        return max(requested_limit, min(80, max(24, min_leads * 2)))
    return max(requested_limit, min(40, max(12, min_leads)))


async def _tool_fetch_page(url: str, needs_javascript: bool, ctx: ToolContext) -> dict:
    url = unwrap_discovery_redirect_url(url)
    logger.info("fetch_page: %s (js=%s)", url, needs_javascript)
    normalized_url = _normalize_url(url)
    domain = _domain_for_url(url)
    budget_key = _fetch_budget_key(url)

    if _already_saved_source_url(url, ctx):
        _mark_terminal_url(url, "duplicate", ctx)
        return {
            "error": "URL already exists in storage for this client. Skip it and continue with a new candidate.",
            "url": url,
        }

    if normalized_url in ctx.exhausted_discovery_urls:
        ctx.fetch_error_count += 1
        return {
            "error": "Discovery/search URL is exhausted for this run. Continue with a different seed or domain.",
            "url": url,
        }

    terminal_outcome = _terminal_outcome_for_url(url, ctx)
    if terminal_outcome:
        ctx.fetch_error_count += 1
        return {
            "error": f"URL is exhausted for this run due to a previous terminal outcome: {terminal_outcome}.",
            "url": url,
        }

    if normalized_url in ctx.url_to_fetch_id:
        fetch_id = ctx.url_to_fetch_id[normalized_url]
        if fetch_id in ctx.exhausted_discovery_fetches:
            ctx.fetch_error_count += 1
            return {
                "error": "Discovery/search URL is exhausted for this run. Continue with a different seed or domain.",
                "url": url,
            }
        cached = ctx.fetch_metadata[fetch_id].copy()
        cached["fetch_id"] = fetch_id
        if fetch_id in ctx.exhausted_discovery_fetches:
            cached["exhausted"] = True
        return cached

    if _is_social_media(url):
        if ctx.source_mode == "web":
            ctx.fetch_error_count += 1
            return {
                "error": "Social-media URLs are not allowed in web mode.",
                "url": url,
            }

        if domain in ctx.unavailable_domains:
            ctx.fetch_error_count += 1
            return {
                "error": f"Social platform '{domain}' is unavailable for this run. Choose another active candidate domain.",
                "url": url,
            }

        broad_mode_rejection = _broad_mode_rejection(url, normalized_url, domain, ctx)
        if broad_mode_rejection:
            ctx.fetch_error_count += 1
            return {"error": broad_mode_rejection, "url": url}

        if _is_low_yield_platform(url, ctx) and _has_alternative_allowed_domain(url, ctx):
            ctx.fetch_error_count += 1
            return {
                "error": (
                    f"Social platform '{domain}' is on a low-yield cooldown for this run. "
                    "Use another candidate domain before retrying it."
                ),
                "url": url,
            }

        if ctx.client_config.get("website", "NA").upper() == "NA" and url not in ctx._logged_sites_chosen:
            ctx._logged_sites_chosen.append(url)
            logger.info("Agent-chosen target site: %s", url)

        result = await _fetch_social_media(url, ctx)
        if "error" in result:
            ctx.fetch_error_count += 1
        return result

    broad_mode_rejection = _broad_mode_rejection(url, normalized_url, domain, ctx)
    if broad_mode_rejection:
        ctx.fetch_error_count += 1
        return {"error": broad_mode_rejection, "url": url}

    if ctx.target_domain and not _same_or_subdomain(domain, ctx.target_domain):
        ctx.fetch_error_count += 1
        return {
            "error": f"URL is outside the configured target website ({ctx.target_domain}).",
            "url": url,
        }

    fetch_budget = _fetch_budget_for_url(url, ctx)
    if ctx.fetch_budget_counts.get(budget_key, 0) >= fetch_budget:
        ctx.fetch_error_count += 1
        return {
            "error": f"Fetch budget reached for domain '{domain}' in this run.",
            "url": url,
        }

    # Log site if website=NA (agent chose this target itself)
    if ctx.client_config.get("website", "NA").upper() == "NA":
        if url not in ctx._logged_sites_chosen:
            ctx._logged_sites_chosen.append(url)
            logger.info("Agent-chosen target site: %s", url)

    try:
        fetch_result = await smart_fetch(
            url,
            needs_javascript,
            get_context=ctx.scraper_browser.new_context,
        )
    except Exception as exc:
        logger.warning("fetch_page error for %s: %s", url, exc)
        ctx.fetch_error_count += 1
        return {"error": str(exc), "url": url}

    fetch_id = str(uuid.uuid4())
    page_info = classify_page(url, fetch_result.final_url, fetch_result.html)
    preview = build_preview(fetch_result.html)
    metadata = {
        "url": url,
        "final_url": page_info.final_url,
        "title": page_info.title,
        "page_kind": page_info.page_kind,
        "preview": preview,
    }
    async with ctx.state_lock:
        ctx.fetch_count += 1
        ctx.domain_fetch_counts[domain] = ctx.domain_fetch_counts.get(domain, 0) + 1
        ctx.fetch_budget_counts[budget_key] = ctx.fetch_budget_counts.get(budget_key, 0) + 1
        ctx.page_cache[fetch_id] = fetch_result.html
        ctx.fetch_metadata[fetch_id] = metadata
        ctx.url_to_fetch_id[normalized_url] = fetch_id
        ctx.url_to_fetch_id[_normalize_url(page_info.final_url)] = fetch_id
        ctx.visited_urls.add(_normalize_url(page_info.final_url))
        ctx.visited_urls.add(normalized_url)
        if page_info.page_kind in {"blocked", "not_found"}:
            _mark_terminal_url(page_info.final_url, page_info.page_kind, ctx)
            _mark_terminal_url(url, page_info.page_kind, ctx)
    _record_fetch_outcome(fetch_id, domain, page_info.page_kind, ctx)
    _record_source_fetch(page_info.final_url, ctx)

    return {"fetch_id": fetch_id, **metadata}


async def _fetch_social_media(url: str, ctx: ToolContext) -> dict:
    """Route a social-media URL through the human emulator."""
    if ctx.emulator_browser is None or ctx.emulator_state is None:
        return {
            "error": "Human emulator not available for this job. Run with --source all or --source human_emulator.",
            "url": url,
        }

    adapter_cls = adapter_for_url(url)
    if adapter_cls is None:
        return {
            "error": "This social-media platform is not supported yet. Supported platforms are LinkedIn and X.",
            "url": url,
        }

    platform = adapter_cls.platform
    enabled_platforms = {
        str(item).strip().lower()
        for item in ctx.client_config.get("social_platforms", [])
        if str(item).strip()
    }
    if ctx.source_mode in {"human_emulator", "all"} and not enabled_platforms:
        return {
            "error": "No social platforms are enabled for this client. Add 'social_platforms' to the client config.",
            "url": url,
        }
    if enabled_platforms and platform not in enabled_platforms:
        return {
            "error": f"Social platform '{platform}' is not enabled for this client.",
            "url": url,
        }

    availability = ctx.emulator_state.availability(platform)
    if availability["status"] in {"missing_credentials", "paused", "unavailable"}:
        reason = availability["reason"] or availability["status"]
        _mark_social_platform_unavailable(platform, reason, ctx)
        return {
            "error": f"Social platform '{platform}' is currently unavailable: {reason}",
            "url": url,
        }

    logger.info("Routing social-media URL to human emulator (%s): %s", platform, url)
    adapter = ctx.social_adapters.get(platform)
    if adapter is None:
        context = await ctx.emulator_browser.get_context(platform)
        adapter = adapter_cls(context, ctx.emulator_state, ctx.client_config["client_id"])
        ctx.social_adapters[platform] = adapter

    try:
        async with _platform_lock(ctx, platform):
            fetch_result = await adapter.fetch(url)
    except RestrictionDetected as exc:
        ctx.emulator_state.record_restriction(platform)
        ctx.emulator_state.set_pause_hours(platform, hours=8, reason=str(exc))
        ctx.emulator_state.set_availability(platform, "paused", str(exc))
        _mark_social_platform_unavailable(platform, str(exc), ctx)
        return {
            "error": f"Social platform '{platform}' hit a checkpoint and has been paused: {exc}",
            "url": url,
        }
    except Exception as exc:
        ctx.emulator_state.set_availability(platform, "unavailable", str(exc))
        _mark_social_platform_unavailable(platform, str(exc), ctx)
        return {
            "error": f"Social fetch failed for platform '{platform}': {exc}",
            "url": url,
        }

    domain = _domain_for_url(fetch_result.final_url or url)
    budget_key = _fetch_budget_key(fetch_result.final_url or url)
    fetch_id = str(uuid.uuid4())
    preview = build_preview(fetch_result.html)
    metadata = {
        "url": url,
        "final_url": fetch_result.final_url or url,
        "title": fetch_result.title or "Social profile",
        "page_kind": fetch_result.page_kind,
        "preview": preview,
        "platform": platform,
        "extracted_data": fetch_result.extracted_data,
    }
    async with ctx.state_lock:
        ctx.fetch_count += 1
        ctx.domain_fetch_counts[domain] = ctx.domain_fetch_counts.get(domain, 0) + 1
        ctx.fetch_budget_counts[budget_key] = ctx.fetch_budget_counts.get(budget_key, 0) + 1
        ctx.page_cache[fetch_id] = fetch_result.html
        ctx.fetch_metadata[fetch_id] = metadata
        ctx.url_to_fetch_id[_normalize_url(url)] = fetch_id
        ctx.url_to_fetch_id[_normalize_url(fetch_result.final_url or url)] = fetch_id
        ctx.visited_urls.add(_normalize_url(url))
        ctx.visited_urls.add(_normalize_url(fetch_result.final_url or url))
        if fetch_result.page_kind in {"blocked", "not_found"}:
            _mark_terminal_url(fetch_result.final_url or url, fetch_result.page_kind, ctx)
            _mark_terminal_url(url, fetch_result.page_kind, ctx)
    _record_fetch_outcome(fetch_id, domain, fetch_result.page_kind, ctx)
    _record_source_fetch(fetch_result.final_url or url, ctx)

    if fetch_result.page_kind == "profile":
        ctx.emulator_state.mark_visited(fetch_result.final_url or url, platform=platform)
        seed_fetch_id = ctx.fetch_parent_seeds.get(_normalize_url(url), "")
        if seed_fetch_id:
            ctx.fetch_parent_seeds[fetch_id] = seed_fetch_id

    if fetch_result.extracted_data:
        if fetch_result.page_kind in {"search_results", "directory"}:
            profile_urls = [
                str(item.get("url"))
                for item in fetch_result.extracted_data.get("results", [])
                if isinstance(item, dict) and item.get("url")
            ]
            if profile_urls:
                ctx.emulator_state.add_profiles(profile_urls, platform=platform)
            for item in fetch_result.extracted_data.get("results", []):
                if not isinstance(item, dict) or not item.get("url"):
                    continue
                candidate_url = _normalize_url(str(item["url"]))
                ctx.social_profile_hints[candidate_url] = {
                    "name": item.get("name"),
                    "job_title": item.get("headline"),
                    "company": item.get("company"),
                    "social_media": item.get("url"),
                }
                ctx.fetch_parent_seeds[candidate_url] = fetch_id
        else:
            hint = ctx.social_profile_hints.get(_normalize_url(fetch_result.final_url or url), {})
            if isinstance(fetch_result.extracted_data, dict) and hint:
                for field_name, field_value in hint.items():
                    if field_name not in fetch_result.extracted_data or not _has_meaningful_value(fetch_result.extracted_data.get(field_name)):
                        fetch_result.extracted_data[field_name] = field_value
                metadata["extracted_data"] = fetch_result.extracted_data
                fetch_result.html = build_profile_html(fetch_result.extracted_data)
                ctx.page_cache[fetch_id] = fetch_result.html
            preview = json.dumps(fetch_result.extracted_data, indent=2)
            metadata["preview"] = preview
            ctx.fetch_metadata[fetch_id]["preview"] = preview

    return {"fetch_id": fetch_id, **metadata, "extracted_data": fetch_result.extracted_data}


def _tool_list_links(
    fetch_id: str,
    selector: str | None,
    limit: int,
    same_domain_only: bool,
    ctx: ToolContext,
) -> dict:
    html = ctx.page_cache.get(fetch_id)
    metadata = ctx.fetch_metadata.get(fetch_id)
    if not html or not metadata:
        return {"error": f"No cached HTML for fetch_id '{fetch_id}'. Call fetch_page first."}

    page_kind = metadata.get("page_kind", "unknown")
    if page_kind not in {"search_results", "directory", "company_directory", "company_page", "unknown"}:
        return {
            "error": (
                f"list_links is for discovery pages. fetch_id '{fetch_id}' is classified as "
                f"'{page_kind}'. Use parse_html or fail_url instead."
            )
        }

    ctx.processed_fetch_ids.add(fetch_id)
    if fetch_id in ctx.exhausted_discovery_fetches:
        return {"links": [], "count": 0, "exhausted": True}

    base_url = metadata.get("final_url") or metadata.get("url")
    effective_same_domain = same_domain_only or bool(ctx.target_domain)
    links = extract_links(
        html,
        base_url,
        selector=selector,
        limit=100,
        same_domain_only=effective_same_domain,
    )
    seen_urls = ctx.discovery_seen_links.setdefault(fetch_id, set())
    eligible_links: list[dict[str, Any]] = []
    for item in links:
        candidate_url = _normalize_url(str(item.get("url", "")))
        if not candidate_url:
            continue
        if candidate_url in seen_urls:
            continue
        if _already_saved_source_url(candidate_url, ctx):
            seen_urls.add(candidate_url)
            _mark_terminal_url(candidate_url, "duplicate", ctx)
            continue
        if candidate_url in ctx.visited_urls or candidate_url in ctx.terminal_url_outcomes:
            seen_urls.add(candidate_url)
            continue
        eligible_links.append({**item, "url": candidate_url})

    unseen_links = eligible_links[: max(1, min(int(limit), 50))]
    for item in unseen_links:
        candidate_url = str(item.get("url", ""))
        seen_urls.add(candidate_url)
        ctx.discovered_link_parents[candidate_url] = fetch_id

    remaining_candidates = eligible_links[len(unseen_links) :]
    exhausted = len(eligible_links) < 2 or not remaining_candidates
    if exhausted:
        ctx.exhausted_discovery_fetches.add(fetch_id)
        ctx.exhausted_discovery_urls.add(_normalize_url(base_url))

    return {"links": unseen_links, "count": len(unseen_links), "exhausted": exhausted}


def _tool_parse_html(
    fetch_id: str,
    field_names: list[str] | None,
    fields: dict[str, str | list[str]],
    ctx: ToolContext,
) -> dict:
    html = ctx.page_cache.get(fetch_id)
    if not html:
        return {"error": f"No cached HTML for fetch_id '{fetch_id}'. Call fetch_page first."}
    metadata = ctx.fetch_metadata.get(fetch_id, {})
    page_kind = metadata.get("page_kind", "unknown")
    if page_kind not in {"profile", "unknown"}:
        return {
            "error": (
                f"parse_html is only for detail/profile pages. "
                f"fetch_id '{fetch_id}' is classified as '{page_kind}'. Use list_links or fail_url instead."
            )
        }
    if ctx.parse_attempt_counts.get(fetch_id, 0) >= 1 and fetch_id in ctx.parsed_results:
        return {"fields": ctx.parsed_results[fetch_id], "cached": True}
    ctx.processed_fetch_ids.add(fetch_id)
    selector_map = fields or _selector_map_for_fetch(fetch_id, field_names, ctx)
    if not selector_map:
        return {"error": "No extractable field selectors available for this page."}
    result = parse_fields(html, selector_map)
    result = _postprocess_extracted_fields(fetch_id, result, field_names, ctx)
    ctx.parse_attempt_counts[fetch_id] = ctx.parse_attempt_counts.get(fetch_id, 0) + 1
    ctx.parsed_results[fetch_id] = result
    final_url = metadata.get("final_url") or metadata.get("url") or ""
    if _is_blank_social_profile_data(final_url, result):
        metadata["page_kind"] = "not_found"
        _mark_terminal_url(final_url, "blank_profile", ctx)
        _record_social_blank_profile(final_url, ctx)
    logger.debug("parse_html: %s → %s", fetch_id, result)
    return {"fields": result}


async def _tool_save_result(url: str, data: dict, ctx: ToolContext) -> dict:
    prior_outcome = _terminal_outcome_for_url(url, ctx)
    if prior_outcome == "saved":
        duplicate_count = getattr(ctx.sheets_writer, "duplicate_count", None)
        if isinstance(duplicate_count, int):
            ctx.sheets_writer.duplicate_count = duplicate_count + 1
        return {"status": "duplicate", "url": url}
    if prior_outcome == "duplicate":
        duplicate_count = getattr(ctx.sheets_writer, "duplicate_count", None)
        if isinstance(duplicate_count, int):
            ctx.sheets_writer.duplicate_count = duplicate_count + 1
        return {"status": "duplicate", "url": url}
    if prior_outcome == "rejected":
        return {"status": "rejected", "url": url, "reason": "Previously rejected earlier in this run."}

    if _already_saved_source_url(url, ctx):
        duplicate_count = getattr(ctx.sheets_writer, "duplicate_count", None)
        if isinstance(duplicate_count, int):
            ctx.sheets_writer.duplicate_count = duplicate_count + 1
        _mark_terminal_url(url, "duplicate", ctx)
        return {"status": "duplicate", "url": url}

    data = _normalize_lead_payload(url, data)
    source_stats = _source_stats_for_url(url, ctx)
    viable, reason = _is_minimally_viable_lead(data, url)
    if not viable:
        async with ctx.state_lock:
            ctx.rejected_weak_count += 1
            source_stats.rejected_count += 1
            _mark_terminal_url(url, "rejected", ctx)
            if _domain_for_url(url) in _SUPPORTED_SOCIAL_PROFILE_DOMAINS:
                _record_social_blank_profile(url, ctx)
        logger.info("Rejected weak lead: %s — %s", url, reason)
        print(f"  • Weak lead rejected: {data.get('name') or url} ({reason})", flush=True)
        return {"status": "rejected", "url": url, "reason": reason}

    source_status = _source_status_for_url(url, ctx)
    if ctx.source_state is not None and source_status == "discovered":
        sample_result = await _sample_discovered_source(url, data, source_stats, ctx)
        if sample_result is not None:
            return sample_result

    try:
        status = await ctx.sheets_writer.append_row(url, data)
        if status == "saved":
            async with ctx.state_lock:
                _record_domain_save(url, ctx)
                source_stats.saved_count += 1
                _record_saved_lead(url, data, source_status, source_stats.family, ctx)
                _mark_terminal_url(url, "saved", ctx)
                social_domain = _domain_for_url(url)
                social_platform = _social_platform_for_domain(social_domain)
                if social_platform:
                    ctx.social_platform_saves[social_platform] = ctx.social_platform_saves.get(social_platform, 0) + 1
            print(f"  ✓ Saved: {data.get('name') or url}", flush=True)
        else:
            async with ctx.state_lock:
                source_stats.duplicate_count += 1
                _mark_terminal_url(url, "duplicate", ctx)
            print(f"  • Duplicate skipped: {data.get('name') or url}", flush=True)
        return {"status": status, "url": url}
    except Exception as exc:
        logger.error("save_result failed for %s: %s", url, exc)
        return {"error": str(exc), "url": url}


def _tool_fail_url(url: str, reason: str, ctx: ToolContext) -> dict:
    logger.info("fail_url: %s — %s", url, reason)
    ctx.failed_url_flag = True
    normalized_url = _normalize_url(url)
    existing_outcome = ctx.terminal_url_outcomes.get(normalized_url)
    if existing_outcome != reason and not any(
        _normalize_url(str(item.get("url", ""))) == normalized_url and str(item.get("reason", "")) == reason
        for item in ctx.failed_urls
    ):
        ctx.failed_urls.append({"url": url, "reason": reason})
    fetch_id = _lookup_fetch_id(url, ctx)
    if fetch_id:
        ctx.processed_fetch_ids.add(fetch_id)
    _mark_terminal_url(url, reason, ctx)
    if existing_outcome != reason:
        _record_domain_failure(url, reason, ctx)
    return {"status": "failed", "url": url, "reason": reason}


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme and parsed.path:
        parsed = urlparse(f"https://{url}")
    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if host == "html.duckduckgo.com":
        host = "duckduckgo.com"

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered_pairs = []
    for key, value in query_pairs:
        key_lower = key.lower()
        if host == "linkedin.com" and key_lower in {
            "miniprofileurn",
            "trk",
            "trkinfo",
            "trackingid",
            "originalsubdomain",
            "lipi",
            "midtoken",
            "midsig",
            "sid",
        }:
            continue
        if host in {"x.com", "twitter.com"} and key_lower in {
            "s",
            "t",
            "ref_src",
            "ref_url",
            "cn",
        }:
            continue
        filtered_pairs.append((key, value))

    query = urlencode(filtered_pairs, doseq=True)
    normalized = parsed._replace(netloc=host, path=path, query=query, fragment="")
    return urlunparse(normalized)


def _domain_for_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme and parsed.path:
        parsed = urlparse(f"https://{url}")
    host = parsed.netloc.lower().split(":", 1)[0].removeprefix("www.")
    if host == "html.duckduckgo.com":
        return "duckduckgo.com"
    return host


def _platform_lock(ctx: ToolContext, platform: str) -> asyncio.Lock:
    """Return the per-platform async lock used to serialize social actions."""
    if platform not in ctx.social_platform_locks:
        ctx.social_platform_locks[platform] = asyncio.Lock()
    return ctx.social_platform_locks[platform]


def _source_sample_lock(ctx: ToolContext, source_key_value: str) -> asyncio.Lock:
    """Return the per-source async lock used for discovered-source sampling."""
    if source_key_value not in ctx.source_sample_locks:
        ctx.source_sample_locks[source_key_value] = asyncio.Lock()
    return ctx.source_sample_locks[source_key_value]


def _mark_terminal_url(url: str, outcome: str, ctx: ToolContext) -> None:
    """Record a URL as terminal for this run so it is not retried endlessly."""
    normalized_url = _normalize_url(url)
    ctx.terminal_url_outcomes[normalized_url] = outcome


def _terminal_outcome_for_url(url: str, ctx: ToolContext) -> str | None:
    """Return the run-local terminal outcome for a URL, if any."""
    return ctx.terminal_url_outcomes.get(_normalize_url(url))


def _lookup_fetch_id(url: str, ctx: ToolContext) -> str:
    """Resolve a fetch_id from a URL, tolerating older unnormalized cache keys."""
    normalized_url = _normalize_url(url)
    fetch_id = ctx.url_to_fetch_id.get(normalized_url)
    if fetch_id:
        return fetch_id
    fetch_id = ctx.url_to_fetch_id.get(url)
    if fetch_id:
        return fetch_id
    for candidate_fetch_id, metadata in ctx.fetch_metadata.items():
        metadata_urls = {
            _normalize_url(str(metadata.get("url") or "")),
            _normalize_url(str(metadata.get("final_url") or "")),
        }
        if normalized_url in metadata_urls:
            return candidate_fetch_id
    return ""


def _search_seed_for_url(url: str, ctx: ToolContext) -> str:
    """Return the discovery/search seed that produced a URL when known."""
    fetch_id = _lookup_fetch_id(url, ctx)
    if not fetch_id:
        return ""
    return ctx.fetch_parent_seeds.get(fetch_id, "")


def _record_social_blank_profile(url: str, ctx: ToolContext) -> None:
    """Track blank/low-yield social profiles by seed and platform."""
    domain = _domain_for_url(url)
    platform = _social_platform_for_domain(domain)
    if not platform:
        return
    seed_fetch_id = _search_seed_for_url(url, ctx)
    if seed_fetch_id:
        ctx.social_blank_seed_counts[seed_fetch_id] = ctx.social_blank_seed_counts.get(seed_fetch_id, 0) + 1
        if ctx.social_blank_seed_counts[seed_fetch_id] >= 2:
            ctx.exhausted_discovery_fetches.add(seed_fetch_id)

    ctx.social_platform_failures[platform] = ctx.social_platform_failures.get(platform, 0) + 1
    if ctx.social_platform_saves.get(platform, 0) == 0 and ctx.social_platform_failures[platform] >= 2:
        ctx.low_yield_platforms.add(platform)


def _is_low_yield_platform(url: str, ctx: ToolContext) -> bool:
    """Return True when a social platform is on run-local low-yield cooldown."""
    platform = _social_platform_for_domain(_domain_for_url(url))
    return bool(platform and platform in ctx.low_yield_platforms)


def _saved_count(ctx: ToolContext) -> int:
    """Return the run-local viable saved count from the active writer."""
    return int(getattr(ctx.sheets_writer, "saved_count", 0))


def _fetch_budget_for_url(url: str, ctx: ToolContext) -> int:
    """Return the per-run fetch budget for a URL based on domain and run size."""
    budget_key = _fetch_budget_key(url)
    min_leads = max(1, int(ctx.client_config.get("min_leads", 3) or 3))
    if budget_key == "github.com:search":
        return max(12, min_leads * 2)
    if budget_key == "github.com:profile":
        return max(24, min_leads * 6)
    if budget_key == "duckduckgo.com:search":
        return max(6, min(12, min_leads))
    if budget_key.endswith(":social"):
        return max(6, min_leads * 2)
    domain = _domain_for_url(url)
    return _MAX_FETCHES_PER_DOMAIN


def _fetch_budget_key(url: str) -> str:
    """Return the budget bucket for a URL so search and profile budgets stay separate."""
    domain = _domain_for_url(url)
    parsed = urlparse(url if "://" in url else f"https://{url}")
    path = parsed.path.lower()
    query = parsed.query.lower()

    if domain == "github.com":
        if path == "/search" and "type=users" in query:
            return "github.com:search"
        return "github.com:profile"
    if domain == "duckduckgo.com":
        return "duckduckgo.com:search"
    if domain in _SUPPORTED_SOCIAL_PROFILE_DOMAINS:
        return f"{domain}:social"
    return domain


def _url_budget_exhausted(url: str, ctx: ToolContext) -> bool:
    """Return True when the URL's fetch bucket is exhausted for the current run."""
    budget_key = _fetch_budget_key(url)
    return ctx.fetch_budget_counts.get(budget_key, 0) >= _fetch_budget_for_url(url, ctx)


def _has_alternative_allowed_domain(url: str, ctx: ToolContext) -> bool:
    """Return True when another allowed candidate domain remains available this run."""
    current_domain = _domain_for_url(url)
    for domain in ctx.candidate_domains or sorted(ctx.allowed_domains):
        if domain == current_domain:
            continue
        if domain in ctx.unavailable_domains:
            continue
        outcome = ctx.domain_outcomes.get(domain)
        if outcome and outcome.banned_for_run:
            continue
        return True
    return False


def _filter_candidate_targets(targets: list[dict[str, Any]], ctx: ToolContext) -> list[dict[str, Any]]:
    """Filter out unavailable or exhausted starter targets for this run."""
    filtered: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        url = str(target.get("url", "")).strip()
        domain = str(target.get("domain", "")).strip().lower() or _domain_for_url(url)
        if not url or not domain:
            continue
        if not _duckduckgo_profile_query_allowed(url, ctx):
            continue
        if _url_budget_exhausted(url, ctx):
            continue
        if domain in ctx.unavailable_domains:
            continue
        outcome = ctx.domain_outcomes.get(domain)
        if outcome and outcome.banned_for_run:
            continue
        if _persistent_source_excluded_for_discovery(domain, ctx):
            continue
        if domain == "duckduckgo.com":
            site_domain = _duckduckgo_target_site_domain(url)
            if site_domain:
                site_outcome = ctx.domain_outcomes.get(site_domain)
                if site_domain in ctx.unavailable_domains or (site_outcome and site_outcome.banned_for_run):
                    continue
                if _persistent_source_excluded_for_discovery(site_domain, ctx):
                    continue
        platform = _social_platform_for_domain(domain)
        if platform and platform in ctx.low_yield_platforms:
            continue
        if _normalize_url(url) in ctx.exhausted_discovery_urls:
            continue
        if _already_saved_source_url(url, ctx):
            continue
        if _terminal_outcome_for_url(url, ctx):
            continue
        filtered.append(target)
    return filtered


def _ordered_target_domains(targets: list[dict[str, Any]]) -> list[str]:
    """Return ordered unique domains from the current target set."""
    ordered: list[str] = []
    seen: set[str] = set()
    for target in targets:
        domain = str(target.get("domain", "")).strip().lower()
        if not domain or domain in seen:
            continue
        seen.add(domain)
        ordered.append(domain)
    return ordered


def _already_saved_source_url(url: str, ctx: ToolContext) -> bool:
    """Return True when the source URL is already present in persistent storage."""
    writer = getattr(ctx, "sheets_writer", None)
    has_source_url = getattr(writer, "has_source_url", None)
    if callable(has_source_url):
        try:
            return bool(has_source_url(url))
        except Exception:
            return False
    return False


def _duckduckgo_query(url: str) -> str:
    """Return the normalized DuckDuckGo query string for a URL."""
    if _domain_for_url(url) != "duckduckgo.com":
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    query_values = parse_qs(parsed.query).get("q", [])
    if not query_values:
        return ""
    return " ".join(str(value).strip() for value in query_values if str(value).strip()).lower()


def _duckduckgo_target_site_domain(url: str) -> str | None:
    """Extract a site:domain restriction from a DuckDuckGo query when present."""
    query = _duckduckgo_query(url)
    if not query:
        return None
    match = re.search(r"site:([a-z0-9.-]+\.[a-z]{2,})", query)
    if not match:
        return None
    return match.group(1).removeprefix("www.")


def _duckduckgo_profile_query_allowed(url: str, ctx: ToolContext) -> bool:
    """Return True only for focused DuckDuckGo discovery queries that can yield profiles."""
    if _domain_for_url(url) != "duckduckgo.com":
        return True

    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.path.rstrip("/") != "/html":
        return False

    site_domain = _duckduckgo_target_site_domain(url)
    if not site_domain:
        return False

    if site_domain in ctx.unavailable_domains:
        return False

    site_outcome = ctx.domain_outcomes.get(site_domain)
    if site_outcome and site_outcome.banned_for_run:
        return False

    return True


def _remove_domain_from_candidate_pool(domain: str, ctx: ToolContext) -> None:
    """Remove a domain from the current run-local candidate pool."""
    ctx.allowed_domains.discard(domain)
    ctx.candidate_domains = [candidate for candidate in ctx.candidate_domains if candidate != domain]
    ctx.suggested_targets = [
        target for target in ctx.suggested_targets
        if str(target.get("domain", "")).strip().lower() != domain
    ]
    ctx.suggested_target_urls = {
        url for url in ctx.suggested_target_urls
        if _domain_for_url(url) != domain
    }


def _persistent_source_excluded_for_discovery(domain: str, ctx: ToolContext) -> bool:
    """Return True when discovery should not revisit a previously classified source."""
    if ctx.source_phase != "discovery" or ctx.source_state is None or not domain:
        return False

    kind, identifier = infer_source_identity(domain)
    status = ctx.source_state.source_status(kind, identifier)
    if status in {"approved", "temporary_seed", "pending_review", "rejected"}:
        return True

    metadata = ctx.source_state.metadata_for(kind, identifier)
    return bool(metadata.get("exhausted"))


def _mark_social_platform_unavailable(platform: str, reason: str, ctx: ToolContext) -> None:  # noqa: ARG001
    """Mark a social platform unavailable for the rest of the current run."""
    ctx.active_social_platforms.discard(platform)
    ctx.unavailable_social_platforms.add(platform)
    domain = domain_for_platform(platform)
    ctx.unavailable_domains.add(domain)
    _remove_domain_from_candidate_pool(domain, ctx)
    if ctx.source_mode == "all" and not ctx.active_social_platforms:
        ctx.effective_source_mode = "web"


def _same_or_subdomain(host: str, target_host: str) -> bool:
    host = host.removeprefix("www.")
    target_host = target_host.removeprefix("www.")
    return host == target_host or host.endswith(f".{target_host}")


def _uses_curated_target_pool(ctx: ToolContext) -> bool:
    """Return True when website=NA runs should stay inside the keyword-driven candidate pool."""
    return (
        ctx.source_mode in {"web", "human_emulator", "all"}
        and str(ctx.client_config.get("website", "NA")).upper() == "NA"
        and ctx.target_domain is None
    )


def _broad_mode_rejection(url: str, normalized_url: str, domain: str, ctx: ToolContext) -> str | None:
    """Return an error message when broad-mode fetch gating should reject a URL."""
    if not _uses_curated_target_pool(ctx):
        return None

    if not ctx.suggest_targets_called:
        return (
            "Broad website=NA runs require a keyword-driven target brief. Call suggest_targets first "
            "before fetching pages."
        )

    if domain in ctx.domain_outcomes and ctx.domain_outcomes[domain].banned_for_run:
        return (
            f"Domain '{domain}' has been banned for this run due to blocked or low-yield pages. "
            "Choose another allowed domain from suggest_targets."
        )

    if domain in ctx.avoid_domains:
        return (
            f"Domain '{domain}' is on the avoid list for this run. "
            "Choose another candidate domain from suggest_targets."
        )

    if domain not in ctx.allowed_domains and not _allow_discovered_follow_on_url(normalized_url, domain, ctx):
        return (
            f"URL domain '{domain}' is outside the candidate domain pool for this run. "
            "Use the suggest_targets brief instead of inventing new domains."
        )

    if _invented_direct_profile_url(url, normalized_url, domain, ctx):
        return (
            "Direct profile/detail URLs on allowed domains must come from suggest_targets or list_links. "
            "Use a suggested search seed or a discovered profile URL instead of inventing handles."
        )

    denied_reason = _broad_mode_denied_url(url)
    if denied_reason:
        return denied_reason

    return None


def _allow_discovered_follow_on_url(normalized_url: str, domain: str, ctx: ToolContext) -> bool:
    """Allow profile/detail follow-on URLs discovered from an approved search page."""
    parent_fetch_id = ctx.discovered_link_parents.get(normalized_url)
    if not parent_fetch_id:
        return False

    metadata = ctx.fetch_metadata.get(parent_fetch_id, {})
    parent_url = str(metadata.get("final_url") or metadata.get("url") or "")
    parent_domain = _domain_for_url(parent_url)
    parent_kind = str(metadata.get("page_kind", ""))

    if not any(_same_or_subdomain(parent_domain, allowed_domain) for allowed_domain in ctx.allowed_domains):
        return False
    if parent_kind not in {"search_results", "directory", "company_directory", "company_page", "unknown"}:
        return False
    if domain in ctx.unavailable_domains or domain in ctx.avoid_domains:
        return False
    outcome = ctx.domain_outcomes.get(domain)
    if outcome and outcome.banned_for_run:
        return False
    if _persistent_source_excluded_for_discovery(domain, ctx):
        return False
    return True


def _invented_direct_profile_url(url: str, normalized_url: str, domain: str, ctx: ToolContext) -> bool:
    """Return True when a direct profile URL was invented instead of discovered."""
    if normalized_url in ctx.suggested_target_urls:
        return False
    if _allow_discovered_follow_on_url(normalized_url, domain, ctx):
        return False
    return _is_profile_like_url(url, domain)


def _is_profile_like_url(url: str, domain: str) -> bool:
    """Return True for direct profile/detail URLs that should be sourced, not invented."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    path_parts = [part for part in parsed.path.split("/") if part]

    if domain == "github.com":
        reserved = {
            "about",
            "account",
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
            "notifications",
            "orgs",
            "organizations",
            "pricing",
            "pulls",
            "search",
            "sessions",
            "settings",
            "site",
            "sponsors",
            "topics",
            "trending",
            "users",
        }
        return len(path_parts) == 1 and path_parts[0].lower() not in reserved

    return False


def _broad_mode_denied_url(url: str) -> str | None:
    """Return a rejection reason for known low-value broad-mode seed URLs."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = _domain_for_url(url)
    path = parsed.path.rstrip("/") or "/"

    if host == "crunchbase.com":
        return "Crunchbase is blocked/authwall-heavy in broad web mode. Use another suggested target."

    if host in {"wellfound.com", "angel.co"} and path in {"/", "/company", "/company/"}:
        return "Wellfound/Angel home and company landing pages are not valid broad-mode starter targets."

    if host == "ycombinator.com" and path in {"/companies", "/companies/"}:
        return "Y Combinator company catalog pages are not valid broad-mode starter targets."

    if host in {"techcrunch.com", "producthunt.com"} and path == "/":
        return "Startup news or product homepages are not valid broad-mode starter targets."

    if host == "duckduckgo.com":
        if path != "/html":
            return "DuckDuckGo starter searches must use the HTML endpoint from suggest_targets, not ad hoc search pages."
        if "site:" not in _duckduckgo_query(url):
            return "DuckDuckGo discovery queries must stay site-restricted to profile domains from suggest_targets."

    return None


def _domain_outcome(ctx: ToolContext, domain: str) -> DomainOutcome:
    """Return the per-run memory object for a domain."""
    if domain not in ctx.domain_outcomes:
        ctx.domain_outcomes[domain] = DomainOutcome()
    return ctx.domain_outcomes[domain]


def _has_positive_domain_signal(outcome: DomainOutcome) -> bool:
    """Return True once a domain has shown useful discovery or saved-lead signals."""
    return bool(outcome.discovery_hits or outcome.profile_hits or outcome.saved_hits)


def _maybe_ban_domain(domain: str, ctx: ToolContext) -> None:
    """Ban a domain for the rest of the run when it is clearly blocked or low-yield."""
    outcome = _domain_outcome(ctx, domain)
    if _has_positive_domain_signal(outcome):
        return
    if outcome.blocked_count >= 1:
        outcome.banned_for_run = True
    if outcome.irrelevant_count >= 2:
        outcome.banned_for_run = True
    if outcome.banned_for_run:
        _remove_domain_from_candidate_pool(domain, ctx)


def _record_fetch_outcome(fetch_id: str, domain: str, page_kind: str, ctx: ToolContext) -> None:
    """Update per-run domain memory from a classified fetch result."""
    if fetch_id in ctx.accounted_fetch_ids or not domain:
        return

    outcome = _domain_outcome(ctx, domain)
    if page_kind == "blocked":
        outcome.blocked_count += 1
        outcome.last_reason = "blocked"
    elif page_kind in {"search_results", "directory", "company_directory", "company_page"}:
        outcome.discovery_hits += 1
        outcome.last_reason = page_kind
    elif page_kind == "profile":
        outcome.profile_hits += 1
        outcome.last_reason = "profile"
    elif page_kind in {"job_board", "landing_page", "article_or_news", "not_found"}:
        outcome.irrelevant_count += 1
        outcome.last_reason = page_kind
    else:
        return

    ctx.accounted_fetch_ids.add(fetch_id)
    _maybe_ban_domain(domain, ctx)


def _record_domain_failure(url: str, reason: str, ctx: ToolContext) -> None:
    """Update per-run domain memory when the agent explicitly fails a URL."""
    normalized_url = _normalize_url(url)
    fetch_id = ctx.url_to_fetch_id.get(normalized_url)
    domain = _domain_for_url(url)
    if not domain:
        return
    if fetch_id and fetch_id in ctx.accounted_fetch_ids:
        return

    outcome = _domain_outcome(ctx, domain)
    lowered_reason = reason.lower()
    if any(term in lowered_reason for term in ("blocked", "captcha", "authwall", "verification", "access denied")):
        outcome.blocked_count += 1
        outcome.last_reason = reason
    else:
        outcome.irrelevant_count += 1
        outcome.last_reason = reason
    if _domain_for_url(url) in _SUPPORTED_SOCIAL_PROFILE_DOMAINS:
        _record_social_blank_profile(url, ctx)
    if fetch_id:
        ctx.accounted_fetch_ids.add(fetch_id)
    _maybe_ban_domain(domain, ctx)


def _record_domain_save(url: str, ctx: ToolContext) -> None:
    """Mark a domain as having produced a saved lead this run."""
    domain = _domain_for_url(url)
    if not domain:
        return
    outcome = _domain_outcome(ctx, domain)
    outcome.saved_hits += 1
    outcome.last_reason = "saved"


def _source_stats_for_url(url: str, ctx: ToolContext) -> SourceRunStats:
    """Return or initialize per-run stats for the source behind a URL."""
    kind, identifier = infer_source_identity(url)
    domain = _domain_for_url(url)
    key = source_key(kind, identifier)
    if key not in ctx.source_run_stats:
        family = ""
        if ctx.source_state is not None:
            family = str(ctx.source_state.metadata_for(kind, identifier).get("family", ""))
        if not family:
            for target in ctx.suggested_targets:
                target_kind = str(target.get("source_kind", ""))
                target_id = str(target.get("source_id", ""))
                if target_kind == kind and target_id == identifier:
                    family = str(target.get("family", ""))
                    break
        if not family:
            family = infer_source_family(kind, identifier)
        ctx.source_run_stats[key] = SourceRunStats(
            kind=kind,
            identifier=identifier,
            domain=domain,
            family=family,
        )
    return ctx.source_run_stats[key]


def _record_source_fetch(url: str, ctx: ToolContext) -> None:
    """Increment fetch counts for the source behind a fetched URL."""
    stats = _source_stats_for_url(url, ctx)
    stats.fetch_count += 1


def _source_status_for_url(url: str, ctx: ToolContext) -> str:
    """Return the configured/persisted source status for a URL."""
    if ctx.source_state is None:
        return "approved"
    kind, identifier = infer_source_identity(url)
    return ctx.source_state.source_status(kind, identifier)


def _record_saved_lead(
    url: str,
    data: dict[str, Any],
    source_status: str,
    source_family: str,
    ctx: ToolContext,
) -> None:
    """Capture saved-lead metadata for baseline comparisons in this run."""
    kind, identifier = infer_source_identity(url)
    ctx.current_run_saved_leads.append(
        {
            "url": url,
            "data": dict(data),
            "source_status": source_status,
            "source_kind": kind,
            "source_identifier": identifier,
            "source_family": source_family or infer_source_family(kind, identifier),
        }
    )


async def _sample_discovered_source(
    url: str,
    data: dict[str, Any],
    source_stats: SourceRunStats,
    ctx: ToolContext,
) -> dict[str, Any] | None:
    """Hold newly discovered-source leads until the source is quality-gated."""
    kind, identifier = infer_source_identity(url)
    key = source_key(kind, identifier)
    async with _source_sample_lock(ctx, key):
        family = source_stats.family or infer_source_family(kind, identifier)
        sample_bucket = ctx.discovery_source_samples.setdefault(key, [])
        if any(existing["url"] == url for existing in sample_bucket):
            return {"status": "sampled", "url": url, "sample_count": len(sample_bucket), "required_samples": 3}

        sample_bucket.append({"url": url, "data": dict(data)})
        if len(sample_bucket) < 3:
            provisional_score, provisional_outcome = _provisional_source_review(sample_bucket, source_stats, ctx)
            print(
                f"  • Sampled discovered source lead {len(sample_bucket)}/3: {data.get('name') or url}",
                flush=True,
            )
            return {
                "status": "sampled",
                "url": url,
                "sample_count": len(sample_bucket),
                "required_samples": 3,
                "comparison_ready": len(sample_bucket) >= 2,
                "provisional_score": provisional_score,
                "provisional_outcome": provisional_outcome,
            }

        baseline = _baseline_leads(ctx, limit=3)
        source_score, lead_scores = _score_candidate_source(sample_bucket, source_stats, ctx)
        baseline_score = _baseline_score(baseline, ctx)
        outcome = _classify_candidate_source(source_score, baseline_score, ctx, exhausted=False)

        if outcome == "approved":
            ctx.source_state.promote_approved(kind, identifier, family, source_score)
            await _flush_sampled_leads(sample_bucket, "approved", family, ctx)
            ctx.discovery_source_samples.pop(key, None)
            return {
                "status": "approved_source",
                "url": url,
                "source": identifier,
                "score": source_score,
                "lead_scores": lead_scores,
            }

        if outcome == "temporary_seed":
            ctx.source_state.promote_temporary_seed(kind, identifier, family, source_score)
            await _flush_sampled_leads(sample_bucket, "temporary_seed", family, ctx)
            ctx.discovery_source_samples.pop(key, None)
            return {
                "status": "temporary_seed",
                "url": url,
                "source": identifier,
                "score": source_score,
                "lead_scores": lead_scores,
            }

        if outcome == "pending_review":
            ctx.source_state.queue_for_review(
                kind,
                identifier,
                family,
                source_score,
                [item["data"] | {"source_url": item["url"]} for item in sample_bucket],
                baseline,
            )
            ctx.discovery_source_samples.pop(key, None)
            return {
                "status": "pending_review",
                "url": url,
                "source": identifier,
                "score": source_score,
            }

        ctx.source_state.reject_source(kind, identifier, family, source_score)
        ctx.discovery_source_samples.pop(key, None)
        return {
            "status": "rejected_source",
            "url": url,
            "source": identifier,
            "score": source_score,
        }


def _provisional_source_review(
    sample_bucket: list[dict[str, Any]],
    source_stats: SourceRunStats,
    ctx: ToolContext,
) -> tuple[int, str]:
    """Score an in-flight discovered source before the full 3-sample gate is complete."""
    if len(sample_bucket) < 2:
        return 0, "insufficient_samples"
    baseline = _baseline_leads(ctx, limit=3)
    source_score, _lead_scores = _score_candidate_source(sample_bucket, source_stats, ctx)
    baseline_score = _baseline_score(baseline, ctx)
    outcome = _classify_candidate_source(source_score, baseline_score, ctx, exhausted=False)
    return source_score, outcome


async def _flush_sampled_leads(
    sample_bucket: list[dict[str, Any]],
    source_status: str,
    source_family: str,
    ctx: ToolContext,
) -> None:
    """Persist sampled leads once a discovered source passes its quality gate."""
    for item in sample_bucket:
        url = item["url"]
        data = item["data"]
        source_stats = _source_stats_for_url(url, ctx)
        status = await ctx.sheets_writer.append_row(url, data)
        if status == "saved":
            async with ctx.state_lock:
                _record_domain_save(url, ctx)
                source_stats.saved_count += 1
                _record_saved_lead(url, data, source_status, source_family, ctx)
                _mark_terminal_url(url, "saved", ctx)
                social_domain = _domain_for_url(url)
                social_platform = _social_platform_for_domain(social_domain)
                if social_platform:
                    ctx.social_platform_saves[social_platform] = ctx.social_platform_saves.get(social_platform, 0) + 1
            print(f"  ✓ Saved: {data.get('name') or url}", flush=True)
        else:
            async with ctx.state_lock:
                source_stats.duplicate_count += 1
                _mark_terminal_url(url, "duplicate", ctx)
            print(f"  • Duplicate skipped: {data.get('name') or url}", flush=True)


async def _finalize_partial_discovery_samples(ctx: ToolContext) -> list[dict[str, Any]]:
    """Resolve leftover sampled discovery sources when a run exhausts before 3 samples."""
    if ctx.source_state is None or not ctx.discovery_source_samples:
        return []

    finalized: list[dict[str, Any]] = []
    for key in list(ctx.discovery_source_samples):
        sample_bucket = ctx.discovery_source_samples.get(key, [])
        if not sample_bucket:
            ctx.discovery_source_samples.pop(key, None)
            continue

        kind, identifier = key.split(":", 1)
        source_stats = _source_stats_for_url(sample_bucket[0]["url"], ctx)
        family = source_stats.family or infer_source_family(kind, identifier)
        baseline = _baseline_leads(ctx, limit=3)
        source_score, lead_scores = _score_candidate_source(sample_bucket, source_stats, ctx)
        baseline_score = _baseline_score(baseline, ctx)
        outcome = _classify_candidate_source(source_score, baseline_score, ctx, exhausted=True)

        if outcome == "approved":
            ctx.source_state.promote_approved(kind, identifier, family, source_score)
            await _flush_sampled_leads(sample_bucket, "approved", family, ctx)
        elif outcome == "pending_review":
            ctx.source_state.queue_for_review(
                kind,
                identifier,
                family,
                source_score,
                [item["data"] | {"source_url": item["url"]} for item in sample_bucket],
                baseline,
            )
        else:
            ctx.source_state.reject_source(kind, identifier, family, source_score)

        ctx.discovery_source_samples.pop(key, None)
        finalized.append(
            {
                "source": identifier,
                "sample_count": len(sample_bucket),
                "outcome": outcome,
                "score": source_score,
                "lead_scores": lead_scores,
            }
        )

    return finalized


def _accuracy_thresholds(ctx: ToolContext) -> dict[str, int]:
    """Return the configured source-accuracy preset thresholds."""
    preset = str(ctx.client_config.get("source_accuracy", "balanced")).strip().lower() or "balanced"
    from source_state import SOURCE_ACCURACY_PRESETS

    return SOURCE_ACCURACY_PRESETS[preset]


def _baseline_leads(ctx: ToolContext, limit: int = 3) -> list[dict[str, Any]]:
    """Collect approved-source leads from this run first, then from recent history."""
    baseline = [
        {"source_url": item["url"], **item["data"]}
        for item in ctx.current_run_saved_leads
        if item.get("source_status") == "approved"
    ][:limit]
    if len(baseline) >= limit:
        return baseline

    if ctx.source_state is None or not hasattr(ctx.sheets_writer, "recent_rows"):
        return baseline

    approved_sources = ctx.source_state.approved_sources()
    approved_domains = set(approved_sources["web_domains"])
    approved_social = {domain_for_platform(platform) for platform in approved_sources["social_platforms"]}
    for row in ctx.sheets_writer.recent_rows(limit=20):
        row_url = str(row.get("source_url", "")).strip()
        domain = _domain_for_url(row_url)
        if domain not in approved_domains and domain not in approved_social:
            continue
        baseline.append(dict(row))
        if len(baseline) >= limit:
            break
    return baseline[:limit]


_RESEED_TITLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("principal software engineer", "Principal Software Engineer"),
    ("principal engineer", "Principal Engineer"),
    ("senior staff software engineer", "Senior Staff Software Engineer"),
    ("staff software engineer", "Staff Software Engineer"),
    ("staff engineer", "Staff Engineer"),
    ("solution architect", "Solution Architect"),
    ("software architect", "Software Architect"),
    ("cloud architect", "Cloud Architect"),
    ("architect", "Architect"),
    ("engineering manager", "Engineering Manager"),
    ("senior backend engineer", "Senior Backend Engineer"),
    ("senior frontend engineer", "Senior Frontend Engineer"),
    ("senior full stack engineer", "Senior Full Stack Engineer"),
    ("senior software engineer", "Senior Software Engineer"),
    ("software engineer", "Software Engineer"),
    ("platform engineer", "Platform Engineer"),
    ("backend engineer", "Backend Engineer"),
    ("frontend engineer", "Frontend Engineer"),
    ("full stack engineer", "Full Stack Engineer"),
)


def _prepare_target_reseed(ctx: ToolContext) -> dict[str, Any] | None:
    """Derive fresh keyword-driven starter terms from recent approved leads."""
    if str(ctx.client_config.get("website", "NA")).strip().upper() != "NA":
        return None
    if ctx.source_phase != "pass1":
        return None
    if (ctx.target_strategy or "") not in {"technical_profiles", ""}:
        return None

    baseline = _baseline_leads(ctx, limit=3)
    if not baseline:
        return None

    terms = _derive_reseed_search_terms(baseline, ctx)
    areas = _derive_reseed_area_variants(baseline, ctx)
    if not terms and not areas:
        return None

    if (
        tuple(terms) == tuple(ctx.reseed_search_terms)
        and tuple(areas) == tuple(ctx.reseed_area_variants)
        and ctx.reseed_generation > 0
    ):
        return None

    ctx.reseed_search_terms = list(terms)
    ctx.reseed_area_variants = list(areas)
    ctx.reseed_generation += 1
    return {
        "baseline": baseline,
        "terms": list(terms),
        "areas": list(areas),
        "generation": ctx.reseed_generation,
    }


def _derive_reseed_search_terms(baseline_leads: list[dict[str, Any]], ctx: ToolContext) -> list[str]:
    """Extract stronger follow-on title queries from recent approved leads."""
    derived: list[str] = []
    for row in baseline_leads[:3]:
        job_title = str(row.get("job_title") or "").strip().lower()
        if not job_title:
            continue
        normalized_title = " ".join(job_title.replace("|", " ").replace("/", " ").replace(",", " ").split())
        for pattern, canonical in _RESEED_TITLE_PATTERNS:
            if pattern in normalized_title:
                derived.append(canonical)

    fallback_terms = [
        str(ctx.client_config.get("job_title", "")).strip(),
        "Staff Engineer",
        "Principal Engineer",
        "Solution Architect",
    ]
    derived.extend(fallback_terms)

    seen: set[str] = set()
    ordered: list[str] = []
    existing_terms = {
        str(ctx.client_config.get("job_title", "")).strip().lower(),
        "senior software engineer",
        "software engineer",
        "architect",
    }
    for term in derived:
        cleaned = str(term).strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        if lowered in existing_terms and ordered:
            continue
        ordered.append(cleaned)
    return ordered[:8]


def _derive_reseed_area_variants(baseline_leads: list[dict[str, Any]], ctx: ToolContext) -> list[str]:
    """Return broader area variants hinted by config and recent saved leads."""
    configured_area = str(ctx.client_config.get("area", "NA")).strip() or "NA"
    variants: list[str] = []
    if configured_area.upper() != "NA":
        variants.append(configured_area)

    text_blob = " ".join(
        str(value).lower()
        for row in baseline_leads[:3]
        for value in (row.get("job_title", ""), row.get("company", ""), row.get("social_media", ""))
    )
    if "san francisco bay area" in text_blob:
        variants.extend(["San Francisco Bay Area", "San Francisco", "Bay Area"])
    elif "san francisco" in text_blob:
        variants.extend(["San Francisco", "Bay Area"])
    elif "bay area" in text_blob:
        variants.extend(["San Francisco Bay Area", "Bay Area"])

    deduped: list[str] = []
    seen: set[str] = set()
    for value in variants:
        cleaned = str(value).strip()
        lowered = cleaned.lower()
        if not cleaned or cleaned.upper() == "NA" or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(cleaned)
    return deduped[:3]


def _lead_quality_score(data: dict[str, Any], ctx: ToolContext) -> int:
    """Deterministically score a lead row from 0 to 100."""
    score = 0
    if _is_plausible_person_name(data.get("name")):
        score += 15

    job_title = str(data.get("job_title") or "").strip()
    if job_title:
        score += 15
        title_lower = job_title.lower()
        combined = " ".join(
            str(value).lower()
            for value in (
                ctx.client_config.get("job_title", ""),
                ctx.client_config.get("job", ""),
            )
        )
        if any(token and token in title_lower for token in combined.split()):
            score += 20
        seniority_terms = ("founder", "cto", "chief", "head", "vp", "director", "principal", "staff", "senior")
        if any(term in title_lower for term in seniority_terms):
            score += 10

    if _has_meaningful_value(data.get("company")):
        score += 20
    if _has_meaningful_value(data.get("email")):
        score += 10
    if _has_meaningful_value(data.get("phone")):
        score += 5
    if _has_meaningful_value(data.get("social_media")):
        score += 10

    location = str(data.get("location") or "").strip().lower()
    area = str(ctx.client_config.get("area", "NA")).strip().lower()
    if location and area not in {"", "na"} and area in location:
        score += 10

    return min(score, 100)


def _score_candidate_source(
    sample_bucket: list[dict[str, Any]],
    source_stats: SourceRunStats,
    ctx: ToolContext,
) -> tuple[int, list[int]]:
    """Score a candidate source based on lead quality and source-level penalties."""
    lead_scores = [_lead_quality_score(item["data"], ctx) for item in sample_bucket[:3]]
    average = int(round(sum(lead_scores) / max(1, len(lead_scores))))
    penalty = min(20, source_stats.duplicate_count * 5 + source_stats.rejected_count * 8)
    return max(0, average - penalty), lead_scores


def _baseline_score(baseline_leads: list[dict[str, Any]], ctx: ToolContext) -> int:
    """Compute the baseline approved-source score for comparison."""
    if not baseline_leads:
        return 0
    scores = [_lead_quality_score(row, ctx) for row in baseline_leads[:3]]
    return int(round(sum(scores) / max(1, len(scores))))


def _classify_candidate_source(
    candidate_score: int,
    baseline_score: int,
    ctx: ToolContext,
    exhausted: bool,
) -> str:
    """Classify a candidate source into approved, temporary_seed, pending_review, or rejected."""
    thresholds = _accuracy_thresholds(ctx)
    gap = candidate_score - baseline_score if baseline_score else 0

    if candidate_score >= thresholds["approved_threshold"] and gap >= thresholds["approved_gap"]:
        return "approved"
    if (
        not exhausted
        and candidate_score >= thresholds["temporary_seed_threshold"]
        and gap >= thresholds["temporary_seed_gap"]
    ):
        return "temporary_seed"
    if candidate_score >= thresholds["review_threshold"] and gap >= thresholds["review_gap"]:
        return "pending_review"
    return "rejected"


def _curated_target_pool_exhausted(ctx: ToolContext) -> bool:
    """Return True when all candidate domains have been tried or banned."""
    if not _uses_curated_target_pool(ctx) or not ctx.suggest_targets_called:
        return False

    if _has_remaining_target_urls(ctx):
        return False

    candidate_domains = ctx.candidate_domains or sorted(ctx.allowed_domains)
    if not candidate_domains:
        return False

    fetched_domains = {
        _domain_for_url(str(metadata.get("final_url") or metadata.get("url") or ""))
        for metadata in ctx.fetch_metadata.values()
    }
    for domain in candidate_domains:
        outcome = ctx.domain_outcomes.get(domain)
        if domain not in fetched_domains and not (outcome and outcome.banned_for_run):
            return False

    return not any(fetch_id not in ctx.processed_fetch_ids for fetch_id in ctx.fetch_metadata)


def _has_remaining_target_urls(ctx: ToolContext) -> bool:
    """Return True when suggest_targets still has starter URLs that have not been tried."""
    for target in ctx.suggested_targets:
        if not isinstance(target, dict):
            continue
        url = str(target.get("url", "")).strip()
        if not url:
            continue
        normalized_url = _normalize_url(url)
        domain = _domain_for_url(url)
        if domain in ctx.unavailable_domains:
            continue
        if _url_budget_exhausted(url, ctx):
            continue
        if _terminal_outcome_for_url(url, ctx):
            continue
        if normalized_url in ctx.exhausted_discovery_urls:
            continue
        if normalized_url in ctx.url_to_fetch_id:
            continue
        return True
    return False


def _is_plausible_person_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    name = value.strip()
    if len(name) < 2 or len(name) > 80:
        return False
    banned = {
        "sign in",
        "log in",
        "login",
        "home",
        "jobs",
        "careers",
        "just a moment",
        "page not found",
        "directory",
        "search results",
    }
    lowered = name.lower()
    if lowered in banned:
        return False
    return any(char.isalpha() for char in name)


def _looks_like_handle_name(value: Any) -> bool:
    """Return True when a name is more likely a username than a real person name."""
    if not isinstance(value, str):
        return False
    name = value.strip()
    if not name:
        return False
    if any(char in name for char in "@/_."):
        return True
    if any(char.isdigit() for char in name):
        return True
    parts = name.split()
    if len(parts) >= 2:
        return False
    token = parts[0] if parts else name
    if token.lower() == token:
        return True
    return token.upper() == token and len(token) > 4


def _has_strong_handle_support(data: dict[str, Any], source_url: str) -> bool:
    """Require stronger evidence before accepting a handle-like name as a viable lead."""
    job_title = str(data.get("job_title") or "").strip()
    distinct_social = False
    social_media = data.get("social_media")
    if _has_meaningful_value(social_media):
        distinct_social = _normalize_url(str(social_media)) != _normalize_url(source_url)

    has_company = _has_meaningful_value(data.get("company"))
    has_direct_contact = any(_has_meaningful_value(data.get(field)) for field in ("email", "phone"))
    has_role = _looks_like_title_fragment(job_title)
    senior_role = any(
        term in job_title.lower()
        for term in ("senior", "staff", "lead", "principal", "architect")
    )

    if has_company and has_role:
        return True
    if has_direct_contact and has_role:
        return True
    if distinct_social and senior_role:
        return True
    return False


def _has_meaningful_value(value: Any) -> bool:
    """Return True when a field value is present enough to count as supporting lead data."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null", "n/a", "na"}
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _same_normalized_text(left: Any, right: Any) -> bool:
    """Return True when two text fields normalize to the same content."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return _clean_field_text(left).casefold() == _clean_field_text(right).casefold()


def _normalize_lead_payload(source_url: str, data: dict[str, Any]) -> dict[str, Any]:
    """Clean noisy extracted lead fields without inventing new facts."""
    normalized = dict(data or {})
    for key, value in list(normalized.items()):
        if isinstance(value, str):
            cleaned = _clean_field_text(value)
            normalized[key] = cleaned if cleaned else None

    raw_job_title = normalized.get("job_title")
    role_hint_title, role_hint_company = _extract_role_company_from_bio_text(raw_job_title)
    normalized["job_title"] = _normalize_job_title_value(normalized.get("job_title"))
    normalized["company"] = _normalize_company_value(normalized.get("company"))
    if _has_meaningful_value(role_hint_title):
        normalized["job_title"] = role_hint_title
    if _has_meaningful_value(role_hint_company) and (
        not _has_meaningful_value(normalized.get("company"))
        or _should_prefer_role_hint_company(raw_job_title)
    ):
        normalized["company"] = role_hint_company
    if not _has_meaningful_value(normalized.get("company")):
        hinted_company = _derive_company_from_job_title_hint(raw_job_title)
        if _has_meaningful_value(hinted_company):
            normalized["company"] = hinted_company
    derived_title, derived_company = _split_title_company(
        normalized.get("job_title"),
        normalized.get("company"),
    )
    normalized["job_title"] = derived_title
    normalized["company"] = derived_company
    normalized["social_media"] = _normalize_social_value(normalized.get("social_media"), source_url)
    normalized["website"] = _normalize_website_value(
        normalized.get("website"),
        fallback_social=normalized.get("social_media"),
    )
    if _same_normalized_text(normalized.get("job_title"), normalized.get("name")):
        normalized["job_title"] = None
    return normalized


def _clean_field_text(value: str) -> str:
    """Collapse noisy whitespace in extracted fields."""
    return " ".join(value.replace("\xa0", " ").split())


def _normalize_job_title_value(value: Any) -> Any:
    """Trim common GitHub-style title noise while preserving a plausible role."""
    if not isinstance(value, str):
        return value
    text = _clean_field_text(value)
    if not text:
        return None

    first_line = value.splitlines()[0].strip()
    if first_line and _looks_like_title_fragment(first_line):
        text = _clean_field_text(first_line)

    if "@" in text:
        first_segment = _clean_field_text(text.split("@", 1)[0])
        if first_segment and _looks_like_title_fragment(first_segment):
            text = first_segment

    if " {" in text and _looks_like_title_fragment(text.split(" {", 1)[0].strip()):
        text = text.split(" {", 1)[0].strip()

    text = re.sub(r"\s+at\s*$", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip(".,;:").strip()

    return text or None


def _normalize_company_value(value: Any) -> Any:
    """Simplify common company strings without guessing across multiple employers."""
    if not isinstance(value, str):
        return value
    text = _clean_field_text(value)
    if not text:
        return None
    if text.casefold() in _LOW_QUALITY_COMPANY_VALUES:
        return None

    tokens = [_humanize_org_token(token) for token in text.split("@") if token.strip()]
    if text.startswith("@") and tokens:
        return tokens[0]

    if len(tokens) >= 2:
        first_token_lower = tokens[0].lower()
        if any(term in first_token_lower for term in _COMPANY_TEAM_HINTS):
            return tokens[-1]

    return text


def _split_title_company(job_title: Any, company: Any) -> tuple[Any, Any]:
    """Extract trailing company text from a title when it is explicitly present."""
    if not isinstance(job_title, str):
        return job_title, company
    title = job_title.strip()
    if not title:
        return None, company

    match = re.match(r"^(?P<title>.+?)\s+at\s+(?P<company>.+)$", title, flags=re.IGNORECASE)
    if not match:
        return title, company

    clean_title = match.group("title").strip()
    explicit_company = _normalize_company_value(match.group("company").strip())
    resolved_company = explicit_company or company
    return clean_title or title, resolved_company


def _extract_role_company_from_bio_text(job_title: Any) -> tuple[Any, Any]:
    """Recover a cleaner role/company pair from multi-segment GitHub-style bio text."""
    if not isinstance(job_title, str):
        return None, None
    text = _clean_field_text(job_title)
    if not text:
        return None, None

    segments = [segment.strip(" -–—>•|·") for segment in re.split(r"[•|·]+", text) if segment.strip()]
    for segment in segments:
        candidate_title, candidate_company = _split_title_company(segment, None)
        if isinstance(candidate_title, str) and "@" in candidate_title:
            left, right = [part.strip() for part in candidate_title.split("@", 1)]
            if _looks_like_title_fragment(left):
                return _normalize_job_title_value(left), _normalize_company_value(f"@{right}")
        if isinstance(candidate_title, str) and _looks_like_title_fragment(candidate_title):
            normalized_title = _normalize_job_title_value(candidate_title)
            normalized_company = _normalize_company_value(candidate_company)
            return normalized_title, normalized_company

    return None, None


def _derive_company_from_job_title_hint(job_title: Any) -> Any:
    """Recover simple handle-style company text embedded in a job-title string."""
    if not isinstance(job_title, str):
        return None
    text = _clean_field_text(job_title)
    if not text or text.count("@") != 1:
        return None

    title_hint, company_hint = [part.strip() for part in text.split("@", 1)]
    if not title_hint or not company_hint:
        return None
    title_hint = re.sub(r"\s+at\s*$", "", title_hint, flags=re.IGNORECASE).strip()
    if not _looks_like_title_fragment(title_hint):
        return None
    return _normalize_company_value(f"@{company_hint}")


def _should_prefer_role_hint_company(job_title: Any) -> bool:
    """Return True when a role-derived company should override a weaker parsed company."""
    if not isinstance(job_title, str):
        return False
    return any(marker in job_title for marker in ("•", "·", "|")) or job_title.count("@") >= 2


def _normalize_social_value(value: Any, source_url: str) -> Any:
    """Normalize social/profile values while preserving the original source URL when helpful."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text.startswith("in/"):
        return f"https://www.linkedin.com/{text}"
    if text.startswith("www."):
        return f"https://{text}"
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if _domain_for_url(source_url) in _SUPPORTED_SOCIAL_PROFILE_DOMAINS:
        return source_url
    return text


def _normalize_website_value(value: Any, fallback_social: Any = None) -> Any:
    """Keep websites normalized and promote non-social URLs into website when useful."""
    if isinstance(value, str) and value.strip():
        website = value.strip()
    elif isinstance(fallback_social, str) and fallback_social.strip() and not _looks_like_social_profile_url(fallback_social):
        website = fallback_social.strip()
    else:
        return value

    if website.startswith("www."):
        return f"https://{website}"
    return website


def _looks_like_title_fragment(value: str) -> bool:
    """Return True when a fragment looks like a plausible job title."""
    lowered = value.strip().lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "engineer",
            "developer",
            "architect",
            "founder",
            "chief",
            "cto",
            "manager",
            "director",
            "lead",
            "scientist",
            "analyst",
            "designer",
            "sales",
            "marketing",
            "product",
            "quality",
        )
    )


def _humanize_org_token(value: str) -> str:
    """Turn simple handle-like company tokens into readable text."""
    token = value.strip().strip("@")
    if not token:
        return token
    token = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
    token = token.replace("_", " ").replace("-", " ")
    if token.islower() or token.istitle():
        return token.title()
    if token.isupper() and len(token) <= 5:
        return token
    return token


def _looks_like_social_profile_url(value: str) -> bool:
    """Return True when a value already points at a social/profile URL."""
    domain = _domain_for_url(value)
    return domain in {*_SUPPORTED_SOCIAL_PROFILE_DOMAINS, "github.com", "gitlab.com"}


def _is_minimally_viable_lead(data: dict[str, Any], source_url: str) -> tuple[bool, str]:
    """Return whether a lead is strong enough to count toward the run target."""
    name = data.get("name")
    if not _is_plausible_person_name(name):
        return False, "missing a plausible person identifier in name"
    if _looks_like_handle_name(name) and not _has_strong_handle_support(data, source_url):
        return False, "name looks like a handle without enough supporting evidence"

    job_title = data.get("job_title")
    company = data.get("company")
    has_job_title = _has_meaningful_value(job_title) and not _same_normalized_text(job_title, name)
    has_company = _has_meaningful_value(company)
    has_direct_contact = any(_has_meaningful_value(data.get(field)) for field in ("email", "phone"))
    if has_job_title or has_company or has_direct_contact:
        return True, ""

    social_media = data.get("social_media")
    if _has_meaningful_value(social_media):
        normalized_social = _normalize_url(str(social_media))
        normalized_source = _normalize_url(source_url)
        if normalized_social != normalized_source:
            return True, ""

    return False, "missing supporting data (job_title, company, email, phone, or distinct social_media)"


def _is_blank_social_profile_data(source_url: str, data: dict[str, Any]) -> bool:
    """Return True when a social profile has no meaningful person or support fields."""
    domain = _domain_for_url(source_url)
    if domain not in _SUPPORTED_SOCIAL_PROFILE_DOMAINS:
        return False

    name = data.get("name")
    job_title = data.get("job_title")
    company = data.get("company")
    email = data.get("email")
    phone = data.get("phone")
    website = data.get("website")

    if _is_plausible_person_name(name):
        return False
    if any(_has_meaningful_value(value) for value in (job_title, company, email, phone, website)):
        return False
    return True


def _selector_map_for_fetch(
    fetch_id: str,
    field_names: list[str] | None,
    ctx: ToolContext,
) -> dict[str, list[str]]:
    """Return built-in selector presets for the given fetched page."""
    metadata = ctx.fetch_metadata.get(fetch_id, {})
    final_url = metadata.get("final_url") or metadata.get("url") or ""
    domain = _domain_for_url(final_url)
    requested_fields = field_names or list(ctx.client_config.get("fields", {}).keys())

    if domain == "github.com":
        return {
            field: _GITHUB_PROFILE_SELECTORS[field]
            for field in requested_fields
            if field in _GITHUB_PROFILE_SELECTORS
        }

    return {
        field: _GENERIC_PROFILE_SELECTORS[field]
        for field in requested_fields
        if field in _GENERIC_PROFILE_SELECTORS
    }


def _coerce_save_result_args(arguments: dict[str, Any], ctx: ToolContext) -> tuple[str, dict[str, Any]]:
    """Normalize save_result arguments from strict or loose Ollama tool-call shapes."""
    if not isinstance(arguments, dict):
        return "", {}

    url = arguments.get("url")
    data = arguments.get("data")
    fetch_id = arguments.get("fetch_id")

    if isinstance(url, str) and isinstance(data, dict):
        return url, data

    if isinstance(fetch_id, str):
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        inferred_url = metadata.get("final_url") or metadata.get("url") or ""
        if isinstance(data, dict):
            return inferred_url, data

        if fetch_id in ctx.parsed_results:
            parsed = ctx.parsed_results[fetch_id].copy()
        else:
            parsed = {}

        control_keys = {"fetch_id", "url", "data"}
        for key, value in arguments.items():
            if key in control_keys:
                continue
            parsed[key] = None if value in {"None", "null"} else value

        return inferred_url, parsed

    if isinstance(url, str):
        if isinstance(data, dict):
            return url, data
        control_keys = {"fetch_id", "url", "data"}
        parsed = {
            key: (None if value in {"None", "null"} else value)
            for key, value in arguments.items()
            if key not in control_keys
        }
        return url, parsed

    return "", {}


def _coerce_fetch_page_args(arguments: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    """Normalize fetch_page arguments from strict or loose tool-call shapes."""
    if not isinstance(arguments, dict):
        return "", True

    url = arguments.get("url")
    fetch_id = arguments.get("fetch_id")
    if not isinstance(url, str) or not url.strip():
        if isinstance(fetch_id, str):
            metadata = ctx.fetch_metadata.get(fetch_id, {})
            inferred_url = metadata.get("final_url") or metadata.get("url") or ""
            if inferred_url:
                url = inferred_url
        if not isinstance(url, str) or not url.strip():
            return "", True

    needs_javascript = arguments.get("needs_javascript")
    if isinstance(needs_javascript, str):
        lowered = needs_javascript.strip().lower()
        if lowered in {"true", "1", "yes"}:
            needs_javascript = True
        elif lowered in {"false", "0", "no"}:
            needs_javascript = False
        else:
            needs_javascript = None

    if not isinstance(needs_javascript, bool):
        domain = _domain_for_url(url)
        needs_javascript = domain in {
            "linkedin.com",
            "instagram.com",
            "snapchat.com",
            "x.com",
            "twitter.com",
            "github.com",
            "gitlab.com",
            "ycombinator.com",
        }

    return str(url).strip(), bool(needs_javascript)


def _coerce_fetch_id(arguments: dict[str, Any], ctx: ToolContext) -> str:
    """Infer a valid fetch_id from tool arguments."""
    if not isinstance(arguments, dict):
        return ""

    fetch_id = arguments.get("fetch_id")
    if isinstance(fetch_id, str) and fetch_id in ctx.fetch_metadata:
        return fetch_id

    url = arguments.get("url")
    if isinstance(url, str) and url.strip():
        return _lookup_fetch_id(url, ctx)

    return ""


def _coerce_fail_url(arguments: dict[str, Any], ctx: ToolContext) -> str:
    """Normalize fail_url arguments from strict or loose Ollama tool-call shapes."""
    if not isinstance(arguments, dict):
        return ""

    url = arguments.get("url")
    if isinstance(url, str) and url.strip():
        return url

    fetch_id = arguments.get("fetch_id")
    if isinstance(fetch_id, str):
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        inferred_url = metadata.get("final_url") or metadata.get("url") or ""
        if inferred_url:
            return inferred_url

    return ""


def _postprocess_extracted_fields(
    fetch_id: str,
    result: dict[str, Any],
    field_names: list[str] | None,
    ctx: ToolContext,
) -> dict[str, Any]:
    """Clean up extracted fields using page-specific metadata."""
    metadata = ctx.fetch_metadata.get(fetch_id, {})
    final_url = metadata.get("final_url") or metadata.get("url") or ""
    title = metadata.get("title", "")
    domain = _domain_for_url(final_url)
    requested_fields = set(field_names or result.keys())

    if domain == "github.com":
        username = _github_username_from_url(final_url)
        derived_title = _github_role_from_title(title, username)
        name = result.get("name")
        job_title = result.get("job_title")
        if isinstance(derived_title, str):
            derived_title = derived_title.strip() or None
        if isinstance(derived_title, str) and name and _same_normalized_text(derived_title, name):
            derived_title = None
        if isinstance(derived_title, str) and not _looks_like_title_fragment(derived_title):
            derived_title = None

        if _looks_like_role_text(name):
            result["name"] = username or name

        if "job_title" in requested_fields and not result.get("job_title") and derived_title:
            result["job_title"] = derived_title

        if "job_title" in requested_fields and result.get("job_title") == result.get("name") and derived_title:
            result["job_title"] = derived_title

        if "social_media" in requested_fields and not result.get("social_media"):
            result["social_media"] = final_url

        if "name" in requested_fields and not result.get("name") and username:
            result["name"] = username

        if job_title and username and str(job_title).strip() == username:
            result["job_title"] = derived_title

    if domain in _SUPPORTED_SOCIAL_PROFILE_DOMAINS:
        extracted_data = metadata.get("extracted_data", {})
        if not isinstance(extracted_data, dict):
            extracted_data = {}
        hint_data = ctx.social_profile_hints.get(_normalize_url(final_url), {})
        if not hint_data:
            hint_data = ctx.social_profile_hints.get(final_url, {})
        fallback_social: dict[str, Any] = {}
        for field_name in {"name", "job_title", "company", "email", "phone", "website", "social_media", "headline"}:
            extracted_value = extracted_data.get(field_name)
            if field_name == "name" and isinstance(extracted_value, str) and extracted_value.strip().lower() == "unknown":
                extracted_value = None
            if field_name == "social_media" and isinstance(extracted_value, str) and extracted_value.strip().lower() in {"profile", "social", ""}:
                extracted_value = None
            fallback_social[field_name] = extracted_value if _has_meaningful_value(extracted_value) else hint_data.get(field_name)

        if "social_media" in requested_fields and (
            not result.get("social_media")
            or str(result.get("social_media")).strip().lower() in {"profile", "social", ""}
        ):
            result["social_media"] = final_url

        if "name" in requested_fields and (
            not result.get("name")
            or str(result.get("name")).strip().lower() == "unknown"
        ):
            result["name"] = (
                fallback_social.get("name")
                or _social_handle_from_url(final_url)
            )

        if "job_title" in requested_fields and not result.get("job_title"):
            result["job_title"] = fallback_social.get("job_title") or fallback_social.get("headline")

        if "company" in requested_fields and not result.get("company"):
            result["company"] = fallback_social.get("company")

        if "email" in requested_fields and not result.get("email"):
            result["email"] = fallback_social.get("email")

        if "phone" in requested_fields and not result.get("phone"):
            result["phone"] = fallback_social.get("phone")

        if "website" in requested_fields and not result.get("website"):
            result["website"] = fallback_social.get("website")

    return result


def _github_username_from_url(url: str) -> str | None:
    """Return the GitHub username from a profile URL."""
    parsed = urlparse(url)
    parts = [segment for segment in parsed.path.split("/") if segment]
    if len(parts) == 1:
        return parts[0]
    return None


def _github_role_from_title(title: str, username: str | None) -> str | None:
    """Extract the role text from a GitHub profile page title."""
    if not title:
        return None
    pattern = r"^(.*?) \((.*?)\) · GitHub$"
    match = re.match(pattern, title)
    if not match:
        return None
    first, second = match.groups()
    if username and first.strip() == username:
        return second.strip() or None
    if username and second.strip() == username:
        return first.strip() or None
    return second.strip() or None


def _looks_like_role_text(value: Any) -> bool:
    """Return True when a value looks more like a role/headline than a person identifier."""
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    if not lowered:
        return False
    role_words = {
        "engineer",
        "developer",
        "designer",
        "manager",
        "founder",
        "consultant",
        "architect",
        "marketer",
        "scientist",
    }
    return any(word in lowered for word in role_words)


def _social_handle_from_url(url: str) -> str | None:
    """Return a username or handle from a social profile URL."""
    parsed = urlparse(url)
    parts = [segment for segment in parsed.path.split("/") if segment]
    if len(parts) == 1:
        return parts[0]
    return None

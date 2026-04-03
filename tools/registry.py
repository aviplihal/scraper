"""Tool registry: tool definitions plus the async dispatcher.

Five tools are exposed to the agent:
  1. fetch_page     — fetch a URL and classify the page
  2. list_links     — extract candidate navigation/profile links from a page
  3. parse_html     — extract fields from previously fetched HTML via CSS selectors
  4. save_result    — write a lead row to the client's SQLite database
  5. fail_url       — log a URL failure and skip it

A ToolContext object is passed to every tool so they can share state
(browser, page cache, metadata, crawler state, human emulator, storage writer)
without globals.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from tools.discovery import build_preview, classify_page, extract_links
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
    "social_media": [
        "a[href*='linkedin.com']",
        "a[href*='twitter.com']",
        "a[href*='x.com']",
        "a[href*='github.com']",
    ],
}

# Social-media domains that must be routed to the human emulator
SOCIAL_MEDIA_DOMAINS: frozenset[str] = frozenset(
    ["linkedin.com", "facebook.com", "instagram.com", "twitter.com", "x.com"]
)


def _is_social_media(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in SOCIAL_MEDIA_DOMAINS)


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


# ---------------------------------------------------------------------------
# Tool context
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Shared state passed to every tool execution."""
    client_config:  dict
    sheets_writer:  Any                              # storage.writer.StorageWriter
    source_mode:    str             = "web"
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
    parsed_results:  dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected_weak_count: int           = 0
    domain_outcomes: dict[str, DomainOutcome] = field(default_factory=dict)
    accounted_fetch_ids: set[str]      = field(default_factory=set)
    suggest_targets_called: bool       = False
    suggested_targets: list[dict[str, Any]] = field(default_factory=list)
    suggested_target_urls: set[str]    = field(default_factory=set)
    allowed_domains: set[str]          = field(default_factory=set)
    target_strategy: str | None        = None


# ---------------------------------------------------------------------------
# Tool definitions (Ollama / OpenAI format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "suggest_targets",
            "description": (
                "Return ranked starter URLs for broad-mode discovery when website is NA. "
                "Use this before your first fetch in web broad mode so you start from curated, "
                "persona-relevant people discovery pages."
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
    if tool_name == "suggest_targets":
        return _tool_suggest_targets(arguments.get("limit", 8), ctx)
    if tool_name == "fetch_page":
        return await _tool_fetch_page(arguments["url"], arguments["needs_javascript"], ctx)
    elif tool_name == "list_links":
        return _tool_list_links(
            arguments["fetch_id"],
            arguments.get("selector"),
            arguments.get("limit", 25),
            arguments.get("same_domain_only", False),
            ctx,
        )
    elif tool_name == "parse_html":
        return _tool_parse_html(
            arguments["fetch_id"],
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
    """Return ranked starter URLs for the current run and arm broad-mode gating."""
    result = suggest_targets(ctx.client_config, ctx.source_mode, limit=max(1, min(int(limit), 20)))
    targets = result.get("targets", [])
    ctx.suggest_targets_called = True
    ctx.suggested_targets = list(targets)
    ctx.target_strategy = str(result.get("strategy", "unknown"))
    ctx.suggested_target_urls = {
        _normalize_url(str(target["url"]))
        for target in targets
        if isinstance(target, dict) and target.get("url")
    }
    ctx.allowed_domains = {
        _domain_for_url(str(target["url"]))
        for target in targets
        if isinstance(target, dict) and target.get("url")
    }
    return result


async def _tool_fetch_page(url: str, needs_javascript: bool, ctx: ToolContext) -> dict:
    logger.info("fetch_page: %s (js=%s)", url, needs_javascript)
    normalized_url = _normalize_url(url)
    domain = _domain_for_url(url)

    if normalized_url in ctx.url_to_fetch_id:
        fetch_id = ctx.url_to_fetch_id[normalized_url]
        cached = ctx.fetch_metadata[fetch_id].copy()
        cached["fetch_id"] = fetch_id
        return cached

    if _is_social_media(url):
        if ctx.source_mode == "web":
            ctx.fetch_error_count += 1
            return {
                "error": "Social-media URLs are not allowed in web mode.",
                "url": url,
            }

        ctx.fetch_count += 1
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

    if ctx.domain_fetch_counts.get(domain, 0) >= _MAX_FETCHES_PER_DOMAIN:
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

    ctx.fetch_count += 1
    ctx.domain_fetch_counts[domain] = ctx.domain_fetch_counts.get(domain, 0) + 1

    fetch_id = str(uuid.uuid4())
    ctx.page_cache[fetch_id] = fetch_result.html
    page_info = classify_page(url, fetch_result.final_url, fetch_result.html)
    preview = build_preview(fetch_result.html)
    metadata = {
        "url": url,
        "final_url": page_info.final_url,
        "title": page_info.title,
        "page_kind": page_info.page_kind,
        "preview": preview,
    }
    ctx.fetch_metadata[fetch_id] = metadata
    ctx.url_to_fetch_id[normalized_url] = fetch_id
    ctx.url_to_fetch_id[_normalize_url(page_info.final_url)] = fetch_id
    ctx.visited_urls.add(_normalize_url(page_info.final_url))
    ctx.visited_urls.add(normalized_url)
    _record_fetch_outcome(fetch_id, domain, page_info.page_kind, ctx)

    return {"fetch_id": fetch_id, **metadata}


async def _fetch_social_media(url: str, ctx: ToolContext) -> dict:
    """Route a social-media URL through the human emulator."""
    if ctx.emulator_browser is None or ctx.emulator_state is None:
        return {
            "error": "Human emulator not available for this job. Run with --source all or --source human_emulator.",
            "url": url,
        }

    logger.info("Routing social-media URL to human emulator: %s", url)
    collected: list[dict] = []

    async def on_result(visited_url: str, data: dict) -> None:
        collected.append(data)

    from human_emulator.linkedin import LinkedInEmulator

    # Instantiate a short-lived emulator just for this URL
    context = getattr(ctx.emulator_browser, "_context", None)
    if context is None:
        context = await ctx.emulator_browser.start()
    emulator = LinkedInEmulator(context, ctx.emulator_state, ctx.client_config["client_id"])
    await emulator.process_profiles([url], on_result)

    if not collected:
        return {"error": "Human emulator returned no data.", "url": url}

    data = collected[0]
    fetch_id = str(uuid.uuid4())
    # Store a synthetic HTML blob of the extracted data for parse_html compatibility
    fake_html = "<html><body>" + "".join(
        f"<div class='field-{k}'>{v}</div>" for k, v in data.items() if v
    ) + "</body></html>"
    ctx.page_cache[fetch_id] = fake_html

    preview = json.dumps(data, indent=2)
    metadata = {
        "url": url,
        "final_url": url,
        "title": data.get("name") or "Social profile",
        "page_kind": "profile",
        "preview": preview,
    }
    ctx.fetch_metadata[fetch_id] = metadata
    ctx.url_to_fetch_id[_normalize_url(url)] = fetch_id
    return {"fetch_id": fetch_id, **metadata, "extracted_data": data}


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
    base_url = metadata.get("final_url") or metadata.get("url")
    effective_same_domain = same_domain_only or bool(ctx.target_domain)
    links = extract_links(
        html,
        base_url,
        selector=selector,
        limit=max(1, min(int(limit), 50)),
        same_domain_only=effective_same_domain,
    )
    return {"links": links, "count": len(links)}


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
    ctx.processed_fetch_ids.add(fetch_id)
    selector_map = fields or _selector_map_for_fetch(fetch_id, field_names, ctx)
    if not selector_map:
        return {"error": "No extractable field selectors available for this page."}
    result = parse_fields(html, selector_map)
    result = _postprocess_extracted_fields(fetch_id, result, field_names, ctx)
    ctx.parsed_results[fetch_id] = result
    logger.debug("parse_html: %s → %s", fetch_id, result)
    return {"fields": result}


async def _tool_save_result(url: str, data: dict, ctx: ToolContext) -> dict:
    viable, reason = _is_minimally_viable_lead(data, url)
    if not viable:
        ctx.rejected_weak_count += 1
        logger.info("Rejected weak lead: %s — %s", url, reason)
        print(f"  • Weak lead rejected: {data.get('name') or url} ({reason})", flush=True)
        return {"status": "rejected", "url": url, "reason": reason}

    try:
        status = await ctx.sheets_writer.append_row(url, data)
        if status == "saved":
            _record_domain_save(url, ctx)
            print(f"  ✓ Saved: {data.get('name') or url}", flush=True)
        else:
            print(f"  • Duplicate skipped: {data.get('name') or url}", flush=True)
        return {"status": status, "url": url}
    except Exception as exc:
        logger.error("save_result failed for %s: %s", url, exc)
        return {"error": str(exc), "url": url}


def _tool_fail_url(url: str, reason: str, ctx: ToolContext) -> dict:
    logger.info("fail_url: %s — %s", url, reason)
    ctx.failed_url_flag = True
    ctx.failed_urls.append({"url": url, "reason": reason})
    fetch_id = ctx.url_to_fetch_id.get(_normalize_url(url))
    if fetch_id:
        ctx.processed_fetch_ids.add(fetch_id)
    _record_domain_failure(url, reason, ctx)
    return {"status": "failed", "url": url, "reason": reason}


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme and parsed.path:
        parsed = urlparse(f"https://{url}")
    normalized = parsed._replace(fragment="")
    return normalized.geturl()


def _domain_for_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme and parsed.path:
        parsed = urlparse(f"https://{url}")
    return parsed.netloc.lower().split(":", 1)[0].removeprefix("www.")


def _same_or_subdomain(host: str, target_host: str) -> bool:
    host = host.removeprefix("www.")
    target_host = target_host.removeprefix("www.")
    return host == target_host or host.endswith(f".{target_host}")


def _is_broad_web_mode(ctx: ToolContext) -> bool:
    """Return True for website=NA web-only runs that need curated target selection."""
    return (
        ctx.source_mode == "web"
        and str(ctx.client_config.get("website", "NA")).upper() == "NA"
        and ctx.target_domain is None
    )


def _broad_mode_rejection(url: str, normalized_url: str, domain: str, ctx: ToolContext) -> str | None:
    """Return an error message when broad-mode fetch gating should reject a URL."""
    if not _is_broad_web_mode(ctx):
        return None

    if not ctx.suggest_targets_called:
        return (
            "Broad web mode requires curated starter targets. Call suggest_targets first before "
            "fetching pages."
        )

    if domain in ctx.domain_outcomes and ctx.domain_outcomes[domain].banned_for_run:
        return (
            f"Domain '{domain}' has been banned for this run due to blocked or low-yield pages. "
            "Use another suggested target."
        )

    if domain not in ctx.allowed_domains:
        return (
            f"URL domain '{domain}' is outside the curated target pool for this run. "
            "Use suggest_targets output instead of inventing new domains."
        )

    denied_reason = _broad_mode_denied_url(url)
    if denied_reason:
        return denied_reason

    return None


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


def _curated_target_pool_exhausted(ctx: ToolContext) -> bool:
    """Return True when all suggested starter targets have been consumed or banned."""
    if not _is_broad_web_mode(ctx) or not ctx.suggest_targets_called:
        return False

    for normalized_url in ctx.suggested_target_urls:
        domain = _domain_for_url(normalized_url)
        outcome = ctx.domain_outcomes.get(domain)
        if normalized_url not in ctx.url_to_fetch_id and not (outcome and outcome.banned_for_run):
            return False

    return not any(fetch_id not in ctx.processed_fetch_ids for fetch_id in ctx.fetch_metadata)


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


def _has_meaningful_value(value: Any) -> bool:
    """Return True when a field value is present enough to count as supporting lead data."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null", "n/a", "na"}
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _is_minimally_viable_lead(data: dict[str, Any], source_url: str) -> tuple[bool, str]:
    """Return whether a lead is strong enough to count toward the run target."""
    if not _is_plausible_person_name(data.get("name")):
        return False, "missing a plausible person identifier in name"

    support_fields = ("job_title", "company", "email", "phone")
    if any(_has_meaningful_value(data.get(field)) for field in support_fields):
        return True, ""

    social_media = data.get("social_media")
    if _has_meaningful_value(social_media):
        normalized_social = _normalize_url(str(social_media))
        normalized_source = _normalize_url(source_url)
        if normalized_social != normalized_source:
            return True, ""

    return False, "missing supporting data (job_title, company, email, phone, or distinct social_media)"


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

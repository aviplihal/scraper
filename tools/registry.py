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
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from tools.discovery import build_preview, classify_page, extract_links
from tools.fetcher import smart_fetch
from tools.parser import parse_fields

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


# ---------------------------------------------------------------------------
# Tool definitions (Ollama / OpenAI format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
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
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for the failure.",
                    },
                },
                "required": ["url", "reason"],
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
        return await _tool_save_result(arguments["url"], arguments.get("data", {}), ctx)
    elif tool_name == "fail_url":
        return _tool_fail_url(arguments["url"], arguments.get("reason", ""), ctx)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

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
    logger.debug("parse_html: %s → %s", fetch_id, result)
    return {"fields": result}


async def _tool_save_result(url: str, data: dict, ctx: ToolContext) -> dict:
    if not _is_plausible_person_name(data.get("name")):
        return {"error": "Lead must include a plausible person name before saving.", "url": url}

    try:
        status = await ctx.sheets_writer.append_row(url, data)
        if status == "saved":
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

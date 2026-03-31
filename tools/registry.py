"""Tool registry: Ollama tool-call definitions + async dispatcher.

Four tools are exposed to the agent:
  1. fetch_page     — fetch a URL (auto-routes social media to human emulator)
  2. parse_html     — extract fields from previously fetched HTML via CSS selectors
  3. save_result    — write a lead row to the client's Google Sheet
  4. fail_url       — log a URL failure and skip it

A ToolContext object is passed to every tool so they can share state
(browser, page cache, human emulator, sheets writer) without globals.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from tools.fetcher import smart_fetch
from tools.parser import parse_fields

logger = logging.getLogger(__name__)

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
    sheets_writer:  Any                              # sheets.writer.SheetsWriter
    scraper_browser: Any | None      = None          # tools.browser.ScraperBrowser
    emulator_browser: Any | None     = None          # human_emulator.browser.EmulatorBrowser
    emulator_state:  Any | None      = None          # human_emulator.state.EmulatorState
    page_cache:      dict[str, str]  = field(default_factory=dict)   # fetch_id → HTML
    failed_url_flag: bool            = False          # set by fail_url to break the step loop
    _logged_sites_chosen: list[str]  = field(default_factory=list)   # for website=NA logging
    fetch_count:     int             = 0
    fetch_error_count: int           = 0
    tool_call_count: int             = 0
    failed_urls:     list[dict[str, str]] = field(default_factory=list)


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
                "Returns a fetch_id that must be passed to parse_html, plus a 3000-character text preview."
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
                    "fields": {
                        "type": "object",
                        "description": (
                            "Mapping of field_name to CSS selector string. "
                            "Example: {\"name\": \"h1.profile-name\", \"email\": \"a[href^='mailto:']\"}."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["fetch_id", "fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_result",
            "description": (
                "Save a lead result to the client's Google Sheet. "
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
    elif tool_name == "parse_html":
        return _tool_parse_html(arguments["fetch_id"], arguments.get("fields", {}), ctx)
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
    ctx.fetch_count += 1

    if _is_social_media(url):
        result = await _fetch_social_media(url, ctx)
        if "error" in result:
            ctx.fetch_error_count += 1
        return result

    # Log site if website=NA (agent chose this target itself)
    if ctx.client_config.get("website", "NA").upper() == "NA":
        if url not in ctx._logged_sites_chosen:
            ctx._logged_sites_chosen.append(url)
            logger.info("Agent-chosen target site: %s", url)

    try:
        html = await smart_fetch(
            url,
            needs_javascript,
            get_context=ctx.scraper_browser.new_context,
        )
    except Exception as exc:
        logger.warning("fetch_page error for %s: %s", url, exc)
        ctx.fetch_error_count += 1
        return {"error": str(exc), "url": url}

    fetch_id = str(uuid.uuid4())
    ctx.page_cache[fetch_id] = html

    # Build a text preview from the first 3000 characters of visible text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    text_preview = soup.get_text(separator=" ", strip=True)[:3000]

    return {"fetch_id": fetch_id, "url": url, "preview": text_preview}


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
    return {"fetch_id": fetch_id, "url": url, "preview": preview, "extracted_data": data}


def _tool_parse_html(fetch_id: str, fields: dict[str, str], ctx: ToolContext) -> dict:
    html = ctx.page_cache.get(fetch_id)
    if not html:
        return {"error": f"No cached HTML for fetch_id '{fetch_id}'. Call fetch_page first."}
    result = parse_fields(html, fields)
    logger.debug("parse_html: %s → %s", fetch_id, result)
    return {"fields": result}


async def _tool_save_result(url: str, data: dict, ctx: ToolContext) -> dict:
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
    return {"status": "failed", "url": url, "reason": reason}

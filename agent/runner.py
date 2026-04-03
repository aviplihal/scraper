"""Job runner — sets up shared context and dispatches to the agent loop.

Each source mode (web, human_emulator, all) wires up the appropriate
browser and emulator instances before running the agent loop.
"""

import logging
from urllib.parse import urlparse

from agent.loop import run_agent_loop
from human_emulator.browser import EmulatorBrowser
from human_emulator.platforms import adapter_for_platform
from human_emulator.state import EmulatorState
from storage.writer import StorageWriter
# To switch back to Google Sheets, replace the line above with:
#   from sheets.writer import SheetsWriter
from tools.browser import ScraperBrowser
from tools.registry import ToolContext

logger = logging.getLogger(__name__)


async def run_job(config: dict, source: str) -> None:
    """Entry point called by run_job.py."""
    client_id = config["client_id"]

    writer = StorageWriter(client_id)
    # To switch back to Google Sheets, replace the line above with:
    #   writer = SheetsWriter(config["sheet_id"])

    if source in ("web", "all"):
        await _run_web(config, source, writer)
    elif source == "human_emulator":
        await _run_emulator(config, writer)


async def _run_web(config: dict, source: str, writer: StorageWriter) -> None:
    """Run the agent with the web-scraper browser active."""
    client_id = config["client_id"]
    scraper   = ScraperBrowser()
    await scraper.start()

    # Set up the human emulator if source == all (so social-media routing works)
    emulator_browser = None
    emulator_state   = None
    if source == "all":
        emulator_browser = EmulatorBrowser(client_id)
        await emulator_browser.start()
        emulator_state = EmulatorState(client_id, _social_platforms_for_config(config))

    ctx = ToolContext(
        client_config    = config,
        sheets_writer    = writer,
        source_mode      = source,
        target_domain    = _target_domain_for_config(config),
        scraper_browser  = scraper,
        emulator_browser = emulator_browser,
        emulator_state   = emulator_state,
    )

    try:
        if source == "all":
            await _preflight_social_platforms(config, ctx)
        run_result = await run_agent_loop(config, source, ctx)
    finally:
        await scraper.close()
        if emulator_browser:
            await emulator_browser.close()

    _print_run_summary(config, source, writer, ctx, run_result)


async def _run_emulator(config: dict, writer: StorageWriter) -> None:
    """Run a human-emulator-only job: process queued social-media profiles."""
    client_id        = config["client_id"]
    platforms        = _social_platforms_for_config(config)
    emulator_browser = EmulatorBrowser(client_id)
    await emulator_browser.start()
    emulator_state   = EmulatorState(client_id, platforms)

    ctx = ToolContext(
        client_config    = config,
        sheets_writer    = writer,
        source_mode      = "human_emulator",
        target_domain    = _target_domain_for_config(config),
        emulator_browser = emulator_browser,
        emulator_state   = emulator_state,
    )

    try:
        if not platforms:
            print("[runner] No social platforms are enabled for this client. Add 'social_platforms' to the config.")
            run_result = _noop_run_result("No social platforms were enabled for this client.")
            _print_run_summary(config, "human_emulator", writer, ctx, run_result)
            return

        await _preflight_social_platforms(config, ctx)
        if not _has_active_social_platform(ctx, platforms):
            print("[runner] No enabled social platforms are currently active. Check credentials or pause state.")
            run_result = _noop_run_result("No enabled social platforms were active for this run.")
            _print_run_summary(config, "human_emulator", writer, ctx, run_result)
            return

        # Run the agent loop — it will call fetch_page with social-media URLs
        # which are auto-routed to the human emulator via the tool dispatcher
        run_result = await run_agent_loop(config, "human_emulator", ctx)
    finally:
        await emulator_browser.close()

    _print_run_summary(config, "human_emulator", writer, ctx, run_result)


def _print_run_summary(
    config: dict,
    source: str,
    writer: StorageWriter,
    ctx: ToolContext,
    run_result: dict,
) -> None:
    """Print a concise end-of-run summary with the output location."""
    summary_status = _derive_summary_status(config, writer, run_result)
    lead_target = int(config["min_leads"])
    target_reached = writer.saved_count >= lead_target

    print("\n=== Job Summary ===")
    print(f"Status     : {summary_status}")
    print(f"Client     : {config['client_id']}")
    print(f"Source     : {source}")
    print(f"Output DB  : {writer.db_path}")
    print(f"Lead target: {lead_target}")
    print(f"Steps run  : {run_result['steps_run']}")
    print(f"Tool calls : {ctx.tool_call_count}")
    print(f"Pages tried: {ctx.fetch_count}")
    print(f"Fetch errs : {ctx.fetch_error_count}")
    print(f"Viable saved: {writer.saved_count}")
    print(f"Rejected weak: {ctx.rejected_weak_count}")
    print(f"Duplicates : {writer.duplicate_count}")
    print(f"Target reached: {'yes' if target_reached else 'no'}")
    print(f"Failed URLs: {len(ctx.failed_urls)}")
    print(f"Why stop   : {run_result['stop_reason']}")

    social_platforms = _social_platforms_for_config(config)
    if social_platforms and ctx.emulator_state is not None:
        print("\nSocial platforms:")
        for platform in social_platforms:
            snapshot = ctx.emulator_state.platform_summary(platform)
            status = snapshot["status"]
            reason = snapshot["reason"]
            visits = snapshot["visits_today"]
            line = f"  - {platform}: {status} ({visits} visited today)"
            if reason:
                line += f" — {reason}"
            if snapshot["paused_until"]:
                line += f" until {snapshot['paused_until']}"
            print(line)

    if ctx._logged_sites_chosen:
        print("\nSites chosen:")
        for url in ctx._logged_sites_chosen[:10]:
            print(f"  - {url}")

    print("\nOutput:")
    if writer.saved_rows:
        for idx, row in enumerate(writer.saved_rows[:10], start=1):
            label = row["name"] or row["source_url"]
            details = " | ".join(
                part for part in [row.get("job_title"), row.get("company")] if part
            )
            if details:
                print(f"  {idx}. {label} | {details}")
            else:
                print(f"  {idx}. {label}")
            print(f"     {row['source_url']}")
    else:
        print("  No new leads were saved in this run.")

    if ctx.failed_urls:
        print("\nFailures:")
        for item in ctx.failed_urls[:5]:
            print(f"  - {item['reason']}: {item['url']}")

    unprocessed = [
        ctx.fetch_metadata[fetch_id]
        for fetch_id in ctx.fetch_metadata
        if fetch_id not in ctx.processed_fetch_ids
    ]
    if unprocessed:
        print("\nUnprocessed pages:")
        for item in unprocessed[:5]:
            print(f"  - {item['page_kind']}: {item['final_url']}")

    print("\nLook here after each run:")
    print(f"  - Terminal summary above")
    print(f"  - Database file: {writer.db_path}")


def _derive_summary_status(config: dict, writer: StorageWriter, run_result: dict) -> str:
    """Map raw run results to a quick human-readable status."""
    if run_result["status"] == "error":
        return "ERROR"
    if writer.saved_count >= int(config["min_leads"]):
        return "SUCCESS"
    return "INCOMPLETE"


def _target_domain_for_config(config: dict) -> str | None:
    """Return the configured target domain, if the client pinned a website."""
    website = str(config.get("website", "NA")).strip()
    if not website or website.upper() == "NA":
        return None

    parsed = urlparse(website if "://" in website else f"https://{website}")
    host = parsed.netloc or parsed.path
    return host.lower().split(":", 1)[0].removeprefix("www.") or None


def _social_platforms_for_config(config: dict) -> list[str]:
    """Return normalized enabled social platforms for the client."""
    return [
        str(platform).strip().lower()
        for platform in config.get("social_platforms", [])
        if str(platform).strip()
    ]


async def _preflight_social_platforms(config: dict, ctx: ToolContext) -> None:
    """Validate social sessions and create adapter instances for active platforms."""
    if ctx.emulator_browser is None or ctx.emulator_state is None:
        return

    for platform in _social_platforms_for_config(config):
        adapter_cls = adapter_for_platform(platform)
        if adapter_cls is None:
            ctx.emulator_state.set_availability(platform, "unavailable", "Unsupported social platform.")
            continue

        context = await ctx.emulator_browser.get_context(platform)
        adapter = adapter_cls(context, ctx.emulator_state, config["client_id"])
        status, reason = await adapter.preflight()
        ctx.emulator_state.set_availability(platform, status, reason)
        if status == "active":
            ctx.social_adapters[platform] = adapter


def _has_active_social_platform(ctx: ToolContext, platforms: list[str]) -> bool:
    """Return True if at least one enabled social platform is active."""
    if ctx.emulator_state is None:
        return False
    return any(
        ctx.emulator_state.availability(platform)["status"] == "active"
        for platform in platforms
    )


def _noop_run_result(stop_reason: str) -> dict[str, object]:
    """Build a no-op run result for early exits that still want a summary."""
    return {
        "status": "done",
        "steps_run": 0,
        "stop_reason": stop_reason,
    }

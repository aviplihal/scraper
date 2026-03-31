"""Job runner — sets up shared context and dispatches to the agent loop.

Each source mode (web, human_emulator, all) wires up the appropriate
browser and emulator instances before running the agent loop.
"""

import logging

from agent.loop import run_agent_loop
from human_emulator.browser import EmulatorBrowser
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
        emulator_state = EmulatorState(client_id)

    ctx = ToolContext(
        client_config    = config,
        sheets_writer    = writer,
        scraper_browser  = scraper,
        emulator_browser = emulator_browser,
        emulator_state   = emulator_state,
    )

    try:
        run_result = await run_agent_loop(config, source, ctx)
    finally:
        await scraper.close()
        if emulator_browser:
            await emulator_browser.close()

    _print_run_summary(config, source, writer, ctx, run_result)


async def _run_emulator(config: dict, writer: StorageWriter) -> None:
    """Run a human-emulator-only job: process queued social-media profiles."""
    client_id        = config["client_id"]
    emulator_browser = EmulatorBrowser(client_id)
    context          = await emulator_browser.start()
    emulator_state   = EmulatorState(client_id)

    ctx = ToolContext(
        client_config    = config,
        sheets_writer    = writer,
        emulator_browser = emulator_browser,
        emulator_state   = emulator_state,
    )

    try:
        if emulator_state.queue_exhausted():
            print("[runner] Human emulator queue is empty. Add profile URLs to the state file.")
            print(f"         State file: state/{client_id}_state.json")
            print("         Add URLs to the 'profiles_queue' array and re-run.")
            return

        if emulator_state.daily_budget_exhausted():
            print(f"[runner] Daily visit budget exhausted ({emulator_state.visits_today} visits today). "
                  "Re-run tomorrow.")
            return

        paused, until = emulator_state.is_paused()
        if paused:
            print(f"[runner] Emulator is paused until {until}. Re-run after that time.")
            return

        print(f"[runner] Human emulator queue: {len(emulator_state._data['profiles_queue'])} total, "
              f"position {emulator_state._data['current_position']}, "
              f"{emulator_state.visits_today}/{75} visited today.")

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
    summary_status = _derive_summary_status(writer, run_result)

    print("\n=== Job Summary ===")
    print(f"Status     : {summary_status}")
    print(f"Client     : {config['client_id']}")
    print(f"Source     : {source}")
    print(f"Output DB  : {writer.db_path}")
    print(f"Steps run  : {run_result['steps_run']}")
    print(f"Tool calls : {ctx.tool_call_count}")
    print(f"Pages tried: {ctx.fetch_count}")
    print(f"Fetch errs : {ctx.fetch_error_count}")
    print(f"Saved new  : {writer.saved_count}")
    print(f"Duplicates : {writer.duplicate_count}")
    print(f"Failed URLs: {len(ctx.failed_urls)}")
    print(f"Why stop   : {run_result['stop_reason']}")

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

    print("\nLook here after each run:")
    print(f"  - Terminal summary above")
    print(f"  - Database file: {writer.db_path}")


def _derive_summary_status(writer: StorageWriter, run_result: dict) -> str:
    """Map raw run results to a quick human-readable status."""
    if run_result["status"] == "error":
        return "ERROR"
    if writer.saved_count > 0:
        return "SUCCESS"
    if run_result["status"] == "max_steps":
        return "INCOMPLETE"
    return "NO RESULTS"

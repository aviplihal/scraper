"""Job runner — sets up shared context and dispatches to the agent loop.

Each source mode (web, human_emulator, all) wires up the appropriate
browser and emulator instances before running the agent loop.
"""

import logging

from agent.loop import run_agent_loop
from human_emulator.browser import EmulatorBrowser
from human_emulator.state import EmulatorState
from sheets.writer import SheetsWriter
from tools.browser import ScraperBrowser
from tools.registry import ToolContext

logger = logging.getLogger(__name__)


async def run_job(config: dict, source: str) -> None:
    """Entry point called by run_job.py."""
    client_id = config["client_id"]
    sheet_id  = config["sheet_id"]

    sheets_writer = SheetsWriter(sheet_id)

    if source in ("web", "all"):
        await _run_web(config, source, sheets_writer)
    elif source == "human_emulator":
        await _run_emulator(config, sheets_writer)


async def _run_web(config: dict, source: str, sheets_writer: SheetsWriter) -> None:
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
        sheets_writer    = sheets_writer,
        scraper_browser  = scraper,
        emulator_browser = emulator_browser,
        emulator_state   = emulator_state,
    )

    try:
        await run_agent_loop(config, source, ctx)
    finally:
        await scraper.close()
        if emulator_browser:
            await emulator_browser.close()


async def _run_emulator(config: dict, sheets_writer: SheetsWriter) -> None:
    """Run a human-emulator-only job: process queued social-media profiles."""
    client_id        = config["client_id"]
    emulator_browser = EmulatorBrowser(client_id)
    context          = await emulator_browser.start()
    emulator_state   = EmulatorState(client_id)

    ctx = ToolContext(
        client_config    = config,
        sheets_writer    = sheets_writer,
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
        await run_agent_loop(config, "human_emulator", ctx)
    finally:
        await emulator_browser.close()

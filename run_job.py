#!/usr/bin/env python3
"""CLI entry point for the lead-generation platform.

Usage:
    python run_job.py --client <client_id> --source <web|human_emulator|all>

The script loads the client config from clients/<client_id>/config.json,
kicks off the agent loop, and streams all output to the terminal.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from agent.runner import run_job  # noqa: E402 — import after dotenv


def _validate_config(config: dict, client_arg: str) -> None:
    """Validate required runtime config before starting a job."""
    if config.get("client_id") != client_arg:
        print(
            f"Warning: config client_id '{config.get('client_id')}' "
            f"does not match --client '{client_arg}'.",
        )

    min_leads = config.get("min_leads")
    if isinstance(min_leads, bool) or not isinstance(min_leads, int) or min_leads < 1:
        raise ValueError(
            "client config must include a positive integer 'min_leads' "
            "so the job knows how many viable leads to target before stopping."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead generation job runner")
    parser.add_argument(
        "--client",
        required=True,
        help="Client ID (must match a directory under clients/)",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["web", "human_emulator", "all"],
        help="Source module to run",
    )
    args = parser.parse_args()

    config_path = Path(f"clients/{args.client}/config.json")
    if not config_path.exists():
        print(
            f"Error: client config not found at '{config_path}'\n"
            f"Create the file and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    try:
        _validate_config(config, args.client)
        asyncio.run(run_job(config, args.source))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[run_job] Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[run_job] Fatal error: {exc}", file=sys.stderr)
        logging.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()

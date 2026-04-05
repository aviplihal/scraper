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
from human_emulator.platforms import supported_social_platforms  # noqa: E402
from source_state import SOURCE_ACCURACY_PRESETS, seed_approved_sources_from_config  # noqa: E402


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

    social_platforms = config.get("social_platforms", [])
    if social_platforms is None:
        social_platforms = []
    if not isinstance(social_platforms, list):
        raise ValueError("client config 'social_platforms' must be a list of platform names if provided.")

    supported = set(supported_social_platforms())
    normalized = []
    for platform in social_platforms:
        if not isinstance(platform, str) or not platform.strip():
            raise ValueError("client config 'social_platforms' must contain non-empty platform names.")
        normalized_name = platform.strip().lower()
        if normalized_name not in supported:
            raise ValueError(
                f"Unsupported social platform '{platform}'. Supported values are: {', '.join(sorted(supported))}."
            )
        normalized.append(normalized_name)
    config["social_platforms"] = list(dict.fromkeys(normalized))

    approved_sources = config.get("approved_sources")
    if approved_sources is not None and not isinstance(approved_sources, dict):
        raise ValueError("client config 'approved_sources' must be an object with 'web_domains' and 'social_platforms'.")

    seeded_sources = seed_approved_sources_from_config(config)
    normalized_approved = {
        "web_domains": list(dict.fromkeys(seeded_sources["web_domains"])),
        "social_platforms": list(dict.fromkeys(seeded_sources["social_platforms"])),
    }
    for platform in normalized_approved["social_platforms"]:
        if platform not in supported:
            raise ValueError(
                f"Unsupported social platform '{platform}' in approved_sources. "
                f"Supported values are: {', '.join(sorted(supported))}."
            )
    config["approved_sources"] = normalized_approved

    combined_social = config["social_platforms"] + normalized_approved["social_platforms"]
    config["social_platforms"] = list(dict.fromkeys(combined_social))

    source_accuracy = str(config.get("source_accuracy", "balanced")).strip().lower() or "balanced"
    if source_accuracy not in SOURCE_ACCURACY_PRESETS:
        raise ValueError(
            "client config 'source_accuracy' must be one of: "
            + ", ".join(sorted(SOURCE_ACCURACY_PRESETS))
            + "."
        )
    config["source_accuracy"] = source_accuracy


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

"""Registry for supported social-platform adapters."""

from __future__ import annotations

from human_emulator.instagram import InstagramAdapter
from human_emulator.linkedin import LinkedInAdapter
from human_emulator.snapchat import SnapchatAdapter
from human_emulator.x import XAdapter

SOCIAL_ADAPTERS = {
    "linkedin": LinkedInAdapter,
    "x": XAdapter,
    "instagram": InstagramAdapter,
    "snapchat": SnapchatAdapter,
}


def adapter_for_platform(platform: str):
    """Return the adapter class for a configured platform name."""
    return SOCIAL_ADAPTERS.get(platform)


def adapter_for_url(url: str):
    """Return the first adapter class that matches the given URL."""
    for adapter in SOCIAL_ADAPTERS.values():
        if adapter.matches_url(url):
            return adapter
    return None


def supported_social_platforms() -> list[str]:
    """Return supported platform names."""
    return list(SOCIAL_ADAPTERS.keys())

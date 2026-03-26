"""Generic human-behaviour primitives shared by all platform modules.

Provides:
  - human_mouse_move   — curved Bezier path mouse movement
  - human_scroll       — natural multi-event scrolling
  - human_type         — character-by-character typing with variable delays
  - wait_reading_time  — pause proportional to visible page text length
  - session_delay      — inter-visit wait (30–100 seconds)
  - SessionRhythmManager — enforces active-period / break rhythm
"""

import asyncio
import logging
import math
import random
from datetime import datetime, timedelta, timezone

from playwright.async_api import Page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mouse movement
# ---------------------------------------------------------------------------

def _bezier_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate a cubic Bezier curve at parameter t ∈ [0, 1]."""
    mt = 1 - t
    x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
    y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
    return x, y


async def human_mouse_move(
    page: Page,
    target_x: float,
    target_y: float,
    from_x: float | None = None,
    from_y: float | None = None,
) -> None:
    """Move the mouse from its current position to (target_x, target_y) along a
    randomised cubic Bezier path that mimics natural hand movement."""
    vp = page.viewport_size or {"width": 1280, "height": 800}
    sx = from_x if from_x is not None else random.uniform(vp["width"] * 0.1, vp["width"] * 0.9)
    sy = from_y if from_y is not None else random.uniform(vp["height"] * 0.1, vp["height"] * 0.9)

    dx, dy = target_x - sx, target_y - sy
    dist = math.hypot(dx, dy)

    # Bow the control points perpendicular to the straight line
    perp_scale = random.uniform(0.1, 0.3) * dist
    angle = math.atan2(dy, dx) + math.pi / 2
    cp_offset = (math.cos(angle) * perp_scale, math.sin(angle) * perp_scale)

    cp1 = (sx + dx * 0.3 + cp_offset[0], sy + dy * 0.3 + cp_offset[1])
    cp2 = (sx + dx * 0.7 - cp_offset[0], sy + dy * 0.7 - cp_offset[1])

    steps = random.randint(25, 45)
    for i in range(steps + 1):
        t = i / steps
        # Ease-in-out: slow at start and end, faster in middle
        t_eased = t * t * (3 - 2 * t)
        px, py = _bezier_point((sx, sy), cp1, cp2, (target_x, target_y), t_eased)
        await page.mouse.move(px, py)
        await asyncio.sleep(random.uniform(0.003, 0.010))


# ---------------------------------------------------------------------------
# Scrolling
# ---------------------------------------------------------------------------

async def human_scroll(
    page: Page,
    direction: str = "down",
    total_px: int | None = None,
) -> None:
    """Scroll the page in multiple events with randomised distances and pauses.

    Args:
        direction: 'down' or 'up'
        total_px:  approximate total pixels to scroll; defaults to a random
                   amount between 300 and 900 px.
    """
    if total_px is None:
        total_px = random.randint(300, 900)

    sign = 1 if direction == "down" else -1
    scrolled = 0
    while scrolled < total_px:
        chunk = random.randint(60, 160)
        chunk = min(chunk, total_px - scrolled)
        await page.mouse.wheel(0, sign * chunk)
        scrolled += chunk
        await asyncio.sleep(random.uniform(0.08, 0.25))

    # Occasional micro-scroll back (like adjusting reading position)
    if random.random() < 0.25:
        back = random.randint(20, 60)
        await page.mouse.wheel(0, -sign * back)
        await asyncio.sleep(random.uniform(0.1, 0.3))


# ---------------------------------------------------------------------------
# Typing
# ---------------------------------------------------------------------------

async def human_type(page: Page, selector: str, text: str) -> None:
    """Type text into the element matching selector, character by character,
    with per-character delays between 50 and 180 ms."""
    element = page.locator(selector).first
    await element.click()
    await asyncio.sleep(random.uniform(0.2, 0.5))
    for char in text:
        await element.press(char)
        await asyncio.sleep(random.uniform(0.05, 0.18))


# ---------------------------------------------------------------------------
# Reading time
# ---------------------------------------------------------------------------

async def wait_reading_time(page: Page) -> None:
    """Wait a realistic amount of time based on the visible text length.

    Models a user skimming at ~500 words per minute with jitter, clamped
    between 1.5 and 20 seconds.
    """
    try:
        text = await page.evaluate("document.body.innerText")
    except Exception:
        text = ""
    word_count = len((text or "").split())
    # Skim reading: 400–600 wpm
    wpm = random.uniform(400, 600)
    seconds = (word_count / wpm) * 60
    seconds = max(1.5, min(20.0, seconds))
    seconds += random.uniform(0.0, 2.0)
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# Inter-visit delay
# ---------------------------------------------------------------------------

async def session_delay() -> None:
    """Wait between 30 and 100 seconds — the inter-visit gap for the emulator."""
    delay = random.uniform(30, 100)
    logger.debug("Inter-visit delay: %.1f s", delay)
    await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Session rhythm manager
# ---------------------------------------------------------------------------

# Active session: 1–2 hours; break: 15–45 minutes
_SESSION_MIN = 60 * 60        # 1 hour
_SESSION_MAX = 2 * 60 * 60    # 2 hours
_BREAK_MIN   = 15 * 60        # 15 minutes
_BREAK_MAX   = 45 * 60        # 45 minutes


class SessionRhythmManager:
    """Manages active-period / break cycles to mimic a human workday.

    Call :meth:`maybe_take_break` before each visit. It will sleep for the
    break duration if an active session has expired.
    """

    def __init__(self) -> None:
        self._session_start = datetime.now(timezone.utc)
        self._session_duration = timedelta(seconds=random.uniform(_SESSION_MIN, _SESSION_MAX))
        logger.info(
            "Session rhythm: active for %.0f min before first break",
            self._session_duration.total_seconds() / 60,
        )

    async def maybe_take_break(self) -> None:
        now = datetime.now(timezone.utc)
        if now - self._session_start >= self._session_duration:
            break_sec = random.uniform(_BREAK_MIN, _BREAK_MAX)
            logger.info(
                "Session rhythm: taking a %.0f-min break.", break_sec / 60
            )
            await asyncio.sleep(break_sec)
            # Start a fresh session
            self._session_start = datetime.now(timezone.utc)
            self._session_duration = timedelta(
                seconds=random.uniform(_SESSION_MIN, _SESSION_MAX)
            )
            logger.info(
                "Session rhythm: resuming — active for %.0f min",
                self._session_duration.total_seconds() / 60,
            )

    async def pace_variation(self) -> None:
        """Occasional short micro-pause within a session (0–3 seconds)."""
        if random.random() < 0.15:
            await asyncio.sleep(random.uniform(1, 3))

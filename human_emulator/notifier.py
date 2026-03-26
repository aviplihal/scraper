"""Telegram alert sender — plain httpx, no extra dependencies."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


async def send_alert(message: str) -> None:
    """Send a Telegram message. Silently logs on failure so it never crashes the job."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram credentials not configured — skipping alert: %s", message)
        return
    url = TELEGRAM_API.format(token=token)
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                logger.warning("Telegram API returned %s: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("Failed to send Telegram alert: %s", exc)

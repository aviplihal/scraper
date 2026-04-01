"""BeautifulSoup HTML parser used by the parse_html agent tool."""

import logging
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_fields(html: str, fields: dict[str, str | list[str]]) -> dict[str, Any]:
    """Extract fields from HTML using CSS selectors.

    Args:
        html:   Full HTML string of the page.
        fields: Mapping of field_name → CSS selector string or ordered selector list.

    Returns:
        Dict of field_name → extracted text (or None if not found).
    """
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, Any] = {}

    for field_name, selector_spec in fields.items():
        value = None
        selectors = selector_spec if isinstance(selector_spec, list) else [selector_spec]
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    value = _extract_element_value(element)
                    if value:
                        break
            except Exception as exc:
                logger.debug("CSS selector '%s' failed: %s", selector, exc)
        result[field_name] = value

    return result


def _extract_element_value(element) -> str | None:
    """Return a useful string value from a matched element."""
    if element.name == "a" and element.get("href"):
        href = element["href"]
        if isinstance(href, str) and href.startswith("mailto:"):
            return href[len("mailto:"):]
        if isinstance(href, str) and href.startswith("tel:"):
            return href[len("tel:"):]
        return element.get_text(strip=True) or href

    return element.get_text(strip=True) or None

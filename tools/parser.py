"""BeautifulSoup HTML parser used by the parse_html agent tool."""

import logging
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_fields(html: str, fields: dict[str, str]) -> dict[str, Any]:
    """Extract fields from HTML using CSS selectors.

    Args:
        html:   Full HTML string of the page.
        fields: Mapping of field_name → CSS selector string.

    Returns:
        Dict of field_name → extracted text (or None if not found).
    """
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, Any] = {}

    for field_name, selector in fields.items():
        value = None
        try:
            element = soup.select_one(selector)
            if element:
                # For anchor tags prefer the href; otherwise use text content
                if element.name == "a" and element.get("href"):
                    href = element["href"]
                    # For mailto: links strip the scheme
                    if isinstance(href, str) and href.startswith("mailto:"):
                        value = href[len("mailto:"):]
                    elif isinstance(href, str) and href.startswith("tel:"):
                        value = href[len("tel:"):]
                    else:
                        value = element.get_text(strip=True) or href
                else:
                    value = element.get_text(strip=True) or None
        except Exception as exc:
            logger.debug("CSS selector '%s' failed: %s", selector, exc)
        result[field_name] = value

    return result

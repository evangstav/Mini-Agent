"""Web fetch tool — retrieve and extract text from URLs."""

import re
from typing import Any

import httpx

from .base import Tool, ToolResult

_MAX_OUTPUT = 25_000  # 25 KB cap
_TIMEOUT = 30


class WebFetchTool(Tool):
    """Fetch a URL and return its text content (HTML stripped)."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a URL and return its text content with HTML tags, scripts, "
            "and styles stripped. Output capped at 25KB."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str) -> ToolResult:
        """Fetch URL, strip HTML, return plain text."""
        if not url.strip():
            return ToolResult(success=False, error="URL cannot be empty.")

        if not url.startswith(("http://", "https://")):
            return ToolResult(success=False, error="URL must start with http:// or https://")

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MiniAgent/1.0)",
        }

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(url, headers=headers, timeout=_TIMEOUT)
                resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"Fetch failed: {e}")

        text = _html_to_text(resp.text)

        if len(text) > _MAX_OUTPUT:
            text = text[:_MAX_OUTPUT] + f"\n\n[... truncated at {_MAX_OUTPUT // 1000}KB]"

        if not text.strip():
            return ToolResult(success=True, content="(Page returned no extractable text.)")

        return ToolResult(success=True, content=text)


def _html_to_text(html: str) -> str:
    """Strip scripts, styles, and HTML tags; collapse whitespace."""
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Replace block elements with newlines
    text = re.sub(r"<(?:br|p|div|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

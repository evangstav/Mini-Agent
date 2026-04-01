"""Web search tool using DuckDuckGo HTML scraping."""

import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from .base import Tool, ToolResult

_MAX_RESULTS = 8
_TIMEOUT = 15


class WebSearchTool(Tool):
    """Search the web via DuckDuckGo HTML and return top results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using DuckDuckGo. Returns top results with "
            "title, URL, and snippet. No API key needed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Max results to return (default: {_MAX_RESULTS}).",
                    "default": _MAX_RESULTS,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_results: int = _MAX_RESULTS) -> ToolResult:
        """Execute web search via DuckDuckGo HTML."""
        if not query.strip():
            return ToolResult(success=False, error="Query cannot be empty.")

        max_results = max(1, min(max_results, 20))
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MiniAgent/1.0)",
        }

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(url, headers=headers, timeout=_TIMEOUT)
                resp.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"Search request failed: {e}")

        results = _parse_results(resp.text, max_results)
        if not results:
            return ToolResult(success=True, content="No results found.")

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            lines.append(f"   {r['snippet']}")
            lines.append("")

        return ToolResult(success=True, content="\n".join(lines).strip())


def _parse_results(html: str, max_results: int) -> list[dict[str, str]]:
    """Extract search results from DuckDuckGo HTML response."""
    results: list[dict[str, str]] = []

    # DuckDuckGo HTML results are in <a class="result__a"> with snippets in
    # <a class="result__snippet">
    link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title_html) in enumerate(links[:max_results]):
        title = _strip_tags(title_html).strip()
        snippet = _strip_tags(snippets[i]).strip() if i < len(snippets) else ""
        url = _extract_url(href)
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results


def _extract_url(href: str) -> str:
    """Extract actual URL from DuckDuckGo redirect link."""
    # DDG wraps URLs: //duckduckgo.com/l/?uddg=<encoded_url>&...
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        from urllib.parse import unquote
        return unquote(match.group(1))
    # Direct URL
    if href.startswith("http"):
        return href
    return ""


def _strip_tags(html: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", html)

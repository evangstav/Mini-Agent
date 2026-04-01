"""Tests for WebSearchTool and WebFetchTool."""

import pytest

from mini_agent.tools.web_fetch import WebFetchTool, _html_to_text
from mini_agent.tools.web_search import WebSearchTool, _parse_results, _strip_tags


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_schema(self):
        tool = WebSearchTool()
        schema = tool.to_schema()
        assert schema["name"] == "web_search"
        assert "query" in schema["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = WebSearchTool()
        result = await tool.execute(query="  ")
        assert not result.success
        assert "empty" in result.error.lower()

    def test_parse_results_empty_html(self):
        results = _parse_results("<html><body></body></html>", 5)
        assert results == []

    def test_parse_results_with_content(self):
        html = """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">Example Title</a>
        <a class="result__snippet">This is a snippet.</a>
        """
        results = _parse_results(html, 5)
        assert len(results) == 1
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == "This is a snippet."

    def test_strip_tags(self):
        assert _strip_tags("<b>hello</b> <i>world</i>") == "hello world"
        assert _strip_tags("no tags") == "no tags"


class TestWebFetchTool:
    """Tests for WebFetchTool."""

    def test_schema(self):
        tool = WebFetchTool()
        schema = tool.to_schema()
        assert schema["name"] == "web_fetch"
        assert "url" in schema["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_empty_url(self):
        tool = WebFetchTool()
        result = await tool.execute(url="")
        assert not result.success
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_scheme(self):
        tool = WebFetchTool()
        result = await tool.execute(url="ftp://example.com")
        assert not result.success
        assert "http" in result.error.lower()

    def test_html_to_text_strips_scripts(self):
        html = "<p>Hello</p><script>alert('x')</script><p>World</p>"
        text = _html_to_text(html)
        assert "alert" not in text
        assert "Hello" in text
        assert "World" in text

    def test_html_to_text_strips_styles(self):
        html = "<style>.foo { color: red; }</style><p>Content</p>"
        text = _html_to_text(html)
        assert "color" not in text
        assert "Content" in text

    def test_html_to_text_entities(self):
        html = "&amp; &lt; &gt; &quot; &#39;"
        text = _html_to_text(html)
        assert "& < > \" '" == text

    def test_html_to_text_whitespace_collapse(self):
        html = "<p>Hello    World</p>"
        text = _html_to_text(html)
        assert "Hello World" in text

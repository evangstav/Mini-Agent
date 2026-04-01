"""Test cases for MCP tool loading."""

import json
import tempfile
from pathlib import Path

import pytest

from mini_agent.tools.mcp_loader import (
    MCPServerConnection,
    cleanup_mcp_connections,
    load_mcp_tools_async,
)


class TestMCPServerConnectionInit:
    """Tests for MCPServerConnection initialization."""

    def test_stdio_connection_init(self):
        conn = MCPServerConnection(
            name="test-stdio",
            connection_type="stdio",
            command="npx",
            args=["-y", "test-server"],
            env={"API_KEY": "test"},
        )
        assert conn.name == "test-stdio"
        assert conn.connection_type == "stdio"
        assert conn.command == "npx"
        assert conn.args == ["-y", "test-server"]

    def test_url_connection_init(self):
        conn = MCPServerConnection(
            name="test-url",
            connection_type="streamable_http",
            url="https://mcp.example.com/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert conn.name == "test-url"
        assert conn.connection_type == "streamable_http"
        assert conn.url == "https://mcp.example.com/mcp"

    def test_default_values(self):
        conn = MCPServerConnection(name="test-default")
        assert conn.connection_type == "stdio"
        assert conn.args == []
        assert conn.env == {}

    def test_timeout_overrides(self):
        conn = MCPServerConnection(
            name="test-timeout",
            connect_timeout=15.0,
            execute_timeout=90.0,
            sse_read_timeout=180.0,
        )
        assert conn.connect_timeout == 15.0
        assert conn.execute_timeout == 90.0
        assert conn.sse_read_timeout == 180.0


@pytest.mark.asyncio
async def test_url_config_validation():
    """Test that URL-based config without url is rejected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {"mcpServers": {"broken-sse": {"type": "sse"}}}
        json.dump(config, f)
        f.flush()

        try:
            tools = await load_mcp_tools_async(f.name)
            assert tools == []
        finally:
            await cleanup_mcp_connections()
            Path(f.name).unlink()


@pytest.mark.asyncio
async def test_stdio_config_validation():
    """Test that STDIO config without command is rejected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {"mcpServers": {"broken-stdio": {"type": "stdio"}}}
        json.dump(config, f)
        f.flush()

        try:
            tools = await load_mcp_tools_async(f.name)
            assert tools == []
        finally:
            await cleanup_mcp_connections()
            Path(f.name).unlink()


@pytest.mark.asyncio
async def test_disabled_servers():
    """Test that disabled servers are skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config = {
            "mcpServers": {
                "disabled-server": {"command": "npx", "args": ["-y", "test"], "disabled": True},
            }
        }
        json.dump(config, f)
        f.flush()

        try:
            tools = await load_mcp_tools_async(f.name)
            assert tools == []
        finally:
            await cleanup_mcp_connections()
            Path(f.name).unlink()


@pytest.mark.asyncio
async def test_nonexistent_config():
    """Test loading from nonexistent config file."""
    tools = await load_mcp_tools_async("nonexistent_config.json")
    assert tools == []

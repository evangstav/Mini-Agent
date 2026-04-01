"""MCP tool loader with real MCP client integration."""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .base import Tool, ToolResult

ConnectionType = Literal["stdio", "sse", "http", "streamable_http"]

# Default timeouts (seconds)
CONNECT_TIMEOUT = 10.0
EXECUTE_TIMEOUT = 60.0
SSE_READ_TIMEOUT = 120.0


class MCPTool(Tool):
    """Wrapper for MCP tools."""

    def __init__(self, name: str, description: str, parameters: dict[str, Any],
                 session: ClientSession, execute_timeout: float = EXECUTE_TIMEOUT):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._session = session
        self._execute_timeout = execute_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs) -> ToolResult:
        try:
            async with asyncio.timeout(self._execute_timeout):
                result = await self._session.call_tool(self._name, arguments=kwargs)

            content_parts = []
            for item in result.content:
                content_parts.append(item.text if hasattr(item, "text") else str(item))

            content_str = "\n".join(content_parts)
            is_error = getattr(result, "isError", False)
            return ToolResult(success=not is_error, content=content_str,
                              error="Tool returned error" if is_error else None)

        except TimeoutError:
            return ToolResult(success=False, error=f"MCP tool timed out after {self._execute_timeout}s")
        except Exception as e:
            return ToolResult(success=False, error=f"MCP tool failed: {e}")


class MCPServerConnection:
    """Manages connection to a single MCP server."""

    def __init__(self, name: str, connection_type: ConnectionType = "stdio",
                 command: str | None = None, args: list[str] | None = None,
                 env: dict[str, str] | None = None, url: str | None = None,
                 headers: dict[str, str] | None = None,
                 connect_timeout: float = CONNECT_TIMEOUT,
                 execute_timeout: float = EXECUTE_TIMEOUT,
                 sse_read_timeout: float = SSE_READ_TIMEOUT):
        self.name = name
        self.connection_type = connection_type
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.url = url
        self.headers = headers or {}
        self.connect_timeout = connect_timeout
        self.execute_timeout = execute_timeout
        self.sse_read_timeout = sse_read_timeout
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack | None = None
        self.tools: list[MCPTool] = []

    async def connect(self) -> bool:
        try:
            self.exit_stack = AsyncExitStack()
            async with asyncio.timeout(self.connect_timeout):
                if self.connection_type == "stdio":
                    # Augment parent env with configured env vars (don't replace)
                    merged_env = {**os.environ, **self.env} if self.env else None
                    params = StdioServerParameters(command=self.command, args=self.args,
                                                   env=merged_env)
                    read_stream, write_stream = await self.exit_stack.enter_async_context(
                        stdio_client(params))
                elif self.connection_type == "sse":
                    read_stream, write_stream = await self.exit_stack.enter_async_context(
                        sse_client(url=self.url, headers=self.headers or None,
                                   timeout=self.connect_timeout, sse_read_timeout=self.sse_read_timeout))
                else:
                    read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
                        streamablehttp_client(url=self.url, headers=self.headers or None,
                                              timeout=self.connect_timeout,
                                              sse_read_timeout=self.sse_read_timeout))

                session = await self.exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream))
                self.session = session
                await session.initialize()
                tools_list = await session.list_tools()

            for tool in tools_list.tools:
                self.tools.append(MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                    session=session,
                    execute_timeout=self.execute_timeout,
                ))

            logger.info("MCP '%s': loaded %d tools", self.name, len(self.tools))
            return True

        except Exception as e:
            logger.error("MCP '%s' failed: %s", self.name, e)
            if self.exit_stack:
                await self.exit_stack.aclose()
                self.exit_stack = None
            return False

    async def disconnect(self):
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
            except Exception:
                pass
            finally:
                self.exit_stack = None
                self.session = None


class MCPManager:
    """Manages MCP server connections with proper lifecycle.

    Replaces the global _mcp_connections list with instance-level state.
    """

    def __init__(self) -> None:
        self.connections: list[MCPServerConnection] = []

    async def load_tools(self, config_path: str = "mcp.json") -> list[Tool]:
        """Load MCP tools from config file.

        Supports STDIO and URL-based (SSE/HTTP) servers.
        Falls back to mcp-example.json if mcp.json not found.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            if config_file.name == "mcp.json":
                example = config_file.parent / "mcp-example.json"
                if example.exists():
                    config_file = example
                else:
                    return []
            else:
                return []

        try:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            mcp_servers = config.get("mcpServers", {})
            if not mcp_servers:
                return []

            all_tools: list[Tool] = []
            for server_name, sc in mcp_servers.items():
                if sc.get("disabled", False):
                    continue

                # Determine connection type
                conn_type: ConnectionType = sc.get("type", "").lower()  # type: ignore[assignment]
                if conn_type not in ("stdio", "sse", "http", "streamable_http"):
                    conn_type = "streamable_http" if sc.get("url") else "stdio"

                connection = MCPServerConnection(
                    name=server_name, connection_type=conn_type,
                    command=sc.get("command"), args=sc.get("args", []),
                    env=sc.get("env", {}), url=sc.get("url"),
                    headers=sc.get("headers", {}),
                    connect_timeout=sc.get("connect_timeout", CONNECT_TIMEOUT),
                    execute_timeout=sc.get("execute_timeout", EXECUTE_TIMEOUT),
                    sse_read_timeout=sc.get("sse_read_timeout", SSE_READ_TIMEOUT),
                )

                if await connection.connect():
                    self.connections.append(connection)
                    all_tools.extend(connection.tools)

            return all_tools

        except Exception as e:
            logger.error("Error loading MCP config: %s", e)
            return []

    async def cleanup(self) -> None:
        """Clean up all MCP connections."""
        for connection in self.connections:
            await connection.disconnect()
        self.connections.clear()


# Backward-compatible module-level functions using a default manager
_default_manager = MCPManager()


async def load_mcp_tools_async(config_path: str = "mcp.json") -> list[Tool]:
    """Load MCP tools from config file (backward-compatible wrapper)."""
    return await _default_manager.load_tools(config_path)


async def cleanup_mcp_connections() -> None:
    """Clean up all MCP connections (backward-compatible wrapper)."""
    await _default_manager.cleanup()

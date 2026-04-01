# Mini Agent

A minimal, readable agent harness in Python. ~2,000 lines of code you can read in an afternoon.

Built on the [MiniMax M2.5](https://www.minimax.io/) model with Anthropic-compatible API, but works with any Anthropic or OpenAI-compatible provider.

## What It Does

- **Agent loop**: Think → Act → Observe cycle with tool use
- **Streaming events**: `run_stream()` yields typed events (`TextChunk`, `ToolStart`, `ToolEnd`, etc.)
- **Tools**: Bash, Read/Write/Edit files, MCP integration
- **Context management**: Large tool results stored to disk, automatic compaction when context gets large
- **System prompt injection**: Auto-injects CLAUDE.md and git info
- **Two LLM providers**: Anthropic and OpenAI protocols
- **Permission gating**: Optional callback to approve/deny tool calls

## Quick Start

```bash
# Clone and install
git clone https://github.com/evangstav/Mini-Agent.git
cd Mini-Agent
uv sync  # or: pip install -e .

# Set your API key
export MINIMAX_API_KEY="your-key-here"
```

### Simple Usage

```python
import asyncio
from mini_agent import Agent, LLMClient
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.file_tools import ReadTool, WriteTool, EditTool

async def main():
    client = LLMClient(
        api_key="your-key",
        provider="anthropic",
        api_base="https://api.minimax.io",
        model="MiniMax-M2.5",
    )
    tools = [BashTool(), ReadTool(), WriteTool(), EditTool()]
    agent = Agent(client, "You are a helpful coding assistant.", tools)
    agent.add_user_message("List the files in this directory")
    result = await agent.run()
    print(result)

asyncio.run(main())
```

### Streaming Events

```python
from mini_agent import TextChunk, ToolStart, ToolEnd, AgentDone

async for event in agent.run_stream():
    if isinstance(event, TextChunk):
        print(event.content, end="")
    elif isinstance(event, ToolStart):
        print(f"\n  -> {event.tool_name}")
    elif isinstance(event, ToolEnd):
        print(f"  <- {'ok' if event.success else event.error}")
    elif isinstance(event, AgentDone):
        print(f"\nDone in {event.steps} steps.")
```

### With Context Management

```python
from mini_agent.context import ToolResultStore, SystemPromptBuilder

agent = Agent(
    llm_client=client,
    system_prompt="You are a helpful assistant.",
    tools=tools,
    tool_result_store=ToolResultStore(),   # Persist large results to disk
    context_window=200_000,                  # Total context window (tokens)
    compact_threshold_pct=0.85,              # Compact at 85% of context window
    project_dir=".",                        # Inject CLAUDE.md + git info
)
```

### Using Other Providers

```python
# Anthropic Claude
client = LLMClient(
    api_key="sk-ant-...",
    provider="anthropic",
    api_base="https://api.anthropic.com",
    model="claude-sonnet-4-6",
)

# OpenAI
client = LLMClient(
    api_key="sk-...",
    provider="openai",
    api_base="https://api.openai.com/v1",
    model="gpt-4o",
)
```

## Architecture

```
mini_agent/
├── agent.py          215 LOC  Core loop: run() + run_stream()
├── context.py        194 LOC  Tool result storage, compaction, system prompt
├── events.py          74 LOC  Event types for streaming
├── schema/            55 LOC  Message, ToolCall, LLMResponse
├── retry.py          138 LOC  Async retry with backoff
├── llm/
│   ├── base.py        84 LOC  Abstract LLM client
│   ├── llm_wrapper.py 127 LOC Provider router
│   ├── anthropic_client.py  307 LOC
│   └── openai_client.py     295 LOC
└── tools/
    ├── base.py         55 LOC  Tool + ToolResult
    ├── bash_tool.py    85 LOC  Shell execution
    ├── file_tools.py  137 LOC  Read/Write/Edit
    └── mcp_loader.py  212 LOC  MCP integration
```

## MCP Integration

Load tools from any MCP server:

```python
from mini_agent.tools.mcp_loader import load_mcp_tools

tools = await load_mcp_tools("path/to/mcp-config.json")
agent = Agent(client, "You are helpful.", tools + [BashTool()])
```

## Running Tests

```bash
uv run pytest tests/
```

## License

MIT
